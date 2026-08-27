# Benchmarks

This page compares AsterVec against embedded vector stores a Python
application could use in-process: Chroma, LanceDB, and sqlite-vec. All
runs happened on the same machine, in the same harness, with the same
HNSW parameters and stock settings for every engine.

**The short version:** at 100K vectors AsterVec is the fastest of the
three and uses the least memory. At 1M vectors it answers queries in a
fraction of the memory an in-RAM engine needs, at the cost of raw query
speed. And under a steady stream of inserts, updates and deletes, it
writes fastest and its memory stays flat, while Chroma's memory climbs
and LanceDB cannot sustain per-operation writes at all.

## Setup

| | |
|---|---|
| Machine | 2× Intel Xeon Silver 4314, 503 GB RAM, Samsung NVMe, Ubuntu 22.04 |
| AsterVec | 0.2.1, default configuration |
| Chroma | 1.5.9, `PersistentClient` |
| LanceDB | 0.25.3, `IVF_HNSW_SQ` index |
| sqlite-vec | 0.1.9, `vec0` virtual table |

Graph parameters are identical for every HNSW engine: M = 16, M0 = 32,
ef_construction = 200, k = 10. Each engine builds the index, answers
2,000 queries at every ef_search setting from 16 to 256, then takes
10,000 single-vector inserts. ef_search is the accuracy knob of graph search;
higher values examine more of the graph, which raises recall and costs
speed. Recall is measured against exact ground truth. The churn test is
separate and described below.

Datasets: SIFT-100K and SIFT-1M (128d, L2) and GloVe-1.2M (200d, cosine).

## Search speed and accuracy

SIFT-1M:

| ef | AsterVec recall | AsterVec QPS | Chroma recall | Chroma QPS |
|---:|---:|---:|---:|---:|
| 16 | 0.830 | 1,453 | 0.801 | 1,983 |
| 32 | 0.918 | 902 | 0.903 | 1,721 |
| 64 | 0.965 | 543 | 0.964 | 1,556 |
| 128 | 0.984 | 319 | 0.990 | 1,249 |
| 256 | 0.990 | 186 | 0.997 | 907 |

GloVe-1.2M, a harder dataset where every engine tops out lower:

| ef | AsterVec recall | AsterVec QPS | Chroma recall | Chroma QPS |
|---:|---:|---:|---:|---:|
| 16 | 0.512 | 790 | 0.483 | 1,690 |
| 64 | 0.688 | 288 | 0.696 | 1,261 |
| 128 | 0.759 | 165 | 0.775 | 964 |
| 256 | 0.814 | 93 | 0.833 | 670 |

SIFT-100K, with the full field:

| ef | AsterVec | Chroma | LanceDB | sqlite-vec |
|---:|---:|---:|---:|---:|
| 16 | 0.880 / 14,699 | 0.874 / 2,014 | 0.835 / 467 | — |
| 64 | 0.984 / 6,002 | 0.989 / 1,689 | 0.974 / 485 | — |
| 128 | 0.991 / 3,408 | 0.998 / 1,422 | 0.982 / 436 | 1.000 / 41 |
| 256 | 0.993 / 1,898 | 1.000 / 1,170 | 0.983 / 416 | — |

Cells are recall@10 / QPS. sqlite-vec is exact brute-force search, so its
recall is 1.0 by construction and ef does not apply.

The pattern is consistent. At 100K, AsterVec is the fastest engine at
every ef point. At 1M, Chroma is faster, because it holds the whole
index in RAM: a hop through the graph costs it a pointer dereference,
where AsterVec pays a cache lookup. What that query speed costs Chroma
in memory is the next section.

## Memory

Resident memory (RSS) of the whole process, measured while queries are
running, at ef 128:

| | AsterVec | Chroma |
|---|---:|---:|
| SIFT-100K | 0.17 GB | 0.31 GB |
| SIFT-1M | 0.78 GB | 1.71 GB |
| GloVe-1.2M | 1.34 GB | 2.80 GB |

These numbers include the benchmark harness itself, which holds the
dataset in memory in both processes, about 0.5 GB at SIFT-1M scale on
each side. Subtract it and the engines alone stand at roughly 0.3 GB
against 1.2 GB, a 4× gap, with the same ratio on GloVe. We checked that
nothing hides outside these numbers: neither engine spawns child
processes, uses shared memory, or touches swap. 97% of Chroma's memory
is plain heap, which is what an index held in RAM looks like.

AsterVec reads vectors through the operating system's file cache, so
its working data lives in memory the OS can hand to any other program
the moment it is needed. Chroma's index lives in heap, which the OS
cannot reclaim. That difference is what the table shows.

## Live updates

The churn test loads half the dataset, then runs ten rounds. Each round
applies operations amounting to 5% of the dataset, split 90% inserts,
5% updates, 5% deletes, and then measures recall at ef 64 against exact
ground truth recomputed for the current state.

SIFT-1M, round 1 → round 10:

| | AsterVec | Chroma |
|---|---|---|
| Recall@10 | 0.964 → 0.956 | 0.968 → 0.961 |
| Insert, per op | 3.1 → 4.6 ms | 5.9 → 6.1 ms |
| Search p50 | 1.4 → 2.0 ms | 0.7 → 0.7 ms |
| RSS | 0.86 → 0.96 GB | 1.38 → 1.79 GB |

Both engines hold recall essentially flat through sustained mutation.
AsterVec takes writes faster and its memory stays close to where it
started; Chroma searches faster and its memory keeps climbing, because
every insert grows the in-RAM index.

LanceDB does not appear in this table because it did not finish. Its
per-operation writes measured 0.92 inserts per second on SIFT-100K, and
the churn run exceeded a 5 GB disk budget before completing a single
round, because each write commits a new table version. LanceDB is built
for batch writes; writing one vector at a time is not its use case.
sqlite-vec takes writes quickly, 3,400 inserts per second at 100K, but
it scans the whole table on every query, so search is 24 ms at 150K
vectors and grows linearly from there.

Benchmark harness publication is planned; the parameters above are
sufficient to reproduce every table.
