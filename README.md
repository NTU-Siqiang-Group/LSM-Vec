# LSM-Vec
LSM-Vec is a research prototype that builds an HNSW-style index on top of **Aster** (a RocksDB fork providing a graph API). Vectors are stored on disk via a pluggable vector storage layer.

## Features

* HNSW graph construction with configurable hyperparameters (`M`, `Mmax`, `Ml`, `efConstruction`)
* Layer-0 edges stored in **Aster RocksGraph**
* Upper layers stored in memory
* Two vector storage backends:

  * **BasicVectorStorage**: contiguous by logical ID
  * **PagedVectorStorage**: 4KB-page managed layout + FIFO page cache (user-space)

---

## Repository layout (expected)

```
LSM-Vec/
  include/
  src/
  lib/
    aster/        # Aster submodule
  CMakeLists.txt
```

---

## Dependencies

### System packages

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake \
  libzstd-dev libsnappy-dev liblz4-dev libbz2-dev zlib1g-dev
```



---

## Build LSM-Vec

To download and build Aster first, from the repo root:

```bash
git submodule update --init --recursive
cd lib/aster
make static_lib -j"$(nproc)" DEBUG_LEVEL=0 DISABLE_WARNING_AS_ERROR=1 EXTRA_CXXFLAGS=-fPIC
cd ../..
```

To build LSM-Vec then:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Binary output at:

```
build/bin/lsm_vec
```

---

## Running

The program requires:

* `--db <path>`: DB directory (required)
* dataset files: either via `--data-dir` (recommended) or explicit `--base/--query/--truth`

### Option A: use `--data-dir` (recommended)

If you pass only `--data-dir`, the tool expects:

* `<data-dir>/input.fvecs`
* `<data-dir>/query.fvecs`
* `<data-dir>/groundtruth.ivecs`

Example:

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --data-dir ./data/sift_100k \
  --out ./run/output.txt
```

### Option B: explicit file paths

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --base ./data/sift_100k_input.fvecs \
  --query ./data/sift_100k_query.fvecs \
  --truth ./data/sift_100k_groundtruth.ivecs \
  --out ./run/output.txt
```

---

## CLI options

### Graph / index parameters

* `--M <int>` (default: 8)
* `--Mmax <int>` (default: 16)
* `--Ml <int>` (default: 1)
* `--efc <float>` (default: 64)

Example:

```bash
./build/bin/lsm_vec \
  --db ./run/db \
  --data-dir ./data/sift_100k \
  --M 16 --Mmax 32 --Ml 1 --efc 200 \
  --out ./run/output.txt
```

### Storage / paths

* `--db <path>`: DB directory (required)
* `--vec <path>`: vector file path (default: `<db>/vector.log`)
* `--vec-storage <int>`:

  * `0` = BasicVectorStorage (default)
  * `1` = PagedVectorStorage (4KB pages + FIFO cache)
* `--out <path>`: output file (default: `output.txt`)

---

## Notes on vector storage

### BasicVectorStorage

* Writes/reads by offset = `id * dim * sizeof(float)`
* Performance depends heavily on OS page cache and access pattern

### PagedVectorStorage

* Manages vectors by 4KB pages
* Vectors never cross page boundaries
* Page-level caching with FIFO eviction (`paged_max_cached_pages` inside `Config.h`)
* Supports optional page prefetch by IDs

---

## Troubleshooting

#### 1) Link errors about RocksGraph methods (e.g. `undefined reference to rocksdb::RocksGraph::AddEdge`)

This usually means you linked against **system RocksDB** instead of Aster’s build.

Make sure:

* Aster is built under `lib/aster`
* CMake resolves the RocksDB library from `lib/aster` (not `/usr/lib/...`)

#### 2) ZSTD undefined references (e.g. `ZSTD_versionNumber`)

You are missing `-lzstd` at link time, or `libzstd-dev` is not installed.

* Install `libzstd-dev`
* Ensure `target_link_libraries(... zstd ...)` is present (your current CMake does this)

