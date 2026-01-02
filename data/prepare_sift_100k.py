"""
Prepare SIFT_100k dataset in the current directory.

It will:
  1) Download 'sift.tar.gz' from the TexMex corpus if not present.
  2) Extract the SIFT1M files:
       sift/sift_base.fvecs
       sift/sift_query.fvecs
       sift/sift_groundtruth.ivecs
  3) Create a 100k subset:
       sift_100k_input.fvecs        (first 100,000 base vectors)
       sift_100k_query.fvecs        (all queries, copied)
       sift_100k_groundtruth.ivecs  (exact k-NN groundtruth on 100k base)

Groundtruth is computed with exact L2 (squared) distances.
This is meant as a reasonably robust but simple reference script.
"""

import os
import tarfile
import urllib.request
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
ARCHIVE_NAME = "sift.tar.gz"
EXTRACTED_DIR = "sift"

BASE_FILE = os.path.join(EXTRACTED_DIR, "sift_base.fvecs")
QUERY_FILE = os.path.join(EXTRACTED_DIR, "sift_query.fvecs")
GT_FILE = os.path.join(EXTRACTED_DIR, "sift_groundtruth.ivecs")

# Output subset files (in current directory)
OUT_BASE_100K = "sift_100k_input.fvecs"
OUT_QUERY_100K = "sift_100k_query.fvecs"
OUT_GT_100K = "sift_100k_groundtruth.ivecs"

# Number of base vectors to keep
N_BASE_100K = 100_000
# Groundtruth top-k (same as TexMex default)
TOPK = 100
# Batch size for query processing when computing groundtruth
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# I/O helpers for fvecs / ivecs (TexMex format)
# ---------------------------------------------------------------------------

def read_fvecs(path: str) -> np.ndarray:
    """
    Read a .fvecs file into a float32 array of shape (n, d).

    Format (little endian):
       For each vector:
         int32 d
         float32[d] components
    """
    data = np.fromfile(path, dtype='int32')
    if data.size == 0:
        raise RuntimeError(f"Empty fvecs file: {path}")

    d = data[0]
    if d <= 0:
        raise RuntimeError(f"Invalid dimension {d} in {path}")

    # Reshape as (n, d+1), then drop the first column (the dimension)
    data = data.reshape(-1, d + 1)
    # Reinterpret the last d entries as float32
    # The first column is 'd' stored as int32, we ignore it.
    vectors = data[:, 1:].view('float32')
    return vectors


def read_ivecs(path: str) -> np.ndarray:
    """
    Read a .ivecs file into an int32 array of shape (n, d).

    Format (little endian):
       For each vector:
         int32 d
         int32[d] components
    """
    data = np.fromfile(path, dtype='int32')
    if data.size == 0:
        raise RuntimeError(f"Empty ivecs file: {path}")

    d = data[0]
    if d <= 0:
        raise RuntimeError(f"Invalid dimension {d} in {path}")

    data = data.reshape(-1, d + 1)
    vectors = data[:, 1:]  # still int32
    return vectors


def write_fvecs(path: str, x: np.ndarray) -> None:
    """
    Write a float32 array x (n, d) to a .fvecs file.

    Format (little endian):
       For each vector:
         int32 d
         float32[d] components
    """
    x = np.asarray(x, dtype='float32')
    n, d = x.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(np.int32(d).tobytes())
            f.write(x[i].tobytes())


def write_ivecs(path: str, x: np.ndarray) -> None:
    """
    Write an int32 array x (n, d) to a .ivecs file.

    Format (little endian):
       For each vector:
         int32 d
         int32[d] components
    """
    x = np.asarray(x, dtype='int32')
    n, d = x.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(np.int32(d).tobytes())
            f.write(x[i].tobytes())


# ---------------------------------------------------------------------------
# Dataset download / extract
# ---------------------------------------------------------------------------

def download_sift_if_needed():
    """Download sift.tar.gz into current directory if it does not exist."""
    if os.path.exists(ARCHIVE_NAME):
        print(f"[info] Archive {ARCHIVE_NAME} already exists, skip download.")
        return

    print(f"[info] Downloading SIFT dataset archive from:\n  {SIFT_URL}")
    try:
        urllib.request.urlretrieve(SIFT_URL, ARCHIVE_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to download {SIFT_URL}: {e}")
    print("[info] Download complete.")


def extract_sift_if_needed():
    """Extract sift.tar.gz if 'sift/' directory is missing."""
    if os.path.isdir(EXTRACTED_DIR) and \
       os.path.exists(BASE_FILE) and \
       os.path.exists(QUERY_FILE) and \
       os.path.exists(GT_FILE):
        print(f"[info] Directory '{EXTRACTED_DIR}' with SIFT files already present.")
        return

    print(f"[info] Extracting {ARCHIVE_NAME} ...")
    if not os.path.exists(ARCHIVE_NAME):
        raise RuntimeError(f"Archive {ARCHIVE_NAME} not found. "
                           "Run download_sift_if_needed() first.")

    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall()
    print("[info] Extraction complete.")


# ---------------------------------------------------------------------------
# Groundtruth computation for 100k base
# ---------------------------------------------------------------------------

def compute_exact_gt_100k(xb_100k: np.ndarray,
                          xq: np.ndarray,
                          k: int = TOPK,
                          batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Compute exact L2 groundtruth for queries on top of xb_100k.

    xb_100k: (nb, d) float32 base vectors (nb = 100k)
    xq:      (nq, d) float32 query vectors
    k:       number of nearest neighbors to keep
    batch_size: number of queries processed per batch to limit memory

    Returns:
        gt: (nq, k) int32, indices in [0, nb-1] sorted by increasing distance.
    """
    xb = xb_100k
    xq = xq
    nb, d = xb.shape
    nq, d2 = xq.shape
    assert d == d2, "Base and query must have the same dimension."

    print(f"[info] Computing exact groundtruth on {nb} base vectors "
          f"and {nq} queries (k={k}, batch_size={batch_size}) ...")

    # Precompute base squared norms once
    xb_norms = (xb ** 2).sum(axis=1)  # (nb,)

    gt = np.empty((nq, k), dtype='int32')

    for i0 in range(0, nq, batch_size):
        i1 = min(i0 + batch_size, nq)
        q_batch = xq[i0:i1]          # (b, d)
        b = q_batch.shape[0]

        # Squared norms of queries
        q_norms = (q_batch ** 2).sum(axis=1)  # (b,)

        # Compute squared L2 distances using the formula:
        #   ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q·x
        #   q_batch @ xb.T has shape (b, nb)
        dot = q_batch @ xb.T                   # (b, nb)
        dists = q_norms[:, None] + xb_norms[None, :] - 2.0 * dot

        # Get indices of k smallest distances using argpartition
        # Note: argpartition with kth=k-1 returns an array where the k
        # smallest elements are in the first k positions (unordered).
        idx_part = np.argpartition(dists, k - 1, axis=1)[:, :k]

        # Now sort those k elements by distance to get a properly ordered list
        part_dists = np.take_along_axis(dists, idx_part, axis=1)
        order = np.argsort(part_dists, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)

        gt[i0:i1] = idx_sorted.astype('int32')

        print(f"[info] Processed queries [{i0}, {i1}) / {nq}")

    print("[info] Groundtruth computation done.")
    return gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1: download and extract SIFT1M if needed
    download_sift_if_needed()
    extract_sift_if_needed()

    # Step 2: load original base and query
    print(f"[info] Loading base from {BASE_FILE}")
    xb = read_fvecs(BASE_FILE)
    print(f"[info] Base shape: {xb.shape}")

    print(f"[info] Loading query from {QUERY_FILE}")
    xq = read_fvecs(QUERY_FILE)
    print(f"[info] Query shape: {xq.shape}")

    nb, d = xb.shape
    nq, d2 = xq.shape
    if nb < N_BASE_100K:
        raise RuntimeError(f"SIFT base has only {nb} vectors; "
                           f"cannot create a 100k subset.")

    # Step 3: take the first 100k base vectors
    xb_100k = xb[:N_BASE_100K].copy()
    print(f"[info] Subset base shape: {xb_100k.shape}")

    # Step 4: compute new groundtruth on the 100k base
    gt_100k = compute_exact_gt_100k(xb_100k, xq, k=TOPK, batch_size=BATCH_SIZE)
    print(f"[info] Groundtruth shape: {gt_100k.shape}")

    # Step 5: write output files in current directory
    print(f"[info] Writing {OUT_BASE_100K}")
    write_fvecs(OUT_BASE_100K, xb_100k)

    print(f"[info] Writing {OUT_QUERY_100K}")
    write_fvecs(OUT_QUERY_100K, xq)

    print(f"[info] Writing {OUT_GT_100K}")
    write_ivecs(OUT_GT_100K, gt_100k)

    print("[info] All done.")
    print(f"Generated files in {os.getcwd()}:")
    print(f"  {OUT_BASE_100K}")
    print(f"  {OUT_QUERY_100K}")
    print(f"  {OUT_GT_100K}")


if __name__ == "__main__":
    main()
