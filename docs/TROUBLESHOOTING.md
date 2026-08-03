# Troubleshooting

Common errors when installing, building, or running AsterVec.

## Installing

- **`No matching distribution found` on `pip install aster-vec`** — no prebuilt
  package for your platform yet (shipped: Linux x86_64/aarch64, macOS). Build
  from source: see [CONTRIBUTING.md](../CONTRIBUTING.md#development-setup).
- **`externally-managed-environment` on `pip install .`** — use a
  virtualenv/conda environment.

## Building from source

- **`Aster RocksDB library or headers not found`** — build Aster first:
  `git submodule update --init --recursive && make aster`.
- **`libzstd not found`** — install zstd (the only required codec):
  `apt-get install libzstd-dev` / `brew install zstd`.
- **`FetchContent` can't download pybind11 during `pip install .`** — install
  pybind11 (`pip install pybind11` or conda) and switch the `FetchContent` block
  in `CMakeLists.txt` to `find_package(pybind11 REQUIRED)`.

## Running

- **`cannot allocate memory in static TLS block` (Linux, jemalloc)** — preload
  jemalloc: `LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 python your_app.py`.
- **`libastervec.so: cannot open shared object file`** — add the build dir to the
  loader path: `export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH`.

Still stuck? Open an [issue](https://github.com/NTU-Siqiang-Group/AsterVec/issues)
with your platform, the failing command, and its output.
