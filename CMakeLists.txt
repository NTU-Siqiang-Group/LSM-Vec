# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(LSMVEC)

# Set C++ standard to C++17 and build type
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O2")

find_package(Boost REQUIRED)
include_directories(PRIVATE ${Boost_INCLUDE_DIRS})
find_library(JEMALLOC_LIB jemalloc HINTS "$(brew --prefix jemalloc)/lib")
include_directories(PRIVATE ${JEMALLOC_LIB})

set(ASTER_ROCKSDB_ROOT ${CMAKE_SOURCE_DIR}/lib/aster)

find_library(ROCKSDB_LIB rocksdb
    HINTS
        "${ASTER_ROCKSDB_ROOT}"
        "${ASTER_ROCKSDB_ROOT}/build"
        "${ASTER_ROCKSDB_ROOT}/build/lib"
        "${ASTER_ROCKSDB_ROOT}/lib"
)

find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h
    HINTS
        "${ASTER_ROCKSDB_ROOT}/include"
        "${ASTER_ROCKSDB_ROOT}"
)

if(NOT ROCKSDB_LIB OR NOT ROCKSDB_INCLUDE_DIR)
    message(FATAL_ERROR "Aster RocksDB library or headers not found")
endif()

message(STATUS "Using RocksDB library at ${ROCKSDB_LIB}")
message(STATUS "Using RocksDB headers at ${ROCKSDB_INCLUDE_DIR}")

include_directories(${ROCKSDB_INCLUDE_DIR})

# find_library(ROCKSDB_LIB rocksdb HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
# set(ROCKSDB_INCLUDE_DIR /usr/include/rocksdb)

if(NOT ROCKSDB_LIB OR NOT ROCKSDB_INCLUDE_DIR)
    message(FATAL_ERROR "RocksDB library or headers not found")
endif()

message(STATUS "Found RocksDB library at ${ROCKSDB_LIB}")
message(STATUS "Found RocksDB headers at ${ROCKSDB_INCLUDE_DIR}")

include_directories(${ROCKSDB_INCLUDE_DIR})

# Specify the source files
set(SOURCES
    src/lsm_vec_index.cc
    src/lsm_vec_db.cc
    src/utils.cc
    src/test.cc
)

# Add the executable target
add_executable(lsm_vec ${SOURCES})

target_include_directories(lsm_vec PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_library(ZSTD_LIB zstd
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)

if(NOT ZSTD_LIB)
    message(FATAL_ERROR "libzstd not found; please install libzstd-dev or provide its path")
endif()

message(STATUS "Using ZSTD library at ${ZSTD_LIB}")

target_link_libraries(lsm_vec PRIVATE
    ${ROCKSDB_LIB}
    ${ZSTD_LIB}  
    snappy
    lz4
    bz2
    z
    pthread
    dl
    jemalloc
)

# Set output directory for the executable
set_target_properties(lsm_vec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
