# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(HNSWAsterDBProject)

# Set C++ standard to C++17 and build type
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O2")

set(ASTER_ROCKSDB_ROOT "/home/dhmo/Aster")

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
    src/HNSWGraph.cpp
    src/HNSWUtils.cpp
    src/main.cpp
)

# Add the executable target
add_executable(hnsw_asterdb ${SOURCES})

target_include_directories(hnsw_asterdb PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_library(ZSTD_LIB zstd
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)

if(NOT ZSTD_LIB)
    message(FATAL_ERROR "libzstd not found; please install libzstd-dev or provide its path")
endif()

message(STATUS "Using ZSTD library at ${ZSTD_LIB}")

target_link_libraries(hnsw_asterdb PRIVATE
    ${ROCKSDB_LIB}
    ${ZSTD_LIB}  
    snappy
    lz4
    bz2
    z
    pthread
    dl
)

# Set output directory for the executable
set_target_properties(hnsw_asterdb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
