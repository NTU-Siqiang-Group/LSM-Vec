CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2

.RECIPEPREFIX := >

REPO_ROOT := ..
ASTER_ROOT := $(REPO_ROOT)/lib/aster

LSM_VEC_SOURCES := \
$(REPO_ROOT)/src/lsm_vec_index.cc \
$(REPO_ROOT)/src/lsm_vec_db.cc \
$(REPO_ROOT)/src/utils.cc

TEST_SOURCES := test.cc

INCLUDES := \
-I$(REPO_ROOT)/include \
-I$(ASTER_ROOT)/include

LIB_DIRS := \
-L$(ASTER_ROOT) \
-L$(ASTER_ROOT)/build \
-L$(ASTER_ROOT)/build/lib \
-L$(ASTER_ROOT)/lib

LIBS := -lrocksdb -lzstd -lsnappy -llz4 -lbz2 -lz -lpthread -ldl -ljemalloc

TARGET := lsm_vec_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(LSM_VEC_SOURCES) $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_DIRS) $^ -o $@ $(LIBS)

clean:
	rm -f $(TARGET)
