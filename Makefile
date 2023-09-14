CXX ?= gcc
MKL_ROOT ?= /opt/intel/oneapi/mkl/latest

CXXFLAGS = -fopenmp -std=c++20 -m64 -DNDEBUG -march=native -O3
# CXXFLAGS = -fopenmp-simd -std=c++20 -m64 -DNDEBUG -march=native -O0 -g -fsanitize=address
LIBS = -lstdc++

CXXFLAGS += -DMKL_ILP64 -m64 -DUSE_MKL
CXXFLAGS += -I $(MKL_ROOT)/include
LDFLAGS += -L $(MKL_ROOT)/lib/intel64 -Wl,-rpath,$(MKL_ROOT)/lib/intel64
LIBS += -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

test: main.o
	$(CXX) $^ $(LDFLAGS) -fopenmp -Wl,--no-as-needed -o $@ -lstdc++ $(LIBS)

%.o: %.cpp Makefile test_utils.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
