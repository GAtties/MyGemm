CXX ?= gcc

CXXFLAGS = -fopenmp-simd -std=c++20 -m64 -DNDEBUG -march=native -O3
# CXXFLAGS = -fopenmp-simd -std=c++20 -m64 -DNDEBUG -march=native -O0 -g -fsanitize=address

test: main.o
	$(CXX) $^ -o $@ -lstdc++

%.o: %.cpp Makefile test_utils.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
