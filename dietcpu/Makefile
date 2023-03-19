CXXFLAGS ?= -O3 -g -DNDEBUG

# Just rebuild every time because its fast & easy
.PHONY: main
main:
	$(CXX) -mavx2 -std=c++20 $(CXXFLAGS) -o main ans/Main.cpp ans/ANSDecode.cpp ans/ANSEncode.cpp ans/ANSStatistics.cpp
