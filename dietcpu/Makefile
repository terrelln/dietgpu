CXXFLAGS ?= -O3 -g -DNDEBUG -mavx2
CXX ?= clang

# Just rebuild every time because its fast & easy
.PHONY: main
main:
	$(CXX) $(CXXFLAGS) -o main ans/Main.cpp ans/ANSDecode.cpp
