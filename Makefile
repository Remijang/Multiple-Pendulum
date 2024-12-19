CXX = g++
CXXFLAGS = -std=c++17 -O3 -pthread -lm -lGL -lGLU -lglfw -lglut

all: main

main: main.cpp
	$(CXX) -o main main.cpp $(CXXFLAGS) 

clean:
	rm -f main

.PHONY: all clean
