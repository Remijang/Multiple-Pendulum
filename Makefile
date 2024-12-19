NVFLAGS := -std=c++11 -O3 -use_fast_math -ftz=true -rdc=true
TARGET := main
LINK := -lGL -lglfw -lGLU

.PHONY: all
all: $(TARGET)

$(TARGET): pendulum.o main.cu
	nvcc $(NVFLAGS) -o main main.cu pendulum.o $(LINK)

pendulum.o: pendulum.cu
	nvcc $(NVFLAGS) -c pendulum.cu

clean:
	rm -rf main *.o
