# Multiple Pendulum Simulator

## Overview
This is a simple multiple pendulum simulator written in CUDA and OpenGL.

## Remark
Since this is nonlinear system, the model uses Langrange equation and rk4.

The equation cannot be solve directly, instead, we use linear regression to aprroximate.

This project now is the basic implementation in parallel aspect, and it's not very efficiency.

## How to compile
Use `Make` or by the command below:
```bash=
nvcc -std=c++11 -O3 -use_fast_math -ftz=true -rdc=true -c pendulum.cu
nvcc -std=c++11 -O3 -use_fast_math -ftz=true -rdc=true -o main main.cu pendulum.o -lGL -lglfw -lGLU
```

## How to run
```bash=
./main [number of pendulum] [time]
```
