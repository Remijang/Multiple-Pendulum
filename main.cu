#include <bits/stdc++.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>  // CUDA header

#define F first
#define S second

const float TARGET_FPS = 60.0f;
const float FRAME_TIME = 1.0f / TARGET_FPS;

void circle(const std::pair<double, double>& center, double radius) {
    const int segments = 20;

    double delta = 2 * M_PI / segments;
    double c     = std::cos(delta);
    double s     = std::sin(delta);

    std::pair<double, double> xy{radius, 0};

    glColor3d(0, 0, 0);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; ++i) {
        glVertex2d(xy.first + center.first, xy.second + center.second);
        glVertex2d(center.first, center.second);

        xy = {c * xy.first - s * xy.second, s * xy.first + c * xy.second};
    }
    glEnd();
}

// CUDA kernel to simulate the pendulum
__global__ void runCUDAKernel(float* posX, float* posY, int n, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        // Example of a pendulum update logic (simplified)
        // Update position based on some pendulum physics (replace with actual equations)
        posX[i] += dt;
        posY[i] += dt;
    }
}

int main() {
    int n;
    std::cin >> n;
    int t_max;
    std::cin >> t_max;

    std::pair<int, int> window_dim = {1080, 1080};
    glfwInit();
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    auto window = glfwCreateWindow(window_dim.F, window_dim.S, "Multiple Pendulum", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glViewport(0, 0, window_dim.F, window_dim.S);
    glClearColor(1, 1, 1, 0);
    glLineWidth(4);

    // Allocate GPU memory using CUDA (for pendulum positions)
    float* posX_A_GPU;
    float* posY_A_GPU;
    float* posX_B_GPU;
    float* posY_B_GPU;

    cudaMalloc(&posX_A_GPU, sizeof(float) * n);
    cudaMalloc(&posY_A_GPU, sizeof(float) * n);
    cudaMalloc(&posX_B_GPU, sizeof(float) * n);
    cudaMalloc(&posY_B_GPU, sizeof(float) * n);

    // Initialize positions (just for demonstration, in a real scenario, initial positions will come from pendulum simulation)
    float* hostPosX = new float[n];
    float* hostPosY = new float[n];
    for (int i = 0; i < n; i++) {
        hostPosX[i] = 0.0f;
        hostPosY[i] = 0.0f;
    }

    // Copy initial positions to GPU memory
    cudaMemcpy(posX_A_GPU, hostPosX, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(posY_A_GPU, hostPosY, sizeof(float) * n, cudaMemcpyHostToDevice);

    auto lastTime = std::chrono::high_resolution_clock::now();

    for (double i = 0; ; i += 0.01) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = currentTime - lastTime;

        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_LINE_STRIP);
        glColor3d(1, 0, 0);
        glVertex2d(0, 0);

        // Run the CUDA kernel to update the pendulum positions in Buffer B
        runCUDAKernel<<<(n + 255) / 256, 256>>>(posX_B_GPU, posY_B_GPU, n, 0.01f);
        cudaDeviceSynchronize();

        // Copy results from GPU Buffer B to Buffer A (for rendering)
        cudaMemcpy(posX_A_GPU, posX_B_GPU, sizeof(float) * n, cudaMemcpyDeviceToDevice);
        cudaMemcpy(posY_A_GPU, posY_B_GPU, sizeof(float) * n, cudaMemcpyDeviceToDevice);

        // Render the pendulum positions from Buffer A (GPU memory)
        float* posX_A_CPU = new float[n];
        float* posY_A_CPU = new float[n];

        cudaMemcpy(posX_A_CPU, posX_A_GPU, sizeof(float) * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(posY_A_CPU, posY_A_GPU, sizeof(float) * n, cudaMemcpyDeviceToHost);

        // Draw pendulums from Buffer A (GPU memory)
        for (int j = 0; j < n; j++) {
            glVertex2d(20 * posX_A_CPU[j] / window_dim.S, 20 * posY_A_CPU[j] / window_dim.S);
        }

        glEnd();
        circle({0, 0}, 0.01);
        for (int j = 0; j < n; j++) {
            circle({20 * posX_A_CPU[j] / window_dim.S, 20 * posY_A_CPU[j] / window_dim.S}, 0.01);
        }

        // Swap buffers: Buffer A becomes Buffer B, and vice versa
        std::swap(posX_A_GPU, posX_B_GPU);
        std::swap(posY_A_GPU, posY_B_GPU);

        glfwSwapBuffers(window);
        glfwPollEvents();

        float frameTime = elapsed.count();
        if (frameTime < FRAME_TIME) {
            std::this_thread::sleep_for(std::chrono::milliseconds(int((FRAME_TIME - frameTime) * 1000)));
        }

        lastTime = currentTime;
    }

    // Clean up CUDA resources
    cudaFree(posX_A_GPU);
    cudaFree(posY_A_GPU);
    cudaFree(posX_B_GPU);
    cudaFree(posY_B_GPU);

    delete[] hostPosX;
    delete[] hostPosY;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

