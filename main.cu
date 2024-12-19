#include <bits/stdc++.h>
#include <GL/glew.h>   // OpenGL extensions
#include <GLFW/glfw3.h> // GLFW windowing
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>  // CUDA-OpenGL interoperability

#define F first
#define S second

const float TARGET_FPS = 60.0f;
const float FRAME_TIME = 1.0f / TARGET_FPS;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define GL_CHECK() \
    do { \
        GLenum err = glGetError(); \
        if (err != GL_NO_ERROR) { \
            std::cerr << "OpenGL error in " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void circle(const std::pair<double, double>& center, double radius) {
    const int segments = 20;
    double delta = 2 * M_PI / segments;
    double c = std::cos(delta);
    double s = std::sin(delta);
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
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    auto window = glfwCreateWindow(window_dim.F, window_dim.S, "Multiple Pendulum", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glViewport(0, 0, window_dim.F, window_dim.S);
    glClearColor(1, 1, 1, 0);
    glLineWidth(4);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    GLuint posX_A_VBO, posY_A_VBO;
    glGenBuffers(1, &posX_A_VBO);
    GL_CHECK();
    glGenBuffers(1, &posY_A_VBO);
    GL_CHECK();

    glBindBuffer(GL_ARRAY_BUFFER, posX_A_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * n, NULL, GL_DYNAMIC_DRAW);
    GL_CHECK();
    glBindBuffer(GL_ARRAY_BUFFER, posY_A_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * n, NULL, GL_DYNAMIC_DRAW);
    GL_CHECK();

    cudaGraphicsResource_t cudaPosX_A, cudaPosY_A;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPosX_A, posX_A_VBO, cudaGraphicsRegisterFlagsNone));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPosY_A, posY_A_VBO, cudaGraphicsRegisterFlagsNone));

    auto lastTime = std::chrono::high_resolution_clock::now();

    for (double i = 0; ; i += 0.01) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = currentTime - lastTime;

        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_LINE_STRIP);
        glColor3d(1, 0, 0);
        glVertex2d(0, 0);

        // Map CUDA to OpenGL buffer
        float *posX_GPU, *posY_GPU;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPosX_A, 0));
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPosY_A, 0));

        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&posX_GPU, NULL, cudaPosX_A));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&posY_GPU, NULL, cudaPosY_A));

        // Run CUDA kernel on GPU data
        runCUDAKernel<<<(n + 255) / 256, 256>>>(posX_GPU, posY_GPU, n, 0.01f);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Render the pendulum positions from GPU buffer (OpenGL)
        glBindBuffer(GL_ARRAY_BUFFER, posX_A_VBO);
        GL_CHECK();
        glVertexPointer(2, GL_FLOAT, 0, (void*)0);  // Enable vertex pointer
        glBindBuffer(GL_ARRAY_BUFFER, posY_A_VBO);
        GL_CHECK();
        glVertexPointer(2, GL_FLOAT, 0, (void*)0);  // Enable vertex pointer

        glBegin(GL_LINE_STRIP);
        for (int j = 0; j < n; j++) {
            glVertex2d(20 * posX_GPU[j] / window_dim.S, 20 * posY_GPU[j] / window_dim.S);
        }
        glEnd();

        circle({0, 0}, 0.01);
        for (int j = 0; j < n; j++) {
            circle({20 * posX_GPU[j] / window_dim.S, 20 * posY_GPU[j] / window_dim.S}, 0.01);
        }

        // Unmap the CUDA buffers after usage
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPosX_A, 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPosY_A, 0));

        glfwSwapBuffers(window);
        glfwPollEvents();

        float frameTime = elapsed.count();
        if (frameTime < FRAME_TIME) {
            std::this_thread::sleep_for(std::chrono::milliseconds(int((FRAME_TIME - frameTime) * 1000)));
        }

        lastTime = currentTime;
    }

    // Clean up CUDA resources
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPosX_A));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaPosY_A));
    glDeleteBuffers(1, &posX_A_VBO);
    GL_CHECK();
    glDeleteBuffers(1, &posY_A_VBO);
    GL_CHECK();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

