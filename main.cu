#include <bits/stdc++.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include "pendulum.h"

#define F first
#define S second

const float TARGET_FPS = 30.0f;
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

	pendulum pp(n);

	auto lastTime = std::chrono::high_resolution_clock::now();

	for (double i = 0; i < t_max ; i += 0.01) {
		pp.next();
		auto currentTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> elapsed = currentTime - lastTime;
		std::cerr << elapsed.count() << std::endl;

		glClear(GL_COLOR_BUFFER_BIT);
		glBegin(GL_LINE_STRIP);
		glColor3d(1, 0, 0);
		glVertex2d(0, 0.5);

		for (int j = 0; j < n; j++) {
			glVertex2d(10 * pp.posx[j] / window_dim.S, 10 * pp.posy[j] / window_dim.S + 0.5);
		}

		glEnd();
		circle({0, 0.5}, 0.01);
		for (int j = 0; j < n; j++) {
			circle({10 * pp.posx[j] / window_dim.S, 10 * pp.posy[j] / window_dim.S + 0.5}, 0.01);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();

		float frameTime = elapsed.count();
		if (frameTime < FRAME_TIME) {
			std::this_thread::sleep_for(std::chrono::milliseconds(int((FRAME_TIME - frameTime) * 1000)));
		}

		lastTime = currentTime;
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

