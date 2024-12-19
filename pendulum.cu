#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include "pendulum.h"

using namespace cooperative_groups;

namespace pp {
	const double g = 9.81;
	const double dt = 0.008;
	const double b = 0.5;
	const int    count = 4;
	const double eps = 0.01;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void derivation (grid_group g, int n, double* st, double* ret, 
		double* mass, double* length, double* suffix_mass, 
		double* arr, double* arr2, double* arr3, double* arr4, 
		double* y, double* ans) {
	int idx = g.thread_rank();
	int stride = g.size();
	double c = cos(st[idx]), s = sin(st[idx]);

	if (idx < n) {
		y[idx] = 0.0;
		for (int i = 0; i < n; ++i) {
			arr[idx * n + i] = suffix_mass[max(idx, i)] * length[i] * (c * cos(st[i]) + s * sin(st[i]));
			y[idx] += suffix_mass[max(idx, i)] * length[i] * st[n + i] * st[n + i] * (s * cos(st[i]) - c * sin(st[i]));
		}
		y[idx] += suffix_mass[idx] * pp::g * s;
		y[idx] *= -1;
	}

	g.sync();

	if (idx < n) {
		for (int i = 0; i < n; ++i) arr2[idx * n + i] = arr[i * n + idx];
		for (int i = 0; i < n; ++i) arr4[idx * n + i] = 0;
		arr4[idx * n + idx] = 1;
	}

	g.sync();

	int t = idx;

	while (t < n * n) {
		int x = t / n, y = t % n;
		double sum = 0.0;
		for (int i = 0; i < n; ++i) sum += arr2[x * n + i] * arr2[y * n + i];
		arr3[t] = sum;
		t += stride;
	}

	g.sync();

	for (int i = 0; i < n; ++i) {
		if (idx < n && idx != i) {
			double d = arr3[idx * n + i] / arr3[i * n + i];
			if (arr3[i * n + i] < pp::eps) d = 0.0;
			for (int j = i; j < n; ++j) {
				arr3[idx * n + j] -= d * arr3[i * n + j];
			}
			for (int j = 0; j < n; ++j) {
				arr4[idx * n + j] -= d * arr4[i * n + j];
			}
		}
		g.sync();
	}

	t = idx;

	while (t < n * n) {
		int x = t / n, y = t % n;
		double sum = 0.0;
		for (int i = 0; i < n; ++i) {
			if (arr3[x * n + x] >= pp::eps) sum += arr4[x * n + i] * arr[y * n + i] / arr3[x * n + x];
		}
		arr2[t] = sum;
		t += stride;
	}

	g.sync();

	if (idx < n) {
		double sum = 0.0;
		for (int i = 0; i < n; ++i) sum += arr2[idx * n + i] * y[i];
		ans[idx] = sum;
	}

	g.sync();

	if (idx < n) {
		ret[    idx] =  st[n + idx];
		ret[n + idx] = ans[    idx];
	}
	g.sync();
	return;
}

__device__ void rk4_func1 (grid_group g, double *st, double dt, double* k, double *res, int n) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < n) {
		res[    id] = st[    id] + dt * k[    id];
		res[n + id] = st[n + id] + dt * k[n + id];
	}
	g.sync();
	return;
}

__device__ void rk4_func2 (grid_group g, double *st, double dt, 
		double *k1, double *k2, double *k3, double *k4 , double *res, int n) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < n) {
		res[    id] = st[    id] + dt * (k1[    id] + 2 * k2[    id] + 2 * k3[    id] + k4[    id]);
		res[n + id] = st[n + id] + dt * (k1[n + id] + 2 * k2[n + id] + 2 * k3[n + id] + k4[n + id]);
	}
	g.sync();
	return;
}

__global__ void rk4 (
		int n, double* mass, double* length, double* suffix_mass, 
		double* theta, double* omega, double* st, 
		double* arr, double* y, double* ans) {
	grid_group g = this_grid();
	double dt  = pp::dt;
	double dt2 = pp::dt / 2;
	double dt6 = pp::dt / 6;
	double *s[4], *k[4], *re;
	for (int i = 0; i < 4; ++i) {
		s[i] = st + 2 * n * i;
		k[i] = st + 2 * n * (i + 4);
	}
	double* arr2 = arr + n * n;
	double* arr3 = arr + n * n * 2;
	double* arr4 = arr + n * n * 3;
	re = st + 16 * n;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// do simulate pp::count times to meet 60fps
	for (int i = 0; i < pp::count; ++i) {
		// s0
		if (idx < n) {
			st[    idx] = theta[idx] - ((int) (theta[idx] / (2 * M_PI))) * (2 * M_PI);
			st[n + idx] = omega[idx];
		}
		g.sync();

		// k1
		derivation(g, n, s[0], k[0], mass, length, suffix_mass, arr, arr2, arr3, arr4, y, ans);
		// s1
		rk4_func1(g, s[0], dt2, k[0], s[1], n);
		// k2
		derivation(g, n, s[1], k[1], mass, length, suffix_mass, arr, arr2, arr3, arr4, y, ans);
		// s2
		rk4_func1(g, s[0], dt2, k[1], s[2], n);
		// k3
		derivation(g, n, s[2], k[2], mass, length, suffix_mass, arr, arr2, arr3, arr4, y, ans);
		// s3
		rk4_func1(g, s[0], dt, k[2], s[3], n);
		// k4
		derivation(g, n, s[3], k[3], mass, length, suffix_mass, arr, arr2, arr3, arr4, y, ans);

		// sum
		rk4_func2(g, s[0], dt6, k[0], k[1], k[2], k[3], re, n);
		// assign

		if (idx < n) {
			theta[idx] = re[    idx];
			omega[idx] = re[n + idx];
		}
		g.sync();
	}
	return;
}

void pendulum::init () {
	cudaMallocHost(       &mass, sizeof(double) * n);
	cudaMallocHost(     &length, sizeof(double) * n);
	cudaMallocHost(&suffix_mass, sizeof(double) * n);
	cudaMallocHost(      &theta, sizeof(double) * n);
	cudaMallocHost(      &omega, sizeof(double) * n);
	cudaMallocHost(       &posx, sizeof(double) * n);
	cudaMallocHost(       &posy, sizeof(double) * n);


	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<> rnd_1(((double)50) / n, ((double)75) / n);
	std::uniform_real_distribution<> rnd_2(-M_PI / 2, M_PI / 2);
	std::uniform_real_distribution<> rnd_3(-1 / n, 1 / n);
	for (int i = 0; i < n; ++i) theta[i]  = rnd_2(generator);
	for (int i = 0; i < n; ++i) omega[i]  = rnd_3(generator);
	for (int i = 0; i < n; ++i) mass[i]   = rnd_1(generator);
	for (int i = 0; i < n; ++i) length[i] = rnd_1(generator);

#ifdef HAND
	for (int i = 0; i < n; ++i) std::cin >> mass[i];
	for (int i = 0; i < n; ++i) std::cin >> length[i];
	for (int i = 0; i < n; ++i) std::cin >> theta[i];
	for (int i = 0; i < n; ++i) std::cin >> omega[i];
#endif

	suffix_mass[n - 1] = mass[n - 1];

	for (int i = n - 2; i >= 0; --i) suffix_mass[i] = suffix_mass[i + 1] + mass[i];

	cudaMalloc(       &d_mass, sizeof(double) * n);
	cudaMalloc(     &d_length, sizeof(double) * n);
	cudaMalloc(&d_suffix_mass, sizeof(double) * n);
	cudaMalloc(      &d_theta, sizeof(double) * n);
	cudaMalloc(      &d_omega, sizeof(double) * n);
	cudaMalloc(         &d_st, sizeof(double) * n * 19);
	cudaMalloc(        &d_arr, sizeof(double) * n * n * 4);
	cudaMalloc(          &d_y, sizeof(double) * n);
	cudaMalloc(        &d_ans, sizeof(double) * n);

	cudaMemcpy(       d_mass,        mass, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(     d_length,      length, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_suffix_mass, suffix_mass, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(      d_theta,       theta, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(      d_omega,       omega, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemset(         d_st,           0, sizeof(double) * n * 19);
}

void pendulum::destory () {
	cudaFreeHost(mass);
	cudaFreeHost(length);
	cudaFreeHost(suffix_mass);
	cudaFreeHost(theta);
	cudaFreeHost(omega);
	cudaFreeHost(posx);
	cudaFreeHost(posy);

	cudaFree(d_mass);
	cudaFree(d_length);
	cudaFree(d_suffix_mass);
	cudaFree(d_theta);
	cudaFree(d_omega);
	cudaFree(d_st);
	cudaFree(d_arr);
	cudaFree(d_y);
	cudaFree(d_ans);
}

void pendulum::update () {
	double tmpx = 0.0, tmpy = 0.0;
	gpuErrchk( cudaMemcpy(theta, d_theta, sizeof(double) * n, cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(omega, d_omega, sizeof(double) * n, cudaMemcpyDeviceToHost) );
	for (int i = 0; i < n; ++i)
		std::cerr << omega[i] << " ";
	std::cerr << std::endl;
	for (int i = 0; i < n; ++i) {
		tmpx += length[i] * sin(theta[i]);
		tmpy -= length[i] * cos(theta[i]);
		posx[i] = tmpx;
		posy[i] = tmpy;
	}
}

void pendulum::next () {
	int threads = 128;
	int blocks = (n + threads - 1) / threads;
	void *params[] = {&(n), &(d_mass), &(d_length), &(d_suffix_mass), &(d_theta), &(d_omega), &(d_st), &(d_arr), &(d_y), &(d_ans)};
	cudaLaunchCooperativeKernel(rk4, blocks, threads, params, 0, 0);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	update();
}
