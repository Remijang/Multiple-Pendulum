#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include "pendulum.h"

using namespace cooperative_groups;

namespace pp {
	const double g = 9.81;
	const double dt = 0.001;
	const double b = 0.5;
}

/*
 * derivation function prototype
 *
 * 1. init_cs (const state *st, double *c, double *s, int n)
 * calculate c, s
 *
 * 2. init_arr (const state *st, const sys *ss, double **arr, double *y, int n)
 * calculate arr, y
 *
 * 3. gauss (double **arr, double *y, double *ans, int n)
 * solve the equation
 *
 * 4. assign (double *ans, state *ret, int n)
 * assign the derivation value
 *
 */

__device__ void derivation(thread_group g, int n, double* st, double* ret, double* mass, double* length, double* suffix_mass, double* arr, double* y, double* ans)
{
	int idx=g.thread_rank();
	double c,s;
	c=cos(st[idx]);
	s=sin(st[idx]);
	y[idx]=0.0;
	for(int i=0;i<n;i++)
	{
		arr[idx * n + i] = suffix_mass[idx] * length[i] * (c * cos(st[i]) + s * sin(st[i]));
		y[idx] += suffix_mass[idx] * length[i] * st[n+i] * st[n+i] * (s * cos(st[i]) - c * sin(st[i]));
	}
	y[idx] += suffix_mass[0] * pp::g * s;
	y[idx] *= -1;
	g.sync();
	for(int i=0;i<n;i++)
	{
		if(idx>i)
		{
			double d = arr[idx * n + i] / arr[i * n + i];
			for(int j=i;j<n;j++) arr[idx * n + j] -= d * arr[i * n + j];
			y[idx] -= d * y[i];
		}
		g.sync();
	}
	for(int i=n-1;i>=0;i--)
	{
		if(idx==i) ans[idx] = y[idx] / arr[idx * n + idx];
		g.sync();
		if(idx<i) y[idx] -= arr[idx * n + i] * ans[i];
		g.sync();
	}
	ret[idx]=st[n+idx];
	ret[n+idx]=ans[idx];
	g.sync();
	return;
}

__device__ void rk4_func1(grid_group g, double *st, double dt, double* k, double *res, int n) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < n)
	{
		res[id] = st[id] + dt * k[id];
		res[n+id] = st[n+id] + dt * k[n+id];
	}
	g.sync();
	return;
}

__device__ void rk4_func2(grid_group g, double *st, double dt, double *k1, double *k2, double *k3, double *k4 , double *res, int n) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id < n)
	{
		res[id] = st[id] + dt * (k1[id] + 2 * k2[id] + 2 * k3[id] + k4[id]);
		res[id] = st[id] + dt * (k1[n+id] + 2 * k2[n+id] + 2 * k3[n+id] + k4[n+id]);
	}
	g.sync();
	return;
}

__global__ void rk4(int n, double* mass, double* length, double* suffix_mass, double* theta, double* omega, double* st, double* arr, double* y, double* ans)
{
	grid_group g = this_grid();
	double dt=pp::dt;
	double dt2=pp::dt/2;
	double dt6=pp::dt/6;
	double *s[4];
	double *k[4];
	double *re;
	for(int i=0;i<4;i++)
	{
		s[i]=st+2*n*i;
		k[i]=st+2*n*(i+4);
	}
	re=st+16*n;
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	st[idx]=theta[idx];
	st[idx+n]=omega[idx];
	g.sync();
	derivation(g, n, s[0], k[0], mass, length, suffix_mass, arr, y, ans);
	rk4_func1(g, s[0],dt2,k[0],s[1],n);
	derivation(g, n, s[1], k[1], mass, length, suffix_mass, arr, y, ans);
	rk4_func1(g, s[0],dt2,k[1],s[2],n);
	derivation(g, n, s[2], k[2], mass, length, suffix_mass, arr, y, ans);
	rk4_func1(g, s[0],dt,k[2],s[3],n);
	derivation(g, n, s[3], k[3], mass, length, suffix_mass, arr, y, ans);
	rk4_func2(g, s[0],dt6,k[0],k[1],k[2],k[3],re,n);
	theta[idx]=re[idx];
	omega[idx]=re[idx+n];
	g.sync();
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
	std::uniform_real_distribution<> rnd(0, 2 * M_PI);
	for (int i = 0; i < n; ++i) theta[i]  = rnd(generator);
	for (int i = 0; i < n; ++i) omega[i]  = rnd(generator);
	for (int i = 0; i < n; ++i) mass[i]   = rnd(generator);
	for (int i = 0; i < n; ++i) length[i] = rnd(generator);

	suffix_mass[n - 1] = mass[n - 1];

	for (int i = n - 2; i >= 0; --i) suffix_mass[i] = suffix_mass[i + 1] + mass[i];

	cudaMalloc(       &d_mass, sizeof(double) * n);
	cudaMalloc(     &d_length, sizeof(double) * n);
	cudaMalloc(&d_suffix_mass, sizeof(double) * n);
	cudaMalloc(      &d_theta, sizeof(double) * n);
	cudaMalloc(      &d_omega, sizeof(double) * n);
	cudaMalloc(         &d_st, sizeof(double) * n * 18);
	cudaMalloc(        &d_arr, sizeof(double) * n * n);
	cudaMalloc(          &d_y, sizeof(double) * n);
	cudaMalloc(        &d_ans, sizeof(double) * n);

	cudaMemcpy(       d_mass,        mass, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(     d_length,      length, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_suffix_mass, suffix_mass, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(      d_theta,       theta, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(      d_omega,       omega, sizeof(double) * n, cudaMemcpyHostToDevice);
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
	cudaMemcpy(theta, d_theta, sizeof(double) * n, cudaMemcpyDeviceToHost);
	for(int i = 0; i < n; ++i) {
		tmpx += length[i] * sin(theta[i]);
		tmpy -= length[i] * cos(theta[i]);
		posx[i] = tmpx;
		posy[i] = tmpy;
	}
}

void pendulum::next () {
	int threads = 64;
	int blocks = (n + threads - 1) / threads;
	void *params[] = {&(n), &(d_mass), &(d_length), &(d_suffix_mass), &(d_theta), &(d_omega), &(d_st), &(d_arr), &(d_y), &(d_ans)};
	cudaLaunchCooperativeKernel(rk4, blocks, threads, params, 0, 0);
	update();
}
