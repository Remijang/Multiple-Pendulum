#ifndef PENDULUM_H
#define PENDULUM_H

#ifdef C_PLUS_PLUS
extern "C" {
#endif

#include <cooperative_groups.h>

	using namespace cooperative_groups;

	__device__ void derivation(grid_group g, int n, double* st, double* ret, double* mass, double* length, double* suffix_mass, double* arr, double* arr2, double* arr3, double* arr4, double* y, double* ans);

	__device__ void rk4_func1(grid_group g, double *st, const double *dt, double* k, double *res, int n);

	__device__ void rk4_func2(grid_group g, double *st, const double* dt, double *k1, double *k2, double *k3, double *k4 , double *res, int n);

	__global__ void rk4(int n, double* mass, double* length, double* suffix_mass, double* theta, double* omega, double* st, double* arr, double* y, double* ans);

#ifdef C_PLUS_PLUS
}
#endif

class pendulum {
	public:
		int n;

		double *mass;
		double *length;
		double *suffix_mass;
		double *theta;
		double *omega;
		double *posx;
		double *posy;

		double *d_mass;
		double *d_length;
		double *d_suffix_mass;
		double *d_theta;
		double *d_omega;
		double *d_st;
		double *d_arr;
		double *d_y;
		double *d_ans;

		void init();
		void destory();
		pendulum (int _n) : n(_n){
			init();
		}
		void update();
		void next();
		~pendulum () {
			destory();
		}
};

#endif

