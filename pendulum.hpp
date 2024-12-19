#ifndef PENDULUM_HPP
#define PENDULUM_HPP

#include <math.h>
#include <vector>

namespace pp {
	const double g = 9.81;
	const double pi = 3.1415926;
	const double dt = 0.005;
}

struct system {
	int n;
	std::vector<double> mass;
	std::vector<double> length;

	init (int _n) {
		n = _n;
		mass.resize(_n);
		length.resize(_n);

		for(int i = 0; i < n; ++i)
			cin >> mass[i];

		for(int i = 0; i < n; ++i)
			cin >> length[i];
	}
};

struct state {
	int n;
	std::vector<double> theta;
	std::vector<double> omega;

	init (int _n) {
		n = _n;
		theta.resize(_n);
		omega.resize(_n);

		for(int i = 0; i < n; ++i)
			cin >> theta[i];

		for(int i = 0; i < n; ++i)
			cin >> omega[i];
	}
};

state operator+ (const state &s1, const state &s2) {
	state ret(s1.n);
	for(int i = 0; i < s1.n; ++i) {
		ret.theta[i] = s1.theta[i] + s2.theta[i];
		ret.omega[i] = s1.omega[i] + s2.omega[i];
	}
}

state operator* (const double d, const state &s) {
	state ret(s.n);
	for(int i = 0; i < s.n; +=i) {
		ret.theta[i] = d * s.theta[i];
		ret.omega[i] = d * s.omega[i];
	}
}

state derivation (const state &st, const system &ss) {
	int n = st.n;
	double cc[n], ss[n];
	for(int i = 0; i < n; ++i) {
		cc[i] = cos(st.theta[i]);
		ss[i] = sin(st.theta[i]);
	}
	int arr[n][n];
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) arr[i][j] = 0.0;
		for(int j = i; j < n; ++j) {

		}
	}
}

state rk4 (const state &st, const system &ss) {
	static double dt2 = pp::dt / 2;
	static double dt6 = pp::dt / 6;
	state k1 = derivation(st, ss);
	state s2 = st + dt2 * k1;

	state k2 = derivation(s2, ss);
	state s3 = st + dt2 * k2;

	state k3 = derivation(s3, ss);
	state s4 = st + pp::dt * k3;

	state k4 = derivation(s4, ss);

	state re = st + dt6 * (k1 + 2 * k2 + 2 * k3 + k4);

	return re;
}

class pendulum {
	system ss;
	state  st;

	pendulum (int n) {
		ss.init(n);
		st.init(n);
	}

	void next () {
		st = rk4(st, ss);
	}
};

#endif
