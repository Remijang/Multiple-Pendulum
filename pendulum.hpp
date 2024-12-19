#ifndef PENDULUM_HPP
#define PENDULUM_HPP

#include <math.h>
#include <vector>
#include <iostream>

namespace pp {
	const double g = 9.81;
	const double dt = 0.005;
}

struct sys {
	int n;
	std::vector<double> mass;
	std::vector<double> length;

    sys (int n) {
        init(n);
    }

	void init (int _n) {
		n = _n;
		mass.resize(_n);
		length.resize(_n);

		for(int i = 0; i < n; ++i)
			std::cin >> mass[i];

		for(int i = 0; i < n; ++i)
			std::cin >> length[i];

        return;
	}
};

struct state {
	int n;
	std::vector<double> theta;
	std::vector<double> omega;

    state (int n) {
        init(n);
    }

	void init (int _n) {
		n = _n;
		theta.resize(_n);
		omega.resize(_n);

		for(int i = 0; i < n; ++i)
			std::cin >> theta[i];

		for(int i = 0; i < n; ++i)
			std::cin >> omega[i];

        return;
	}
};

state operator+ (const state &s1, const state &s2) {
	state ret(s1.n);
	for(int i = 0; i < s1.n; ++i) {
		ret.theta[i] = s1.theta[i] + s2.theta[i];
		ret.omega[i] = s1.omega[i] + s2.omega[i];
	}
    return ret;
}

state operator* (const double d, const state &s) {
	state ret(s.n);
	for(int i = 0; i < s.n; ++i) {
		ret.theta[i] = d * s.theta[i];
		ret.omega[i] = d * s.omega[i];
	}
    return ret;
}

state derivation (const state &st, const sys &ss) {
	int n = st.n;
	double c[n], s[n];

	for(int i = 0; i < n; ++i) {
		c[i] = cos(st.theta[i]);
		s[i] = sin(st.theta[i]);
	}

	double arr[n][n], y[n];
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) arr[i][j] = 0.0;
        y[i] = 0.0;
		for(int j = i; j < n; ++j) {
            for(int k = 0; k <= j; ++k) {
				/*
                arr[i][k] += ss.mass[j] * ss.length[k] * (c[i] * c[k] + s[i] * s[k]);
                y[i]      += ss.mass[j] * ss.length[k] * st.omega[k] * st.omega[k] * (s[i] * c[k] - c[i] * s[k]);
				*/
                arr[i][k] += ss.mass[j] * ss.length[k] * cos(st.theta[i] - st.theta[k]);
                y[i]      += ss.mass[j] * ss.length[k] * st.omega[k] * st.omega[k] * sin(st.theta[i] - st.theta[k]);
            }
            y[i] += ss.mass[j] * pp::g * s[i];
		}
        y[i] *= -1;
	}

    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            double d = arr[j][i] / arr[i][i];
            for(int k = i; k < n; ++k)
                arr[j][k] -= d * arr[i][k];
            y[j] -= d * y[i];
        }
    }

    double ans[n];
    for(int i = n - 1; i >= 0; --i) {
        ans[i] = y[i] / arr[i][i];
        for(int j = 0; j < i; ++j) y[j] -= arr[j][i] * ans[i];
    }

    state ret(n);
    for(int i = 0; i < n; ++i) {
        ret.theta[i] = st.omega[i];
        ret.omega[i] = ans[i];
    }

    return ret;
}

state rk4 (const state &st, const sys &ss) {
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

struct Pos {
    std::vector<double> x;
    std::vector<double> y;

    Pos (int _n) {
        x.resize(_n);
        y.resize(_n);
    }
};

class pendulum {
    public:
        int n;
        sys   ss;
        state st;
        Pos   pos;

        pendulum (int _n) : n(_n), ss(_n), st(_n), pos(_n) {}

        void update() {
            double tmpx = 0.0, tmpy = 0.0;
            for(int i = 0; i < n; ++i) {
                tmpx += ss.length[i] * sin(st.theta[i]);
                tmpy -= ss.length[i] * cos(st.theta[i]);
                pos.x[i] = tmpx;
                pos.y[i] = tmpy;
            }
        }

        void next () {
            st = rk4(st, ss);
            update();
        }

        friend std::ostream& operator<< (std::ostream &o, const pendulum &pp) {
            for(int i = 0; i < pp.n; ++i)
                o << "(" << pp.st.theta[i] << ", " << pp.st.omega[i] << ") ";
            o << std::endl;
            return o;
        }
};

#endif
