#include <bits/stdc++.h>
#include "pendulum.hpp"

int main() {
	int n;
	cin >> n;

	int t_max;
	cin >> t_max;

	pendulum pp(n);

	for(int i = 0; i < t_max; i += pp::dt){
		pp.next();
		for(int i = 0; i < n; ++i)
			cout << "(" << st.theta << ", " << st.omega << ") ";
		cout << endl;
	}

	return 0;
}

