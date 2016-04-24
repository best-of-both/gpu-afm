#include <cmath>
#include <iostream>
#include <iomanip>

#include "typedefs.h"
#include "mq.cuh"


namespace rbfs {

	__device__ data_type
	mq::differentiate(data_type r, unsigned int d) const
	{
		data_type phi2 = 1 + eps * eps * r * r;
		data_type phi_p = sqrt(phi2);
		int pref = 1;
		int exp = 1;
		data_type eps_p = 1;

		while (d -- > 0) {
			eps_p *= eps * eps;
			phi_p /= phi2;
			pref *= exp;
			exp -= 2;
		}

		return pref * eps_p * phi_p;
	}

}
