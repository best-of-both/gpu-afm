#ifndef RBFS_RBF_H
#define RBFS_RBF_H

#include <stdio.h>
#include "typedefs.h"
#include "functor.cuh"

namespace rbfs {

	class rbf : public df::functor {
		protected:
			const data_type eps;
		public:
			__device__ rbf(data_type eps) :
				eps(eps) {}
	};

}

#endif
