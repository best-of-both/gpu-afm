#ifndef RBFS_MQ_CUH
#define RBFS_MQ_CUH

#include "rbf.cuh"

namespace rbfs {

	class mq : public rbf {
		private:
			__device__ data_type differentiate(data_type, unsigned int) const;
		protected:
			using rbf::eps;
		public:
			__device__ mq(data_type eps) : 
				rbf(eps) {}
	};

}

#endif
