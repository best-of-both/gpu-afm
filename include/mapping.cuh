#ifndef DT_MAPPING_CUH
#define DT_MAPPING_CUH

#include "typedefs.h"
#include "matrix.cuh"
#include "vector.cuh"

namespace dt {

	__device__ extern index_type compute_index(size_type, unsigned int*);

	class mapping : public matrix {
		private:
			bool is_forcing;
			index_type forcing_index;
		public:
			__device__ mapping(size_type, size_type, bool, index_type);
			__device__ data_type operator*(vector&);
	};

}

#endif
