#ifndef DT_MAPPING_CUH
#define DT_MAPPING_CUH

#include "typedefs.h"
#include "matrix.cuh"
#include "vector.cuh"

namespace dt {

	__device__ extern index_type* p_row_indices;

	__device__ extern size_type initialize_mapping(size_type, unsigned char*);

	class mapping : public matrix {
		public:
			__device__ mapping(size_type, unsigned char*);
			__device__ ~mapping();  // needed to free p_row_indices
			__device__ data_type operator*(vector&);
	};

}

#endif
