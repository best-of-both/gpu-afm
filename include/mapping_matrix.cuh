#ifndef MAPPING_MATRIX_CUH
#define MAPPING_MATRIX_CUH

// stuff for P matrix

#include "vector.cuh"
#include "matrix.cuh"
#include "forcing.cuh"

namespace dt {

	class mapping_matrix {
		private:
			unsigned int rows, cols, *mapping;
		public:
			__device__ void vector_multiply(vector&, vector&);
			__host__ mapping_matrix(forcing_data);
			__host__ vector operator*(vector);
	};

}

#endif
