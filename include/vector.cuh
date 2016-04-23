#ifndef DT_VECTOR_CUH
#define DT_VECTOR_CUH

#include "typedefs.h"

namespace dt {

	class vector {
		private:
			data_type* const values;
			data_type sentinel;  // safety mechanism
			const size_type length;
		public:
			__device__ vector(data_type* values, size_type length) :
				values(values), length(length) {}
			__device__ size_type size() { return length; }
			__device__ data_type& operator[](index_type);
	};

};

#endif
