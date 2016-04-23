#ifndef DT_POINT_CUH
#define DT_POINT_CUH

#include "internal_typedefs.h"

namespace dt {

	class point {
		public:
			index_type index;
			data_type x;
			data_type y;

			__host__ __device__ point() : index(0), x(0), y(0) {}
			__host__ __device__ point(index_type index, data_type x, data_type y) :
				index(index), x(x), y(y) {}
	};

}

#endif
