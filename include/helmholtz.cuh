#ifndef DT_HELMHOLTZ_H
#define DT_HELMHOLTZ_H

#include "typedefs.h"
#include "matrix.cuh"
#include "vector.cuh"

namespace dt {

	class helmholtz : public matrix {
		private:
			const size_type nx, ny, n;
			const data_type scale;
		public:
			__device__ helmholtz(size_type nx, size_type ny, size_type n, data_type scale) :
				matrix(nx * ny, nx * ny), nx(nx), ny(ny), n(n), scale(scale) {}
			__device__ data_type operator*(vector&);
	};

}

#endif
