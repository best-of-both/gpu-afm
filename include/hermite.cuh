#ifndef DT_HERMITE_CUH
#define DT_HERMITE_CUH

#include "typedefs.h"
#include "matrix.cuh"
#include "vector.cuh"
#include "mq.cuh"
#include "l2.cuh"

#define NSPECIES 1

using metrics::norm;

namespace dt {

	class hermite : public matrix {
		private:
			typedef point_type normal_type;

			data_type coeffs[3];
			index_type indices[4];
			const rbfs::mq phi;

			__device__ data_type boundary_op(point_type&, point_type&, normal_type&,
					data_type, data_type);
			__device__ data_type boundary_opop(point_type&, point_type&,
					normal_type&, normal_type&, data_type, data_type, data_type, data_type);

		public:
			__device__ hermite(size_type, size_type, index_type,
					data_type, data_type*, data_type,
					point_type*, point_type*, point_type*,
					point_type*, point_type*, point_type*,
					normal_type*, normal_type*, normal_type*,
					point_type*);
			__device__ data_type operator*(vector&);
	};

}

#endif
