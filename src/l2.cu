#include "typedefs.h"
#include "l2.cuh"

namespace metrics {

	__device__ data_type
	norm(point_type& l, point_type& r)
	{
		data_type dx = l.x - r.x,
				  dy = l.y - r.y;
		return sqrt(dx * dx + dy * dy);
	}

}
