#include "functor.cuh"

namespace df {

	__device__ functor_val<0>
	functor::operator()(data_type r) const
	{
		return functor_val<0>(*this, r);
	}

}
