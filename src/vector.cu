#include "vector.cuh"

namespace dt {

	__device__ data_type&
	vector::operator[](index_type index)
	{
		if (index < length)
			return values[index];
		sentinel = 0;
		return sentinel;
	}

}
