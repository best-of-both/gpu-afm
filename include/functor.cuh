#ifndef DF_FUNCTOR_CUH
#define DF_FUNCTOR_CUH

#include "typedefs.h"

namespace df {  // differentiable functors

	typedef unsigned int derivative;

	class functor;
	template<derivative N> class functor_val;
	template<derivative N> __device__ functor_val<N+1> diff(functor_val<N>);

	/* class representing symbolic derivatives */
	template<derivative N>
	class functor_val {
		private:
			const functor& fn;
			const data_type r;

			__device__ functor_val(const functor& fn, data_type r) :
				fn(fn), r(r) {}
		public:
			__device__ operator data_type();

		friend __device__ functor_val<N+1> diff<>(functor_val<N>);  // to access fn, r
		friend __device__ functor_val<N> diff<>(functor_val<N-1>);  // to access constructor
	};

	template<>
	class functor_val<0> {
		private:
			const functor& fn;
			const data_type r;

			__device__ functor_val(const functor& fn, const data_type r) :
				fn(fn), r(r) {}
		public:
			__device__ operator data_type();

		friend __device__ functor_val<1> diff<>(functor_val<0>);  // to access fn, r
		friend class functor;  // to access constructor

	};

	/* functor_val factory class */
	class functor {
		private:
			virtual __device__ data_type differentiate(data_type, derivative) const = 0;
		public:
			__device__ functor_val<0> operator()(data_type) const;

		template<derivative N> friend class functor_val;
	};

	/* cast to data_type (probably float or double) */
	template<derivative N>
	__device__
	functor_val<N>::operator data_type()
	{
		return fn.differentiate(r, N);
	}

	__device__ inline
	functor_val<0>::operator data_type()
	{
		return fn.differentiate(r, 0);
	}

	/* lazy-compute N+1st derivative */
	template<derivative N>
	__device__ functor_val<N+1>
	diff(functor_val<N> v)
	{
		return functor_val<N+1>(v.fn, v.r);
	}

}

using df::diff;  // bring diff into the global namespace

#endif
