#ifndef DT_VECTOR_CUH
#define DT_VECTOR_CUH

namespace dt {

	class vector {
		private:
			unsigned int ref;
			const unsigned int n;
			float* values;
		public:
			__host__ vector(unsigned int n, float* h_values) :
				ref(0), n(n)
			{
				cudaMalloc((void**) &values, n * sizeof(float));
				cudaMemcpy(values, h_values, n * sizeof(float), cudaMemcpyHostToDevice);
			}

			__host__ vector(unsigned int n) :
				ref(0), n(n)
			{
				cudaMalloc((void**) &values, n * sizeof(float));
				cudaMemset(values, 0x00, n * sizeof(float));
			}

			__host__ __device__ vector(const vector& v) :
				ref(v.ref + 1), n(v.n), values(v.values) {}

			__host__ __device__ ~vector()
			{
				if (ref == 0)
					cudaFree(values);
			}

			__device__ float& operator[](unsigned int i) { return values[i]; }
			__host__ __device__ unsigned int size() { return n; }

			void get(float* rec) {
				cudaMemcpy(rec, values, n * sizeof(float), cudaMemcpyDeviceToHost);
			}
	};

};

#endif
