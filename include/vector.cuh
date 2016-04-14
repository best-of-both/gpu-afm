#ifndef DT_VECTOR_CUH
#define DT_VECTOR_CUH

namespace dt {

	class vector {
		private:
			const unsigned int n;
			float* values;
		public:
			__host__ vector(unsigned int n, float* h_values) : n(n){
				cudaMalloc((void**) &values, n * sizeof(float));
				cudaMemcpy(values, h_values, n * sizeof(float), cudaMemcpyHostToDevice);
			}

			__host__ vector(unsigned int n) : n(n)  {
				cudaMalloc((void**) &values, n * sizeof(float));
				cudaMemset(values, 0x00, n * sizeof(float));
			}

			__device__ float& operator[](unsigned int i) { return values[i]; }
			__host__ __device__ unsigned int size() { return n; }

			void get(float* rec) {
				cudaMemcpy(rec, values, n * sizeof(float), cudaMemcpyDeviceToHost);
			}
	};

};

#endif
