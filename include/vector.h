#ifndef DT_VECTOR_H
#define DT_VECTOR_H

namespace dt {

	template<unsigned int N>
	class vector {
		private:
			float* values;

		public:
			__host__ vector(float* h_values) {
				cudaMalloc((void**) &values, N * sizeof(float));
				cudaMemcpy(values, h_values, N * sizeof(float), cudaMemcpyHostToDevice);
			}

			__host__ vector() {
				cudaMalloc((void**) &values, N * sizeof(float));
				cudaMemset(values, 0x00, N * sizeof(float));
			}

			__device__ float& operator[](unsigned int n) { return values[n]; }

			void get(float* rec) {
				cudaMemcpy(rec, values, N * sizeof(float), cudaMemcpyDeviceToHost);
			}
	};

};

#endif
