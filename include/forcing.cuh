#ifndef DT_FORCING_CUH
#define DT_FORCING_CUH

namespace dt {

	class forcing_data {
		private:
			const unsigned int length;
			unsigned char* data;
		public:
			__host__ unsigned char* get();
			__host__ __device__ unsigned int size() { return length; }
			__host__ __device__ unsigned char& operator[](unsigned int i) { return data[i]; }

			forcing_data(unsigned int);
	};

}

#endif
