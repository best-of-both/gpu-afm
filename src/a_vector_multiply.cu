__device__ void
device_a_vector_multiply(int* a, float* v_in, float* v_out, unsigned int nx, unsigned int ny)
{
	unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < nx * ny) {
		int down  = a[4 * thread + 0],
		    left  = a[4 * thread + 1],
		    right = a[4 * thread + 2],
		    up    = a[4 * thread + 3];
		
		v_out[thread] = down * v_in[thread - nx] +
		                left * v_in[thread - 1] +
		                right * v_in[thread + 1] +
		                up * v_in[thread + nx]
		                - 4 * v_in[thread];
	}
}
