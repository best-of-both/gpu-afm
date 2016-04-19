#include <iostream>
#include "cuda.h"
#include "mapping_matrix.cuh"
#include "forcing.cuh"
#include "vector.cuh"

int main(int argc, char **argv) {
	const int s = 64;
	const int n = s * s;
	dt::forcing_data data(n);
	dt::mapping_matrix m(data);

	float* vec_data = new float[68];
	float* vec_out = new float[n];
	for (unsigned int i = 0; i < 68; ++i)
		vec_data[i] = 1;
	dt::vector v(68, vec_data);
	dt::vector o = m * v;
	delete[] vec_data;

	o.get(vec_out);

	for (unsigned int i = 0; i < o.size(); ++i)
		std::cout << vec_out[i] << " " << std::endl;
}
