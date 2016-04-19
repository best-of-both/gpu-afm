#include <fstream>

#include "forcing.cuh"

namespace dt {

	forcing_data::forcing_data(unsigned int length) :
		length(length)
	{
		unsigned int datum;
		unsigned char* h_data = new unsigned char[length];
		cudaMalloc((void **) &data, length * sizeof(unsigned char));

		std::ifstream fd;
		fd.open("data/forcing");
		for (unsigned int i = 0; i < length; ++i) {
			fd >> datum;
			h_data[i] = (unsigned char) datum;
		}
		fd.close();

		cudaMemcpy(data, h_data, length * sizeof(unsigned char), cudaMemcpyHostToDevice);
		delete[] h_data;
	}

	unsigned char*
	forcing_data::get()
	{
		unsigned char* h_data = new unsigned char[length];
		cudaMemcpy(h_data, data, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		return h_data;
	}

}
