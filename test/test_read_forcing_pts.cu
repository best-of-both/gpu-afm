#include <iostream>
#include "forcing.cuh"

int
main(void)
{
	dt::forcing_data data(64 * 64);

	unsigned char* is_forcing = data.get();

	for (unsigned int i = 0; i < 64 * 64; ++i) {
		std::cout << is_forcing[i];
		if (i % 64 == 63)
			std::cout << std::endl;
		else
			std::cout << " ";
	}

	return 0;
}
