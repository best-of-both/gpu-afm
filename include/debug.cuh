#ifndef DEBUG_H
#define DEBUG_H

#include "cuda.h"

#define CHECK(e) _cudaErrorCheck(e, __FILE__, __LINE__)

void _cudaErrorCheck(cudaError_t, const char*, int);

#endif
