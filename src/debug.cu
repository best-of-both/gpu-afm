#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"
#include "debug.cuh"

void _cudaErrorCheck(cudaError_t e, const char* file, int line){
	if(e != cudaSuccess){
		printf("Failed to run statement (%s:%d): %s \n",
		       file, line, cudaGetErrorString(e));
		exit(1);
	}
}
