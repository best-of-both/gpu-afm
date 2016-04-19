#define CHECK(e) _errorCheck(e, __FILE__, __LINE__)

void _errorCheck(cudaError_t e, const char* file, int line){
	if(e != cudaSuccess){
		printf("Failed to run statement (%s:%d): %s \n",
		       file, line, cudaGetErrorString(e));
		exit(1);
	}
}
