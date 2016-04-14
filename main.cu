#include<stdio.h>
#include "cuda.h"
#include<string.h>
#include<stdlib.h>

#define BLOCK_DIM_X 1024 
#define BLOCK_DIM_Y 512  

#define DEBUG 0                 //@@ toggle for debug messages
#define REPS 1			//Max number of runs for timing analysis

/* Choose which kernel to run:
  1) 
  2) 
  3) 
  4) 
*/
#define KERNEL_SELECT 1

char *inputFile;
char *outputFile;

void _errorCheck(cudaError_t e){
	if(e != cudaSuccess){
		printf("Failed to run statement \n");
	}
}

//GPU Function.  Called from host, runs on device
//Input:
//	
//output:
//	
//Desc:	
__global__ void func1( ) {

//VARIABLES
	int tid = blockIdx.x * blockDim.x + threadIdx.x;


//GPU Function.  Called from host, runs on device
//Input:
//	
//	
//Desc:	

__global__ void func2( ) {

//VARIABLES
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	

}// end void func1



//writes command line arguments to program variables
void parseInput(int argc, char **argv){
	if(argc < 2){
		printf("Not enough arguments\n");
		printf("Usage: reduction -i inputFile -o outputFile\n");
		exit(1);
	}
	int i=1;
	while(i<argc){
		if(!strcmp(argv[i],"-i") ){
			++i;
			inputFile = argv[i];
			//printf("Inputfile: %s\n", argv[i]);
		}
		else if(!strcmp(argv[i],"-o")){
			++i;
		 	outputFile = argv[i];
		}
		else{
			printf("Wrong input");
			exit(1);
		}
		i++;
	}//end while
} // end parse input


void getSize(int &size, char *file){
	FILE *fp;
	fp = fopen(file,"r");

	if(fp == NULL){
		perror("Error opening File\n");
		exit(1);
	}

	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	fclose(fp);
}


void readFromFile(int &size,float *v, char *file){
	FILE *fp;
	fp = fopen(file,"r");
	if(fp == NULL){
		printf("Error opening File %s\n",file);
		exit(1);
	}

	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	int i=0;
	float t;
	while(i < size){
		if(fscanf(fp,"%f",&t)==EOF){
			printf("Error reading file\n");
			exit(1);
		}
		v[i++]=t;
	}//end while
	fclose(fp);

}//end readFromFile


int main(int argc, char **argv) {

//VARAIBLES
  int ii;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list

  float *deviceInput;
  float *deviceOutput;

  float *solution;


  //Kernel pointer & descriptor
    void (*kernel)(float *, float *, int);
    const char *kernelName;

//STATEMENTS
  parseInput(argc, argv);

  //  Define list length, allocate for hostInput, fill hostInput
  getSize(numInputElements,inputFile);
  hostInput = (float*) malloc(numInputElements*sizeof(float));
  readFromFile(numInputElements,hostInput,inputFile);

  int opsz;
  getSize(opsz,outputFile);
  solution = (float*) malloc(opsz*sizeof(float));  //pre-allocate space for 'opsz' length array
  readFromFile(opsz,solution,outputFile);



  if(numOutputElements==0) {
    numOutputElements++;
  }

  //Pre-allocate memory for host output variable based on
  //size of output array?
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

	int memSizeIn = numInputElements * sizeof(float);
	int memSizeOut = numOutputElements * sizeof(float);


  //Set kernel Pointer
  switch(KERNEL_SELECT)
  {
    case 1:
      kernel = &func1;
      kernelName = "Function 1		";
      break;
    case 2:
      kernel = &func2;
      kernelName = "Function 2 ";
      break;
    default:
      kernel = &func1;
      kernelName = "Function 1		";
      break;
   }// end switch

  //@@ Allocate GPU memory here

    cudaMalloc( (void **)&deviceInput, memSizeIn );
    cudaMalloc( (void **) &deviceOutput, memSizeOut );

  //@@ Copy memory to the GPU here

    cudaMemcpy(deviceInput,hostInput, memSizeIn, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostOutput, memSizeOut, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here

  // Initialize timer
    cudaEvent_t start,stop;
    float elapsed_time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

  for(int timingLoop = 0; timingLoop<REPS; timingLoop++){	
  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
    kernel<<<numBlocks, numThreads>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, memSizeOut, cudaMemcpyDeviceToHost);



  }//end for timingLoop
  
cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time,start, stop);

if(DEBUG){
  printf("Elapsed time: %f\n", elapsed_time);
}

  //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);


 //Free Host memory
  free(hostInput);
  free(hostOutput);

  //clean slate
  cudaDeviceReset();

  return 0;
}//end main
