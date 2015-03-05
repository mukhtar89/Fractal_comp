#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "mutil.h"

using namespace std;

cudaError_t cuda_decoder(int *h_r, int *h_g, int *h_b, int *h_Tr, int *h_Tg, int *h_Tb, int rsize, int nd, int nr, int sv, int sh)
{

	int *d_r, *d_g, *d_b, *d_Tr, *d_Tg, *d_Tb, *d_M1_r, *d_M1_g, *d_M1_b, *d_bigM_r, *d_bigM_g, *d_bigM_b;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&d_r, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_g, sv * sh*sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_b, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_Tr, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_Tg, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_Tb, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_M1_r, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_M1_g, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_M1_b, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_bigM_r, nd*nd*rsize*rsize * 8 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_bigM_g, nd*nd*rsize*rsize * 8 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_bigM_b, nd*nd*rsize*rsize * 8 * sizeof(int));

	cudaStatus = cudaMemcpy(d_r, h_r, sv * sh* sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_g, h_g, sv * sh* sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_b, h_b, sv * sh* sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_Tr, h_Tr, sv * sh* sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_Tg, h_Tg, sv * sh* sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_Tb, h_Tb, sv * sh* sizeof(int), cudaMemcpyHostToDevice);

	//kernel call here

	cudaStatus = cudaMemcpy(h_Tr, d_Tr, sv * sh* sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(h_Tg, d_Tg, sv * sh* sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(h_Tb, d_Tb, sv * sh* sizeof(int), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	return cudaStatus;
}