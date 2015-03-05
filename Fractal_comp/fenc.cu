#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <assert.h>
#include <malloc.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "mutil.h"

#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

using namespace std;

__global__ void cuda_M1(int *M, int *M1, int rsize, int nd, int sv)
{
	__shared__ int *temp;
	__shared__ int stride;
	__shared__ int block;

	temp = new int[4];
	block = 1;
	stride = (rsize*nd) / (1024 * blockDim.x);
	int j = threadIdx.x % (rsize*nd);
	int k = threadIdx.x / (rsize*nd);

	for (int i = k; i < (rsize*nd); i += stride)
	{
		maccess (M, temp, i * 2, i * 2 + 1, j * 2, j * 2 + 1, sv);
		M1[((i*rsize*nd) + j)] = mavg(temp, 4);
	}

	delete[] temp;
}

__global__  void cuda_bigM(int *d_M1, int *d_bigM, int rsize, int nd, int nr, int sv, int sh)
{
	__shared__ int stride;
	stride = (rsize*nd) / (1024 * blockDim.x);
	__shared__ int *D, *temp, *temp2;
	__shared__ int i1, i2, j1, j2;
	D = new int[rsize];
	temp = new int[rsize];
	temp2 = new int[rsize];

	int j = threadIdx.x % (rsize*nd);
	int k = threadIdx.x / (rsize*nd);

	for (int i = k; i < (rsize*nd); i += stride)
	{
		i1 = i*rsize;
		i2 = (i + 1)*rsize - 1;
		j1 = j*rsize;
		j2 = (j + 1)*rsize - 1;
		maccess(d_M1, D, i1, i2, j1, j2, rsize*nd);
		msave(D, d_bigM, i1, i2, j1, j2, 0, 8, rsize*nd);
		rotmat(D, temp, rsize);
		msave(temp, d_bigM, i1, i2, j1, j2, 1, 8, rsize*nd);
		rotmat(temp, temp2, rsize);
		msave(temp2, d_bigM, i1, i2, j1, j2, 2, 8, rsize*nd);
		rotmat(temp2, temp, rsize);
		msave(temp, d_bigM, i1, i2, j1, j2, 3, 8, rsize*nd);
		fliph(D, temp, rsize);
		msave(temp, d_bigM, i1, i2, j1, j2, 4, 8, rsize*nd);
		flipv(D, temp, rsize);
		msave(temp, d_bigM, i1, i2, j1, j2, 5, 8, rsize*nd);
		transpose(D, temp, rsize);
		msave(temp, d_bigM, i1, i2, j1, j2, 6, 8, rsize*nd);
		rotmat(temp, temp2, rsize);
		rotmat(temp2, temp, rsize);
		msave(temp, d_bigM, i1, i2, j1, j2, 7, 8, rsize*nd);
	}

	delete[] temp;
	delete[] temp2;
	delete[] D;
}

__global__  void search(int *d_M, int *d_M1, int *T, int *d_bigM, int rsize, int nd, int nr, int sv, int sh)
{
	__shared__ int i1, i2, j1, j2, k1, k2, l1, l2, off, i0, j0, m0, s0, g0, del_g, sum_dist;
	__shared__ float dist, dmin;
	float s[4] = { 0.45, 0.60, 0.80, 1.00 };
	/*s[0] = 0.45;
	s[1] = 0.60;
	s[2] = 0.80;
	s[3] = 1.00;*/
	int domainDim = nd*nd*rsize*rsize * 8;
	__shared__ int *sharedBigM, *R, *D, *temp, *temp2;

	sharedBigM = new int[domainDim];
	R = new int[rsize*rsize];
	D = new int[rsize*rsize];
	temp = new int[rsize*rsize];
	temp2 = new int[rsize*rsize];

	for (int i = 0; i< domainDim; i++) 
		sharedBigM[i] = d_bigM[i];

	int k = blockIdx.x;
	int l = blockIdx.y;

	k1 = k*rsize;
	k2 = (k + 1)*rsize - 1;
	l1 = l*rsize;
	l2 = (l + 1)*rsize - 1;

	maccess(d_M, R, k1, k2, l1, l2, sv);
	off = mavg(R, rsize*rsize);

	dmin = (int)powf(10, 6);
	i0 = 0;
	j0 = 0;
	m0 = 0;

	for (int i = 0; i < nd; i++)
	{
		i1 = i*rsize;
		i2 = (i + 1)*rsize - 1;
		for (int j = 0; j < nd; j++)
		{
			j1 = j*rsize;
			j2 = (j + 1)*rsize - 1;
			// Test each transformation
			for (int n = 0; n < 4; n++)
			{
				for (int m = 0; m < 8; m++)
				{
					maccess(sharedBigM, D, i1, i2, j1, j2, m, 8, rsize*nd);
					scale(D, s[n], rsize*rsize);
					del_g = off - mavg(D, rsize*rsize);
					increment(D, del_g, rsize*rsize);
					diff(R, D, temp, rsize*rsize);
					msquare(temp, temp2, rsize*rsize);
					sum_dist = msum(temp2, rsize*rsize);
					dist = sqrtf(sum_dist);
					if (dist < dmin)
					{
						dmin = dist;
						i0 = i;
						j0 = j;
						m0 = m;
						s0 = n;
						g0 = del_g;
					}
				}
			}
		}
		T[((k*nr + l) * 5 + 0)] = i0;
		T[((k*nr + l) * 5 + 1)] = j0;
		T[((k*nr + l) * 5 + 2)] = m0;
		T[((k*nr + l) * 5 + 3)] = s0;
		T[((k*nr + l) * 5 + 4)] = g0;
	}
}

cudaError_t cuda_encoder(int *h_r, int *h_g, int *h_b, int *h_Tr, int *h_Tg, int *h_Tb, int rsize, int nd, int nr, int sv, int sh)
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

	//kernel call here
	dim3 grid1 = (64, 1, 1);
	dim3 block1 = (1024, 1, 1);

	dim3 grid2 = (sv / rsize, sh / rsize, 1);
	dim3 block2 = (1, 1, 1);

	cuda_M1 <<<grid1,block1>>> (d_r, d_M1_r, rsize, nd, sv);

	cudaStatus = cudaMemcpy(h_Tr, d_Tr, nr * nr * 5 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(h_Tg, d_Tg, nr * nr * 5 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(h_Tb, d_Tb, nr * nr * 5 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	return cudaStatus;
}