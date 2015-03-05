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

void fenc(int *M, int *T, int rsize, int nd, int nr, int sv, int sh)
{
	int *M1, *temp, *D, *bigM, *temp2, *R, *temp3, count=0;

	//Begin batch runs
	int min0 = 100;

	//Scale the Domain Blocks
	M1 = (int*)malloc((int)pow(rsize*nd,2) * sizeof(int));
	temp3 = (int*)malloc(4 * sizeof(int));
	for (int i = 0; i < rsize*nd; i++)
	{
		for (int j = 0; j < rsize*nd; j++)
		{
			maccess(M, temp3, i * 2, i * 2 + 1, j * 2, j * 2 + 1, sv);
			M1[((i*rsize*nd) + j)] = mavg(temp3,4);
		}
	}
	cout << "M1 created\n";

	//Matrix of 4 possible scalings to transform grayscale
	float s[4] = { 0.45, 0.60, 0.80, 1.00 };

	// Create monster matrix containing all possible 2D transformations
	// of the domain blocks.Store in multidimensional matrix bigM.
	int i1, i2, j1, j2;
	temp = (int*)malloc(rsize * rsize * sizeof(int));
	temp2 = (int*)malloc(rsize * rsize * sizeof(int));
	D = (int*)malloc(rsize * rsize * sizeof(int));
	bigM = (int*)malloc(nd*nd*rsize*rsize*8 * sizeof(int));
	int g = 0;
	for (int i = 0; i < nd; i++)
	{
		i1 = i*rsize;
		i2 = (i + 1)*rsize - 1;
		for (int j = 0; j < nd; j++)
		{
			j1 = j*rsize;
			j2 = (j + 1)*rsize - 1;
			maccess(M1, D, i1, i2, j1, j2, rsize*nd);
			msave(D, bigM, i1, i2, j1, j2, 0, 8, rsize*nd);
			rotmat(D, temp, rsize*rsize);
			msave(temp, bigM, i1, i2, j1, j2, 1, 8, rsize*nd);
			rotmat(temp, temp2, rsize*rsize);
			msave(temp2, bigM, i1, i2, j1, j2, 2, 8, rsize*nd);
			rotmat(temp2, temp, rsize*rsize);
			msave(temp, bigM, i1, i2, j1, j2, 3, 8, rsize*nd);
			fliph(D, temp, rsize*rsize);
			msave(temp, bigM, i1, i2, j1, j2, 4, 8, rsize*nd);
			flipv(D, temp, rsize*rsize);
			msave(temp, bigM, i1, i2, j1, j2, 5, 8, rsize*nd);
			transpose(D, temp, rsize*rsize);
			msave(temp, bigM, i1, i2, j1, j2, 6, 8, rsize*nd);
			rotmat(temp, temp2, rsize*rsize);
			rotmat(temp2, temp, rsize*rsize);
			msave(temp, bigM, i1, i2, j1, j2, 7, 8, rsize*nd);
		}
	}
	free(D);
	cout << "BigM created\n";
				


	// Compare the range blocks and scaled domain blocks.
	// k, l - used to cycle through blocks Rkl.
	int k1, k2, l1, l2, off, i0, j0, m0, s0, g0, del_g, sum_dist, ming0=256, maxg0=0;
	float dist, dmin;
	R = (int*)malloc(rsize * rsize * sizeof(int));
	D = (int*)malloc(rsize * rsize * sizeof(int));
	for (int k = 0; k < nr; k++)
	{
		k1 = k*rsize;
		k2 = (k + 1)*rsize - 1;
		for (int l = 0; l < nr; l++)
		{
			l1 = l*rsize;
			l2 = (l + 1)*rsize - 1;
			maccess(M, R, k1, k2, l1, l2, sv);
			off = mavg(R, rsize*rsize);
			// Initialize error to large value
			dmin = (int)pow(10, 6);
			i0 = 0;
			j0 = 0;
			m0 = 0;
			// Now cycle through each Domain Dij
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
							maccess(bigM, D, i1, i2, j1, j2, m, 8, rsize*nd);
							scale(D, s[n], rsize*rsize);
							del_g = off - mavg(D, rsize*rsize);
							increment(D, del_g, rsize*rsize);
							diff(R, D, temp, rsize*rsize);
							msquare(temp, temp2, rsize*rsize);
							sum_dist = msum(temp2, rsize*rsize);
							dist = sqrt(sum_dist);
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
			}
			T[((k*nr + l) * 5 + 0)] = i0;
			T[((k*nr + l) * 5 + 1)] = j0;
			T[((k*nr + l) * 5 + 2)] = m0;
			T[((k*nr + l) * 5 + 3)] = s0;
			T[((k*nr + l) * 5 + 4)] = g0;
		}
	}
	cout << "T created\n";
	free(temp);
	free(temp2);
	free(D);
	free(R);
	free(temp3);
}

__global__ void cuda_M1(int *M, int *M1, int rsize, int nd, int sv)
{
	__shared__ int *temp;
	__constant__ int stride = (rsize*nd) / (1024 * blockDim.x);
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**) temp, 4 * sizeof(int));

	int j = threadIdx.x % (rsize*nd);
	int k = threadIdx.x / (rsize*nd);

	for (int i = k; i < (rsize*nd); i += stride)
	{
		maccess(M, temp, i * 2, i * 2 + 1, j * 2, j * 2 + 1, sv);
		M1[((i*rsize*nd) + j)] = mavg(temp, 4);
	}
}

void cuda_encoder(int *h_r, int *h_g, int *h_b, int *h_Tr, int *h_Tg, int *h_Tb, int rsize, int nd, int nr, int sv, int sh)
{

	int *d_r, *d_g, *d_b , *d_Tr, *d_Tg, *d_Tb,  *d_M1_r, *d_M1_g, *d_M1_b, *d_bigM_r, *d_bigM_g, *d_bigM_b;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc ((void**) &d_r, sv * sh* sizeof (int));
	cudaStatus = cudaMalloc ((void**) &d_g, sv * sh*sizeof (int));
	cudaStatus = cudaMalloc ((void**) &d_b, sv * sh* sizeof (int));
	cudaStatus = cudaMalloc((void**)&d_Tr, sv * sh* sizeof(int));
	cudaStatus = cudaMalloc ((void**) &d_Tg, sv * sh* sizeof (int));
	cudaStatus = cudaMalloc ((void**) &d_Tb, sv * sh* sizeof (int));
	cudaStatus = cudaMalloc ((void**) &d_M1_r, sv * sh* sizeof (int));
	cudaStatus = cudaMalloc ((void**) &d_M1_g, sv * sh* sizeof (int));
	cudaStatus = cudaMalloc ((void**) &d_M1_b, sv * sh* sizeof (int));
	cudaStatus = cudaMalloc ((void**) &d_bigM_r, nd*nd*rsize*rsize*8 * sizeof(int));
	cudaStatus = cudaMalloc ((void**) &d_bigM_g, nd*nd*rsize*rsize*8 * sizeof(int));
	cudaStatus = cudaMalloc ((void**) &d_bigM_b, nd*nd*rsize*rsize*8 * sizeof(int));
  
	cudaStatus = cudaMemcpy (d_r, h_r,sv * sh* sizeof (int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy (d_g, h_g,sv * sh* sizeof (int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy (d_b, h_b, sv * sh* sizeof (int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy (d_Tr, h_Tr,sv * sh* sizeof (int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy (d_Tg, h_Tg,sv * sh* sizeof (int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy (d_Tb, h_Tb, sv * sh* sizeof (int), cudaMemcpyHostToDevice);

  //kernel call here
  
	cudaStatus = cudaMemcpy(h_Tr, d_Tr,  sv * sh* sizeof (int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(h_Tg, d_Tg,  sv * sh* sizeof (int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(h_Tb, d_Tb,  sv * sh* sizeof (int), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize ();
}