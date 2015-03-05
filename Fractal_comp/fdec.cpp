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

void fdec(int *M, int *T, int rsize, int nd, int nr, int sv, int sh)
{
	int *R, *M1, *temp, *D, *bigM, *temp2, *MM, *temp3, count = 0;
	int i, j, i1, i2, j1, j2, k1, k2, l1, l2, off, i0, j0, m0, g0, s0, del_g, sum_dist;
	MM = (int*)malloc(sv*sh * sizeof(int));
	M1 = (int*)malloc((int)pow(rsize*nd, 2) * sizeof(int));

	ones(MM, 100, sv*sh);

	D = (int*)malloc(rsize * rsize * sizeof(int));
	R = (int*)malloc(rsize * rsize * sizeof(int));
	temp2 = (int*)malloc(rsize * rsize * sizeof(int));
	temp = (int*)malloc(rsize * rsize * sizeof(int));
	temp3 = (int*)malloc(4 * sizeof(int));


	//Begin batch runs
	//Matrix of 4 possible scalings to transform grayscale
	float s[4] = { 0.45, 0.60, 0.80, 1.00 };
	int min0 = 100;

	for (int iter = 0; iter < 10; iter++)
	{
		//Scale the Domain Blocks
		for (int i = 0; i < rsize*nd; i++)
		{
			for (int j = 0; j < rsize*nd; j++)
			{
				maccess(MM, temp3, i * 2, i * 2 + 1, j * 2, j * 2 + 1, sv);
				M1[((i*rsize*nd) + j)] = mavg(temp3, 4);
			}
		}
		

		// Compare the range blocks and scaled domain blocks.
		// k, l - used to cycle through blocks Rkl.
		for (int k = 0; k < nr; k++)
		{
			k1 = k*rsize;
			k2 = (k + 1)*rsize - 1;
			for (int l = 0; l < nr; l++)
			{
				l1 = l*rsize;
				l2 = (l + 1)*rsize - 1;
				i0 = T[((k*nr + l) * 5 + 0)];
				j0 = T[((k*nr + l) * 5 + 1)];
				m0 = T[((k*nr + l) * 5 + 2)];
				s0 = T[((k*nr + l) * 5 + 3)];
				g0 = T[((k*nr + l) * 5 + 4)];
				i1 = i0*rsize;
				i2 = (i0 + 1)*rsize - 1;
				j1 = j0*rsize;
				j2 = (j0 + 1)*rsize - 1;
				maccess(M1, D, i1, i2, j1, j2, rsize*nd);
				matcpy(temp, D, rsize*rsize);
				if (m0 == 1)
					rotmat(temp, D, rsize*rsize);
				else if (m0 == 2)
				{
					rotmat(temp, temp2, rsize*rsize);
					rotmat(temp2, D, rsize*rsize);
				}
				else if (m0 == 3)
				{
					rotmat(temp, temp2, rsize*rsize);
					rotmat(temp2, temp, rsize*rsize);
					rotmat(temp, D, rsize*rsize);
				}
				else if (m0 == 4)
					fliph(temp, D, rsize*rsize);
				else if (m0 == 5)
					flipv(temp, D, rsize*rsize);
				else if (m0 == 6)
					transpose(temp, D, rsize*rsize);
				else if (m0 == 7)
				{
					transpose(temp, D, rsize*rsize);
					rotmat(D, temp, rsize*rsize);
					rotmat(temp, D, rsize*rsize);
				}
				scale(D, s[s0], rsize*rsize);
				ones(temp, g0, rsize*rsize);
				matsum(D, temp, R, rsize*rsize);
				msave(R, MM, k1, k2, l1, l2, sv);
			}
		}
	}
	matcpy(M, MM, sv*sh);
	free(D);
	free(temp2);
	free(R);
	free(MM);
	free(M1);
	free(temp);
	free(temp3);
}

void cuda_decoder(int *h_r, int *h_g, int *h_b, int *h_Tr, int *h_Tg, int *h_Tb, int rsize, int nd, int nr, int sv, int sh)
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
}