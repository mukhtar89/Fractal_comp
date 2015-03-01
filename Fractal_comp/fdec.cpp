#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <assert.h>

#include "mutil.h"

using namespace std;

/*% Set timers
begrun = clock
cpu = cputime
M = imread('lena_gray_256.tif');
[sv sh] = size(M);*/


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