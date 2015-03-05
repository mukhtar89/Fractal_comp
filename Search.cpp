/*
I have created 64*64 blocks inside 1 grid. Each block will only contain 1 thread.
I could have created 1 block with 64*64 threads also.
But this way , A thread does not have to wait to access the functional unit.
*/


dim3 grid = (sv/rsize,sh/rsize,1);
dim3 block = (1,1,1);

__global__  void search(int *d_M1,int *T, int *d_bigM, int rsize, int nd, int nr, int sv, int sh)
{
	int k1, k2, l1, l2, off, i0, j0, m0, s0, g0, del_g, sum_dist;
	float dist, dmin;
	int domainDim = nd*nd*rsize*rsize*8;
	__shared__ int sharedBigM[domainDim];

	for (int i = 0; i< domainDim; i++) sharedBigM[i] = d_bigM[i];

	int R[rsize][rsize];
	int D[rsize][rsize];

	int k = blockIdx.x;
	int l = blockIdx.y;

	k1 = k*rsize;
	k2 = (k+1)*rsize - 1;
	l1 = l*rsize;
	l2 = (l+1)*rsize - 1;

	maccess(M, R, k1, k2, l1, l2, sv);
	off = mavg(R, rsize*rsize);

	dmin = (int)pow(10, 6);
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
		T[((k*nr + l) * 5 + 0)] = i0;
		T[((k*nr + l) * 5 + 1)] = j0;
		T[((k*nr + l) * 5 + 2)] = m0;
		T[((k*nr + l) * 5 + 3)] = s0;
		T[((k*nr + l) * 5 + 4)] = g0;
	}	
}