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

__device__ void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int size)
{
	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[((a - a1)*(b2 - b1 + 1) + (b - b1))] = A[(a*size + b)];
}

__device__ void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size)
{
	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[((a - a1)*(b2 - b1 + 1) + (b - b1))] = A[((a*size + b)*depth + c)];
}

__device__ void msave(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size)
{
	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[((a*size + b)*depth + c)] = A[((a - a1)*(b2 - b1 + 1) + (b - b1))];
}

__device__  void msave(int *A, int *R, int a1, int a2, int b1, int b2, int size)
{

	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[(a*size + b)] = A[((a - a1)*(b2 - b1 + 1) + (b - b1))];
}

__device__ int mavg(int *R, int size)
{
	int sum = msum(R, size);
	int avg = sum / size;
	return avg;
}

__device__ void diff(int *A, int *B, int *C, int size) //each element of A-B = C
{
	for (int i = 0; i < size; i++)
		C[i] = A[i] - B[i];
}

__device__ void msquare(int *A, int *B, int size) //B = A.^2
{
	for (int i = 0; i < size; i++)
		B[i] = A[i] * A[i];
}

__device__ int msum(int *A, int size) //Sum of all elements of A
{
	int sum = 0;
	for (int i = 0; i < size; i++)
		sum += A[i];
	return sum;
}

__device__ void ones(int *A, int k, int size)   //initialize Array A with scalar 'k'
{
	for (int i = 0; i < size; i++)
		A[i] = k;
}

__device__ void matsum(int *A, int *B, int *C, int size)  // add array A + B = C
{
	for (int i = 0; i < size; i++)
		C[i] = A[i] + B[i];
}

__device__ void rotmat(int *D, int *rot, int size)
{
	for (int i = 0; i <size; i++)
		for (int j = 0; j < size; j++)
			rot[i*size + j] = D[(size - j - 1)*size + i];
}

__device__ void transpose(int *D, int *rot, int size)
{
	for (int i = 0; i < size; i++)
		for (int j = i; j < size; j++)
			rot[i*size + j] = D[j*size + i];
}

__device__ void flipv(int *D, int *rot, int size)
{
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			rot[i*size + j] = D[i*size + size - j - 1];
}

__device__ void fliph(int *D, int *rot, int size)
{
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			rot[i*size + j] = D[(size - i - 1)*size + j];
}

__device__ void scale(int *D, float k, int size)
{
	for (int i = 0; i < size; i++)
		D[i] = D[i] * k;
}

__device__ void increment(int *D, float k, int size)
{
	for (int i = 0; i < size; i++)
		D[i] = D[i] + k;
}

__device__ void matcpy(int *A, int *B, int size)
{
	for (int i = 0; i < size; i++)
		A[i] = B[i];
}