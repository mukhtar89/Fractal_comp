#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <assert.h>

#include "mutil.h"

void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int size)
{
	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[((a - a1)*(b2 - b1 + 1) + (b - b1))] = A[(a*size + b)];
}

void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size)
{
	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[((a - a1)*(b2 - b1 + 1) + (b - b1))] = A[((a*size + b)*depth + c)];
}
void msave(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size)
{
	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[((a*size + b)*depth + c)] = A[((a - a1)*(b2 - b1 + 1) + (b - b1))];
}

void msave(int *A, int *R, int a1, int a2, int b1, int b2, int size)
{

	for (int a = a1; a <= a2; a++)
		for (int b = b1; b <= b2; b++)
			R[(a*size + b)] = A[((a - a1)*(b2 - b1 + 1) + (b - b1))];
}

int mavg(int *R, int size)
{
	int sum = msum(R, size);
	int avg = sum / size;
	return avg;
}

void diff(int *A, int *B, int *C, int size) //each element of A-B = C
{
	for (int i = 0; i < size; i++)
		C[i] = A[i] - B[i];
}

void msquare(int *A, int *B, int size) //B = A.^2
{
	for (int i = 0; i < size; i++)
		B[i] = A[i] * A[i];
}

int msum(int *A, int size) //Sum of all elements of A
{
	int sum = 0;
	for (int i = 0; i < size; i++)
		sum+=A[i];
	return sum;
}

void ones(int *A, int k, int size)   //initialize Array A with scalar 'k'
{
	for (int i = 0; i < size; i++)
		A[i] = k;
}

void matsum(int *A, int *B, int *C, int size)  // add array A + B = C
{
	for (int i = 0; i < size; i++)
		C[i] = A[i] + B[i];
}

void rotmat(int *D, int *rot, int size)
{
	int n = sqrt(size);
	for (int i = 0; i <n ; i++)
		for (int j = 0; j < n; j++)
			rot[i*n + j] = D[(n - j - 1)*n + i];
}

void transpose(int *D, int *rot, int size)
{
	int n = sqrt(size);
	for (int i = 0; i < n; i++)
		for (int j = i; j < n; j++)
			rot[i*n + j] = D[j*n + i];
}

void flipv(int *D, int *rot, int size)
{
	int n = sqrt(size);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			rot[i*n + j] = D[i*n + n - j - 1];
}

void fliph(int *D, int *rot, int size)
{
	int n = sqrt(size);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			rot[i*n + j] = D[(n - i - 1)*n + j];
}

void scale(int *D, float k, int size)
{
	for (int i = 0; i < size; i++)
		D[i] = D[i] * k;
}

void increment(int *D, float k, int size)
{
	for (int i = 0; i < size; i++)
		D[i] = D[i] + k;
}

void matcpy(int *A, int *B, int size)
{
	for (int i = 0; i < size; i++)
		A[i] = B[i];
}