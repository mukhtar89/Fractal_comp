#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

__device__  void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int size);     //access A, save into R

__device__  void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size);    //access A, save into R

__device__  void msave(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size);    //access A, save into R

__device__  void msave(int *A, int *R, int a1, int a2, int b1, int b2, int size);    //access A, save into R

__device__  int mavg(int *R, int size);

__device__  void rotmat(int *D, int *rot, int size);   //rotate matrix D, save it into rot

__device__  void fliph(int *D, int *rot, int size);   //rotate matrix D, save it into rot

__device__  void flipv(int *D, int *rot, int size);   //rotate matrix D, save it into rot

__device__  void transpose(int *D, int *rot, int size);   //rotate matrix D, save it into rot

__device__  void scale(int *D, float k, int size); //scale matrix with k

__device__  void increment(int *D, float k, int size); //increment each matrix element with k

__device__  void diff(int *A, int *B, int *C, int size); //each element of A-B = C

__device__  void msquare(int *A, int *B, int size); //B = A.^2

__device__  int msum(int *A, int size); //Sum of all elements of A

__device__  void ones(int *A, int k, int size); //initialize Array A with scalar 'k'

__device__  void matsum(int *A, int *B, int *C, int size);  // add array A + B = C

__device__  void matcpy(int *A, int *B, int size);  //copy one matrix to another
