#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <math.h>


void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int size);     //access A, save into R

void maccess(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size);    //access A, save into R

void msave(int *A, int *R, int a1, int a2, int b1, int b2, int c, int depth, int size);    //access A, save into R

void msave(int *A, int *R, int a1, int a2, int b1, int b2, int size);    //access A, save into R

int mavg(int *R, int size);

void rotmat(int *D, int *rot, int size);   //rotate matrix D, save it into rot

void fliph(int *D, int *rot, int size);   //rotate matrix D, save it into rot

void flipv(int *D, int *rot, int size);   //rotate matrix D, save it into rot

void transpose(int *D, int *rot, int size);   //rotate matrix D, save it into rot

void scale(int *D, float k, int size); //scale matrix with k

void increment(int *D, float k, int size); //increment each matrix element with k

void diff(int *A, int *B, int *C, int size); //each element of A-B = C

void msquare(int *A, int *B, int size); //B = A.^2

int msum(int *A, int size); //Sum of all elements of A

void ones(int *A, int k, int size); //initialize Array A with scalar 'k'

void matsum(int *A, int *B, int *C, int size);  // add array A + B = C

void matcpy(int *A, int *B, int size);  //copy one matrix to another
