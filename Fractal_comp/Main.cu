#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <conio.h>
#include <fstream>
#include <malloc.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "fenc.h"
#include "fdec.h"

#define IMAGE "lena256.BMP"
#define FILE "lena256.frct"

using namespace std;
using namespace cv;

int main()
{
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		_getch();
		return -1;
	}
	
	Mat img = imread(IMAGE, CV_LOAD_IMAGE_UNCHANGED);
	if (img.data == NULL)
	{
		cout << "Image cannot be loaded..!!" << endl;
		system("pause");
		return -1;
	}

	int sv, sh;
	sv = img.rows;
	sh = img.cols;
	if (sv != sh)
	{
		cout << "\nImage is not square";
		return 1;
	}

	int count = 0;
	int *r, *g, *b, *Tr, *Tb, *Tg;
	r = (int*)malloc(img.total()*sizeof(int));
	b = (int*)malloc(img.total()*sizeof(int));
	g = (int*)malloc(img.total()*sizeof(int));
	int k = 0;
	for (int i = 0; i<img.rows; i++) {
		for (int j = 0; j<img.cols; j++) {
			Vec3b p = img.at<Vec3b>(i, j);
			r[k] = int(p[2]);
			g[k] = int(p[1]);
			b[k] = int(p[0]);
			k++;

		}
	}

	int rsize = 4;
	int nd, nr;
	nd = (sv / rsize) / 2;
	nr = sv / rsize;

	Tr = (int*)malloc(nr*nr * 5 * sizeof(int));
	Tb = (int*)malloc(nr*nr * 5 * sizeof(int));
	Tg = (int*)malloc(nr*nr * 5 * sizeof(int));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize();
	cudaEventRecord(start, 0);

	cudaStatus = cuda_encoder(r, b, g, Tr, Tb, Tg, rsize, nd, nr, sv, sh);

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, stop, start);

	cout << "\nEncoding time: " << elapsedTime << " milliseconds\n";

	ofstream fout;
	fout.open(FILE);
	k = nr*nr * 5;
	for (int i = 0; i < k; i++)
		fout << Tr[i] << " "; //writing ith character of array in the file
	fout << "\n";
	for (int i = 0; i < k; i++)
		fout << Tg[i] << " "; //writing ith character of array in the file
	fout << "\n";
	for (int i = 0; i < k; i++)
		fout << Tb[i] << " "; //writing ith character of array in the file
	fout << "\n";

	fout.close();

	int *r1, *g1, *b1, *Tr1, *Tb1, *Tg1;

	Tr1 = (int*)malloc(nr*nr * 5 * sizeof(int));
	Tb1 = (int*)malloc(nr*nr * 5 * sizeof(int));
	Tg1 = (int*)malloc(nr*nr * 5 * sizeof(int));
	r1 = (int*)malloc(img.total()*sizeof(int));
	b1 = (int*)malloc(img.total()*sizeof(int));
	g1 = (int*)malloc(img.total()*sizeof(int));

	ifstream fin;
	fin.open(FILE);
	string line;
	int value;

	k = 0;
	if (getline(fin, line))
	{
		std::istringstream iss(line);
		while (iss >> value)
		{
			Tr1[k] = value;
			k++;
		}
	}
	k = 0;
	if (getline(fin, line))
	{
		std::istringstream iss(line);
		while (iss >> value)
		{
			Tg1[k] = value;
			k++;
		}
	}
	k = 0;
	if (getline(fin, line))
	{
		std::istringstream iss(line);
		while (iss >> value)
		{
			Tb1[k] = value;
			k++;
		}
	}

	cudaDeviceSynchronize();
	cudaEventRecord(start, 0);

	cudaStatus = cuda_decoder(r1, b1, g1, Tr1, Tb1, Tg1, rsize, nd, nr, sv, sh);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "\nDecoding time: " << elapsedTime << " milliseconds\n";

	Mat A(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

	k = 0;
	for (int i = 0; i<A.rows; i++){
		for (int j = 0; j<A.cols; j++){
			A.data[A.channels()*(A.cols*i + j) + 2] = r1[k]; 
			A.data[A.channels()*(A.cols*i + j) + 1] = g1[k]; 
			A.data[A.channels()*(A.cols*i + j) + 0] = b1[k];
			k++;
		}
	}

	namedWindow("MyImage", CV_WINDOW_AUTOSIZE);

	imshow("MyImage", A);

	

    // Add vectors in parallel.
    /*cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }*/


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	waitKey(0);
	free(r1);
	free(b1);
	free(g1);
	free(Tr1);
	free(Tb1);
	free(Tg1);
	free(r);
	free(b);
	free(g);
	free(Tr);
	free(Tb);
	free(Tg);

	return 0;
}