#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <conio.h>
#include <fstream>
#include <malloc.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

cudaError_t cuda_encoder(int *h_r, int *h_g, int *h_b, int *h_Tr, int *h_Tg, int *h_Tb, int rsize, int nd, int nr, int sv, int sh);

__global__ void cuda_M1(int *M, int *M1, int rsize, int nd, int sv);