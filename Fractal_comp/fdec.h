//#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <conio.h>
#include <fstream>
#include <malloc.h>

cudaError_t cuda_decoder(int *h_r, int *h_g, int *h_b, int *h_Tr, int *h_Tg, int *h_Tb, int rsize, int nd, int nr, int sv, int sh);