
__global__ void nbody_kernel_shared(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux, int bsize, int bnum);
__global__ void nbody_kernel_2D(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux);
__global__ void nbody_kernel(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux);