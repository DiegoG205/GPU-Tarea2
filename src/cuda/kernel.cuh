
//__global__ void nbody_kernel(int n, double4 *posData, double4 *velData, int steps);
__global__ void nbody_kernel_shared(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux, int bsize, int bnum);
__device__ double3 batch_calculation(double4 pos, double3 acc, double4* data, int bsize);
__global__ void nbody_kernel_2D(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux, int nx);
__global__ void nbody_kernel_shared_2D(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux, int bsize, int bnumx, int bnumy);
__global__ void nbody_kernel(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux);