
//__global__ void nbody_kernel(int n, double4 *data, double4 *aux);
//__global__ void copyAux(double4 *data, double4 *aux);
__global__ void test(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux, int steps);
__global__ void nbody_kernel(int n, double4 *posData, double4 *velData, int steps);
__global__ void nbody_kernel_shared(int n, double4 *posData, double4 *velData, int steps, int bsize, int bnum);
__device__ float3 batch_calculation(double4 pos, float3 acc, double4* data, int bsize);
__global__ void nbody_kernel_2D(int n, double4 *posData, double4 *velData, int steps, int nx);
__global__ void nbody_kernel_shared_2D(int n, double4 *posData, double4 *velData, int steps, int bsize, int bnumx, int bnumy);

__global__ void nbody_kernel_single(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux);
__global__ void nbody_copy(double4 *posData, double4 *posAux, double4 *velData, double4 *velAux);