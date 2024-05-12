
//__global__ void nbody_kernel(int n, double4 *data, double4 *aux);
//__global__ void copyAux(double4 *data, double4 *aux);
__global__ void nbody_kernel(int n, int4 *posData, double4 *velData, int steps);
__global__ void nbody_kernel_shared(int n, int4 *posData, double4 *velData, int steps, int bsize, int bnum);
__device__ float3 batch_calculation(int4 pos, float3 acc, int4* data, int bsize);
