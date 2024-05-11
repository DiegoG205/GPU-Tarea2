
//__global__ void nbody_kernel(int n, float4 *data, float4 *aux);
//__global__ void copyAux(float4 *data, float4 *aux);
__global__ void nbody_kernel(int n, float4 *data, int steps);
__global__ void nbody_kernel_shared(int n, float4 *data, int steps, int bsize, int bnum);
__device__ float3 batch_calculation(float4 pos, float3 acc, float4* data, int bsize);
