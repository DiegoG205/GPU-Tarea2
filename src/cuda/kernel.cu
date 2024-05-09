#include "kernel.cuh"

// __global__ void vec_sum(int *a, int *b, int *c, int n){
//   int idx = blockDim.x * blockIdx.x + threadIdx.x;
//   if(idx < n) {
//     c[idx] = a[idx] + b[idx];
//   }
// };

// __global__ void nbody_kernel(int n, float4 *data, int step) {

//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int pos_index = 2*index;
//   unsigned int vel_index = 2*index + 1;

//   // position and velocity (last frame)
//   float4 pos = data[pos_index];
//   float4 vel = data[vel_index];
//   float4 r, acc;

//   for (int i = 0; i < n; i++) {

//     r = data[i];
//     r.x -= pos.x;
//     r.y -= pos.y;
//     r.z -= pos.z;

//     double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
//     double dist = std::sqrt(distSqr);
//     double distCube = dist * dist * dist;
//     double s = r.w / distCube;

//     acc.x += r.x * s;
//     acc.y += r.y * s;
//     acc.z += r.z * s;

//   }

//   vel.x += acc.x; //* step;
//   vel.y += acc.y; //* step;
//   vel.z += acc.z; //* step;

//   pos.x += vel.x; //* step;
//   pos.y += vel.y; //* step;
//   pos.z += vel.z; //* step;

//   __syncthreads();

//   data[pos_index] = pos;
//   data[vel_index] = vel;
// };

__global__ void nbody_kernel(int n, float4 *data, int steps) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int pos_index = 2*index;
  unsigned int vel_index = 2*index + 1;

  while(steps--) {
    // position and velocity (last frame)
    float4 pos = data[pos_index];
    float4 vel = data[vel_index];
    float4 r, acc;

    for (int i = 0; i < n; i++) {

      r = data[i];
      r.x -= pos.x;
      r.y -= pos.y;
      r.z -= pos.z;

      double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
      double dist = std::sqrt(distSqr);
      double distCube = dist * dist * dist;
      double s = r.w / distCube;

      acc.x += r.x * s;
      acc.y += r.y * s;
      acc.z += r.z * s;

    }

    vel.x += acc.x;
    vel.y += acc.y;
    vel.z += acc.z;

    pos.x += vel.x;
    pos.y += vel.y;
    pos.z += vel.z;

    __syncthreads();

    data[pos_index] = pos;
    data[vel_index] = vel;

    __syncthreads();
  }

};

__device__ float3 batch_calculation(float4 pos, float3 acc, float4* data, int bsize) {
  
  float4 r;
  for (int i = 0; i < bsize; i++) {
    r = data[i];
    r.x -= pos.x;
    r.y -= pos.y;
    r.z -= pos.z;

    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
    double dist = std::sqrt(distSqr);
    double distCube = dist * dist * dist;
    double s = r.w / distCube;

    acc.x += r.x * s;
    acc.y += r.y * s;
    acc.z += r.z * s;
  }
  return acc;
}

__global__ void nbody_kernel_shared(int n, float4 *data, int steps, int bsize, int bnum) {

  __shared__ float4 batchData[32];

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int pos_index = 2*index;
  unsigned int vel_index = 2*index + 1;

  while(steps--) {
    // position and velocity (last frame)
    float4 pos = data[pos_index];
    float4 vel = data[vel_index];
    float3 acc;

    for (int i = 0; i < bnum; i++) {

      batchData[threadIdx.x] = data[threadIdx.x + i * blockDim.x];

      __syncthreads();

      acc = batch_calculation(pos, acc, batchData, bsize);

      __syncthreads();
    }

    vel.x += acc.x;
    vel.y += acc.y;
    vel.z += acc.z;

    pos.x += vel.x;
    pos.y += vel.y;
    pos.z += vel.z;

    __syncthreads();

    data[pos_index] = pos;
    data[vel_index] = vel;

    __syncthreads();
  }

};