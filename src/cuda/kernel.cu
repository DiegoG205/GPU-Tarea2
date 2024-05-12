#include "kernel.cuh"
#include <stdio.h>

// __global__ void nbody_kernel(int n, double4 *data) {

//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int pos_index = 2*index;
//   unsigned int vel_index = 2*index + 1;

//   // position and velocity (last frame)
//   double4 pos = data[pos_index];
//   double4 vel = data[vel_index];
//   double4 r, acc;

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

//   vel.x += acc.x;
//   vel.y += acc.y;
//   vel.z += acc.z;

//   pos.x += vel.x;
//   pos.y += vel.y;
//   pos.z += vel.z;

//   __syncthreads();

//   data[pos_index] = pos;
//   data[vel_index] = vel;
// };

// __global__ void nbody_kernel(int n, double4 *data, double4 *aux) {

//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int pos_index = 2*index;
//   unsigned int vel_index = 2*index + 1;

//   // position and velocity (last frame)
//   double4 pos = data[pos_index];
//   double4 vel = data[vel_index];
//   double4 r, acc;

//   __threadfence_system();

//   for (int i = 0; i < n; i++) {

//     r = data[i];
//     r.x = r.x - pos.x;
//     r.y = r.y - pos.y;
//     r.z = r.z - pos.z;

//     double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
//     double dist = std::sqrt(distSqr);
//     double distCube = dist * dist * dist;
//     double s = r.w / distCube;

//     acc.x = acc.x + r.x * s;
//     acc.y = acc.y + r.y * s;
//     acc.z = acc.z + r.z * s;

//   }

//   vel.x = vel.x + acc.x;
//   vel.y = vel.y + acc.y;
//   vel.z = vel.z + acc.z;

//   pos.x = pos.x + vel.x;
//   pos.y = pos.y + vel.y;
//   pos.z = pos.z + vel.z;

//   aux[pos_index] = pos;
//   aux[vel_index] = vel;
// };

// __global__ void copyAux(double4 *data, double4 *aux) {
//   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int pos_index = 2*index;
//   data[pos_index] = aux [pos_index];
//   data[pos_index + 1] = aux [pos_index + 1];
// };

__global__ void nbody_kernel(int n, int4 *posData, double4 *velData, int steps) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while(steps--) {
    // position and velocity (last frame)
    int4 pos = posData[index];
    double4 vel = velData[index];

    int4 r;
    double4 acc;

    for (int i = 0; i < n; i++) {

      r = posData[i];
      r.x = r.x - pos.x;
      r.y = r.y - pos.y;
      r.z = r.z - pos.z;

      double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
      double dist = std::sqrt(distSqr);
      double distCube = dist * dist * dist;
      double s = r.w / distCube;

      acc.x = acc.x + r.x * s;
      acc.y = acc.y + r.y * s;
      acc.z = acc.z + r.z * s;

      // No tengo idea de por que, pero este print evita que la simulacion explote
      printf("");


    }

    vel.x = vel.x + acc.x;
    vel.y = vel.y + acc.y;
    vel.z = vel.z + acc.z;

    pos.x = pos.x + lround(vel.x);
    pos.y = pos.y + lround(vel.y);
    pos.z = pos.z + lround(vel.z);

    __syncthreads();
    __threadfence_system();

    posData[index] = pos;
    velData[index] = vel;

    //printf("Particula %d: (%f,%f,%f)\n", index, vel.x, vel.y, vel.z);

  }

};

__device__ float3 batch_calculation(int4 pos, float3 acc, int4* data, int bsize) {
  
  int4 r;
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

extern __shared__ int4 batchData[];
__global__ void nbody_kernel_shared(int n, int4 *posData, double4 *velData, int steps, int bsize, int bnum) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while(steps--) {
    // position and velocity (last frame)
    int4 pos = posData[index];
    double4 vel = velData[index];
    float3 acc;

    for (int i = 0; i < bnum; i++) {

      batchData[threadIdx.x] = posData[threadIdx.x + i * blockDim.x];

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
    __threadfence_system();

    posData[index] = pos;
    velData[index] = vel;

    __syncthreads();
  }

};