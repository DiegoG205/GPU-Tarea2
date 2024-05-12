#include "kernel.cuh"
#include <stdio.h>

__global__ void nbody_kernel(int n, double4 *posData, double4 *velData, int steps) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while(steps--) {
    // position and velocity (last frame)
    double4 pos = posData[index];
    double4 vel = velData[index];

    double4 r;
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

      // No tengo idea de por que, pero este print a veces evita que la simulacion explote
      printf("");


    }

    vel.x = vel.x + acc.x;
    vel.y = vel.y + acc.y;
    vel.z = vel.z + acc.z;

    pos.x = pos.x + vel.x;
    pos.y = pos.y + vel.y;
    pos.z = pos.z + vel.z;

    __syncthreads();
    __threadfence_system();

    posData[index] = pos;
    velData[index] = vel;

    //printf("Particula %d: (%f,%f,%f)\n", index, vel.x, vel.y, vel.z);

  }

};

__device__ float3 batch_calculation(double4 pos, float3 acc, double4* data, int bsize) {
  
  double4 r;
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

    // No tengo idea de por que, pero este print evita que la simulacion explote
      printf("");
  }
  return acc;
}

extern __shared__ double4 batchData[];
__global__ void nbody_kernel_shared(int n, double4 *posData, double4 *velData, int steps, int bsize, int bnum) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while(steps--) {
    // position and velocity (last frame)
    double4 pos = posData[index];
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

__global__ void nbody_kernel_2D(int n, double4 *posData, double4 *velData, int steps, int nx) {

  unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int index = index_x + index_y*nx;

  //printf("Index_X: %d, Index_Y: %d, Index: %d\n", index_x, index_y, index);

  while(steps--) {
    // position and velocity (last frame)
    double4 pos = posData[index];
    double4 vel = velData[index];

    double4 r;
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

    pos.x = pos.x + vel.x;
    pos.y = pos.y + vel.y;
    pos.z = pos.z + vel.z;

    __syncthreads();
    __threadfence_system();

    posData[index] = pos;
    velData[index] = vel;
  }

};

__global__ void nbody_kernel_shared_2D(int n, double4 *posData, double4 *velData, int steps, int bsize, int bnumx, int bnumy) {

  unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int index = index_x + index_y*bsize*bnumx;
  unsigned int thread_index = threadIdx.x + threadIdx.y*bsize;

  while(steps--) {
    // position and velocity (last frame)
    double4 pos = posData[index];
    double4 vel = velData[index];
    float3 acc;

    for (int i = 0; i < bnumx; i++) {
      for (int j = 0; j < bnumy; j++) {
        printf("%d\n", thread_index + i * blockDim.x + j * blockDim.y);
        batchData[thread_index] = posData[thread_index + i * blockDim.x + j * blockDim.y];

        __syncthreads();

        acc = batch_calculation(pos, acc, batchData, bsize);

        __syncthreads();
      } 
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