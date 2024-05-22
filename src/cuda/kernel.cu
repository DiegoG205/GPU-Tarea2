#include "kernel.cuh"
#include <stdio.h>


__global__ void nbody_kernel(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  // position and velocity (last frame)
  double4 pos = posData[index];
  double4 vel = velData[index];

  double4 r;
  double3 acc = {0,0,0};

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

  }

  vel.x = vel.x + acc.x;
  vel.y = vel.y + acc.y;
  vel.z = vel.z + acc.z;

  pos.x = pos.x + vel.x;
  pos.y = pos.y + vel.y;
  pos.z = pos.z + vel.z;
  
  posAux[index] = pos;
  velAux[index] = vel;
};

__device__ double3 batch_calculation(double4 pos, double3 acc, double4* data, int bsize) {
  
  double4 r;
  for (int i = 0; i < blockDim.x; i++) {
    r = data[i];
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
  }
  return acc;
}

extern __shared__ double4 batchData[];

__global__ void nbody_kernel_shared(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux, int bsize, int bnum) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // position and velocity (last frame)
  double4 pos = posData[index];
  double4 vel = velData[index];

  double3 acc = {0,0,0};

  for (int i = 0; i < bnum; i++) {

    batchData[threadIdx.x] = posData[threadIdx.x + i * blockDim.x];
    __syncthreads();

    for (int i = 0; i < blockDim.x; i++) {
      double4 r = batchData[i];
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
    }
    __syncthreads();
  }

  vel.x = vel.x + acc.x;
  vel.y = vel.y + acc.y;
  vel.z = vel.z + acc.z;

  pos.x = pos.x + vel.x;
  pos.y = pos.y + vel.y;
  pos.z = pos.z + vel.z;

  posAux[index] = pos;
  velAux[index] = vel;
};

__global__ void nbody_kernel_2D(int n, double4 *posData, double4 *posAux, double4 *velData, double4 *velAux) {

  unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = index_x + index_y*blockDim.x*gridDim.x;

  // position and velocity (last frame)
  double4 pos = posData[index];
  double4 vel = velData[index];

  double4 r;
  double3 acc = {0,0,0};

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
  }

  vel.x = vel.x + acc.x;
  vel.y = vel.y + acc.y;
  vel.z = vel.z + acc.z;

  pos.x = pos.x + vel.x;
  pos.y = pos.y + vel.y;
  pos.z = pos.z + vel.z;

  posAux[index] = pos;
  velAux[index] = vel;

  //printf("%d %f %f %f\n", index, pos.x, pos.y, pos.z);
};