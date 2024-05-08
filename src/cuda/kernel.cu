#include "kernel.cuh"

__global__ void vec_sum(int *a, int *b, int *c, int n){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < n) {
    c[idx] = a[idx] + b[idx];
  }
};

__global__ void nbody_kernel(int n, float4 *data, int step) {

  // index for vertex (pos)
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = 2*index;
  unsigned int y = 2*index + 1;

  // position and velocity (last frame)
  float4 pos = data[x];
  float4 vel = data[y];
  float3 r, acc;

  for (int i = 0; i < n; i++) {

    r.x = pos.x - data[i].x;
    r.y = pos.y - data[i].y;
    r.z = pos.z - data[i].z;

    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    double dist = std::sqrt(distSqr);
    double distCube = dist * dist * dist;
    double s = pos.w / distCube;

    acc.x += r.x * s;
    acc.y += r.y * s;
    acc.z += r.z * s;

  }

  float4 p, v;
  vel.x += acc.x * step;
  vel.y += acc.y * step;
  vel.z += acc.z * step;

  pos.x += vel.x * step;
  pos.y += vel.y * step;
  pos.z += vel.z * step;

  __syncthreads();

  data[x] = pos;
  data[y] = vel;
};