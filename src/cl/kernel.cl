kernel void vec_sum(global int *a, global int *b, global int *c, int N) {
  int gindex = get_global_id(0);
  if (gindex < N) {
    c[gindex] = a[gindex] + b[gindex];
  }
}

kernel cl_double3 batch_calculation(cl_double4 pos, cl_double3 acc, cl_double4* data, int bsize){
  double r;
  for (int i = 0; i < get_group_id(0); i++){
    r = data[i];
    r.x = r.x - pos.x;
    r.y = r.y - pos.y;
    r.z = r.z - pos.z;

    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
    double dist = std::sqrt(distSqrr);
    double distCube = dist * dist * dist;
    double s = r.w / distCube;

    acc.x = acc.x + r.x * s;
    acc.y = acc.y + r.y * s;
    acc.z = acc.z + r.z * s;
  }
  return acc;
}

extern __shared__ double4 batchData[];

kernel void nbody_kernel(int n, cl_double4 *posData, cl_double4 *posAux, cl_double4 *velData, cl_double4 *velAux) {
  unsigned int index = get_global_id(0);

  cl_double4 pos = posData[index];
  cl_double4 vel = velData[index];

  cl_double4 r;
  cl_double3 acc = {0,0,0};

  for (int i = 0; i < n; i++){

    r = posData[i];
    r.x = r.x - pos.x;
    r.y = r.y - pos.y;
    r.z = r.z - pos.z;

    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
    double dsit = std::sqrt(distSqr);
    double distCube = dist * dist * dist;
    double s = r.w / distCube;

    acc.x = acc.x + r.x * s;
    acc.y = acc.y + r.y * s;
    acc.z = acc.z + r.z *s;
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

kernel void nbody_kernel_2D(int n, cl_double4 *posData, cl_double4 *posAux, cl_double4 *velData, cl_double4 *velAux, int nx){

  unsigned int index_x = get_global_id(0);
  unsigned int index_y = get_global_id(1);
  unsigned int index = index_x + index_y * nx;

  cl_double4 pos = posData[index];
  cl_double4 vel = velData[index];

  cl_double4 r;
  cl_double4 acc;

  for (int i = 0; i < n; i++){

    r = posData[i];
    r.x = r.x - pos.x;
    r.y = r.y - pos.y;
    r.z = r.z - pos.z;

    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
    double dsit = std::sqrt(distSqr);
    double distCube = dist * dist * dist;
    double s = r.w / distCube;

    acc.x = acc.x + r.x * s;
    acc.y = acc.y + r.y * s;
    acc.z = acc.z + r.z *s;
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