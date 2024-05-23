kernel void vec_sum(global int *a, global int *b, global int *c, int N) {
  int gindex = get_global_id(0);
  if (gindex < N) {
    c[gindex] = a[gindex] + b[gindex];
  }
}

kernel void nbody_kernel(int n, global double4 *posData, global double4 *posAux, global double4 *velData, global double4 *velAux) {
  unsigned int index = get_global_id(0);

  double4 pos = posData[index];
  double4 vel = velData[index];

  double4 r;
  double3 acc = {0,0,0};

  for (int i = 0; i < n; i++){

    r = posData[i];
    r.x = r.x - pos.x;
    r.y = r.y - pos.y;
    r.z = r.z - pos.z;

    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
    double dist = sqrt(distSqr);
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

kernel void nbody_kernel_2D(int n, global double4 *posData, global double4 *posAux, global double4 *velData, global double4 *velAux, int nx){

  unsigned int index_x = get_global_id(0);
  unsigned int index_y = get_global_id(1);
  unsigned int index = index_x + index_y * nx;

  double4 pos = posData[index];
  double4 vel = velData[index];

  double4 r;
  double3 acc = {0,0,0};

  for (int i = 0; i < n; i++){

    r = posData[i];
    r.x = r.x - pos.x;
    r.y = r.y - pos.y;
    r.z = r.z - pos.z;

    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
    double dist = sqrt(distSqr);
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

kernel void nbody_kernel_shared(int n, global double4 *posData, global double4 *posAux, global double4 *velData, global double4 *velAux, local double4 *batchData, int bnum) {

  unsigned int index = get_global_id(0);
  unsigned int l_index = get_local_id(0);
  unsigned int l_size = get_local_size(0);

  double4 pos = posData[index];
  double4 vel = velData[index];

  double3 acc = {0,0,0};

  for (int i = 0; i < bnum; i++) {

    batchData[l_index] = posData[l_index + i * l_size];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < l_size; i++) {
      double4 r = batchData[i];
      r.x = r.x - pos.x;
      r.y = r.y - pos.y;
      r.z = r.z - pos.z;

      double distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.1;
      double dist = sqrt(distSqr);
      double distCube = dist * dist * dist;
      double s = r.w / distCube;

      acc.x = acc.x + r.x * s;
      acc.y = acc.y + r.y * s;
      acc.z = acc.z + r.z * s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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