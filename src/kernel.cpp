#include "kernel.h"
#include <vector>
#include <cmath>
#include <iostream>

void vec_sum(int* a, int* b, int* c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
};

void nbody_sec(int n, std::vector<pData> &data, int steps) {

  std::vector<pData> temp(n);

  for (int k = 1; k <= steps; k++) {
    for (int i = 0; i < n; i++) {

      std::vector<float> acc(3,0);

      for (int j = 0; j < n; j++) {

        if (j == i) continue;

        float rx,ry,rz;

        rx = data[j].x - data[i].x;
        ry = data[j].y - data[i].y;
        rz = data[j].z - data[i].z;

        double distSqr = rx * rx + ry * ry + rz * rz + 0.1;

        double dist = std::sqrt(distSqr);
        double distCube = dist * dist * dist;

        double s = data[j].m / distCube;

        acc[0] += rx * s;
        acc[1] += ry * s;
        acc[2] += rz * s;
      }
      float vx, vy, vz;
      vx = data[i].vx + acc[0];
      vy = data[i].vy + acc[1];
      vz = data[i].vz + acc[2];

      temp[i] = {data[i].x + vx, data[i].y + vy, data[i].z + vz, data[i].m, vx, vy, vz};
    }

    for (int i = 0; i < n; i++) {
      data[i] = temp[i];
    }
  }
};