#include "kernel.h"
#include <vector>
#include <cmath>
#include <iostream>

void vec_sum(int* a, int* b, int* c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
};

void nbody_sec(int n, std::vector<pData> data, int steps) {

  std::vector<pData> temp(n);
  double dt = 0.1;

  for (int k = 1; k <= steps; k++) {
    for (int i = 0; i < n; i++) {

      std::vector<double> acc(3,0);

      for (int j = 0; j < n; j++) {

        if (j == i) continue;

        double rx,ry,rz;

        rx = data[i].x - data[j].x;
        ry = data[i].y - data[j].y;
        rz = data[i].z - data[j].z;

        double distSqr = rx * rx + ry * ry + rz * rz;

        double dist = std::sqrt(distSqr);
        double distCube = dist * dist * dist;

        double s = data[i].m / distCube;

        acc[0] += rx * s;
        acc[1] += ry * s;
        acc[2] += rz * s;
      }

      double vx, vy, vz;
      vx = data[i].vx + acc[0] * k;
      vy = data[i].vy + acc[1] * k;
      vz = data[i].vz + acc[2] * k;

      temp[i] = {data[i].x + vx*k, data[i].y + vy*k, data[i].z + vz*k, data[i].m, vx, vy, vz};
    }

    std::cout << "Step: " << k << "\n";
    for (int i = 0; i < n; i++) {
      data[i] = temp[i];
      std::cout << "P " << i << ": (" << data[i].x << "," << data[i].y << "," << data[i].z << ")\n";
    }
  }

  std::cout << "Simulation finished\n";
  for (int i = 0; i < n; i++) {
    std::cout << "P " << i << ": (" << data[i].x << "," << data[i].y << "," << data[i].z << ")\n";
  }

};