#include "kernel.h"
#include <vector>
#include <cmath>
#include <iostream>

void nbody_sec(int n, std::vector<pData> &data, int steps) {

  std::vector<pData> temp(n);

  while(steps--) {
    for (int i = 0; i < n; i++) {

      double ax, ay, az;
      ax = 0;
      ay = 0;
      az = 0;

      for (int j = 0; j < n; j++) {

        if (j == i) continue;

        int rx,ry,rz;

        rx = data[j].x - data[i].x;
        ry = data[j].y - data[i].y;
        rz = data[j].z - data[i].z;

        double distSqr = rx * rx + ry * ry + rz * rz + 0.1;

        double dist = std::sqrt(distSqr);
        double distCube = dist * dist * dist;

        double s = data[j].m / distCube;

        ax += rx * s;
        ay += ry * s;
        az += rz * s;
      }
      double vx, vy, vz;
      vx = data[i].vx + ax;
      vy = data[i].vy + ay;
      vz = data[i].vz + az;

      //std::cout << "Particula " << i << ": (" << vx << ", " << vy << ", " << vz << ")\n";

      temp[i] = {data[i].x + lround(vx), data[i].y + lround(vy), data[i].z + lround(vz), data[i].m, vx, vy, vz};
    }

    for (int i = 0; i < n; i++) {
      data[i] = temp[i];
    }
  }
};