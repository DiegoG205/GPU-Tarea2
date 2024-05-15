#include <vector>

typedef struct {
    int x, y, z, m;
    double vx, vy, vz;
} pData;

void nbody_sec(int n, std::vector<pData> &data, int steps);