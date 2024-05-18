#include <vector>

typedef struct {
    double x, y, z, m;
    double vx, vy, vz;
} pData;

void nbody_sec(int n, std::vector<pData> &data, int steps);