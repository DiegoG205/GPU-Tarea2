#include <vector>

typedef struct {
    double x, y, z, m, vx, vy, vz;
} pData;


void vec_sum(int* a, int* b, int* c, int n);

void nbody_sec(int n, std::vector<pData> &data, int steps);