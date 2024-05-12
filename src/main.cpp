#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>

#include "kernel.h"

struct Times {
  long create_data;
  long execution;

  long total() { return create_data + execution; }
};

Times t;

bool simulate(int N, int Steps, int seed) {
  using std::chrono::microseconds;
  if (!seed) {
    std::cout << "Random seed\n";
    std::srand(std::time(0));
  } 
  else std::srand(seed);

  std::vector<pData> data(N);
  std::vector<pData> aux(N);

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    data[i] = {(std::rand() % 10000), (std::rand() % 10000), (std::rand() % 10000), (std::rand() % 50000 + 50000), 0, 0, 0};
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  std::cout << "INITIAL: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << "Particula " << i << " (" << data[i].x << ", " << data[i].y << ", " << data[i].z << ") m = " << data[i].m << "\n";

  t_start = std::chrono::high_resolution_clock::now();
  nbody_sec(N, data, Steps);
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "RESULTS: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << "Particula " << i << " (" << data[i].x << ", " << data[i].y << ", " << data[i].z << ")\n";

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";

  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    std::cerr << "Uso: " << argv[0] << " <particle_count> <step_amount> <seed (optional)>"
              << std::endl;
    return 2;
  }

  int n = std::stoi(argv[1]);
  int s = std::stoi(argv[2]);
  int seed = 0;
  if (argc == 4) seed = std::stoi(argv[3]);
  if (!simulate(n, s, seed)) {
    std::cerr << "Error while executing the simulation" << std::endl;
    return 3;
  }

  // std::ofstream out;
  // out.open(argv[2], std::ios::app | std::ios::out);
  // if (!out.is_open()) {
  //   std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
  //   return 4;
  // }
  // out << n << "," << t.create_data << "," << t.execution << "," << t.total()
  //     << "\n";

  // std::cout << "Data written to " << argv[2] << std::endl;
  return 0;
}
