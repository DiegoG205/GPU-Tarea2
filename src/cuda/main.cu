#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "kernel.cuh"

struct Times {
  long create_data;
  long copy_to_host;
  long execution;
  long copy_to_device;
  inline long total() {
    return create_data + copy_to_host + execution + copy_to_device; 
  }
};

Times t;

bool simulate(int N, int Steps, int blockSize, int gridSize) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(float4) * N * 2;
  std::vector<float4> data(2*N);

  // Create the memory buffers
  float4 *dataDev;
  cudaMalloc(&dataDev, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    data[2*i].x = std::rand() % 1000;
    data[2*i].y = std::rand() % 1000;
    data[2*i].z = std::rand() % 1000;
    data[2*i].w = std::rand() % 25000 + 50000;
    data[2*i + 1] = {0,0,0,0};
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  std::cout << "INITIAL: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << " Particula " << i << ": (" << data[2*i].x << ", " << data[2*i].y << ", " << data[2*i].z << ")\n";

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(dataDev, data.data(), size, cudaMemcpyHostToDevice);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();


  // Execute the function on the device (using 32 threads here)
  t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < Steps; i++) {
    nbody_kernel<<<blockSize, gridSize>>>(N, dataDev, i);
    cudaDeviceSynchronize();
  }
  
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(data.data(), dataDev, size, cudaMemcpyDeviceToHost);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

  // Print the result
  std::cout << "RESULTS: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << " Particula " << i << ": (" << data[2*i].x << ", " << data[2*i].y << ", " << data[2*i].z << ")\n";

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to copy data to device: " << t.copy_to_device
            << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to copy data to host: " << t.copy_to_host
            << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";

  cudaFree(dataDev);

  return true;

}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Uso: " << argv[0] << " <array size> <step_count> <block size> <grid size>"
              << std::endl;
    return 2;
  }
  int n = std::atoi(argv[1]);
  int s = std::atoi(argv[2]);
  int bs = std::atoi(argv[3]);
  int gs = std::atoi(argv[4]);

  if (!simulate(n, s, bs, gs)) {
    std::cerr << "CUDA: Error while executing the simulation" << std::endl;
    return 3;
  }

  // std::ofstream out;
  // out.open(argv[5], std::ios::app | std::ios::out);
  // if (!out.is_open()) {
  //   std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
  //   return 4;
  // }
  // // params
  // out << n << "," << bs << "," << gs << ",";
  // // times
  // out << t.create_data << "," << t.copy_to_device << "," << t.execution << "," << t.copy_to_host << "," << t.total() << "\n";

  // std::cout << "Data written to " << argv[4] << std::endl;
  return 0;
}
