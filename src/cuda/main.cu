#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
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

bool simulate(int N, int Steps, int blockSize, int sharedMem, int threads2D, int seed) {
  using std::chrono::microseconds;

  cudaSetDevice(0);

  std::default_random_engine gen;

  if (!seed) {
    std::cout << "Random seed\n";
    gen.seed(std::time(0));
    //std::srand(std::time(0));
  } 
  else gen.seed(seed);//std::srand(seed);

  std::size_t size = sizeof(double4) * N;
  std::vector<double4> posData(N);
  std::vector<double4> velData(N);

  // Create the memory buffers
  double4 *posDev;
  double4 *velDev;
  double4 *auxPosDev;
  double4 *auxVelDev;
  cudaMalloc(&posDev, size);
  cudaMalloc(&auxPosDev, size);
  cudaMalloc(&velDev, size);
  cudaMalloc(&auxVelDev, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();

  std::uniform_real_distribution<double> pos(0.0, 10000.0);
  std::uniform_real_distribution<double> mass(50000.0, 100000.0);

  for (int i = 0; i < N; i++) {
    posData[i].x = pos(gen);
    posData[i].y = pos(gen);
    posData[i].z = pos(gen);
    posData[i].w = mass(gen);
  }
  for (int i = 0; i < N; i++) velData[i] = {0,0,0,0};
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  std::cout << "INITIAL: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << " Particula " << i << ": (" << posData[i].x << ", " << posData[i].y << ", " << posData[i].z << ") m = " << posData[i].w << "\n";

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(posDev, posData.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(velDev, velData.data(), size, cudaMemcpyHostToDevice);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  t_start = std::chrono::high_resolution_clock::now();
      
  // test<<<1, 1>>>(N, posDev, auxPosDev, velDev, auxVelDev, Steps);
  // cudaDeviceSynchronize();

  if (threads2D) {
    const dim3 threads(8, 8, 1);
    int blocknum = (N + 63)/64;
	  const dim3 blocks((blocknum+1)/2, 2, 1);
    std::cout << blocks.x << " " << blocks.y << '\n';
    if (sharedMem) {
      while(Steps--){
        nbody_kernel_shared_2D<<<blocks, threads, (sizeof(double4)*64)>>>(N, posDev, auxPosDev, velDev, auxPosDev, 8, blocks.x, blocks.y);
        cudaDeviceSynchronize();
        cudaMemcpy(posDev, auxPosDev, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(velDev, auxVelDev, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
      }
      //std::cerr << "Can't activate both options at the same time\n";
      //nbody_kernel_shared_2D<<<blocks, threads, (sizeof(double4)*64)>>>(N, posDev, velDev, Steps, 8, blocks.x, blocks.y);
    }
    else {
      while(Steps--){
        nbody_kernel_2D<<<blocks, threads>>>(N, posDev, auxPosDev, velDev, auxVelDev, blocks.x * threads.x);
        cudaDeviceSynchronize();
        cudaMemcpy(posDev, auxPosDev, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(velDev, auxVelDev, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
      }
    }
  }
  else {
    int gridSize = (N + blockSize - 1)/blockSize;

    // Execute the function on the device (using 32 threads here)
    if (sharedMem) {
      // Shared memory
      std::cout << "Using shared memory: \n";
      t_start = std::chrono::high_resolution_clock::now();
      while(Steps--){
        nbody_kernel_shared<<<blockSize, gridSize, (sizeof(double4)*blockSize)>>>(N, posDev, auxPosDev, velDev, auxVelDev, blockSize, gridSize);
        cudaDeviceSynchronize();
        cudaMemcpy(posDev, auxPosDev, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(velDev, auxVelDev, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
      }
    } 
    else {
      // No shared memory
      std::cout << "Not using shared memory: \n";
      t_start = std::chrono::high_resolution_clock::now();

      while(Steps--){
        nbody_kernel<<<blockSize, gridSize>>>(N, posDev, auxPosDev, velDev, auxVelDev);
        cudaDeviceSynchronize();
        cudaMemcpy(posDev, auxPosDev, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(velDev, auxVelDev, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
      }
    }
  }
  
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(posData.data(), posDev, size, cudaMemcpyDeviceToHost);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

  // Print the result
  std::cout << "RESULTS: \n";
  for (int i = 0; i < N; i++)
    std::cout << " Particula " << i << ": (" << posData[i].x << ", " << posData[i].y << ", " << posData[i].z << ")\n";

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to copy data to device: " << t.copy_to_device
            << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to copy data to host: " << t.copy_to_host
            << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";

  cudaFree(posDev);
  cudaFree(auxPosDev);
  cudaFree(velDev);
  cudaFree(auxVelDev);

  return true;

}

int main(int argc, char* argv[]) {
  if (argc != 7 && argc != 8) {
    std::cerr << "Uso: " << argv[0] << " <particle_count> <step_count> <block size> <shared_mem> <2d_threads> <output_file> <seed (optional)>"
              << std::endl;
    return 2;
  }
  int n = std::atoi(argv[1]);
  int s = std::atoi(argv[2]);
  int bs = std::atoi(argv[3]);
  int shm = std::atoi(argv[4]);
  int th2d = std::atoi(argv[5]);
  int seed = 0;
  if (argc == 8) seed = std::atoi(argv[7]);

  if (!simulate(n, s, bs, shm, th2d, seed)) {
    std::cerr << "CUDA: Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[6], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[6] << "'" << std::endl;
    return 4;
  }
  // params
  // out << n << "," << bs << "," << gs << ",";
  // times
  out << n << "," << s << "," << bs << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[6] << std::endl;
  return 0;
}
