
#include <cstddef>
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif  // DEBUG
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

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
cl::Program prog;
cl::CommandQueue queue;

bool init() {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Platform::get(&platforms);
  for (auto& p : platforms) {
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() > 0) break;
  }
  if (devices.size() == 0) {
    std::cerr << "Not GPU device found" << std::endl;
    return false;
  }

  std::cout << "GPU Used: " << devices.front().getInfo<CL_DEVICE_NAME>()
            << std::endl;

  cl::Context context(devices.front());
  queue = cl::CommandQueue(context, devices.front());

  std::ifstream sourceFile("kernel.cl");
  std::stringstream sourceCode;
  sourceCode << sourceFile.rdbuf();

  prog = cl::Program(context, sourceCode.str(), true);

  return true;
}

/*
* @param N : 
* @param localSize : 
* @param globalSize : 
*********************/

bool simulate(int N, int Steps, int blocksize, int sharedMem, int threads2D, int seed) {
  using std::chrono::microseconds;
  
  if (!seed) std::srand(std::time(0));
  else std::srand(seed);

  std::size_t size = sizeof(cl_double4) * N;
  std::size_t size = sizeof(cl_double4) * N;
  std::vector<cl_double4> posData(N);
  std::vector<cl_double4> velData(N);

  // Create the memory buffers
  cl::Buffer posDev(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer velDev(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    posData[i].x = double(std::rand() % 10000);
    posData[i].y = double(std::rand() % 10000);
    posData[i].z = double(std::rand() % 10000);
    posData[i].w = double(std::rand() % 25000 + 50000);
    velData[i] = {0,0,0,0};
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  //Print initial state of particles
  std::cout << "INITIAL: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << " Particula " << i << ": (" << posData[i].x << ", " << posData[i].y << ", " << posData[i].z << ") m = " << posData[i].w << "\n";

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  queue.enqueueWriteBuffer(posDev, CL_TRUE, 0, size, a.data());
  queue.enqueueWriteBuffer(velDev, CL_TRUE, 0, size, b.data());
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Make kernel
  if (threads2D) {
    const dim3 threads(8, 8, 1);
    int blocknum = (N + 63)/64;
	  const dim3 blocks((blocknum+1)/2, 2, 1);
    if (sharedMem) {
      std::cerr << "Can't activate both options at the same time\n";
      //nbody_kernel_shared_2D<<<blocks, threads, (sizeof(int4)*64)>>>(N, posDev, velDev, Steps, 8, blocks.x, blocks.y);
    }
    else {
      //kernel w threads2D and no shared memory
      nbody_kernel_2D<<<blocks, threads>>>(N, posDev, velDev, Steps, blocks.x * threads.x);
    }
  }
  else {
    int gridSize = (N + blockSize - 1)/blockSize;

    // Execute the function on the device (using 32 threads here)
    if (sharedMem) {
      // Shared memory
      std::cout << "Using shared memory: \n";
      t_start = std::chrono::high_resolution_clock::now();
      nbody_kernel_shared<<<blockSize, gridSize, (sizeof(int4)*blockSize)>>>(N, posDev, velDev, Steps, blockSize, gridSize);
      cudaDeviceSynchronize();
    } 
    else {
      // No shared memory
      std::cout << "Not using shared memory: \n";
      t_start = std::chrono::high_resolution_clock::now();
      
      nbody_kernel<<<blockSize, gridSize>>>(N, posDev, velDev, Steps);
      cudaDeviceSynchronize();
    }
  }
  cl::Kernel kernel(prog, "vec_sum");

  // Set the kernel arguments
  kernel.setArg(0, aBuff);
  kernel.setArg(1, bBuff);
  kernel.setArg(2, cBuff);
  kernel.setArg(3, N);

  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize(globalSize);
  cl::NDRange lSize(localSize);

  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.finish();
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueReadBuffer(cBuff, CL_TRUE, 0, size, c.data());
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "RESULTS: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << "  out[" << i << "]: " << c[i] << " (" << a[i] << " + " << b[i]
              << ")\n";

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to copy data to device: " << t.copy_to_device
            << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to copy data to host: " << t.copy_to_host
            << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";
  return true;
}

bool simulate(int N, int localSize, int globalSize) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(int) * N;
  std::vector<int> a(N), b(N), c(N);

  // Create the memory buffers
  cl::Buffer aBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer bBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer cBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    a[i] = std::rand() % 2000;
    b[i] = std::rand() % 2000;
    c[i] = 0;
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  queue.enqueueWriteBuffer(aBuff, CL_TRUE, 0, size, a.data());
  queue.enqueueWriteBuffer(bBuff, CL_TRUE, 0, size, b.data());
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Make kernel
  cl::Kernel kernel(prog, "vec_sum");

  // Set the kernel arguments
  kernel.setArg(0, aBuff);
  kernel.setArg(1, bBuff);
  kernel.setArg(2, cBuff);
  kernel.setArg(3, N);

  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize(globalSize);
  cl::NDRange lSize(localSize);

  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.finish();
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueReadBuffer(cBuff, CL_TRUE, 0, size, c.data());
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "RESULTS: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << "  out[" << i << "]: " << c[i] << " (" << a[i] << " + " << b[i]
              << ")\n";

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to copy data to device: " << t.copy_to_device
            << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to copy data to host: " << t.copy_to_host
            << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";
  return true;
}

int main(int argc, char* argv[]) {
  if (!init()) return 1;

  if (argc != 5) {
    std::cerr << "Uso: " << argv[0]
              << " <array size> <local size> <global size> <output file>"
              << std::endl;
    return 2;
  }
  int n = std::stoi(argv[1]);
  int ls = std::stoi(argv[2]);
  int gs = std::stoi(argv[3]);

  if (!simulate(n, ls, gs)) {
    std::cerr << "CL: Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[4], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
    return 4;
  }
  // params
  out << n << "," << ls << "," << gs << ",";
  // times
  out << t.create_data << "," << t.copy_to_device << "," << t.execution << ","
      << t.copy_to_host << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[4] << std::endl;
  return 0;
}
