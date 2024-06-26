
#include <cstddef>
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <OpenCL/opencl.hpp>
#endif  // DEBUG
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>


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

  std::ifstream sourceFile("src/cl/kernel.cl");
  std::stringstream sourceCode;
  sourceCode << sourceFile.rdbuf();

  cl_int err;

  prog = cl::Program(context, sourceCode.str(), true, &err);

  if(err != CL_SUCCESS) {
    std::cerr << err <<"\n";
    // Get the build log
    std::string name     = devices.front().getInfo<CL_DEVICE_NAME>();
    std::string buildlog = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices.front());
    std::cerr << "Build log for " << name << ":" << std::endl
                << buildlog << std::endl;
    };
  

  return true;
}

/*
* @param N : 
* @param Steps : 
* @param sharedMem : 
* @param threads2D : 
* @param seed : 
*********************/

bool simulate(int N, int Steps, int blockSize, int sharedMem, int threads2D, int seed) {
  using std::chrono::microseconds;
  
  std::default_random_engine gen;

  if (sharedMem && threads2D) {
      std::cerr << "Can't activate both options at the same time\n";
      return false;
  }

  if (!seed){ 
    std::cout << "Random seed\n";
    gen.seed(std::time(0));
  }
  else gen.seed(seed);

  std::size_t size = sizeof(cl_double4) * N;
  std::vector<cl_double4> posData(N);
  std::vector<cl_double4> velData(N);

  // Create the memory buffers
  cl::Buffer posDev(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer velDev(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer auxPosDev(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer auxVelDev(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

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

  //Print initial state of particles
  std::cout << "INITIAL: " << std::endl;
  for (int i = 0; i < N; i++)
    std::cout << " Particula " << i << ": (" << posData[i].x << ", " << posData[i].y << ", " << posData[i].z << ") m = " << posData[i].w << "\n";

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  queue.enqueueWriteBuffer(posDev, CL_TRUE, 0, size, posData.data());
  queue.enqueueWriteBuffer(velDev, CL_TRUE, 0, size, velData.data());
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Make kernel
  if (threads2D) {
    const cl_uint3 threads = {8, 8, 1};
    cl_uint blocknum = (N + 63)/64;
	  const cl_uint3 blocks={(blocknum+1)/2, 2, 1};
    if (sharedMem) {
      std::cerr << "Can't activate both options at the same time\n";
      return false;
    }
    else {
      while(Steps--){
        // Make kernel
        //nbody_kernel_2D
          // <<<blocks Tamaño de un bloque,
          // threads Cantidad de bloques>>>
            // (N, posDev, auxPosDev, velDev, auxVelDev, blocks.x * threads.x);
        cl::Kernel kernel(prog, "nbody_kernel_2D");

        // Set the kernel arguments
        
        kernel.setArg(0, N);
        kernel.setArg(1, posDev);
        kernel.setArg(2, auxPosDev);
        kernel.setArg(3, velDev);
        kernel.setArg(4, auxVelDev);
        kernel.setArg(5, blocks.x * threads.x);

        // Execute the function on the device (using 32 threads here)
        cl::NDRange gSize(threads.x*blocks.x, threads.y*blocks.y);//Cantidad Total WorkItems
        cl::NDRange lSize(blocks.x, blocks.y);//WorkItems por WorkGroup

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, gSize, lSize);
        queue.enqueueCopyBuffer(auxPosDev, posDev, 0, 0, size);
        queue.enqueueCopyBuffer(auxVelDev, velDev, 0, 0, size);
      }
    }
  }
  else {
    int gridSize = (N + blockSize -1)/blockSize;

    //Execute the function on the device (using 32 threads here)
    if (sharedMem) {
      //Shared memory
      std::cout << "Using shared memory: \n";
      t_start = std::chrono::high_resolution_clock::now();
      while(Steps--){
        // Make kernel
        // nbody_kernel_shared
          // <<<blockSize Tamaño de un bloque,
          // gridSize Cantidad de bloques,
          // (sizeof(double4)*blockSize) Tamaño de memoria compartida>>>
            // (N, posDev, auxPosDev, velDev, auxVelDev, blockSize, gridSize);
        cl::Kernel kernel(prog, "nbody_kernel_shared");

        // Set the kernel arguments
        kernel.setArg(0, N);
        kernel.setArg(1, posDev);
        kernel.setArg(2, auxPosDev);
        kernel.setArg(3, velDev);
        kernel.setArg(4, auxVelDev);
        kernel.setArg(5, sizeof(cl_double4) * blockSize, NULL);
        kernel.setArg(6, gridSize);

        // Execute the function on the device (using 32 threads here)
        cl::NDRange gSize(gridSize*blockSize);//Cantidad Total WorkItems
        cl::NDRange lSize(blockSize);//WorkItems por WorkGroup
        //memoria compartida (sizeof(cl_float4)*blockSize)

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, gSize, lSize);
        queue.enqueueCopyBuffer(auxPosDev, posDev, 0, 0, size);
        queue.enqueueCopyBuffer(auxVelDev, velDev, 0, 0, size);
      }
    }
    else {
      //No shared memory
      std::cout << "Not using shared memory: \n";
      t_start = std::chrono::high_resolution_clock::now();
      while(Steps--){
        // Make kernel
        // nbody_kernel
          // <<<blockSize Tamaño de un bloque,
          // gridSize cantidad de bloques>>>
            // (N, posDev, auxPosDev, velDev, auxVelDev);
        cl::Kernel kernel(prog, "nbody_kernel");

        // Set the kernel arguments
        kernel.setArg(0, N);
        kernel.setArg(1, posDev);
        kernel.setArg(2, auxPosDev);
        kernel.setArg(3, velDev);
        kernel.setArg(4, auxVelDev);
        // kernel.setArg(5, blockSize);
        // kernel.setArg(6, gridSize);

        // Execute the function on the device (using 32 threads here)
        cl::NDRange gSize(gridSize*blockSize);//Cantidad Total WorkItems
        cl::NDRange lSize(blockSize);//WorkItems por WorkGroup

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, gSize, lSize);
        queue.enqueueCopyBuffer(auxPosDev, posDev, 0, 0, size);
        queue.enqueueCopyBuffer(auxVelDev, velDev, 0, 0, size);
      }
    }
  }

  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueReadBuffer(posDev, CL_TRUE, 0, size, posData.data());
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "RESULTS: " << std::endl;
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

  return true;
}

int main(int argc, char* argv[]) {
  
  if (!init()) return 1;

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
    std::cerr << "OpenCL: Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[6], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[6] << "'" << std::endl;
    return 4;
  }

  out << n << "," << s << "," << bs << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[6] << std::endl;
  return 0;
}
