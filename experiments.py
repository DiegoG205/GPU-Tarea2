import random
import subprocess


n = 8
s = 1000
c = 5
t = 64

# fs = open("out1Seq.txt", "wt")
# fs.write("n,steps,time\n")
# fs.close()
# fcu = open("out1Cuda.txt", "wt")
# fcu.write("n,steps,threads,time\n")
# fcu.close()
# fcl = open("out1CL.txt", "wt")
# fcl.write("n,steps,threads,time\n")
# fcl.close()

# print("Test 1: Hardware comparison")
# for i in range(0, 9): # Desde 16 hasta 4096
#     n *= 2
#     print(f'Using {n} particles:')
#     for j in range(0, c):
#         print(f'Experiment {j+1}/{c}')
#         seed = random.randint(1000,9999)
#         if i < 6:
#             print('Sequential')
#             p = subprocess.Popen(['.\\build\\src\\Debug\\MyProjectCPU.exe', str(n), str(s), "out1Seq.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
#             p.wait()
#         print('Cuda64')
#         p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '0', "out1Cuda.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
#         p.wait()
#         print('OpenCL64')
#         p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(t), '0', '0', "out1CL.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
#         p.wait()
#         print('Cuda128')
#         p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(2*t), '0', '0', "out1Cuda.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
#         p.wait()
#         print('OpenCL128')
#         p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(2*t), '0', '0', "out1CL.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
#         p.wait()

print("Test 2: Cuda Comparison")
n = 512
t = 128

fcu = open("out2Cuda.txt", "wt")
fcu.write("n,steps,threads,time\n")
fcu.close()
fcui = open("out2CudaIrregular.txt", "wt")
fcui.write("n,steps,threads,time\n")
fcui.close()
fcush = open("out2CudaShared.txt", "wt")
fcush.write("n,steps,threads,time\n")
fcush.close()
fcu2d = open("out2Cuda2D.txt", "wt")
fcu2d.write("n,steps,threads,time\n")
fcu2d.close()


for i in range(0, 3): # Desde 1024 a 4096
    n *= 2
    print(f'Using {n} particles:')
    for j in range(0, c):
        print(f'Experiment {j+1}/{c}')
        seed = random.randint(1000,9999)
        print('Cuda')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '0', "out2Cuda.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('Cuda 100')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(100), '0', '0', "out2CudaIrregular.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('Cuda Shared')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '1', '0', "out2CudaShared.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('Cuda 2D')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '1', "out2Cuda2D.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()

print("Test 3: OpenCL Comparison")
n = 512

fcl = open("out3CL.txt", "wt")
fcl.write("n,steps,threads,time\n")
fcl.close()
fcli = open("out3CLIrregular.txt", "wt")
fcli.write("n,steps,threads,time\n")
fcli.close()
fclsh = open("out3CLShared.txt", "wt")
fclsh.write("n,steps,threads,time\n")
fclsh.close()
fcl2d = open("out3CL2D.txt", "wt")
fcl2d.write("n,steps,threads,time\n")
fcl2d.close()

for i in range(0, 3): # Desde 1024 a 4096
    n *= 2
    print(f'Using {n} particles:')
    for j in range(0, c):
        print(f'Experiment {j+1}/{c}')
        seed = random.randint(1000,9999)
        print('OpenCL')
        p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(t), '0', '0', "out3CL.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('OpenCL 100')
        p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(100), '0', '0', "out3CLIrregular.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('OpenCL Shared')
        p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(t), '1', '0', "out3CLShared.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('OpenCL 2D')
        p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(t), '0', '1', "out3CL2D.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()

