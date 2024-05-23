import random
import subprocess


n = 8
s = 1000
c = 5
t = 64

fs = open("out1Seq.txt", "wt")
fs.write("n,steps,time\n")
fs.close()
fcu = open("out1Cuda.txt", "wt")
fcu.write("n,steps,threads,time\n")
fcu.close()
fcl = open("out1CL.txt", "wt")
fcl.write("n,steps,threads,time\n")
fcl.close()

for i in range(0, 9): # 7 iteraciones llega hasta n=1024
    n *= 2
    print(f'Using {n} particles:')
    for j in range(0, c):
        print(f'Experiment {j+1}/{c}')
        seed = random.randint(1000,9999)
        if i < 6:
            print('Sequential')
            p = subprocess.Popen(['.\\build\\src\\Debug\\MyProjectCPU.exe', str(n), str(s), "out1Seq.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            p.wait()
        print('Cuda64')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '0', "out1Cuda.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('OpenCL64')
        p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(t), '0', '0', "out1CL.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('Cuda128')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(2*t), '0', '0', "out1Cuda.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('OpenCL128')
        p = subprocess.Popen(['.\\build\\src\\cl\\Debug\\MyProjectCL.exe', str(n), str(s), str(2*t), '0', '0', "out1CL.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        # print('Cuda Shared')
        # p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '1', '0', "outCudaShared.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        # p.wait()
        # print('Cuda 2D')
        # p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '1', "outCuda2D.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        # p.wait()

