import random
import subprocess


n = 8
s = 1000
c = 5
t = 64

fs = open("outSeq.txt", "wt")
fs.write("n,steps,time\n")
fs.close()
fc = open("outCuda.txt", "wt")
fc.write("n,steps,threads,time\n")
fc.close()
fcsh = open("outCudaShared.txt", "wt")
fcsh.write("n,steps,threads,time\n")
fcsh.close()
fc2d = open("outCuda2D.txt", "wt")
fc2d.write("n,steps,threads,time\n")
fc2d.close()

for i in range(0, 5): # 7 iteraciones llega hasta n=1024
    n *= 2
    print(f'Using {n} particles:')
    for j in range(0, c):
        print(f'Experiment {j+1}/{c}')
        seed = random.randint(1000,9999)
        print('Sequential')
        p = subprocess.Popen(['.\\build\\src\\Debug\\MyProjectCPU.exe', str(n), str(s), "outSeq.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('Cuda')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '0', "outCuda.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('Cuda Shared')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '1', '0', "outCudaShared.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()
        print('Cuda 2D')
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '1', "outCuda2D.txt", str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        p.wait()

