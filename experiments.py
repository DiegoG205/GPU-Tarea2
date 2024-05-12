import random
import subprocess


n = 8
s = 1000
c = 10
t = 64


for i in range(0, 4): # 7 iteraciones llega hasta n=1024
    n *= 2
    for j in range(0, c):
        seed = random.randint(1000,9999)
        p = subprocess.Popen(['.\\build\\src\\Debug\\MyProjectCPU.exe', str(n), str(s), "outSeq.txt", str(seed)])
        p.wait()
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '0', "outCuda.txt", str(seed)])
        p.wait()
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '1', '0', "outCudaShared.txt", str(seed)])
        p.wait()
        p = subprocess.Popen(['.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe', str(n), str(s), str(t), '0', '1', "outCuda2D.txt", str(seed)])
        p.wait()

