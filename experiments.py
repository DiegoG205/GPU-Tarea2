import os
import random


n = 64
s = 1000
c = 10
t = 64

for i in range(0,c):
    seed = random.randint(1000,9999)
    print(f'Current exp: {i + 1}')
    os.popen(f'.\\build\\src\\Debug\\MyProjectCPU.exe {n} {s} outcpu.txt {seed}')
    os.popen(f'.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe {n} {s} {t} 0 0 outCuda.txt {seed}')
    os.popen(f'.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe {n} {s} {t} 1 0 outCudaShared.txt {seed}')
    os.popen(f'.\\build\\src\\cuda\\Debug\\MyProjectCUDA.exe {n} {s} {t} 0 1 outCuda2D.txt {seed}')

