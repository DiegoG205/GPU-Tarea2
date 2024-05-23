# Tarea 2 - _nbody problem_
Esta tarea incluye implementaciones del nbody-problem en cpu, cuda y opencl.
Todas las implementaciones fueron testeadas en un computador con una tarjeta
gráfica Nvidia y sistema operativo Windows 10.

Para compilar las versiones, existe un Makefile que simplifica el proceso, pero
si no funciona, ejecutar los comandos directamente debería compilar. Los comandos
son:
- all: Construye los ejecutables para CUDA, OpenCL y CPU.
- init: Inicializa el directorio de `build` utilizando CMake.
- cuda: Construye el ejecutable para CUDA.
- cl: Construye el ejecutable para OpenCL.
- cpu: Construye el ejecutable para CPU.
- clean: Elimina los artefactos de construcción y los directorios de pruebas.

## Ejecutables

La ruta específica del ejecutable compilado puede variar un poco de pc a pc. En el pc que se realizaron las pruebas, las rutas eran:
- Cpu: build/src/Debug/MyProjectCPU.exe
- Cuda: build/src/cuda/MyProjectCUDA.exe
- OpenCL: build/src/cl/MyProjectCL.exe

### Argumentos

Los argumentos que reciben cada versión son:

`MyProjectCPU.exe <particle_count> <step_count> <output_file> <seed>`
- particle_count: Cantidad de partículas a simular
- step_count: Cantidad de pasos de la simulación
- output_file: Archivo donde se escriben los resultados
- seed: Semilla para generar los valores iniciales, se puede dejar vacía para usar una semilla aleatoria

`MyProjectCUDA.exe <particle_count> <step_count> <block size> <shared_mem> <2d_threads> <output_file> <seed>`
- particle_count: Cantidad de partículas a simular
- step_count: Cantidad de pasos de la simulación
- block_size: Cantidad de threads por bloque
- shared_mem: 0 para no usar memoria compartida, 1 para usarla. No se puede activar a la vez que 2d_threads
- 2d_threads: 0 para no usar threads en 2 dimensiones, 1 para usarlos. No se puede activar a la vez que shared_mem
- output_file: Archivo donde se escriben los resultados
- seed: Semilla para generar los valores iniciales, se puede dejar vacía para usar una semilla aleatoria

`MyProjectCL.exe <particle_count> <step_count> <block size> <shared_mem> <2d_threads> <output_file> <seed>`
- particle_count: Cantidad de partículas a simular
- step_count: Cantidad de pasos de la simulación
- block_size: Cantidad de threads por bloque
- shared_mem: 0 para no usar memoria compartida, 1 para usarla. No se puede activar a la vez que 2d_threads
- 2d_threads: 0 para no usar threads en 2 dimensiones, 1 para usarlos. No se puede activar a la vez que shared_mem
- output_file: Archivo donde se escriben los resultados
- seed: Semilla para generar los valores iniciales, se puede dejar vacía para usar una semilla aleatoria

## Experimentos

Para ejecutar los experimentos, se debe correr el archivo `experiments.py`. Esto escribirá los resultados de todos los experimentos a múltiples archivos de texto. Luego, para graficar estos resultados, se debe ejecutar el archivo `graphics.ipynb`, que se encarga de procesar los datos y generar los gráficos para cada experimento.