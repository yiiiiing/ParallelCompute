# Parallel Computing

###  Exercises Directory
- Analyzing satellite.c program and checking how the compiler vectorizes and optimazing it uisng different compile flags, and adding OpenMP multicore parallelization to it (in satellite_openMP.c).

- Creating an OpenCL kernel for the graphics engine routine (setting values to each pixel) to make drawing part execute in parallel and try the different work group sizes (in satellite_openCL.c).

### Compiling flags

1. Compiling in Linux
 - no optimizations
`gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm`

 - most optimizations, no vectorization
`gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2`

 - Also vectorize and show what loops get vectorized
`gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec`

 - Also allow math relaxations
`gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math`

 - Also allow AVX2 SIMD instructions
`gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -mavx2`

 - Also support OpenMP
`gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -mavx2 -fopenmp`

 - Also support openCL
`gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize  -fopt-info-vec -ffast-math -mavx2 -fopenmp -lOpenCL`

2. Compiling on MacOS X
 - No optimizations
`clang -o parallel parallel.c -framework GLUT -framework OpenGL`

 - Full optimizations
`clang -O3 -o parallel parallel.c -framework GLUT -framework OpenGL`

 - With OpenCL libraries
`clang -O3 -o parallel parallel.c -framework GLUT -framework OpenGL -framework OpenCL`

3. Note: OpenMP is unfortunately not trivially working on Macos X, and requires installing custom compiling instead of the one that comes with XCode
