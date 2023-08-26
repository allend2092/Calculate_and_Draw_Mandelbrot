
---

## Mandelbrot Set Generator in Python

This Python program generates the Mandelbrot set using three different approaches:

1. **Pure Python Implementation**
2. **Numba JIT-compiled Implementation**
3. **GPU-accelerated Implementation**

### Importing Libraries

- `numpy` for numerical operations
- `matplotlib.pyplot` for plotting
- `time` for measuring execution time
- `numba` for Just-In-Time (JIT) compilation and CUDA support

### Pure Python Implementation (`mandelbrot_python`)

- Takes parameters defining the region of the complex plane to visualize, the dimensions of the image, and the maximum number of iterations.
- Uses nested loops to iterate over each pixel in the image.
- For each pixel, it performs Mandelbrot set calculations.
- Measures the execution time and plots the result.

### Numba JIT-compiled Implementation (`mandelbrot_numba`)

- Same as the pure Python implementation but uses Numba's `@jit` decorator for JIT compilation.
- This speeds up the execution time.

### GPU-accelerated Implementation (`mandelbrot_gpu`)

- Uses Numba's CUDA support to run the Mandelbrot set calculations on the GPU.
- Defines block and grid dimensions for parallel execution on the GPU.
- Transfers the image array to GPU memory, performs the calculations, and then transfers it back to CPU memory.

### Execution Time Measurement

- Measures and prints the execution time for each approach.

### Plotting

- Uses `matplotlib` to plot the Mandelbrot set generated by each approach.

---

### Summary

The program demonstrates how to generate the Mandelbrot set using pure Python, Numba JIT compilation, and GPU acceleration. It also measures and prints the execution time for each approach to show the performance benefits of JIT compilation and GPU acceleration.

---

