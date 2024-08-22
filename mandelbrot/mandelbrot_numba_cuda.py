import numpy as np
import numba as nb
from numba import cuda
import math

# Simple LCG random number generator
@cuda.jit(device=True)
def lcg_random(seed):
    a = 1664525
    c = 1013904223
    m = 2**32
    seed = (a * seed + c) % m
    return seed, seed / m

# CUDA kernel for is_in_mandelbrot
@cuda.jit(device=True)
def is_in_mandelbrot(x, y):
    c = complex(x, y)
    z = 0j
    for i in range(1000):  # Adjust the number of iterations as needed
        if abs(z) > 2:
            return 0  # Return 0 for False
        z = z * z + c
    return 1  # Return 1 for True

# Modified CUDA kernel for count_mandelbrot with debug prints
@cuda.jit
def count_mandelbrot_kernel(xmin, width, ymin, height, num_samples, out, seeds, debug_out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        local_count = 0
        seed = seeds[i, j]
        for _ in range(num_samples):
            seed, rand_x = lcg_random(seed)
            seed, rand_y = lcg_random(seed)
            x = xmin + (rand_x * width * out.shape[1])
            y = ymin + (rand_y * height * out.shape[0])
            local_count += is_in_mandelbrot(x, y)
        out[i, j] = local_count
        seeds[i, j] = seed
        
        # Debug: Save some sample points
        if i == 0 and j == 0:
            for k in range(10):
                seed, rand_x = lcg_random(seed)
                seed, rand_y = lcg_random(seed)
                x = xmin + (rand_x * width * out.shape[1])
                y = ymin + (rand_y * height * out.shape[0])
                debug_out[k, 0] = x
                debug_out[k, 1] = y
                debug_out[k, 2] = is_in_mandelbrot(x, y)

# Host function to launch CUDA kernel
def count_mandelbrot_cuda(xmin, width, ymin, height, num_samples, num_tiles_1d):
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (num_tiles_1d + threads_per_block[0] - 1) // threads_per_block[0],
        (num_tiles_1d + threads_per_block[1] - 1) // threads_per_block[1]
    )
    
    out = cuda.device_array((num_tiles_1d, num_tiles_1d), dtype=np.int32)
    seeds = cuda.to_device(np.random.randint(1, 2**32, size=(num_tiles_1d, num_tiles_1d), dtype=np.uint32))
    debug_out = cuda.device_array((10, 3), dtype=np.float64)
    
    count_mandelbrot_kernel[blocks_per_grid, threads_per_block](xmin, width, ymin, height, num_samples, out, seeds, debug_out)
    return out.copy_to_host(), debug_out.copy_to_host()

# Main execution
if __name__ == "__main__":
    print("Starting main execution")
    
    NUM_TILES_1D = 1000
    SAMPLES_IN_BATCH = 1000
    
    xmin, xmax = -2, 1
    ymin, ymax = -1.5, 1.5
    width = (xmax - xmin) / NUM_TILES_1D
    height = (ymax - ymin) / NUM_TILES_1D
    
    print("Starting computation...")
    new_counts, debug_points = count_mandelbrot_cuda(xmin, width, ymin, height, SAMPLES_IN_BATCH, NUM_TILES_1D)
    
    # print("Computation finished. Calculating final results.")
    # print(f"new_counts shape: {new_counts.shape}, min: {np.min(new_counts)}, max: {np.max(new_counts)}")
    # print("Sample debug points (x, y, in_set):")
    # for point in debug_points:
    #     print(f"({point[0]:.4f}, {point[1]:.4f}): {'In set' if point[2] == 1 else 'Not in set'}")
    
    final_value = (np.sum(new_counts) / (NUM_TILES_1D * NUM_TILES_1D * SAMPLES_IN_BATCH)) * (xmax - xmin) * (ymax - ymin)
    print(f"The estimated area of the Mandelbrot set is {final_value}")

    print("Main execution completed")