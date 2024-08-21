from production.mandelbrot_steps import mandelbrot, count_mandelbrot, draw_mandelbrot

import argparse
import jax
import numpy as np
import os

from random import SystemRandom


def main():
    parser = argparse.ArgumentParser(description="Calculate the area of the Mandelbrot set.")
    parser.add_argument("--xmin", type=float, help="Minimum x-value of the region")
    parser.add_argument("--xmax", type=float, help="Maximum x-value of the region")
    parser.add_argument("--ymin", type=float, help="Minimum y-value of the region")
    parser.add_argument("--ymax", type=float, help="Maximum y-value of the region")
    parser.add_argument("--num-samples", type=int, help="Number of samples to use in the Monte Carlo simulation")
    parser.add_argument("--num-batches", type=int, help="Number of pixels in the x-direction")
    parser.add_argument("--out-directory", type=str, help="Output directory for the results.")
    args = parser.parse_args()

    if args.out_directory is None:
        args.out_directory = "results"
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)
    # Variables
    xmin, xmax = args.xmin, args.xmax
    ymin, ymax = args.ymin, args.ymax
    assert(xmin < xmax)
    assert(ymin < ymax)
    num_samples = args.num_samples 
    num_batches = args.num_batches

    seed = SystemRandom().randint(0, 1000000000) 
    print(f"Running with seed: {seed}")

    JAX_MOTHER_RNG_KEY = jax.random.PRNGKey(seed)

    # ToDo: test if faster witht vmap
    # I don't expect it to
    #inside_count  = jax.vmap(count_mandelbrot, in_axes=[0, None,None,None,None,None, ])(keys, num_samples, xmin, xmax - xmin, ymin, ymax - ymin)

    total_hits = 0
    total_samples = 0

    for i, random_key in enumerate(jax.random.split(JAX_MOTHER_RNG_KEY, num_batches)):
        print(f"Starting run {i} with seed {random_key}")
        inside_hits = count_mandelbrot(random_key, num_samples, xmin, xmax - xmin, ymin, ymax - ymin)
        total_hits += inside_hits
        total_samples += num_samples

    tile_size = (xmax - xmin) * (ymax - ymin) 
    hit_precentage = total_hits / total_samples
    area = hit_precentage * tile_size 
    print(f"Area of the Mandelbrot set is {area}, with {hit_precentage:2.2f}% hits at {total_samples} samples")
    result = np.array([tile_size, total_hits,total_samples])

    out_path = os.path.join(args.out_directory, f"result_{seed}.npy")
    np.save(out_path, result)

if __name__ == "__main__":
    main()
"""
# Zeichnen der Mandelbrotmenge für eine 1000x1000-Pixel-Darstellung
pixels = draw_mandelbrot(num_x, num_y)
fig, _, _ = plot_pixels(pixels)
fig.savefig("pixels_jax.png")

# Regionen für die Unsicherheitsberechnung
regions = [
    {"xmin": -1.5, "ymin": 0.5, "width": 0.5, "height": 0.5},
    {"xmin": -0.4, "ymin": 0.5, "width": 0.5, "height": 0.5},
    {"xmin": -0.4, "ymin": -0.25, "width": 0.5, "height": 0.5},
]
for region in regions:
    numerator  = count_mandelbrot(
        JAX_MOTHER_RNG_KEY,
        num_samples,
        region["xmin"],
        region["width"],
        region["ymin"],
        region["height"],
    )
    valid_samples = num_samples 
    ci = confidence_interval(
        0.05, numerator, valid_samples, region["width"] * region["height"]
    )
    print(f"Region: {region} --> CI: {ci}")

# Parallelisierung mit JAX
NUM_TILES_1D = 100
width = 3 / NUM_TILES_1D
height = 3 / NUM_TILES_1D

@jit
def compute_tile(rng, j, i):
    denom = 100
    numer  = count_mandelbrot(rng, denom, xmin + j * width, width, ymin + i * height, height)
    valid_samples = denom
    return numer, valid_samples

# Verteilte Berechnung über Tiles
rngs = random.split(rng, NUM_TILES_1D * NUM_TILES_1D)
numer, valid_samples = jax.vmap(lambda idx: compute_tile(rngs[idx], idx // NUM_TILES_1D, idx % NUM_TILES_1D))(jnp.arange(NUM_TILES_1D * NUM_TILES_1D))
numer = numer.reshape(NUM_TILES_1D, NUM_TILES_1D)
valid_samples = valid_samples.reshape(NUM_TILES_1D, NUM_TILES_1D)

# Unsicherheitsberechnung pro Tile
ci_low, ci_high = confidence_interval(
    0.05, numer, valid_samples, width * height
)
final_uncertainty = combine_uncertainties(ci_low, ci_high, valid_samples)

# Finale Fläche berechnen
final_value = jnp.sum(numer / valid_samples) * width * height
print(f"The total area of the Mandelbrot set is {final_value}")
print(f"The uncertainty on the total area is {final_uncertainty}")
"""

