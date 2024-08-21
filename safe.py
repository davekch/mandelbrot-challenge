from production.mandelbrot_steps import mandelbrot, count_mandelbrot, draw_mandelbrot

import argparse
import jax
import numpy as np
import os
import json
import jax.numpy as jnp

from random import SystemRandom
from functools import partial

print("Runnign on CPU")
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())


def main():
    parser = argparse.ArgumentParser(description="Calculate the area of the Mandelbrot set.")
    """
    parser.add_argument("--xmin", type=float, help="Minimum x-value of the region")
    parser.add_argument("--xmax", type=float, help="Maximum x-value of the region")
    parser.add_argument("--ymin", type=float, help="Minimum y-value of the region")
    parser.add_argument("--ymax", type=float, help="Maximum y-value of the region")
    """
    parser.add_argument("--first-tile", type=int, help="First tile to calculate")
    parser.add_argument("--last-tile", type=int, help="Last tile to calculate")
    parser.add_argument("--num-samples", type=int, help="Number of samples to use in the Monte Carlo simulation")
    parser.add_argument("--num-batches", type=int, help="Number of pixels in the x-direction")
    parser.add_argument("--out-directory", type=str, help="Output directory for the results.")
    args = parser.parse_args()

    with open("tiles.json") as f:
        tile_config = json.load(f)

    tiles = tile_config[args.first_tile:args.last_tile+1]
    n_tiles = len(tiles)

    if args.out_directory is None:
        args.out_directory = "results"
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)
    # Variables
    """
    xmin, xmax = args.xmin, args.xmax
    ymin, ymax = args.ymin, args.ymax
    assert(xmin < xmax)
    assert(ymin < ymax)
    """
    num_samples = args.num_samples 
    num_batches = args.num_batches

    seed = SystemRandom().randint(0, 1000000000) 
    print(f"Running with seed: {seed}")

    JAX_MOTHER_RNG_KEY = jax.random.PRNGKey(seed)

    # ToDo: test if faster witht vmap
    # I don't expect it to
    #inside_count  = jax.vmap(count_mandelbrot, in_axes=[0, None,None,None,None,None, ])(keys, num_samples, xmin, xmax - xmin, ymin, ymax - ymin)

    total_hits = np.zeros(len(tiles))
    total_samples = np.zeros(len(tiles))
    tile_indices = np.arange(args.first_tile, args.last_tile+1)
    
    def unpack_tiles(tiles):
        tiles_arr = []
        for tile in tiles:
            tiles_arr.append((tile["xmin"], tile["xmax"], tile["ymin"], tile["ymax"]))
        return np.array(tiles_arr)
    tiles_arr = unpack_tiles(tiles)

    """
    @jax.jit
    for i, random_key in enumerate(jax.random.split(JAX_MOTHER_RNG_KEY, num_batches)):
        print(f"Starting run {i} with seed {random_key}")
        for i_tile, tile_config in enumerate(tiles):
            inside_hits = count_mandelbrot(random_key, num_samples, tile_config["xmin"], tile_config["xmax"] - tile_config["xmin"], tile_config["ymin"], tile_config["ymax"] - tile_config["ymin"],)
            total_hits[i_tile] += inside_hits
            total_samples[i_tile] += num_samples
    """

    total_hits, total_samples = optimized_run(JAX_MOTHER_RNG_KEY, num_batches, tiles_arr, num_samples)
    result = np.array([tile_indices, total_hits, total_samples]).T

    out_path = os.path.join(args.out_directory, f"result_{args.first_tile:03d}_{args.last_tile:03d}_{seed}.npy")
    np.save(out_path, result)


#@partial(jax.jit, static_argnames=["tile_config", "random_key", "num_samples"])
def process_tile(random_key, tile_config num_samples):
    inside_hits = count_mandelbrot(
        random_key, num_samples,
        tile_config[0],
        tile_config[1] - tile_config[0],
        tile_config[2],
        tile_config[3] - tile_config[2],
    )
    return inside_hits 

#@partial(jax.jit, static_argnames=["random_key","num_samples", "tiles"])
def process_batch(random_keys, tile, num_samples):
    hits = []
    hits =  jax.vmap(process_tile, in_axes=(0, None, None)(random_keys, tile, num_samples))
    return hits 

def optimized_run(JAX_MOTHER_RNG_KEY, num_batches, tiles, num_samples):
    # Generate all random keys at once
    random_keys = jax.random.split(JAX_MOTHER_RNG_KEY, num_batches)

    # Vectorize over all random keys (batches)
    batch_hits = []
    for tile in tiles:
        batch_hits = process_batch(random_keys, tile, num_samples)

    batch_hits = np.array(batch_hits)
    batch_samples = num_samples * np.ones((num_batches, len(tiles)))

    # Initialize lists to store total hits and samples per tile_config
    total_hits_per_tile = []
    total_samples_per_tile = []

    # Calculate total hits and samples per tile_config
    for i in range(len(tiles)):
        total_hits = jnp.sum(batch_hits[:, i])
        total_samples = jnp.sum(batch_samples[:,i ])
        total_hits_per_tile.append(total_hits)
        total_samples_per_tile.append(total_samples)

    return total_hits_per_tile, total_samples_per_tile

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


