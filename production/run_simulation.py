from production.mandelbrot_steps import mandelbrot, count_mandelbrot, draw_mandelbrot
from jax import random, jit, vmap, lax

import jax
import jax.numpy as jnp

# Utility functions
from utils import (
    combine_uncertainties,
    plot_pixels,
    confidence_interval,
    wald_uncertainty,
)

# Variables
xmin, xmax = -2, 1
ymin, ymax = -1.5, 1.5
rng = random.PRNGKey(0)
num_samples = 10000
num_x = 1000
num_y = 1000

pixels = draw_mandelbrot(num_x, num_y)

# Setze Parameter für die Monte-Carlo-Simulation
print("Calculating the area of the Mandelbrot set...")
inside_count, max_iter_count = count_mandelbrot(rng, num_samples, xmin, xmax - xmin, ymin, ymax - ymin)

# Berücksichtige nur die Punkte, die nicht MAX_ITER erreicht haben
valid_samples = num_samples - max_iter_count

# Berechnung der Fläche
area = (inside_count / valid_samples) * (xmax - xmin) * (ymax - ymin)
print(f"Area of the Mandelbrot set is {area}")

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
    numerator, max_iter_count = count_mandelbrot(
        rng,
        num_samples,
        region["xmin"],
        region["width"],
        region["ymin"],
        region["height"],
    )
    valid_samples = num_samples - max_iter_count
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
    numer, MAX_ITER_count = count_mandelbrot(rng, denom, xmin + j * width, width, ymin + i * height, height)
    valid_samples = denom - MAX_ITER_count
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

