import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from jax import random, jit, vmap,lax 
from functools import partial

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
MAX_ITER = 10000


@partial(jit, static_argnames=['max_iter'])
def mandelbrot(c, max_iter=100):
    def cond_fn(state):
        z_tortoise, z_hare, iter_count, diverged, converged = state
        return jnp.logical_and(iter_count < max_iter, jnp.logical_not(diverged | converged))

    def body_fn(state):
        z_tortoise, z_hare, iter_count, diverged, converged = state
        z_tortoise = z_tortoise * z_tortoise + c
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c  # Hare macht zwei Schritte

        # Prüfen auf Divergenz (Betrag > 2)
        diverged = jnp.abs(z_hare) > 2.0

        # Prüfen auf Zyklus (Tortoise-Hare-Vergleich)
        converged = jnp.isclose(z_tortoise, z_hare)

        return z_tortoise, z_hare, iter_count + 1, diverged, converged

    z0 = jnp.zeros_like(c)
    initial_state = (z0, z0, 0, False, False)
    
    final_state = lax.while_loop(cond_fn, body_fn, initial_state)
    _, _, _, diverged, converged = final_state
    
    # Ein Punkt gehört zur Mandelbrotmenge, wenn er nicht divergiert und nicht konvergiert
    return ~(diverged) | converged

# Funktion zur Zählung von Punkten innerhalb der Mandelbrotmenge
@partial(jit, static_argnames=["num_samples", "xmin", "width", "ymin", "height", "max_iter"])
def count_mandelbrot(rng, num_samples, xmin, width, ymin, height, max_iter=100):
    x_norm = random.uniform(rng, (num_samples,))
    y_norm = random.uniform(rng, (num_samples,))
    x = xmin + x_norm * width
    y = ymin + y_norm * height
    c = x + 1j * y
    in_set = vmap(lambda z: mandelbrot(z, max_iter))(c)
    return jnp.sum(in_set)

# Funktion zum Zeichnen der Mandelbrotmenge unter Verwendung von mandelbrot und count_mandelbrot
def draw_mandelbrot(max_iter=100):
    xmin, xmax = -2, 1
    ymin, ymax = -1.5, 1.5
    
    # Erzeuge ein Raster von komplexen Zahlen
    x = jnp.linspace(xmin, xmax, num_x)
    y = jnp.linspace(ymin, ymax, num_y)
    xv, yv = jnp.meshgrid(x, y, indexing='xy')
    c = xv + 1j * yv
    
    # Berechne die Mandelbrotmenge für jedes Pixel im Raster
    pixels = vmap(vmap(lambda z: mandelbrot(z, max_iter), in_axes=0), in_axes=0)(c)
    
    return pixels

# Beispiel: Berechnung der Mandelbrotmenge für eine 1000x1000-Pixel-Darstellung
pixels = draw_mandelbrot()

# Setze Parameter für die Monte-Carlo-Simulation

print("Calculating the area of the Mandelbrot set...")
numerator = count_mandelbrot(rng, num_samples, xmin, xmax - xmin, ymin, ymax - ymin, max_iter=MAX_ITER)
area = (numerator / num_samples) * (xmax - xmin) * (ymax - ymin)
print(f"Area of the Mandelbrot set is {area}")

# Zeichnen der Mandelbrotmenge für eine 1000x1000-Pixel-Darstellung
pixels = draw_mandelbrot()
fig, _, _ = plot_pixels(pixels)
fig.savefig("pixels_jax.png")

# Regionen für die Unsicherheitsberechnung
regions = [
    {"xmin": -1.5, "ymin": 0.5, "width": 0.5, "height": 0.5},
    {"xmin": -0.4, "ymin": 0.5, "width": 0.5, "height": 0.5},
    {"xmin": -0.4, "ymin": -0.25, "width": 0.5, "height": 0.5},
]

for region in regions:
    numerator = count_mandelbrot(
        rng,
        num_samples,
        region["xmin"],
        region["width"],
        region["ymin"],
        region["height"],
    )
    ci = confidence_interval(
        0.05, numerator, num_samples, region["width"] * region["height"]
    )
    print(f"Region: {region} --> CI: {ci}")

# Parallelisierung mit JAX
NUM_TILES_1D = 100
width = 3 / NUM_TILES_1D
height = 3 / NUM_TILES_1D

@jit
def compute_tile(rng, j, i):
    denom = 100
    numer = count_mandelbrot(rng, denom, xmin + j * width, width, ymin + i * height, height)
    return numer, denom

# Verteilte Berechnung über Tiles
rngs = random.split(rng, NUM_TILES_1D * NUM_TILES_1D)
numer, denom = jax.vmap(lambda idx: compute_tile(rngs[idx], idx // NUM_TILES_1D, idx % NUM_TILES_1D))(jnp.arange(NUM_TILES_1D * NUM_TILES_1D))
numer = numer.reshape(NUM_TILES_1D, NUM_TILES_1D)
denom = denom.reshape(NUM_TILES_1D, NUM_TILES_1D)

# Unsicherheitsberechnung pro Tile
ci_low, ci_high = confidence_interval(
    0.05, numer, denom, width * height
)
final_uncertainty = combine_uncertainties(ci_low, ci_high, denom)

# Finale Fläche berechnen
final_value = jnp.sum(numer / denom) * width * height
print(f"The total area of the Mandelbrot set is {final_value}")
print(f"The uncertainty on the total area is {final_uncertainty}")

