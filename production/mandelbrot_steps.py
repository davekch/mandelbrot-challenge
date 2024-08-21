import jax.numpy as jnp

from functools import partial
from jax import random, jit, vmap, lax

MAX_ITER = 10000

@partial(jit)
def mandelbrot(c):
    def cond_fn(state):
        z_tortoise, z_hare, iter_count, diverged, converged = state
        return jnp.logical_and(iter_count < MAX_ITER, jnp.logical_not(diverged | converged))

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
    _, _, iter_count, diverged, converged = final_state
    
    # Bestimme, ob der Punkt Teil der Mandelbrotmenge ist
    in_set = jnp.logical_not(diverged) & converged & (iter_count < MAX_ITER)
    
    # Rückgabe von `in_set` und ob der Punkt die maximale Iterationszahl erreicht hat
    return in_set, iter_count == MAX_ITER

# Funktion zur Zählung von Punkten innerhalb der Mandelbrotmenge
@partial(jit, static_argnames=["num_samples", "xmin", "width", "ymin", "height"])
def count_mandelbrot(rng, num_samples, xmin, width, ymin, height):
    x_norm = random.uniform(rng, (num_samples,))
    y_norm = random.uniform(rng, (num_samples,))
    x = xmin + x_norm * width
    y = ymin + y_norm * height
    c = x + 1j * y
    
    results = vmap(lambda z: mandelbrot(z))(c)
    in_set = results[0]
    MAX_ITER_reached = results[1]
    
    # Zählen der Punkte, die zur Mandelbrotmenge gehören
    inside_count = jnp.sum(in_set)
    
    # Zählen der Punkte, die die maximale Iterationszahl erreicht haben
    MAX_ITER_count = jnp.sum(MAX_ITER_reached)
    
    return inside_count, MAX_ITER_count

# Funktion zum Zeichnen der Mandelbrotmenge unter Verwendung von mandelbrot und count_mandelbrot
@partial(jit, static_argnames=["num_x", "num_y"])
def draw_mandelbrot(num_x, num_y):
    xmin, xmax = -2, 1
    ymin, ymax = -1.5, 1.5
    
    # Erzeuge ein Raster von komplexen Zahlen
    x = jnp.linspace(xmin, xmax, num_x)
    y = jnp.linspace(ymin, ymax, num_y)
    xv, yv = jnp.meshgrid(x, y, indexing='xy')
    c = xv + 1j * yv
    
    # Berechne die Mandelbrotmenge für jedes Pixel im Raster
    pixels = vmap(vmap(lambda z: mandelbrot(z)[0], in_axes=0), in_axes=0)(c)
    
    return pixels
