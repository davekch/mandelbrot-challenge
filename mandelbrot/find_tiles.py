import numpy as np
import numba as nb

SAMPLES_IN_TILE = 100


@nb.jit
def is_in_mandelbrot(x, y):
    """Toirtoise and Hare approach to check if point (x,y) is in Mandelbrot set."""
    c = np.complex64(x) + np.complex64(y) * np.complex64(1j)
    z_hare = z_tortoise = np.complex64(0)  # tortoise and hare start at same point
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = (
            z_hare * z_hare + c
        )  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real**2 + z_hare.imag**2 > 4:
            return False  # diverging to infinity


@nb.jit
def count_mandelbrot(rng, num_samples, xmin, width, ymin, height):
    """Draw num_samples random numbers uniformly between (xmin, xmin+width)
    and (ymin, ymin+height).
    Raise `out` by one if the number is part of the Mandelbrot set.
    """
    out = np.int32(0)
    for x_norm, y_norm in rng.random((num_samples, 2), np.float32):
        x = xmin + (x_norm * width)
        y = ymin + (y_norm * height)
        out += is_in_mandelbrot(x, y)
    return out


def find_edge_tiles(xmin, ymin, width, height, uncert_target, depth):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculate with the Wald approximation in each tile.
    """
    rng = np.random.default_rng()
    # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
    denom = SAMPLES_IN_TILE
    numer = count_mandelbrot(
        rng, SAMPLES_IN_TILE, xmin, width, ymin, height
    )

    if numer > 0 and denom != numer and depth > 0:
        # split the tile
        tiles = []
        tiles.append(find_edge_tiles(xmin,          ymin,          width/2, height/2, uncert_target, depth-1))
        tiles.append(find_edge_tiles(xmin+width/2,  ymin,          width/2, height/2, uncert_target, depth-1))
        tiles.append(find_edge_tiles(xmin,          ymin+height/2, width/2, height/2, uncert_target, depth-1))
        tiles.append(find_edge_tiles(xmin+width/2,  ymin+height/2, width/2, height/2, uncert_target, depth-1))
    else:    
        tiles = [(xmin, xmin+width, ymin, ymin+height)]
    return tiles

# Knill limits
xmin, xmax = -2, 1
ymin, ymax = 0, 3 / 2   # ymin = 0 because symmetry
tiles = find_edge_tiles(xmin, ymin, xmax-xmin, ymax-ymin, 1, 5)
print(tiles)