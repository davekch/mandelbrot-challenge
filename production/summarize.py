import numpy as np
from scipy.stats import beta

def confidence_interval(confidence_level, numerator, denominator, area):
    """Calculate confidence interval based on Clopper-Pearson.
    `beta.ppf` is the Percent Point function of the Beta distribution.
    Check out
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    """

    low = (
        np.nan_to_num(
            beta.ppf(confidence_level / 2, numerator, denominator - numerator + 1),
            nan=0,
        )
        * area
    )
    high = (
        np.nan_to_num(
            beta.ppf(1 - confidence_level / 2, numerator + 1, denominator - numerator),
            nan=1,
        )
        * area
    )
    # catch nan cases
    low = np.nan_to_num(np.asarray(low), nan=0)
    high = np.nan_to_num(np.asarray(high), nan=area)

    return 0.5  * (high - low)


def summarize(files, tiles):
    tiles = {idx:{"total_hits":0, "total_samples":0, "tile_size":None, "uncertainty2":0} for idx in range(len(tiles))}
    tile_size = 100
    confidence_level = 0.05

    for fname in files:
        result = np.load(fname)

        for tile in result:
            idx = int(tile[0])
            tiles[idx]["total_hits"] += tile[1]
            tiles[idx]["total_samples"] += tile[2]
            tiles[idx]["tile_size"] = tile_size

            # area and uncertainty estimate of tile[idx]
            area_i = tile[1]/tile[2] * tile_size
            sigma_i = confidence_interval(confidence_level, tile[1], tile[2], tile_size)

            # squared uncertainty
            tiles[idx]["uncertainty2"] += (sigma_i/area_i)**2

    for k, tile in tiles.items():
        print(k, tile)
    areas = [tile["total_hits"]/tile["total_samples"]*tile["tile_size"] for k, tile in tiles.items()]
    area = sum(areas)
    uncertainty = area*np.sqrt(sum([tile["uncertainty2"]/areas[i]**2 for i, tile in tiles.items()]))

    return area, uncertainty 



if __name__ == "__main__":
    area, uncertainty = summarize(["e1.npy","e2.npy"], range(0,3))

    print(area, uncertainty)