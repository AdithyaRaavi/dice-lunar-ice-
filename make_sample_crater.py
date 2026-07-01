"""
make_sample_crater.py
---------------------
Writes a SAMPLE crater as GeoTIFF rasters, so you can see exactly what input the
pipeline expects and test the real-data path without PRADAN access:

    examples/sample_crater/
        cpr_l.tif   cpr_s.tif   dop.tif   vol_frac.tif   dem.tif

Then run:  python -m dice run --crater examples/sample_crater --name SampleCrater

To use YOUR crater, replace these GeoTIFFs with rasters exported from MIDAS / PolSARPro
(same names), keep the folder, and run the same command. (Data here is synthetic.)
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dice import synth

OUT = os.path.join(os.path.dirname(__file__), "sample_crater")
os.makedirs(OUT, exist_ok=True)

sc = synth.generate_scene(size=64, seed=11)

# a plausible 25 m/pixel transform near the south pole (values are illustrative)
transform = from_origin(-5000.0, 5000.0, sc["pixel_m"], sc["pixel_m"])


def _write(name, arr):
    h, w = arr.shape
    path = os.path.join(OUT, name)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", transform=transform, crs="EPSG:3031") as ds:
        ds.write(arr.astype("float32"), 1)
    print("wrote", path)


_write("cpr_l.tif", sc["cpr_l"])
_write("cpr_s.tif", sc["cpr_s"])
_write("dop.tif", sc["dop"])
_write("vol_frac.tif", sc["vol_frac"])
_write("dem.tif", sc["dem"])
print("Done. Try:  python -m dice run --crater examples/sample_crater --name SampleCrater")
