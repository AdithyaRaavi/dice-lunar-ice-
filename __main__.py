"""
Command-line interface for DICE.

    # run the validated synthetic demo
    python -m dice demo

    # run on ANY crater: a folder of parameter rasters (or an .npz)
    python -m dice run --crater path/to/crater_folder --name Faustini
    python -m dice run --crater faustini.npz --pixel-m 25 --landing 40 52

A crater folder should contain rasters whose names contain:
    cpr_l*  dop*   [required]      cpr_s*  vol_frac*  dem*   [optional]
as GeoTIFF (.tif, needs rasterio) or NumPy (.npy).
"""
import argparse
from . import synth, io_dfsar, pipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="dice", description="DICE lunar subsurface-ice pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run on a synthetic scene with known ground truth")
    d.add_argument("--out", default="results")
    d.add_argument("--seed", type=int, default=7)

    r = sub.add_parser("run", help="run on a real crater (folder of rasters or .npz)")
    r.add_argument("--crater", required=True, help="folder of parameter rasters, or an .npz")
    r.add_argument("--name", default=None, help="crater name (used for the output folder)")
    r.add_argument("--pixel-m", type=float, default=25.0, help="ground sample distance, m/pixel")
    r.add_argument("--landing", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="optional landing pixel; auto-chosen if omitted")
    r.add_argument("--steps", type=int, default=12, help="rover traverse waypoints")
    r.add_argument("--out", default="results")

    args = p.parse_args(argv)

    if args.cmd == "demo":
        scene = synth.generate_scene(size=64, seed=args.seed)
        scene["name"] = "demo_synthetic"
        pipeline.run_pipeline(scene, out_dir=args.out)
    elif args.cmd == "run":
        scene = io_dfsar.load_crater(args.crater, pixel_m=args.pixel_m, name=args.name)
        landing = tuple(args.landing) if args.landing else None
        pipeline.run_pipeline(scene, out_dir=args.out, landing=landing, n_steps=args.steps)


if __name__ == "__main__":
    main()
