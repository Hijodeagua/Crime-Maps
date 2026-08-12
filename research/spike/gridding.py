"""Step 2: fixed spatial units — H3 resolution 9 clipped to the Charlotte
city boundary. Emits the cell inventory (count + area) BEFORE any model runs;
walkforward.py refuses to start without it.

Boundary source: Charlotte city limits polygon as GeoJSON. Default URL points
at the City of Charlotte open data ArcGIS layer; it is validated at runtime
(this spike was authored in a sandbox that could not reach the host, so pass
--boundary-url or a local --boundary-file if the default 404s).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import numpy as np
import requests
from shapely.geometry import shape
from shapely.ops import unary_union

DATA = Path(__file__).parent / "data"
H3_RES = 9  # default; audits.py may demote to 8 as primary


def cells_path(res: int) -> Path:
    return DATA / f"cells_res{res}.json"

DEFAULT_BOUNDARY_URL = (
    # City of Charlotte open data — city limits layer (verify at first run;
    # authored offline). Any polygon GeoJSON of the city boundary works.
    "https://gis.charlottenc.gov/arcgis/rest/services/PLN/CityBoundary/"
    "MapServer/0/query?where=1%3D1&outFields=*&f=geojson&returnGeometry=true"
)


def load_boundary(url: str | None, path: str | None):
    if path:
        gj = json.loads(Path(path).read_text())
    else:
        r = requests.get(url or DEFAULT_BOUNDARY_URL, timeout=120)
        r.raise_for_status()
        gj = r.json()
    geoms = [shape(f["geometry"]) for f in gj["features"]]
    return unary_union(geoms)


def cells_for_boundary(boundary, res: int) -> list[str]:
    gj = boundary.__geo_interface__
    cells = h3.geo_to_cells(gj, res)  # h3-py v4 API
    return sorted(cells)


def build(url: str | None = None, path: str | None = None,
          res: int = H3_RES) -> dict:
    boundary = load_boundary(url, path)
    cells = cells_for_boundary(boundary, res)
    areas_km2 = [h3.cell_area(c, unit="km^2") for c in cells]
    inventory = {
        "h3_resolution": res,
        "n_cells": len(cells),
        "mean_cell_area_km2": round(float(np.mean(areas_km2)), 6),
        "total_area_km2": round(float(np.sum(areas_km2)), 2),
        "cells": cells,
        "neighbors": {c: [n for n in h3.grid_ring(c, 1) if n in set(cells)]
                      for c in cells},
    }
    DATA.mkdir(exist_ok=True)
    cells_path(res).write_text(json.dumps(inventory))
    return inventory


def load_cells(res: int = H3_RES) -> dict:
    if not cells_path(res).exists():
        raise SystemExit(f"Run gridding.py --res {res} first: cell inventory "
                         "must be fixed and written to RESULTS.md before any "
                         "model runs.")
    return json.loads(cells_path(res).read_text())


def assign(df, cells: list[str], res: int = H3_RES):
    """Map incident lat/lon -> H3 cell; drop points outside the boundary."""
    cell_set = set(cells)
    ids = [h3.latlng_to_cell(la, lo, res)
           for la, lo in zip(df["Latitude"], df["Longitude"])]
    df = df.assign(cell=ids)
    inside = df["cell"].isin(cell_set)
    return df[inside], round(100 * (1 - inside.mean()), 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary-url")
    ap.add_argument("--boundary-file")
    ap.add_argument("--res", type=int, default=H3_RES)
    args = ap.parse_args()
    inv = build(args.boundary_url, args.boundary_file, args.res)
    print(json.dumps({k: v for k, v in inv.items()
                      if k not in ("cells", "neighbors")}, indent=2))
