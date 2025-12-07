import pandas as pd
import numpy as np
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad

META_PATH = "JWST_NIRCam_Triple_Filter/nircam_F200W-F277W-F444W_meta.csv"
OUT_PATH = "JJWST_NIRCam_Triple_Filter/nircam_F200W-F277W-F444W_labeled.csv"

RA_COL = "ra_src"
DEC_COL = "dec_src"

GAIA_RAD = 0.5 * u.arcsec
SIMBAD_RAD = 0.5 * u.arcsec

N_WORKERS = 3

simbad = Simbad()
simbad.add_votable_fields("otypes", "otype")
Simbad.ROW_LIMIT = 1

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

meta = pd.read_csv(META_PATH)

if RA_COL not in meta.columns or DEC_COL not in meta.columns:
    raise SystemExit(f"Expected columns {RA_COL}, {DEC_COL} in {META_PATH} !")

coords = SkyCoord(meta[RA_COL].values * u.deg, meta[DEC_COL].values * u.deg)

meta["gaia_source_id"] = pd.Series([None] * len(meta), dtype="object")
meta["gaia_g"] = np.nan
meta["gaia_bp_rp"] = np.nan

meta["simbad_main_id"] = pd.Series([None] * len(meta), dtype="object")
meta["simbad_type"] = pd.Series([None] * len(meta), dtype="object")

def query_catalogs_for_index(idx):
    c = coords[idx]
    out = {
        "i": idx,
        "gaia_source_id": None,
        "gaia_g": np.nan,
        "gaia_bp_rp": np.nan,
        "simbad_main_id": None,
        "simbad_type": None
    }

    try:
        job = Gaia.cone_search_async(c, radius = GAIA_RAD)
        r = job.get_results()
        if len(r) > 0:
            gaia_coords = SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg)
            sep = c.separation(gaia_coords)
            j_min = np.argmin(sep)

            row = r[j_min]
            out["gaia_source_id"] = str(row["source_id"])
            out["gaia_g"] = float(row["phot_g_mean_mag"]) if row["phot_g_mean_mag"] else np.nan
            bp_rp = row["bp_rp"]
            out["gaia_bp_rp"] = float(bp_rp) if bp_rp and not np.ma.is_masked(bp_rp) else np.nan
    except Exception as e:
        print(f"[GAIA] index {idx}: {e}")
        pass

    try:
        res = simbad.query_region(c, radius=SIMBAD_RAD)
        if res is not None and len(res) > 0:
            s_coords = SkyCoord(res["RA"], res["DEC"], unit=(u.hourangle, u.deg))
            sep = c.separation(s_coords)
            j_min = np.argmin(sep)

            row = res[j_min]
            out["simbad_main_id"] = row["MAIN_ID"].decode() if isinstance(row["MAIN_ID"], bytes) else row["MAIN_ID"]
            out["simbad_type"] = row["OTYPE"].decode() if isinstance(row["OTYPE"], bytes) else row["OTYPE"]
    except Exception as e:
        print(f"[SIMBAD] index {idx}: {e}")
        pass

    return out

def run_labeling():
    indices = list(range(len(meta)))
    results = []

    print(f"[Info] Querying catalogs for {len(indices)} sources\n[Info] Workers: {N_WORKERS}")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(query_catalogs_for_index, i): i for i in indices}
        for fut in tqdm(as_completed(futures), total = len(futures), desc="Labeling", unit="src"):
            res = fut.result()
            results.append(res)

    for r in results:
        i = r["i"]
        meta.at[i, "gaia_source_id"] = r["gaia_source_id"]
        meta.at[i, "gaia_g"] = r["gaia_g"]
        meta.at[i, "gaia_bp_rp"] = r["gaia_bp_rp"]
        meta.at[i, "simbad_main_id"] = r["simbad_main_id"]
        meta.at[i, "simbad_type"] = r["simbad_type"]

def decide_target(row):
    st = row["simbad_type"]

    if isinstance(st, str):
        st_upper = st.upper()
        if "STAR" in st_upper or st_upper in ["*", "**", "SB*"]:
            return "star"
        if "GALAXY" in st_upper or st_upper in ["G", "GIN", "GiC"]:
            return "galaxy"
        if "QSO" in st_upper or "AGN" in st_upper or "BLL" in st_upper:
            return "agn"
        
    if pd.notna(row["gaia_g"]):
        return "star"
    
    return "unlabeled"

if __name__ == "__main__":
    run_labeling()
    print("[Info] Catalog columns filled\n[Info] Deciding object targets")
    meta["target"] = meta.apply(decide_target, axis=1)

    meta.to_csv(OUT_PATH, index=False)
    print(f"[Done] Saved labeled catalog to {OUT_PATH}")