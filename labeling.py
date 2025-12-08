import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy import units as u

from concurrent.futures import ThreadPoolExecutor, as_completed

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.ipac.irsa import Irsa
from astroquery.sdss import SDSS
from astroquery.mast import Catalogs
from astroquery.ipac.ned import Ned

META_PATH = "JWST_NIRCam_Triple_Filter/nircam_F200W-F277W-F444W_meta.csv"
OUT_PATH = "JJWST_NIRCam_Triple_Filter/nircam_F200W-F277W-F444W_labeled.csv"

RA_COL = "ra_src"
DEC_COL = "dec_src"

GAIA_RAD = 0.5 * u.arcsec
SIMBAD_RAD = 0.5 * u.arcsec
WISE_RAD = 0.5 * u.arcsec
SDSS_RAD = 0.5 * u.arcsec
PS1_RAD = 0.5 * u.arcsec
TMASS_RAD = 0.5 * u.arcsec
NED_RAD = 1.0 * u.arcsec

simbad = Simbad()
simbad.add_votable_fields("otypes", "otype")

Simbad.ROW_LIMIT = 1
Irsa.ROW_LIMIT = 1
Ned.ROW_LIMIT = 5

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
WISE_CATALOG = "allwise_p3as_psd"
PS1_CATALOG = "Panstarrs"
TMASS_CATALOG = "fp_psc"

Gaia.TIMEOUT = 3600
Simbad.TIMEOUT = 3600
Irsa.TIMEOUT = 3600
SDSS.TIMEOUT = 3600
Ned.TIMEOUT = 3600
Catalogs.TIMEOUT = 3600

N_WORKERS = 4

meta = pd.read_csv(META_PATH)

if RA_COL not in meta.columns or DEC_COL not in meta.columns:
    raise SystemExit(f"Expected columns {RA_COL}, {DEC_COL} in {META_PATH} !")

coords = SkyCoord(meta[RA_COL].values * u.deg, meta[DEC_COL].values * u.deg)

meta["gaia_source_id"] = pd.Series([None] * len(meta), dtype="object")
meta["gaia_g"] = np.nan
meta["gaia_bp_rp"] = np.nan

meta["simbad_main_id"] = pd.Series([None] * len(meta), dtype="object")
meta["simbad_type"] = pd.Series([None] * len(meta), dtype="object")

meta["wise_designation"] = pd.Series([None] * len(meta), dtype="object")
meta["wise_w1"] = np.nan
meta["wise_w2"] = np.nan
meta["wise_w1_w2"] = np.nan

meta["sdss_objid"] = pd.Series([None] * len(meta), dtype="object")
meta["sdss_class"] = pd.Series([None] * len(meta), dtype="object")
meta["sdss_z"] = np.nan

meta["ps1_objid"] = pd.Series([None] * len(meta), dtype="object")
meta["ps1_g"] = np.nan
meta["ps1_r"] = np.nan
meta["ps1_i"] = np.nan

meta["tmass_designation"] = pd.Series([None] * len(meta), dtype="object")
meta["tmass_j"] = np.nan
meta["tmass_h"] = np.nan
meta["tmass_ks"] = np.nan
meta["tmass_j_ks"] = np.nan

meta["ned_name"] = pd.Series([None] * len(meta), dtype="object")
meta["ned_type"] = pd.Series([None] * len(meta), dtype="object")
meta["ned_z"] = np.nan

def query_gaia_with_retry(c, max_retries=3, sleep_base=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            job = Gaia.cone_search_async(c, radius=GAIA_RAD)
            r = job.get_results()
            return r, None
        except Exception as e:
            last_err = e
            print(f"[GAIA] attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)
    return None, last_err

def query_simbad_with_retry(c, max_retries=3, sleep_base=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = simbad.query_region(c, radius=SIMBAD_RAD)
            return res, None
        except Exception as e:
            last_err = e
            print(f"[SIMBAD] attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)
    return None, last_err

def query_sdss_with_retry(c, max_retries=3, sleep_base=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = SDSS.query_region(c, radius=SDSS_RAD, spectro=True)
            return res, None
        except Exception as e:
            last_err = e
            print(f"[SDSS] attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)
    return None, last_err

def query_ps1_with_retry(c, max_retries=3, sleep_base=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = Catalogs.query_region(
                c,
                radius=PS1_RAD,
                catalog=PS1_CATALOG,
                data_release="dr2",
                table="mean",
            )
            return res, None
        except Exception as e:
            last_err = e
            print(f"[PS1] attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)
    return None, last_err

def query_tmass_with_retry(c, max_retries=3, sleep_base=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = Irsa.query_region(
                c,
                catalog=TMASS_CATALOG,
                spatial="Cone",
                radius=TMASS_RAD,
            )
            return res, None
        except Exception as e:
            last_err = e
            print(f"[2MASS] attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)
    return None, last_err

def query_ned_with_retry(c, max_retries=3, sleep_base=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = Ned.query_region(c, radius=NED_RAD)
            return res, None
        except Exception as e:
            last_err = e
            print(f"[NED] attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)
    return None, last_err

def query_wise_with_retry(c, max_retries=3, sleep_base=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = Irsa.query_region(
                c,
                catalog=WISE_CATALOG,
                spatial="Cone",
                radius=WISE_RAD,
            )
            return res, None
        except Exception as e:
            last_err = e
            print(f"[WISE] attempt {attempt} failed: {e}")
            time.sleep(sleep_base * attempt)
    return None, last_err


def query_catalogs_for_index(idx):
    c = coords[idx]
    out = {
        "i": idx,
        "gaia_source_id": None,
        "gaia_g": np.nan,
        "gaia_bp_rp": np.nan,
        "simbad_main_id": None,
        "simbad_type": None,

        "sdss_objid": None,
        "sdss_class": None,
        "sdss_z": np.nan,

        "ps1_objid": None,
        "ps1_g": np.nan,
        "ps1_r": np.nan,
        "ps1_i": np.nan,

        "tmass_designation": None,
        "tmass_j": np.nan,
        "tmass_h": np.nan,
        "tmass_ks": np.nan,
        "tmass_j_ks": np.nan,

        "ned_name": None,
        "ned_type": None,
        "ned_z": np.nan,
    }

    r, g_err = query_gaia_with_retry(c)
    if r is not None and len(r) > 0:
        gaia_coords = SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg)
        sep = c.separation(gaia_coords)
        j_min = np.argmin(sep)

        row = r[j_min]
        out["gaia_source_id"] = str(row["source_id"])

        gmag = row["phot_g_mean_mag"]
        out["gaia_g"] = float(gmag) if (gmag is not None and not np.ma.is_masked(gmag)) else np.nan

        bp_rp = row["bp_rp"]
        out["gaia_bp_rp"] = float(bp_rp) if (bp_rp is not None and not np.ma.is_masked(bp_rp)) else np.nan
    elif g_err is not None:
        print(f"[GAIA] index {idx}: giving up after retries: {g_err}")

    res, s_err = query_simbad_with_retry(c)
    if res is not None and len(res) > 0:
        s_coords = SkyCoord(res["RA"], res["DEC"], unit=(u.hourangle, u.deg))
        sep = c.separation(s_coords)
        j_min = np.argmin(sep)

        row = res[j_min]
        main_id = row["MAIN_ID"]
        otype = row["OTYPE"]

        out["simbad_main_id"] = main_id.decode() if isinstance(main_id, bytes) else main_id
        out["simbad_type"] = otype.decode() if isinstance(otype, bytes) else otype
    elif s_err is not None:
        print(f"[SIMBAD] index {idx}: giving up after retries: {s_err}")

    res_wise, w_err = query_wise_with_retry(c)
    if res_wise is not None and len(res_wise) > 0:
        try:
            w_coords = SkyCoord(res_wise["ra"] * u.deg, res_wise["dec"] * u.deg)
        except KeyError:
            w_coords = SkyCoord(res_wise["RAJ2000"], res_wise["DEJ2000"], unit=(u.deg, u.deg))

        sep = c.separation(w_coords)
        j_min = np.argmin(sep)
        row = res_wise[j_min]

        desig = row.get("designation", None)
        if desig is not None:
            out["wise_designation"] = desig.decode() if isinstance(desig, bytes) else desig

        w1 = row.get("w1mpro", None)
        w2 = row.get("w2mpro", None)

        if w1 is not None and not np.ma.is_masked(w1):
            out["wise_w1"] = float(w1)
        if w2 is not None and not np.ma.is_masked(w2):
            out["wise_w2"] = float(w2)

        if np.isfinite(out["wise_w1"]) and np.isfinite(out["wise_w2"]):
            out["wise_w1_w2"] = out["wise_w1"] - out["wise_w2"]
    elif w_err is not None:
        print(f"[WISE] index {idx}: giving up after retries: {w_err}")
    
    res_sdss, sdss_err = query_sdss_with_retry(c)
    if res_sdss is not None and len(res_sdss) > 0:
        sdss_coords = SkyCoord(res_sdss["ra"] * u.deg, res_sdss["dec"] * u.deg)
        sep = c.separation(sdss_coords)
        j_min = np.argmin(sep)
        row_sdss = res_sdss[j_min]

        objid = row_sdss.get("objid", None)
        sd_class = row_sdss.get("class", None)
        z = row_sdss.get("z", None)

        out["sdss_objid"] = str(objid) if objid is not None else None
        out["sdss_class"] = sd_class.decode() if isinstance(sd_class, bytes) else sd_class
        if z is not None and not np.ma.is_masked(z):
            out["sdss_z"] = float(z)
    elif sdss_err is not None:
        print(f"[SDSS] index {idx}: giving up after retries: {sdss_err}")

    res_ps1, ps1_err = query_ps1_with_retry(c)
    if res_ps1 is not None and len(res_ps1) > 0:
        ps_coords = SkyCoord(res_ps1["ra"] * u.deg, res_ps1["dec"] * u.deg)
        sep = c.separation(ps_coords)
        j_min = np.argmin(sep)
        row_ps = res_ps1[j_min]

        objid = row_ps.get("objID", None)
        out["ps1_objid"] = str(objid) if objid is not None else None

        for col_name, key in [("gmag", "ps1_g"), ("rmag", "ps1_r"), ("imag", "ps1_i")]:
            val = row_ps.get(col_name, None)
            if val is not None and not np.ma.is_masked(val):
                out[key] = float(val)
    elif ps1_err is not None:
        print(f"[PS1] index {idx}: giving up after retries: {ps1_err}")

    res_tm, tm_err = query_tmass_with_retry(c)
    if res_tm is not None and len(res_tm) > 0:
        tm_coords = SkyCoord(res_tm["ra"] * u.deg, res_tm["dec"] * u.deg)
        sep = c.separation(tm_coords)
        j_min = np.argmin(sep)
        row_tm = res_tm[j_min]

        desig = row_tm.get("designation", None)
        out["tmass_designation"] = desig.decode() if isinstance(desig, bytes) else desig

        j = row_tm.get("j_m", None)
        h = row_tm.get("h_m", None)
        k = row_tm.get("k_m", None)

        if j is not None and not np.ma.is_masked(j):
            out["tmass_j"] = float(j)
        if h is not None and not np.ma.is_masked(h):
            out["tmass_h"] = float(h)
        if k is not None and not np.ma.is_masked(k):
            out["tmass_ks"] = float(k)

        if np.isfinite(out["tmass_j"]) and np.isfinite(out["tmass_ks"]):
            out["tmass_j_ks"] = out["tmass_j"] - out["tmass_ks"]
    elif tm_err is not None:
        print(f"[2MASS] index {idx}: giving up after retries: {tm_err}")

    res_ned, ned_err = query_ned_with_retry(c)
    if res_ned is not None and len(res_ned) > 0:
        ned_coords = SkyCoord(res_ned["RA"] * u.deg, res_ned["DEC"] * u.deg)
        sep = c.separation(ned_coords)
        j_min = np.argmin(sep)
        row_ned = res_ned[j_min]

        name = row_ned.get("Object Name", None)
        otype = row_ned.get("Type", None)
        z = row_ned.get("Redshift", None)

        out["ned_name"] = name
        out["ned_type"] = otype
        if z is not None and not np.ma.is_masked(z):
            out["ned_z"] = float(z)
    elif ned_err is not None:
        print(f"[NED] index {idx}: giving up after retries: {ned_err}")

    return out

def run_labeling():
    indices = list(range(len(meta)))
    results = {}

    print(f"[Info] Querying catalogs for {len(indices)} sources")
    print(f"[Info] Workers: {N_WORKERS}")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(query_catalogs_for_index, i): i
            for i in indices
        }

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Labeling",
                        unit="src"):
            i = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[Err] index {i} failed: {e}")
                continue
            results[i] = res

    for i, r in results.items():
        meta.at[i, "gaia_source_id"] = r["gaia_source_id"]
        meta.at[i, "gaia_g"] = r["gaia_g"]
        meta.at[i, "gaia_bp_rp"] = r["gaia_bp_rp"]
        meta.at[i, "simbad_main_id"] = r["simbad_main_id"]
        meta.at[i, "simbad_type"] = r["simbad_type"]
        meta.at[i, "wise_designation"] = r["wise_designation"]
        meta.at[i, "wise_w1"] = r["wise_w1"]
        meta.at[i, "wise_w2"] = r["wise_w2"]
        meta.at[i, "wise_w1_w2"] = r["wise_w1_w2"]
        meta.at[i, "sdss_objid"] = r["sdss_objid"]
        meta.at[i, "sdss_class"] = r["sdss_class"]
        meta.at[i, "sdss_z"] = r["sdss_z"]

        meta.at[i, "ps1_objid"] = r["ps1_objid"]
        meta.at[i, "ps1_g"] = r["ps1_g"]
        meta.at[i, "ps1_r"] = r["ps1_r"]
        meta.at[i, "ps1_i"] = r["ps1_i"]

        meta.at[i, "tmass_designation"] = r["tmass_designation"]
        meta.at[i, "tmass_j"] = r["tmass_j"]
        meta.at[i, "tmass_h"] = r["tmass_h"]
        meta.at[i, "tmass_ks"] = r["tmass_ks"]
        meta.at[i, "tmass_j_ks"] = r["tmass_j_ks"]

        meta.at[i, "ned_name"] = r["ned_name"]
        meta.at[i, "ned_type"] = r["ned_type"]
        meta.at[i, "ned_z"] = r["ned_z"]


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

    nt = row.get("ned_type", None)
    if isinstance(nt, str):
        ntu = nt.upper()
        if "GALAXY" in ntu or "HII" in ntu:
            return "galaxy"
        if "QSO" in ntu or "AGN" in ntu or "BL LAC" in ntu:
            return "agn"

    sdclass = row.get("sdss_class", None)
    if isinstance(sdclass, str):
        sdc = sdclass.upper()
        if "STAR" in sdc:
            return "star"
        if "GALAXY" in sdc:
            return "galaxy"
        if "QSO" in sdc:
            return "agn"

    w1w2 = row.get("wise_w1_w2", np.nan)
    if pd.notna(w1w2) and w1w2 > 0.8:
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