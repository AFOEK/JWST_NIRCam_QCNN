import pandas as pd
import numpy as np
import time, os, logging, warnings

from tqdm import tqdm
from collections import Counter

from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.ipac.irsa import Irsa
from astroquery.sdss import SDSS
from astroquery.mast import Catalogs
from astroquery.ipac.ned import Ned

from astroquery.exceptions import NoResultsWarning

logging.getLogger("astroquery").setLevel(logging.ERROR)
logging.getLogger("astropy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=NoResultsWarning)

META_PATH = "JWST_NIRCam_Triple_Filter/nircam_F200W-F277W-F444W_meta.csv"
OUT_PATH = "JWST_NIRCam_Triple_Filter/nircam_F200W-F277W-F444W_labeled.csv"

out_dir = os.path.dirname(OUT_PATH)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

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

Gaia.TIMEOUT = 60
Simbad.TIMEOUT = 60
Irsa.TIMEOUT = 45
SDSS.TIMEOUT = 50
Ned.TIMEOUT = 45
Catalogs.TIMEOUT = 40

if os.path.exists(OUT_PATH):
    print(f"[Info] Resuming from existing labeled file: {OUT_PATH}")
    meta = pd.read_csv(OUT_PATH)
else:
    print(f"[Info] Starting from base meta file: {META_PATH}")
    meta = pd.read_csv(META_PATH)

if RA_COL not in meta.columns or DEC_COL not in meta.columns:
    raise SystemExit(f"Expected columns {RA_COL}, {DEC_COL} in {META_PATH} !")

meta = meta.reset_index(drop=True)
ra = pd.to_numeric(meta[RA_COL], errors="coerce").to_numpy()
dec = pd.to_numeric(meta[DEC_COL], errors="coerce").to_numpy()

coords = SkyCoord(ra * u.deg, dec * u.deg)

def ensure_col(name, default=np.nan, dtype=None):
    if name not in meta.columns:
        if dtype == "object":
            meta[name] = pd.Series([None] * len(meta), dtype="object")
        else:
            meta[name] = default

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
            if "500" in str(e) or "502" in str(e):
                time.sleep(10 * attempt)
                continue
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
            if "500" in str(e) or "502" in str(e):
                time.sleep(10 * attempt)
                continue
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
            if "500" in str(e) or "502" in str(e):
                time.sleep(10 * attempt)
                continue
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
            if "500" in str(e) or "502" in str(e):
                time.sleep(10 * attempt)
                continue
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
            if "500" in str(e) or "502" in str(e):
                time.sleep(10 * attempt)
                continue
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
            if "500" in str(e) or "502" in str(e):
                time.sleep(10 * attempt)
                continue
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
            if "500" in str(e) or "502" in str(e):
                time.sleep(10 * attempt)
                continue
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

        "wise_designation": None,
        "wise_w1": np.nan,
        "wise_w2": np.nan,
        "wise_w1_w2": np.nan,
        "wise_sep_arcsec": np.nan,

        "gaia_sep_arcsec": np.nan,
        "simbad_sep_arcsec": np.nan,
        "wise_sep_arcsec": np.nan,
        "sdss_sep_arcsec": np.nan,
        "ps1_sep_arcsec": np.nan,
        "tmass_sep_arcsec": np.nan,
        "ned_sep_arcsec": np.nan,
    }

    r, g_err = query_gaia_with_retry(c)
    if r is not None and len(r) > 0:
        tqdm.write(f"[DBG] idx {idx} -> GAIA")
        ra_g = np.array(r["ra"], dtype=float)
        dec_g = np.array(r["dec"], dtype=float)
        gaia_coords = SkyCoord(ra_g, dec_g, unit="deg")

        sep = c.separation(gaia_coords)
        j_min = np.argmin(sep)
        out["gaia_sep_arcsec"] = float(sep[j_min].arcsec)

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
        tqdm.write(f"[DBG] idx {idx} -> SIMBAD")
        if ("ra" in res.colnames) and ("dec" in res.colnames):
            ra_s = np.array(res["ra"], dtype=str)
            dec_s = np.array(res["dec"], dtype=str)
            s_coords = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))

            sep = c.separation(s_coords)
            j_min = int(np.argmin(sep))
            out["simbad_sep_arcsec"] = float(sep[j_min].arcsec)
            row = res[j_min]
        else:
            out["simbad_sep_arcsec"] = np.nan
            row = res[0]

        main_id = row["main_id"]
        otype = row["otype"]
        out["simbad_main_id"] = main_id.decode() if isinstance(main_id, bytes) else main_id
        out["simbad_type"] = otype.decode() if isinstance(otype, bytes) else otype
    elif s_err is not None:
        print(f"[SIMBAD] index {idx}: giving up after retries: {s_err}")

    res_wise, w_err = query_wise_with_retry(c)
    if res_wise is not None and len(res_wise) > 0:
        tqdm.write(f"[DBG] idx {idx} -> WISE")
        try:
            ra_w = np.array(res_wise["ra"], dtype=float)
            dec_w = np.array(res_wise["dec"], dtype=float)
        except KeyError:
            ra_w = np.array(res_wise["RAJ2000"], dtype=float)
            dec_w = np.array(res_wise["DEJ2000"], dtype=float)

        w_coords = SkyCoord(ra_w, dec_w, unit="deg")
        sep = c.separation(w_coords)
        j_min = np.argmin(sep)
        out["wise_sep_arcsec"] = float(sep[j_min].arcsec)

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
        tqdm.write(f"[DBG] idx {idx} -> SDSS")
        ra_sd = np.array(res_sdss["ra"], dtype=float)
        dec_sd = np.array(res_sdss["dec"], dtype=float)
        sdss_coords = SkyCoord(ra_sd, dec_sd, unit="deg")

        sep = c.separation(sdss_coords)
        j_min = np.argmin(sep)
        out["sdss_sep_arcsec"] = float(sep[j_min].arcsec)

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
        tqdm.write(f"[DBG] idx {idx} -> PS1")
        ra_ps = np.array(res_ps1["ra"], dtype=float)
        dec_ps = np.array(res_ps1["dec"], dtype=float)
        ps_coords = SkyCoord(ra_ps, dec_ps, unit="deg")

        sep = c.separation(ps_coords)
        j_min = np.argmin(sep)
        out["ps1_sep_arcsec"] = float(sep[j_min].arcsec)

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
        tqdm.write(f"[DBG] idx {idx} -> TMASS")
        ra_tm = np.array(res_tm["ra"], dtype=float)
        dec_tm = np.array(res_tm["dec"], dtype=float)
        tm_coords = SkyCoord(ra_tm, dec_tm, unit="deg")

        sep = c.separation(tm_coords)
        j_min = np.argmin(sep)
        out["tmass_sep_arcsec"] = float(sep[j_min].arcsec)

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
        tqdm.write(f"[DBG] idx {idx} -> NED")
        ra_ned = np.array(res_ned["RA"], dtype=float)
        dec_ned = np.array(res_ned["DEC"], dtype=float)
        ned_coords = SkyCoord(ra_ned, dec_ned, unit="deg")

        sep = c.separation(ned_coords)
        j_min = np.argmin(sep)
        out["ned_sep_arcsec"] = float(sep[j_min].arcsec)

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

def run_labeling(save_every=50, retry_unlabeled_only=False):
    if retry_unlabeled_only:
        if "target" in meta.columns:
            pending_indices = meta.index[meta["target"].eq("unlabeled")].tolist()
        else:
            pending_indices = meta.index[
                meta["gaia_source_id"].isna()
                & meta["simbad_main_id"].isna()
                & meta["ned_name"].isna()
                & meta["sdss_objid"].isna()
            ].tolist()
    else:
        pending_indices = meta.index[~meta["labeled_done"].astype(bool)].tolist()

    if not pending_indices:
        print("[Info] All rows already labeled, nothing to do.")
        return

    print(f"[Info] Querying catalogs for {len(pending_indices)} sources")
    processed_since_save = 0
    last_save_t = time.time()

    for i in tqdm(pending_indices, desc="Labeling", unit="src"):
        try:
            r = query_catalogs_for_index(i)
        except Exception as e:
            tqdm.write(f"[Err] index {i} failed: {e}")
            meta.at[i, "labeled_done"] = True
            processed_since_save += 1
            now = time.time()

            if processed_since_save >= save_every or (now - last_save_t) >= 60:
                tqdm.write(f"[DBG] Saved files to {OUT_PATH}")
                meta.to_csv(OUT_PATH, index=False)
                processed_since_save = 0
                last_save_t = now
            continue

        meta.at[i, "gaia_source_id"]    = r["gaia_source_id"]
        meta.at[i, "gaia_g"]            = r["gaia_g"]
        meta.at[i, "gaia_bp_rp"]        = r["gaia_bp_rp"]

        meta.at[i, "simbad_main_id"]    = r["simbad_main_id"]
        meta.at[i, "simbad_type"]       = r["simbad_type"]

        meta.at[i, "wise_designation"]  = r["wise_designation"]
        meta.at[i, "wise_w1"]           = r["wise_w1"]
        meta.at[i, "wise_w2"]           = r["wise_w2"]
        meta.at[i, "wise_w1_w2"]        = r["wise_w1_w2"]

        meta.at[i, "sdss_objid"]        = r["sdss_objid"]
        meta.at[i, "sdss_class"]        = r["sdss_class"]
        meta.at[i, "sdss_z"]            = r["sdss_z"]

        meta.at[i, "ps1_objid"]         = r["ps1_objid"]
        meta.at[i, "ps1_g"]             = r["ps1_g"]
        meta.at[i, "ps1_r"]             = r["ps1_r"]
        meta.at[i, "ps1_i"]             = r["ps1_i"]

        meta.at[i, "tmass_designation"] = r["tmass_designation"]
        meta.at[i, "tmass_j"]           = r["tmass_j"]
        meta.at[i, "tmass_h"]           = r["tmass_h"]
        meta.at[i, "tmass_ks"]          = r["tmass_ks"]
        meta.at[i, "tmass_j_ks"]        = r["tmass_j_ks"]

        meta.at[i, "ned_name"]          = r["ned_name"]
        meta.at[i, "ned_type"]          = r["ned_type"]
        meta.at[i, "ned_z"]             = r["ned_z"]

        meta.at[i, "gaia_sep_arcsec"]   = r["gaia_sep_arcsec"]
        meta.at[i, "simbad_sep_arcsec"] = r["simbad_sep_arcsec"]
        meta.at[i, "wise_sep_arcsec"]   = r["wise_sep_arcsec"]
        meta.at[i, "sdss_sep_arcsec"]   = r["sdss_sep_arcsec"]
        meta.at[i, "ps1_sep_arcsec"]    = r["ps1_sep_arcsec"]
        meta.at[i, "tmass_sep_arcsec"]  = r["tmass_sep_arcsec"]
        meta.at[i, "ned_sep_arcsec"]    = r["ned_sep_arcsec"]

        meta.at[i, "labeled_done"] = True
        
        processed_since_save += 1
        now = time.time()
        if processed_since_save >= save_every or (now - last_save_t) >= 60:
            tqdm.write(f"[DBG] Saved files to {OUT_PATH}")
            meta.to_csv(OUT_PATH, index=False)
            processed_since_save = 0
            last_save_t = now

    meta.to_csv(OUT_PATH, index=False)
    print(f"[Info] Saved progress to {OUT_PATH}")


def decide_target(row):
    weights = {
        "gaia":   2.5,   # GAIA: strong but not absolute
        "simbad": 3.0,   # SIMBAD type: very trusted
        "ned":    3.0,   # NED type: very trusted
        "sdss":   3.0,   # SDSS spectroscopic class
        "wise":   1.5,   # WISE color hint (AGN-ish)
        "tmass":  1.0,   # 2MASS color hint (star-ish)
        "ps1":    1.0,   # PS1 color hint (star-ish)
    }

    label_scores = {"star": 0.0, "galaxy": 0.0, "agn": 0.0}

    simbad_vote = None
    st = row.get("simbad_type", None)
    if isinstance(st, str):
        t = st.strip().upper()
        # --- STAR-ish (SIMBAD has lots of compact codes) ---
        if (t in {"*", "**", "SB*", "WD*", "BD*", "PM*", "V*"} or
            "STAR" in t or t.endswith("*")):
            simbad_vote = "star"
        # --- AGN/QSO-ish ---
        elif (t in {"QSO", "AGN", "BLL", "BLAZAR", "SY1", "SY2", "LIN", "SEYFERT"} or
            "QSO" in t or "AGN" in t or "BLAZAR" in t or "SEYFERT" in t):
            simbad_vote = "agn"
        # --- GALAXY-ish ---
        elif (t in {"G", "GAL", "GALAXY", "EMG", "HII", "IG"} or
            "GALAXY" in t or "HII" in t):
            simbad_vote = "galaxy"
        elif t in {"X", "RAD", "IR", "IRS"}:
            simbad_vote = None  

    if simbad_vote is not None:
        label_scores[simbad_vote] += weights["simbad"]

    ned_vote = None
    nt = row.get("ned_type", None)
    if isinstance(nt, str):
        t = nt.strip().upper()
        # direct
        if "STAR" in t:
            ned_vote = "star"
        elif t in {"QSO", "AGN", "BLLAC", "BL LAC", "SEYFERT", "SY1", "SY2", "LINER"} or "QSO" in t or "AGN" in t:
            ned_vote = "agn"
        elif t in {"G", "GALAXY", "HII", "EMG"} or "GALAXY" in t or "HII" in t:
            ned_vote = "galaxy"
        elif t in {"IRS", "IRSRC", "IRSOURCE", "IR", "IRAS", "WISE"} or "IR" in t:
            # NED "IrS" (Infrared Source) is often a galaxy/AGN host.
            # If it has redshift -> almost surely extragalactic.
            if pd.notna(row.get("ned_z", np.nan)):
                ned_vote = "galaxy"
            else:
                ned_vote = "galaxy"

    if ned_vote is None and pd.notna(row.get("ned_z", np.nan)):
        ned_vote = "galaxy"
        label_scores["galaxy"] += 2.0

    sdss_vote = None
    sdclass = row.get("sdss_class", None)
    if isinstance(sdclass, str):
        sdc = sdclass.upper()
        if "STAR" in sdc:
            sdss_vote = "star"
        elif "GALAXY" in sdc:
            sdss_vote = "galaxy"
        elif "QSO" in sdc:
            sdss_vote = "agn"

    if sdss_vote is not None:
        label_scores[sdss_vote] += weights["sdss"]

    gaia_vote = None
    if pd.notna(row.get("gaia_g", np.nan)):
        gaia_vote = "star"
        label_scores[gaia_vote] += weights["gaia"]

    wise_vote = None
    w1w2 = row.get("wise_w1_w2", np.nan)
    if pd.notna(w1w2) and w1w2 > 0.8:
        wise_vote = "agn"
        label_scores[wise_vote] += weights["wise"]

    tmass_vote = None
    j_ks = row.get("tmass_j_ks", np.nan)
    if pd.notna(j_ks) and 0.0 <= j_ks <= 1.0:
        tmass_vote = "star"
        label_scores[tmass_vote] += weights["tmass"]

    ps1_vote = None
    g = row.get("ps1_g", np.nan)
    r = row.get("ps1_r", np.nan)
    i = row.get("ps1_i", np.nan)
    if pd.notna(g) and pd.notna(r) and pd.notna(i):
        gr = g - r
        ri = r - i
        if -0.5 <= gr <= 1.5 and -0.5 <= ri <= 1.5:
            ps1_vote = "star"
            label_scores[ps1_vote] += weights["ps1"]

    if all(score == 0.0 for score in label_scores.values()):
        return "unlabeled"

    sorted_scores = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1]

    if best_score > 0 and best_score >= 1.2 * second_score:
        return best_label

    if gaia_vote is not None:
        return gaia_vote
    if simbad_vote is not None:
        return simbad_vote
    if ned_vote is not None:
        return ned_vote
    if sdss_vote is not None:
        return sdss_vote

    if best_score > 0:
        return best_label

    return "unlabeled"

ensure_col("gaia_source_id", dtype="object")
ensure_col("gaia_g")
ensure_col("gaia_bp_rp")
ensure_col("gaia_sep_arcsec")

ensure_col("simbad_main_id", dtype="object")
ensure_col("simbad_type", dtype="object")
ensure_col("simbad_sep_arcsec")

ensure_col("wise_designation", dtype="object")
ensure_col("wise_w1")
ensure_col("wise_w2")
ensure_col("wise_w1_w2")
ensure_col("wise_sep_arcsec")

ensure_col("sdss_objid", dtype="object")
ensure_col("sdss_class", dtype="object")
ensure_col("sdss_z")
ensure_col("sdss_sep_arcsec")

ensure_col("ps1_objid", dtype="object")
ensure_col("ps1_g")
ensure_col("ps1_r")
ensure_col("ps1_i")
ensure_col("ps1_sep_arcsec")

ensure_col("tmass_designation", dtype="object")
ensure_col("tmass_j")
ensure_col("tmass_h")
ensure_col("tmass_ks")
ensure_col("tmass_j_ks")
ensure_col("tmass_sep_arcsec")

ensure_col("ned_name", dtype="object")
ensure_col("ned_type", dtype="object")
ensure_col("ned_z")
ensure_col("ned_sep_arcsec")

if "labeled_done" not in meta.columns:
    meta["labeled_done"] = False
else:
    meta["labeled_done"] = meta["labeled_done"].astype(bool)

if __name__ == "__main__":
    print("[Info] Fetching Catalogs data...")
    run_labeling(save_every=10, retry_unlabeled_only=False)
    meta["target"] = meta.apply(decide_target, axis=1)
    meta.to_csv(OUT_PATH, index=False)
    print(f"[Info] Saved progress + targets to {OUT_PATH}")

    if meta["labeled_done"].all():
        print("[Done] All rows labeled.")
    else:
        print("[Info] Partial labeling saved; resume later to finish.")

