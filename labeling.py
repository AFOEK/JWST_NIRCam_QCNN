import pandas as pd
import numpy as np
import time, os, logging, warnings, re

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
from astropy.table import Table
from astroquery.xmatch import XMatch
from astroquery.vizier import Vizier

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

XRAD_DEFAULT_CHUNK = 1200
XRAD_SAVE_EVERY_CHUNKS = 1
XRAD_SLEEP = 0.25

_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

XRAD_CATALOGS = {
    "lotss": {
        "table": "J/A+A/634/A133/table",
        "ra2": "RAJ2000", "dec2": "DEJ2000",
        "max_dist": 4.0 * u.arcsec,
        "chunk": 1500,
    },
    "vlass": {
        "table": "J/ApJS/255/30/comp",
        "ra2": "RAJ2000", "dec2": "DEJ2000",
        "max_dist": 2.0 * u.arcsec,
        "chunk": 600,
    },
    "chandra": {
        "table": "IX/70/csc21mas",
        "ra2": "RAICRS", "dec2": "DEICRS",
        "max_dist": 2.5 * u.arcsec,
        "chunk": 1200,
    },
    "xmm": {
        "table_try": [
            "IX/69/xmm4d13s",
            "IX/69/xmm4d13s/xmm4d13s",
            "IX/69/xmm4d13s/summary",
        ],
        "max_dist": 4.0 * u.arcsec,
        "chunk": 900,
        "probe_radec": True,
    },
    "erosita_s": {"table": "J/A+A/682/A34/erass1-s", "ra2": "RA_ICRS", "dec2": "DE_ICRS", "max_dist": 5.0 * u.arcsec, "chunk": 1200},
    "erosita_m": {"table": "J/A+A/682/A34/erass1-m", "ra2": "RA_ICRS", "dec2": "DE_ICRS", "max_dist": 5.0 * u.arcsec, "chunk": 1200},
    "erosita_h": {"table": "J/A+A/682/A34/erass1-h", "ra2": "RA_ICRS", "dec2": "DE_ICRS", "max_dist": 5.0 * u.arcsec, "chunk": 1200},
}

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

def xrad_to_upload_table(df_slice: pd.DataFrame) -> Table:
    t = Table.from_pandas(df_slice[["row_id", RA_COL, DEC_COL]].copy())
    t.rename_column(RA_COL, "RA")
    t.rename_column(DEC_COL, "DEC")
    return t

def xrad_angdist_to_arcsec(x):
    if hasattr(x, "to_value"):
        return float(x.to_value(u.arcsec))
    if isinstance(x, (bytes, bytearray)):
        x = x.decode(errors="ignore")
    if isinstance(x, str):
        m = _num.search(x)
        if not m:
            return np.nan
        val = float(m.group(0))
        s = x.lower()
        if "deg" in s:
            return val * 3600.0
        if "arcmin" in s:
            return val * 60.0
        return val
    try:
        return float(x)
    except Exception:
        return np.nan

def xrad_keep_nearest(md: pd.DataFrame) -> pd.DataFrame:
    if md is None or len(md) == 0:
        return pd.DataFrame(columns=["row_id"])
    if "angDist" not in md.columns:
        return pd.DataFrame(columns=["row_id"])
    md = md.copy()
    md["angDist"] = [xrad_angdist_to_arcsec(v) for v in md["angDist"]]
    md = md.dropna(subset=["angDist"])
    if len(md) == 0:
        return pd.DataFrame(columns=["row_id"])
    md = md.sort_values("angDist").groupby("row_id", as_index=False).first()
    return md

def xrad_choose_id_col(md: pd.DataFrame):
    cols = {c.lower(): c for c in md.columns}
    preferred = ["4xmm", "2cxo", "srcid", "iauname", "iau_name", "name", "source_name", "source", "id", "designation", "objid"]
    for k in preferred:
        if k in cols:
            return cols[k]
    ban = {"row_id", "angdist", "ra", "dec", "raj2000", "dej2000", "ra_icrs", "de_icrs", "raicrs", "deicrs"}
    for c in md.columns:
        if c.lower() not in ban:
            return c
    return None

def xrad_probe_cat2_radec(cat2_table: str, max_dist: u.Quantity):
    upload = Table({"row_id": [0], "RA": [10.0], "DEC": [10.0]})
    candidates = [
        ("RAJ2000", "DEJ2000"),
        ("_RAJ2000", "_DEJ2000"),
        ("RA_ICRS", "DE_ICRS"),
        ("RAICRS", "DEICRS"),
        ("_RA.icrs", "_DE.icrs"),
        ("RA", "DEC"),
    ]
    last_err = None
    for ra2, dec2 in candidates:
        try:
            XMatch.query(
                cat1=upload,
                cat2=f"vizier:{cat2_table}",
                max_distance=max_dist,
                colRA1="RA",
                colDec1="DEC",
                colRA2=ra2,
                colDec2=dec2,
            )
            return ra2, dec2
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Probe failed for {cat2_table}. Last error: {last_err}")

def xrad_atomic_save():
    meta.to_csv(OUT_PATH, index=False)

def xrad_xmatch(upload: Table, table_id: str, max_dist: u.Quantity, ra2: str, dec2: str):
    res = XMatch.query(
        cat1=upload,
        cat2=f"vizier:{table_id}",
        max_distance=max_dist,
        colRA1="RA",
        colDec1="DEC",
        colRA2=ra2,
        colDec2=dec2,
    )
    if res is None or len(res) == 0:
        return pd.DataFrame(columns=["row_id"])
    return res.to_pandas()

def run_xray_radio(save_every_chunks=1):
    if "row_id" not in meta.columns:
        meta["row_id"] = np.arange(len(meta), dtype=np.int64)

    for name, cfg in XRAD_CATALOGS.items():
        id_col = f"{name}_object_id"
        sep_col = f"{name}_sep_arcsec"
        ensure_col(id_col, dtype="object")
        ensure_col(sep_col)

        if "labeled_done" in meta.columns:
            pending = meta.index[meta[id_col].isna() & (~meta["labeled_done"].astype(bool))].tolist()
        else:
            pending = meta.index[meta[id_col].isna()].tolist()

        if not pending:
            print(f"[XRAD] {name}: nothing to do.")
            continue

        max_dist = cfg.get("max_dist", 3.0 * u.arcsec)
        chunk = int(cfg.get("chunk", XRAD_DEFAULT_CHUNK))

        # Resolve table + radec (XMM probing)
        if cfg.get("probe_radec", False):
            table_try = cfg.get("table_try", [])
            table_ok = None
            ra2_ok = None
            dec2_ok = None
            for tid in table_try:
                try:
                    ra2, dec2 = xrad_probe_cat2_radec(tid, max_dist=max_dist)
                    table_ok, ra2_ok, dec2_ok = tid, ra2, dec2
                    break
                except Exception as e:
                    print(f"[XRAD] {name}: probe failed for {tid}: {e}")
            if table_ok is None:
                print(f"[XRAD] {name}: could not find working table for XMatch, skipping.")
                continue
            table_id, ra2, dec2 = table_ok, ra2_ok, dec2_ok
        else:
            table_id, ra2, dec2 = cfg["table"], cfg["ra2"], cfg["dec2"]

        print(f"[XRAD] {name}: table={table_id} ra2={ra2} dec2={dec2} max_dist={max_dist} pending={len(pending)}/{len(meta)}")

        chunks_done = 0
        for start in tqdm(range(0, len(pending), chunk), desc=f"XRAD {name}", unit="chunk"):
            idxs = pending[start:start + chunk]

            sl = meta.loc[idxs, ["row_id", RA_COL, DEC_COL]].copy()
            sl[RA_COL] = pd.to_numeric(sl[RA_COL], errors="coerce")
            sl[DEC_COL] = pd.to_numeric(sl[DEC_COL], errors="coerce")
            sl = sl.dropna(subset=[RA_COL, DEC_COL])
            if sl.empty:
                continue

            upload = xrad_to_upload_table(sl)

            md = None
            for attempt in range(3):
                try:
                    md = xrad_xmatch(upload, table_id, max_dist, ra2, dec2)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"\n[XRAD] {name} chunk failed: {e}")
                    time.sleep(1.0 * (attempt + 1))

            if md is None or len(md) == 0:
                chunks_done += 1
                if chunks_done % save_every_chunks == 0:
                    xrad_atomic_save()
                time.sleep(XRAD_SLEEP)
                continue

            md = xrad_keep_nearest(md)
            if len(md) == 0:
                chunks_done += 1
                if chunks_done % save_every_chunks == 0:
                    xrad_atomic_save()
                time.sleep(XRAD_SLEEP)
                continue

            cat_id_col = xrad_choose_id_col(md)

            # IMPORTANT: avoid pandas dtype warnings by ensuring object dtype for IDs
            if meta[id_col].dtype != "object":
                meta[id_col] = meta[id_col].astype("object")

            # merge by row_id
            upd = md[["row_id", "angDist"]].copy()
            upd.rename(columns={"angDist": sep_col}, inplace=True)
            if cat_id_col is not None:
                upd[id_col] = md[cat_id_col].astype(str)

            meta.set_index("row_id", inplace=True)
            upd = upd.set_index("row_id")

            meta.loc[upd.index, sep_col] = upd[sep_col].astype(float)
            if id_col in upd.columns:
                meta.loc[upd.index, id_col] = upd[id_col].astype("object")

            meta.reset_index(inplace=True)

            chunks_done += 1
            if chunks_done % save_every_chunks == 0:
                xrad_atomic_save()

            time.sleep(XRAD_SLEEP)

        xrad_atomic_save()
        print(f"[XRAD] {name}: done + saved.")


def _get_float(row, key):
    try:
        v = row[key]
    except Exception:
        return np.nan
    if v is None:
        return np.nan
    try:
        if np.ma.is_masked(v):
            return np.nan
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return np.nan

def _get_str(row, key):
    try:
        v = row[key]
    except Exception:
        return None
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode()
        except Exception:
            return str(v)
    return str(v)

def compute_pm_and_sig(row):
    pmra = _get_float(row, "pmra")
    pmdec = _get_float(row, "pmdec")

    if np.isfinite(pmra) and np.isfinite(pmdec):
        pm = float(np.hypot(pmra, pmdec))
    else:
        pm = _get_float(row, "pm")

    pm_over_error = np.nan
    pmra_e = _get_float(row, "pmra_error")
    pmdec_e = _get_float(row, "pmdec_error")

    if np.isfinite(pm) and np.isfinite(pmra_e) and np.isfinite(pmdec_e):
        pm_err = float(np.hypot(pmra_e, pmdec_e))
        if pm_err > 0:
            pm_over_error = pm / pm_err
    else:
        pm_over_error = _get_float(row, "pm_over_error")

    return pm, pm_over_error


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
        "gaia_parallax": np.nan,
        "gaia_parallax_over_error": np.nan,
        "gaia_pm": np.nan,
        "gaia_pm_over_error": np.nan,
        
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
        gaia_coords = SkyCoord(ra_g * u.deg, dec_g * u.deg)

        sep = c.separation(gaia_coords)
        j_min = int(np.argmin(sep))
        out["gaia_sep_arcsec"] = float(sep[j_min].arcsec)

        row = r[j_min]

        out["gaia_source_id"] = _get_str(row, "source_id")

        out["gaia_g"] = _get_float(row, "phot_g_mean_mag")
        out["gaia_bp_rp"] = _get_float(row, "bp_rp")

        out["gaia_parallax"] = _get_float(row, "parallax")
        poe = _get_float(row, "parallax_over_error")
        if np.isfinite(poe):
            out["gaia_parallax_over_error"] = poe
        else:
            pe = _get_float(row, "parallax_error")
            if np.isfinite(out["gaia_parallax"]) and np.isfinite(pe) and pe > 0:
                out["gaia_parallax_over_error"] = out["gaia_parallax"] / pe

        pm, pm_over = compute_pm_and_sig(row)
        out["gaia_pm"] = pm
        out["gaia_pm_over_error"] = pm_over

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
        meta.at[i, "gaia_parallax"]            = r["gaia_parallax"]
        meta.at[i, "gaia_parallax_over_error"] = r["gaia_parallax_over_error"]
        meta.at[i, "gaia_pm"]                  = r["gaia_pm"]
        meta.at[i, "gaia_pm_over_error"]       = r["gaia_pm_over_error"]

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
        "gaia":   3.5,  # GAIA: strong star signal (parallax/pm/photometry), but not absolute
        "simbad": 3.0,  # SIMBAD type: usually very reliable when present
        "ned":    3.0,  # NED type/redshift: very reliable for extragalactic IDs
        "sdss":   3.0,  # SDSS spectroscopic class: extremely strong when available
        "wise":   1.5,  # WISE color (e.g., W1-W2): decent AGN hint, but can be messy
        "tmass":  1.0,  # 2MASS colors: mild star-ish hint
        "ps1":    1.0,  # PS1 colors: mild star-ish hint

        "chandra":   2.0,  # Chandra X-ray: strong high-energy evidence (often AGN, sometimes XRB/CV)
        "xmm":       1.5,  # XMM X-ray: strong but a bit broader/shallower than Chandra (more confusion)
        "erosita_s": 1.5,  # eROSITA soft band: good X-ray evidence, but soft can include stars/CVs too
        "erosita_m": 1.0,  # eROSITA medium band: evidence, but you have few matches; keep moderate
        "erosita_h": 1.0,  # eROSITA hard band: strong AGN-ish when present, but you have ~0 matches now

        "lotss": 1.5,  # LoTSS radio: good AGN/jet hint, but radio galaxies/SF galaxies exist too
        "vlass": 1.0,  # VLASS radio: higher-res radio; useful, but you currently have very few matches
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
        if "STAR" in t:
            ned_vote = "star"
        elif t in {"QSO", "AGN", "BLLAC", "BL LAC", "SEYFERT", "SY1", "SY2", "LINER"} or "QSO" in t or "AGN" in t:
            ned_vote = "agn"
        elif t in {"G", "GALAXY", "HII", "EMG"} or "GALAXY" in t or "HII" in t:
            ned_vote = "galaxy"
        elif t in {"IRS", "IRSRC", "IRSOURCE", "IR", "IRAS", "WISE"} or "IR" in t:
            if pd.notna(row.get("ned_z", np.nan)):
                ned_vote = "galaxy"
    
    if ned_vote is not None:
        label_scores[ned_vote] += weights["ned"]

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
    plx_snr = row.get("gaia_parallax_over_error", np.nan)
    pm_snr  = row.get("gaia_pm_over_error", np.nan)
    plx     = row.get("gaia_parallax", np.nan) 

    strong_astrometry = (pd.notna(plx_snr) and plx_snr >= 5.0) or (pd.notna(pm_snr) and pm_snr >= 5.0)
    parallax_sane = (pd.notna(plx) and plx > 0.2)

    if strong_astrometry and (parallax_sane or (pd.notna(pm_snr) and pm_snr >= 10.0)):
        gaia_vote = "star"
        label_scores["star"] += weights["gaia"]

        gaia_sep = row.get("gaia_sep_arcsec", np.nan)
        if pd.notna(gaia_sep) and gaia_sep > 0.7:
            label_scores["star"] -= 1.0

    if gaia_vote == "star":
        if label_scores["agn"] < label_scores["star"] + 1.0:
            return "star"

    

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
    
    def has_match(obj_col, sep_col=None, max_sep_arcsec=None):
        v = row.get(obj_col, None)
        if v is None or pd.isna(v):
            return False
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "" or s in {"nan", "none", "null"}:
                return False
        if sep_col is None or max_sep_arcsec is None:
            return True
        s = row.get(sep_col, np.nan)
        if pd.isna(s):
            return True
        
        try:
            return float(s) <= float(max_sep_arcsec)
        except Exception:
            return True
    
    XRAY_MAX_SEP = 3.0
    RADIO_MAX_SEP = 3.0

    if has_match("chandra_object_id", "chandra_sep_arcsec", XRAY_MAX_SEP):
        label_scores["agn"] += weights["chandra"]

    if has_match("xmm_object_id", "xmm_sep_arcsec", XRAY_MAX_SEP):
        label_scores["agn"] += weights["xmm"]

    if has_match("erosita_s_object_id", "erosita_s_sep_arcsec", XRAY_MAX_SEP):
        label_scores["agn"] += weights["erosita_s"]

    if has_match("erosita_m_object_id", "erosita_m_sep_arcsec", XRAY_MAX_SEP):
        label_scores["agn"] += weights["erosita_m"]

    if has_match("erosita_h_object_id", "erosita_h_sep_arcsec", XRAY_MAX_SEP):
        label_scores["agn"] += weights["erosita_h"]

    if has_match("lotss_object_id", "lotss_sep_arcsec", RADIO_MAX_SEP):
        if wise_vote == "agn":
            label_scores["agn"] += weights["lotss"]
        else:
            label_scores["galaxy"] += 0.6 * weights["lotss"]

    if has_match("vlass_object_id", "vlass_sep_arcsec", RADIO_MAX_SEP):
        if wise_vote == "agn":
            label_scores["agn"] += weights["vlass"]
        else:
            label_scores["galaxy"] += 0.6 * weights["vlass"]

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
ensure_col("gaia_parallax")
ensure_col("gaia_parallax_over_error")
ensure_col("gaia_pm")
ensure_col("gaia_pm_over_error")
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
    print("[Info] Fetching X-ray / Radio crossmatches...")
    #run_xray_radio(save_every_chunks=1)
    meta["target"] = meta.apply(decide_target, axis=1)
    meta.to_csv(OUT_PATH, index=False)
    print(f"[Info] Saved progress + targets to {OUT_PATH}")

    if meta["labeled_done"].all():
        print("[Done] All rows labeled.")
    else:
        print("[Info] Partial labeling saved; resume later to finish.")

