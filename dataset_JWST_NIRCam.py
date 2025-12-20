import os, io, pathlib, warnings, time, warnings, requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from requests import Session

from astroquery.mast import Observations
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table import vstack
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from reproject import reproject_interp

from photutils.segmentation import detect_threshold, detect_sources, deblend_sources
from photutils.segmentation import SourceCatalog
from skimage.transform import resize

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout

warnings.filterwarnings("ignore", category=FITSFixedWarning)

Observations._session = Session()
HTTP_SESSION = requests.Session()
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"

OUT_DIR = pathlib.Path("JWST_NIRCam_Triple_Filter")
REFENCES_BAND = "F200W" #References filter
BANDS = ["F200W", "F277W", "F444W"] #it should have 16,242 images catalogues. Public access only, with calibiration level 2-4
MAX_OBS_PER_BAND = 4000
MAX_MOSAICS_PER_BAND = 450
MAX_TRIPLETS = 450
MAX_SEGM_LABEL = 8500

CALIB_MIN, CALIB_MAX = 2, 4
MAX_MATCH_ARCSEC = 1.0
CUTOUT_PIX = 64
MIN_AREA = 10
THRESHOLD_SIGMA = 2.0
MAX_SOURCES_PER_FIELD = 500
POOL_GRID = 4
WOKERS_COUNT = 3
BATCH_SIZE = 250
CHUNK_SIZE = 1024 * 1024

OUT_DIR.mkdir(parents=True, exist_ok=True)
FITS_DIR = OUT_DIR / "fits"
FITS_DIR.mkdir(exist_ok=True)

CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

Observations.cache_location = FITS_DIR

def cache_save_df(df, band):
    df.to_parquet(CACHE_DIR / f"{band}_products.parquet", index=False)
    print(f"[Cache] Saved {band} ({len(df)}) to cache/{band}_products.parquet")

def load_band_df(band):
    path = CACHE_DIR / f"{band}_products.parquet"
    if path.exists():
        print(f"[Cache] Loaded cached {band} products")
        return pd.read_parquet(path)
    return None

def arcsec_sep(ra1, dec1, ra2, dec2):
    d2r = np.pi/180.0
    x = (ra2 - ra1) * np.cos(0.5 * (dec1 + dec2) * d2r)
    y = dec2 - dec1
    return 3600.0 * np.sqrt(x*x + y*y)

def robust_scale(im):
    print(f"[Scaling] for downloaded .fits")
    im = np.nan_to_num(im, copy=False)
    if not np.isfinite(im).any():
        return np.zeros_like(im, dtype=np.float32)
    im = im.astype(np.float32, copy=False)
    p1, p99 = np.percentile(im, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        return np.zeros_like(im, dtype=np.float32)
    im = im - np.median(im)
    scale = (p99 - p1)
    im = (im - p1) / (scale + 1e-12)
    return np.clip(im, -5, 5)

def z_score(im):
    im = np.nan_to_num(im, copy=False)
    s = np.std(im)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(im, dtype=np.float32)
    return (im - np.mean(im)) / s

def is_mosaic_filename(filename):
    return (("_i2d.fits" in filename) or ("_drc.fits" in filename))

def filename_has_filter(filename, filter):
    return filter.lower() in filename.lower()

def query_band_products(filter_name, step=1000):
    print(f"[Query] JWST NIRCAM/IMAGE, filter = {filter_name}")
    Observations.TIMEOUT = 3600
    obs = Observations.query_criteria(
        obs_collection="JWST",
        instrument_name="NIRCAM/IMAGE",
        filters=filter_name,
        dataproduct_type="image",
        dataRights="PUBLIC"
    )
    if len(obs) == 0:
        return pd.DataFrame()
    
    if len(obs) > MAX_OBS_PER_BAND:
        print(f"[Query] {filter_name}: {len(obs)} observations total, "
              f"capping to first {MAX_OBS_PER_BAND} to avoid timeouts.")
        obs = obs[:MAX_OBS_PER_BAND]

    obs_df = obs.to_pandas()
    print(f"[Query] {filter_name}: {len(obs_df)} observations;\nFetching products in batches !")
    print(f"[Query] Fetching products in batches of {step} ...")

    products_table = None

    total = len(obs)
    n_batches = (total + step - 1) // step

    for b in tqdm(range(n_batches), desc=f"Querying {filter_name}", unit="batch"):
        i = b * step
        batch = obs[i:i+step]
        try:
            products = Observations.get_product_list(batch)
            
            if products_table is None:
                products_table = products
            else:
                products_table = vstack([products_table, products])
        except Exception as e:
            print(f"Batch {b+1}/{n_batches} failed: {e}")
            continue

    if products_table is None or len(products_table) == 0:
        print(f"[Warn] No products for {filter_name}")
        return pd.DataFrame()

    df = products_table.to_pandas()
    df =  df[df["productFilename"].str.endswith(".fits", na=False)].copy()
    
    if "calib_level" in df.columns:
        df = df[df["calib_level"].isna() | df["calib_level"].between(CALIB_MIN, CALIB_MAX, inclusive="both")]

    if "productType" in df.columns:
        df = df[df["productType"].str.upper() == "SCIENCE"]
    
    df = df[df["productFilename"].apply(lambda func: filename_has_filter(func, filter_name))]

    if "productSubGroupDescription" in df.columns:
        df = df[df["productSubGroupDescription"].str.lower().str.contains("i2d", na=False)]

    if "obsid" in obs_df.columns:
        obs_pos = obs_df[["obsid", "s_ra", "s_dec"]].copy()
        if "obsID" in df.columns:
            df = df.merge(obs_pos, left_on="obsID", right_on="obsid", how="left")
            df.drop(columns=["obsid"], inplace=True)
        else:
            print(f"[Warn] obsID column not found in the products list for {filter_name}, RA/Dec may be missing")
    else:
        print(f"[Warn] obsid/s_ra/s_dec not found in obs table for {filter_name}")

    df.drop_duplicates(subset=["productFilename"], inplace=True)
    df = df.sort_values(["s_ra", "s_dec"]).reset_index(drop=True)
    df = df.iloc[:MAX_MOSAICS_PER_BAND]

    print(f"[Query] {filter_name}: {len(df)} mosaic products kept")
    return df

def split_by_band(df):
    out = {}
    for b in BANDS:
        mask = df["productFilename"].apply(lambda f: filename_has_filter(f,b))
        sub = df[mask].copy().reset_index(drop=True)
        out[b] = sub
        print(f"[band={b}] mosaics found: {len(sub)}")
    return out

def match_triplets(df_F200, df_F277, df_F444, tol_arcsec=1.0):
    print(f"[Info] Matching triplets")
    triplets = []
    used_277 = set()
    used_444 = set()

    for i, a in df_F200.iterrows():
        ra, dec = a.get("s_ra", np.nan), a.get("s_dec", np.nan)
        if not np.isfinite(ra) or not np.isfinite(dec):
            continue

        j_best, d_best = None, 1e9
        for j, b in df_F277.iterrows():
            if j in used_277:
                continue
            ra2, dec2 = b.get("s_ra", np.nan), b.get("s_dec", np.nan)
            if not np.isfinite(ra2) or not np.isfinite(dec2):
                continue
            d= arcsec_sep(ra, dec, ra2, dec2)
            if d < d_best:
                d_best, j_best = d, j
        if j_best is None or d_best > tol_arcsec:
            continue

        k_best, d2_best = None, 1e9
        for k, c in df_F444.iterrows():
            if k in used_444:
                continue
            ra3, dec3 = c.get("s_ra", np.nan), c.get("s_dec", np.nan)
            if not np.isfinite(ra3) or not np.isfinite(dec3):
                continue
            d2 = arcsec_sep(ra, dec, ra3, dec3)
            if d2 < d2_best:
                d2_best, k_best = d2, k
        if k_best is None or d2_best > tol_arcsec:
            continue

        triplets.append((i, j_best, k_best))
        used_277.add(j_best)
        used_444.add(k_best)
    
    return triplets

def download_if_needed(row):
    uri = row["dataURI"]
    fname = uri.split("/")[-1]
    dest = FITS_DIR / fname

    if dest.exists():
        head = HTTP_SESSION.head(url, timeout=60)
        head.raise_for_status()
        total = head.headers.get("Content-Length")
        if total is not None:
            total = int(total)
            local_size = dest.stat().st_size
            if local_size == total:
                return dest
            else:
                print(f"[Warn] {fname} size mismatch ({local_size} vs {total}), redownloading")
                dest.unlink()

    url = f"{MAST_DOWNLOAD_URL}?uri={uri}"

    try:
        resp = HTTP_SESSION.get(url, stream=True, timeout=600)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Err] Request failed for {fname}: {e}")
        raise

    total = resp.headers.get("Content-Length")
    total = int(total) if total is not None else None

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            f.write(chunk)

    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"[Err] Download produced empty file: {dest}")

    if total is not None and dest.stat().st_size != total:
        raise RuntimeError(
            f"[Err] Final size mismatch for {fname}:"
            f"{dest.stat().st_size} vs {total}"
        )

    return dest

def prefetch_all_fits(df_F200, df_F277, df_F444, max_workers=4):
    all_df = pd.concat([df_F200, df_F277, df_F444], ignore_index=True)
    uris = all_df["dataURI"].dropna().unique()

    uri_to_row = {row["dataURI"]: row for _, row in all_df.iterrows()}

    def worker(uri):
        row = uri_to_row[uri]
        try:
            path = download_if_needed(row)
            return (uri, True, str(path))
        except Exception as e:
            return (uri, False, str(e))

    print(f"[Prefetch] Downloading {len(uris)} FITS files with {max_workers} workers...")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, uri): uri for uri in uris}

        with tqdm(
            total=len(futures),
            desc="Prefetch FITS",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for fut in as_completed(futures):
                uri = futures[fut]
                ok_uri, ok_flag, msg = fut.result()

                status = "OK" if ok_flag else "ERR"
                pbar.update(1)
                pbar.set_postfix_str(f"{status} {os.path.basename(ok_uri)[:25]}")
                results.append((ok_uri, ok_flag, msg))

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_err = len(results) - n_ok
    print(f"[Prefetch] Done: {n_ok} downloaded, {n_err} errors")

def read_sci(fits_path):
    if isinstance(fits_path, (tuple, list)):
        fits_path = fits_path[0]
    fits_path = pathlib.Path(fits_path)

    with fits.open(fits_path, memmap=True) as hdul:
        if "SCI" in hdul:
            data = hdul["SCI"].data
            hdr = hdul["SCI"].header
            if data is not None and data.ndim == 2:
                return data.astype(np.float32), hdr.copy()

        for h in hdul[1:]:
            if h.data is not None and h.data.ndim == 2:
                return h.data.astype(np.float32), h.header.copy()

    raise ValueError(f"No 2D SCI image found in {fits_path}")

def reproject_to(ref_img, ref_hdr, mov_img, mov_hdr):
    try:
        w_ref = WCS(ref_hdr)
        w_mov = WCS(mov_hdr)
        out_shape = ref_img.shape
        mov_on_ref, _ = reproject_interp((mov_img, w_mov), w_ref, shape_out=out_shape)
        if not np.isfinite(mov_on_ref).any():
            raise RuntimeError("Reproject produces all-NaN images.")
        mov_on_ref = np.nan_to_num(mov_on_ref, copy=False)
        return mov_on_ref
    except Exception as e:
        warnings.warn(f"reproject_interp failed ({e}), falling back to resize().")
        mov_resized = resize(
            mov_img,
            out_shape,
            preserve_range=True,
            anti_aliasing=True
        )
        mov_resized = np.nan_to_num(mov_resized, copy=False)
        return mov_resized

def detect_sources_on(img):
    mean, med, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
    thres = detect_threshold(img, nsigma=THRESHOLD_SIGMA)
    segm = detect_sources(img, thres, npixels=MIN_AREA)
    if segm is None:
        return None, None
    
    if segm.nlabels > MAX_SEGM_LABEL:
        print(f"[Detect] Too many labels ({segm.nlabels}), skipping field")
        return None, None

    segm = deblend_sources(img, segm, npixels=MIN_AREA, nlevels=16, contrast=0.001)
    cat = SourceCatalog(img, segm)

    tbl = cat.to_table()
    tbl.sort("segment_flux", reverse=True)
    if len(tbl) > MAX_SOURCES_PER_FIELD:
        tbl = tbl[:MAX_SOURCES_PER_FIELD]
    return segm, tbl

def cutout_stack(ref_img, hdr_ref, imgs_dict, xy, size_pix=64):
    pos = (xy[0], xy[1])
    wref = WCS(hdr_ref)
    stacks = []
    for band in BANDS:
        im = imgs_dict[band]
        co = Cutout2D(im, position=pos, size=(size_pix, size_pix), wcs=wref, mode="partial", fill_value=0.0)
        data = np.nan_to_num(co.data, copy=False)
        x = z_score(resize(data, (size_pix, size_pix), anti_aliasing=True, preserve_range=True))
        stacks.append(x)
    return np.stack(stacks, axis=-1).astype(np.float32)

def avg_pool_grid(img, grid=4):
    H, W = img.shape
    gh = H // grid
    gw = W // grid
    img = img[:gh*grid, :gw*grid]
    pooled = img.reshape(grid, gh, grid, gw).mean(axis=(1,3))
    return pooled

def pooled_features_from_stamp(stamp3, grid=4):
    features = []
    stats = []
    for ch in range(stamp3.shape[-1]):
        pooled = avg_pool_grid(stamp3[:, :, ch], grid=grid)
        features.append(pooled.flatten())
        stats.extend([np.mean(stamp3[:, :, ch]), np.std(stamp3[:, :, ch])])
    
    vectors = np.concatenate(features + [np.array(stats, dtype=np.float32)], axis=0).astype(np.float32)
    return vectors

def main():
    print("[Info] Trying load existing caches")
    df_F200 = load_band_df("F200W")
    df_F277 = load_band_df("F277W")
    df_F444 = load_band_df("F444W")

    if df_F200 is None:
        print("[Info] No existing filter cache can be found\n[Info] Querying band from MAST")
        df_F200 = query_band_products("F200W", step=BATCH_SIZE)
        cache_save_df(df_F200, "F200W")
    if df_F277 is None:
        print("[Info] No existing filter cache can be found\n[Info] Querying band from MAST")
        df_F277 = query_band_products("F277W", step=BATCH_SIZE)
        cache_save_df(df_F277, "F277W")
    if df_F444 is None:
        print("[Info] No existing filter cache can be found\n[Info] Querying band from MAST")
        df_F444 = query_band_products("F444W", step=BATCH_SIZE)
        cache_save_df(df_F444, "F444W")
    
    print(f"Filter F200W counts: {len(df_F200)}\nFilter F277W counts: {len(df_F277)}\nFilter F444W counts: {len(df_F444)}")

    if df_F200.empty or df_F277.empty or df_F444.empty:
        raise SystemExit("One of the bands still empty. Check filename and verify again !")
    
    print(f"Products: F200W = {len(df_F200)}, F277W = {len(df_F277)}, and F444W = {len(df_F444)}")
    triplets = match_triplets(df_F200, df_F277, df_F444, tol_arcsec=MAX_MATCH_ARCSEC)
    print(f"Matched triplets (fields): {len(triplets)}")
    if len(triplets) > MAX_TRIPLETS:
        idx = np.random.choice(len(triplets), size=MAX_TRIPLETS, replace=False)
        triplets = [triplets[i] for i in idx]
        print(f"[Info] Subsampled triplets to {len(triplets)}")

    idx_200 = [i200 for (i200, _, _) in triplets]
    idx_277 = [i277 for (_, i277, _) in triplets]
    idx_444 = [i444 for (_, _, i444) in triplets]

    df_F200_sub = df_F200.loc[idx_200].reset_index(drop=True)
    df_F277_sub = df_F277.loc[idx_277].reset_index(drop=True)
    df_F444_sub = df_F444.loc[idx_444].reset_index(drop=True)

    prefetch_all_fits(df_F200_sub, df_F277_sub, df_F444_sub, max_workers=WOKERS_COUNT)

    X_imgs = []
    X_pooled = []
    meta_rows = []
    for i200, i277, i444 in tqdm(triplets, desc="Fields (triplets)"):
        r200 = df_F200.loc[i200]
        r277 = df_F277.loc[i277]
        r444 = df_F444.loc[i444]

        try:
            f200 = FITS_DIR / r200["productFilename"]
            f277 = FITS_DIR / r277["productFilename"]
            f444 = FITS_DIR / r444["productFilename"]
            if not (f200.exists() and f277.exists() and f444.exists()):
                print(f"[Info] Ignoring missing triplets")
                continue

            img200, hdr200 = read_sci(f200)
            img277, hdr277 = read_sci(f277)
            img444, hdr444 = read_sci(f444)
            
            _, tbl = detect_sources_on(img200)
            if tbl is None or len(tbl) == 0:
                continue

            print(f"[Detect] obsID {r200['obsID']}: {len(tbl)} sources found")

            img200 = robust_scale(img200)
            img277 = robust_scale(img277)
            img444 = robust_scale(img444)

            on200_277 = reproject_to(img200, hdr200, img277, hdr277)
            on200_444 = reproject_to(img200, hdr200, img444, hdr444)            

            for row in tbl:
                try:
                    xc = float(row["xcentroid"])
                    yc = float(row["ycentroid"])
                    stamp = cutout_stack(
                        ref_img=img200,
                        hdr_ref=hdr200,
                        imgs_dict={"F200W":img200, "F277W":on200_277, "F444W":on200_444},
                        xy=(xc, yc),
                        size_pix=CUTOUT_PIX
                    )
                    sky = WCS(hdr200).pixel_to_world(xc, yc)
                    ra_src = float(sky.ra.deg)
                    dec_src = float(sky.dec.deg)
                    X_imgs.append(stamp)
                    X_pooled.append(pooled_features_from_stamp(stamp, grid=POOL_GRID))
                    meta_rows.append({
                        "ra":float(r200["s_ra"]),
                        "dec":float(r200["s_dec"]),
                        "ra_src": ra_src,
                        "dec_src": dec_src,
                        "obsID_200": r200["obsID"],
                        "obsID_277": r277["obsID"],
                        "obsID_444": r444["obsID"],
                        "uri_200": r200["dataURI"],
                        "uri_277": r277["dataURI"],
                        "uri_444": r444["dataURI"],
                        "filename_200": r200["productFilename"],
                        "filename_277": r277["productFilename"],
                        "filename_444": r444["productFilename"]
                    })
                except Exception as ex:
                    warnings.warn(f"Cutout failed for one source in obsID {r200['obsID']}: {ex})")

        except Exception as e:
            warnings.warn(f"Triplets failed: {e}")
            continue
    
    if len(X_imgs) == 0:
        raise SystemExit("No cutouts produced. Try relaxing MIN_AREA/THRESH_SIG, or this field has few sources")
    
    X_imgs = np.stack(X_imgs, axis=0)
    X_pooled = np.stack(X_pooled, axis=0)
    meta = pd.DataFrame(meta_rows)

    out_npz = OUT_DIR/f"nircam_{'-'.join(BANDS)}_{CUTOUT_PIX}px_triplets.npz"
    np.savez_compressed(out_npz, X_imgs=X_imgs.astype(np.float32), X_pooled=X_pooled.astype(np.float32))
    meta.to_csv(OUT_DIR/f"nircam_{'-'.join(BANDS)}_meta.csv", index=False)

    print(f"Saved: {out_npz} with X_imgs shape {X_imgs.shape}, X_polled shape {X_pooled.shape}")
    print(f"Saved meta CSV with {len(meta)} rows")

if __name__ == "__main__":
    main()