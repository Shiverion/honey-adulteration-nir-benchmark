"""Retrieve, verify and harmonize the public honey spectral data."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


GRID_NM = np.linspace(780.0, 1000.0, 48)
SOURCES = {
    "auckland": (
        "adulteration_dataset_26_08_2021.csv",
        "https://ndownloader.figshare.com/files/30455640",
        "27c1ccf0c0e6aeeb144d4a58ee2da85a826bcd112a025cab6985a9774551893f",
    ),
    "nirpy": (
        "Honey2021.csv",
        "https://raw.githubusercontent.com/nevernervous78/nirpyresearch/0efbcf0fcd97580dbffd79e857d70d8f001cf5ff/data/Honey2021.csv",
        "7127fdec5994515481c205ee45231fb947b9ea798ae4ba8e152f5fa618534a92",
    ),
    "padua": (
        "Padua_Dataset_Honey.xlsx",
        "https://researchdata.cab.unipd.it/1770/1/Dataset_Honey.xlsx",
        "ed15ca1e667d1c10f8d315b0d3abae6906f0136da79a1361c3c10361799379f3",
    ),
}


@dataclass
class Spectra:
    x: np.ndarray
    meta: pd.DataFrame
    audit: dict

    @property
    def y(self):
        return self.meta["label"].to_numpy(dtype=int)


def get_file(directory, key, download):
    filename, url, expected = SOURCES[key]
    path = Path(directory) / filename
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {digest}")
    elif download:
        with urlopen(url, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    else:
        raise FileNotFoundError(f"{path} is missing. Run with --download or fetch {url}.")
    return path


def resample(values, wavelengths, grid=GRID_NM):
    if not np.isfinite(values).all() or not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Expected finite measurements and a strictly ascending spectral axis.")
    if grid[0] < wavelengths[0] or grid[-1] > wavelengths[-1]:
        raise ValueError("The wavelength grid requests extrapolation.")
    return np.stack([np.interp(grid, wavelengths, row) for row in values])


def load_auckland(path, grid=GRID_NM):
    raw = pd.read_csv(path)
    bands = [name for name in raw if re.fullmatch(r"\d+\.\d+nm", name)]
    wavelengths = np.array([float(name[:-2]) for name in bands])
    classes = {0, 5, 10, 25, 50}
    honey = raw.loc[raw["Brand"].ne("Sugar")].copy()
    if set(honey["Concentration_Class"]) != classes or honey[bands].duplicated().any():
        raise ValueError("Unexpected Auckland class labels or exact duplicate spectra.")

    # Some keys cover two images whose source IDs coincide. Keep the joint mean.
    keys = ["Brand", "Class", "Acquisition", "Concentration_Class", "Concentration"]
    grouped = honey.groupby(keys, sort=True, dropna=False)
    sizes = grouped.size()
    if not sizes.isin([25, 50]).all() or sizes.sum() != len(honey):
        raise ValueError("Unexpected Auckland image segment counts.")
    means = grouped[bands].mean().reset_index()
    meta = pd.DataFrame({
        "sample_id": means[keys].astype(str).agg("|".join, axis=1),
        "group": means["Brand"],
        "product": means["Class"],
        "acquisition": means["Acquisition"],
        "nominal_pct": means["Concentration_Class"],
        "concentration_pct": means["Concentration"],
        "concentration_relation": "=",
        "adulterant": np.where(means["Concentration_Class"].eq(0), "none", "sugar syrup"),
        "label": means["Concentration_Class"].ne(0).astype(int),
        "source_rows": sizes.to_numpy(),
    })
    audit = {
        "raw_rows": len(raw), "pure_sugar_excluded": len(raw) - len(honey),
        "native_bands_in_range": int(np.sum((wavelengths >= GRID_NM[0]) & (wavelengths <= GRID_NM[-1]))),
        "merged_image_keys": int(sizes.eq(50).sum()), "products": honey.groupby(["Brand", "Class"]).ngroups,
    }
    return Spectra(resample(means[bands].to_numpy(), wavelengths, grid), meta, audit)


def load_nirpy(path):
    raw = pd.read_csv(path, sep=";")
    bands = raw.columns[1:]
    # The header's unit is reversed. Reciprocal conversion gives an unrounded nm axis.
    wavelengths = 1e7 / bands.astype(float).to_numpy()
    reference = raw.loc[0, bands].to_numpy(dtype=float)
    if not np.allclose(wavelengths, reference, rtol=0.006):
        raise ValueError("NIRPY's wavelength and wavenumber axes disagree.")
    spectra = raw.iloc[1:].rename(columns={raw.columns[0]: "source_id"}).copy()
    ids = spectra["source_id"].str.split().str.join(" ")
    preparations = ids.str.replace(r"\s*%", "%", regex=True).str.replace(r"\s[a-z]$", "", regex=True)
    pure = preparations.str.startswith("100% HONEY")
    syrup = preparations.str.match(r"100% (GLU|RS|JMAF) \(")
    parsed = preparations.str.extract(r"(?P<relation>>?)(?P<dose>\d+(?:\.\d+)?)% (?P<syrup>GLU|RS|JMAF) \(")
    eligible = pure | (parsed["dose"].notna() & ~syrup)
    eligible_spectra = spectra.loc[eligible, bands].copy()
    grouped = eligible_spectra.groupby(preparations.loc[eligible], sort=True)
    means, sizes = grouped.mean(), grouped.size()
    meta = pd.DataFrame({"sample_id": means.index, "source_rows": sizes.to_numpy()})
    meta["group"] = "unverified"
    prepared_pure = meta["sample_id"].str.startswith("100% HONEY")
    dose = meta["sample_id"].str.extract(r"(>?)(\d+(?:\.\d+)?)% (GLU|RS|JMAF) \(")
    meta["label"] = (~prepared_pure).astype(int)
    meta["concentration_pct"] = pd.to_numeric(dose[1]).fillna(0)
    meta["concentration_relation"] = dose[0].replace("", "=").fillna("=")
    meta["adulterant"] = dose[2].fillna("none")
    audit = {
        "raw_rows": len(spectra), "pure_syrup_excluded": int(syrup.sum()),
        "ambiguous_rows_excluded": int((~eligible & ~syrup).sum()),
        "native_bands_in_range": int(np.sum((wavelengths >= GRID_NM[0]) & (wavelengths <= GRID_NM[-1]))),
    }
    return Spectra(resample(means.to_numpy(), wavelengths), meta, audit)
