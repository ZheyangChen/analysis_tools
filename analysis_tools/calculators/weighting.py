import numpy as np
import pandas as pd

def apply_flux_weight(
    df: pd.DataFrame,
    model: str = "single_power_law",
    norm: float = 1.0,
    index: float = 2.5,
    gamma1: float = 2.0,
    gamma2: float = 3.0,
    ebreak: float = 1e6,  # in GeV
    livetimeseconds: float = 1.0,
    energy_col: str = "I3MCWeightDict_PrimaryNeutrinoEnergy",
    oneweight_col: str = "I3MCWeightDict_OneWeight",
    nevents_col: str = "I3MCWeightDict_NEvents",
    nfiles_col: str = "nfiles",
    weight_colname: str = "astro_weight"
) -> pd.DataFrame:
    """
    Apply astrophysical flux reweighting using single or NuFit-style broken power law.

    The normalization (norm) is defined as the flux at 100 TeV.
    For the broken power law, continuity is enforced at E_break.
    """

    E = df[energy_col]
    E0 = 1e5  # 100 TeV pivot (GeV)

    # --- Select flux model ---
    if model == "single_power_law":
        # Simple SPL: Φ(E) = norm * 1e-18 * (E / E0)^(-index)
        flux_phi = (E / E0) ** (-index)

    elif model == "broken_power_law":
        # Implement NuFit convention:
        #   Φ0 defined at 100 TeV, continuity at E_break enforced
        if E0 < ebreak:
            norm_eff = (E0 / ebreak) ** gamma1
        else:
            norm_eff = (E0 / ebreak) ** gamma2

        flux_phi = np.where(
            E < ebreak,
            norm_eff * (E / ebreak) ** (-gamma1),
            norm_eff * (E / ebreak) ** (-gamma2),
        )

    else:
        raise ValueError(f"Unknown model: {model}")

    # --- Compute event weights ---
    weight = (
        norm * 1e-18 *
        df[oneweight_col] /
        (df[nevents_col] * df[nfiles_col]) *
        flux_phi *
        livetimeseconds
    )

    df[weight_colname] = weight
    return df