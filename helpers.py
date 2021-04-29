"""Miscellaneous helper functions for use in main.py"""

import numpy as np
import pandas as pd
import random


def calculate_klein_nishina_cross_section(photon_energy: float) -> float:

    THOMPSON_CROSS_SECTION: float = 6.6525 * 10 ** -29  # (m^2)
    ELECTRON_REST_MASS: float = 511.0  # (keV)

    energy_to_rest_mass_ratio = photon_energy / ELECTRON_REST_MASS
    hydrogen_to_electron_density_ratio = 1.1 / 0.9

    klein_nishina_cross_section = (
        hydrogen_to_electron_density_ratio
        * THOMPSON_CROSS_SECTION
        * (3.0 / (8.0 * energy_to_rest_mass_ratio))
        * (
            (1 - 2 * (1 + energy_to_rest_mass_ratio) / energy_to_rest_mass_ratio ** 2)
            * np.log(1 + 2 * energy_to_rest_mass_ratio)
            + 0.5
            + 4.0 / energy_to_rest_mass_ratio
            - 1 / (2 * (1 + 2 * energy_to_rest_mass_ratio) ** 2)
        )
    )

    return klein_nishina_cross_section


def import_cross_sections_dataframe(
    total_absorbtion_filepath: str = "./input_data/total_abs.dat",
    fek_absorbtion_filepath: str = "./input_data/fek_abs.dat",
) -> pd.core.frame.DataFrame:

    total_absorbtion_cross_section_df: pd.core.frame.DataFrame = pd.read_csv(
        total_absorbtion_filepath,
        header=None,
        names=["Energy", "Total Absorbtion Cross-Section"],
    ).astype(float)

    FeK_absorbtion_cross_section_df: pd.core.frame.DataFrame = pd.read_csv(
        fek_absorbtion_filepath,
        header=None,
        names=["Energy", "FeK Absorbtion Cross-Section"],
    ).astype(float)

    interaction_cross_sections: pd.core.frame.DataFrame = (
        total_absorbtion_cross_section_df.merge(
            FeK_absorbtion_cross_section_df, how="outer", on=["Energy"]
        ).fillna(0.0)
    )

    interaction_cross_sections["Non-FeK Absorbtion Cross-Section"] = (
        interaction_cross_sections["Total Absorbtion Cross-Section"]
        - interaction_cross_sections["FeK Absorbtion Cross-Section"]
    )

    for col in [
        "Total Absorbtion Cross-Section",
        "FeK Absorbtion Cross-Section",
        "Non-FeK Absorbtion Cross-Section",
    ]:
        interaction_cross_sections[col] = (
            interaction_cross_sections[col] * 10 ** -28
        )  # (m^2)

    interaction_cross_sections[
        "Klein-Nishina Scattering Cross-Section"
    ] = interaction_cross_sections["Energy"].apply(
        calculate_klein_nishina_cross_section
    )

    return interaction_cross_sections.set_index("Energy").sort_index()
