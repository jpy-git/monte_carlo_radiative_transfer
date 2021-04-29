"""Miscellaneous helper functions for use in main.py"""

import numpy as np
import pandas as pd
import random


def calculate_klein_nishina_cross_section(photon_energy: float) -> float:
    """Function to calculate Klein-Nishina scattering cross-section at a given photon energy.

    Args:
        photon_energy (float): Photon energy in keV.

    Returns:
        float: Klein-Nishina scattering cross-section in m^2
    """

    # Constants.
    THOMPSON_CROSS_SECTION: float = 6.6525 * 10 ** -29  # (m^2)
    ELECTRON_REST_MASS: float = 511.0  # (keV)

    # Calculate ratio of photon energy to electron rest mass.
    energy_to_rest_mass_ratio = photon_energy / ELECTRON_REST_MASS

    # Calculate ratio of hydrogen to electron density.
    hydrogen_to_electron_density_ratio = 1.1 / 0.9

    # Calculate Klein-Nisina scattering cross-section.
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
    """Function to import Total and FeK absorbtion cross-sections from .dat files
    and then combine them with the Klein-Nishina scattering cross-section.

    Args:
        total_absorbtion_filepath (str, optional): Path to Total absorbtion cross-section data.
        Defaults to "./input_data/total_abs.dat".
        fek_absorbtion_filepath (str, optional): Path to FeK absorbtion cross-section data.
        Defaults to "./input_data/fek_abs.dat".

    Returns:
        pd.core.frame.DataFrame: Pandas DataFrame of interaction cross-sections.
    """

    # Total absorbtion cross-section data.
    total_absorbtion_cross_section_df: pd.core.frame.DataFrame = pd.read_csv(
        total_absorbtion_filepath,
        header=None,
        names=["Energy", "Total Absorbtion Cross-Section"],
    ).astype(float)

    # FeK absorbtion cross-section data.
    FeK_absorbtion_cross_section_df: pd.core.frame.DataFrame = pd.read_csv(
        fek_absorbtion_filepath,
        header=None,
        names=["Energy", "FeK Absorbtion Cross-Section"],
    ).astype(float)

    # Join DataFrames.
    interaction_cross_sections: pd.core.frame.DataFrame = (
        total_absorbtion_cross_section_df.merge(
            FeK_absorbtion_cross_section_df, how="outer", on=["Energy"]
        ).fillna(0.0)
    )

    # Calculate non-FeK absorbtion cross-sections.
    interaction_cross_sections["Non-FeK Absorbtion Cross-Section"] = (
        interaction_cross_sections["Total Absorbtion Cross-Section"]
        - interaction_cross_sections["FeK Absorbtion Cross-Section"]
    )

    # Convert cross-section units to m^2.
    for col in [
        "Total Absorbtion Cross-Section",
        "FeK Absorbtion Cross-Section",
        "Non-FeK Absorbtion Cross-Section",
    ]:
        interaction_cross_sections[col] = (
            interaction_cross_sections[col] * 10 ** -28
        )  # (m^2)

    # Calculate Klein-Nishina scattering cross-sections.
    interaction_cross_sections[
        "Klein-Nishina Scattering Cross-Section"
    ] = interaction_cross_sections["Energy"].apply(
        calculate_klein_nishina_cross_section
    )

    # Set Energy as the index for quick lookups and return.
    return interaction_cross_sections.set_index("Energy").sort_index()
