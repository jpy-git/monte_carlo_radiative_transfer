"""Analysis of results of MCRT model in main.py"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import Constants, Gas, Photon


def import_MCRT_results(use_pickle: bool = False) -> pd.core.frame.DataFrame:
    """Function to read either .csv files or .pkl file from ./results folder.

    Args:
        use_pickle (bool, optional): Option to use a .pkl file if it exists. Defaults to False.

    Returns:
        pd.core.frame.DataFrame: Pandas DataFrame of MCRT results.
    """

    # Check if pickle option has been selected.
    if use_pickle:

        # If pickle file does not exist then raise FileNotFoundError.
        if not os.path.exists("./results/results.pkl"):
            raise FileNotFoundError(
                "./results/results.pkl not found please run with use_pickle=False first to generate .pkl file."
            )

        # Read pickle file.
        return pd.read_pickle("./results/results.pkl")

    # If pickle option is not selected then create empty master table.
    df = pd.DataFrame(
        columns=["x", "y", "z", "theta", "phi", "energy", "weight", "distance", "event"]
    )

    # Iterate through .csv files and append to master table.
    for result_file in os.listdir("./results"):
        if result_file.rpartition(".")[2] == "csv":
            df = df.append(
                pd.read_csv(f"./results/{result_file}", delimiter=", ", engine="python")
            )

    # If no data is loaded then raise FileNotFoundError and inform user to run main.py.
    if not df:
        raise FileNotFoundError(
            "No .csv files found. Please run main.py first to generate results of MCRT simulation."
        )

    # Save DataFrame to a .pkl file.
    df.to_pickle("./results/results.pkl")

    return df


if __name__ == "__main__":

    # Parameters:
    bin_width = 0.01
    gas = Gas()

    print("Importing Data")
    df = import_MCRT_results(use_pickle=True)

    print("Begin Analysis")
    print("Energy Flux Histogram")
    # Create energy bins.
    hist_bins = list(
        np.arange(Photon.MIN_ENERGY, Photon.MAX_ENERGY + bin_width, bin_width)
    )

    # Calculate weighted histogram values.
    histogram_tuple = np.histogram(df["energy"], bins=hist_bins, weights=df["weight"])

    # Convert histogram tuple to Pandas DataFrame.
    histogram_df = pd.DataFrame(
        data={"Histogram Count": histogram_tuple[0], "Energy": histogram_tuple[1][:-1]}
    )
    # Remove empty bins.
    histogram_df = histogram_df.loc[histogram_df["Histogram Count"] != 0.0]

    # Convert histogram to energy flux.
    histogram_df["Energy Flux"] = (
        histogram_df["Histogram Count"] * histogram_df["Energy"] ** 2
    )
    histogram_df.set_index("Energy", inplace=True)
    histogram_df.sort_index(inplace=True)

    # Plot energy flux.
    histogram_df.plot(y="Energy Flux", loglog=True, style="r")

    # Add initial energy spectrum reference line.
    initial_energy_flux = (
        histogram_df.at[Photon.MIN_ENERGY, "Energy Flux"]
        * histogram_df.index ** (-Photon.GAMMA + 2)
        / Photon.MIN_ENERGY ** (-Photon.GAMMA + 2)
    )

    plt.loglog(
        histogram_df.index,
        initial_energy_flux,
        "k--",
        label="Initial Energy Flux",
        zorder=0,
    )

    # Add K-Alpha and K-Beta reference lines.
    plt.vlines(
        Photon.FEK_ALPHA_ENERGY,
        min(histogram_df["Energy Flux"]),
        max(histogram_df["Energy Flux"]) * 1.5,
        colors="b",
        linestyles="dashed",
        label="K-Alpha Line",
    )
    plt.vlines(
        Photon.FEK_BETA_ENERGY,
        min(histogram_df["Energy Flux"]),
        max(histogram_df["Energy Flux"]) * 1.5,
        colors="g",
        linestyles="dashed",
        label="K-Beta Line",
    )

    # Add legend.
    plt.legend(loc="best")

    # Save figure.
    plt.savefig(
        "./graphs/energy_flux_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("Time Lag Analysis")

    df["direct_distance"] = (
        (df["x"] - (gas.box_length // 2 - 1)) * np.sin(df["phi"]) * np.cos(df["theta"])
        + (df["y"] - (gas.box_length // 2 - 1))
        * np.sin(df["phi"])
        * np.sin(df["theta"])
        + df["z"] * np.cos(df["phi"])
    ).abs()

    df["time_lag"] = (
        (df["distance"] - df["direct_distance"])
        * (2 * gas._max_radius / gas.box_length)
        / Constants.SPEED_OF_LIGHT
    )

    delta_angle = 0.04
    observation_angle = math.radians(5)

    time_lag_df = df.loc[(df['phi']>=observation_angle-delta_angle)&(df['phi']<observation_angle+delta_angle)]
    fig = plt.figure()
    time_lag_df['time_lag'].plot.hist(bins=5000)
    plt.show()