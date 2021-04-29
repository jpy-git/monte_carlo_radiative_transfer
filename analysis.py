"""Analysis of results of MCRT model in main.py"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Importing Data")
df = pd.DataFrame(
    columns=["x", "y", "z", "theta", "phi", "energy", "weight", "distance", "event"]
)

for i in range(4):
    df = df.append(
        pd.read_csv(f"./results/results{i+1}.csv", delimiter=", ", engine="python")
    )

print("Begin Analysis")
bin_width = 0.01

hist_bins = list(np.arange(0.4, 400 + bin_width, bin_width))

histogram_tuple = np.histogram(df["energy"], bins=hist_bins, weights=df["weight"])

histogram_df = pd.DataFrame(
    data={"Histogram Count": histogram_tuple[0], "Energy": histogram_tuple[1][:-1]}
)
histogram_df = histogram_df.loc[histogram_df["Histogram Count"] != 0.0]
histogram_df["Spectral Density"] = (
    histogram_df["Histogram Count"] * histogram_df["Energy"] ** 2
)
histogram_df.set_index("Energy", inplace=True)
histogram_df.sort_index(inplace=True)

histogram_df.plot(y="Spectral Density", loglog=True)

plt.show()
