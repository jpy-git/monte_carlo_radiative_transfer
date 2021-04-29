"""Modelling the Variable X-ray Spectrum of an Accreting Black Hole Binary System.

A Monte-Carlo Radiative Transfer (MCRT) simulation of the spectrum of X-Ray photons 
emitted from a Narrow-Line Seyfert 1 (NLS1) Active Galactic Nuclei (AGN).

Photons are subjected to various scattering and absorbtion probabilities 
as they traverse through a circumnuclear gas distribution.
"""

import concurrent.futures
from itertools import accumulate, product
import math
import random
from time import time
from typing import Union

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from helpers import import_cross_sections_dataframe


class Constants:
    """Class containing Physics constants to be used in the simulation.

    Class attributes:
        1) SPEED_OF_LIGHT
        2) GRAVITATIONAL_CONSTANT
        3) SOLAR_MASS
        4) THOMPSON_CROSS_SECTION
        5) ELECTRON_REST_MASS
        6) INTERACTION_CROSS_SECTIONS
    """

    # Constants.
    SPEED_OF_LIGHT: float = 2.9979 * 10 ** 8  # (m s^-1)
    GRAVITATIONAL_CONSTANT: float = 6.6741 * 10 ** -11  # (N m^2 kg^-2)
    SOLAR_MASS: float = 1.988 * 10 ** 30  # (kg)
    THOMPSON_CROSS_SECTION: float = 6.6525 * 10 ** -29  # (m^2)
    ELECTRON_REST_MASS: float = 511.0  # (keV)

    # Absorbtion cross-section DataFrame.
    INTERACTION_CROSS_SECTIONS: pd.core.frame.DataFrame = (
        import_cross_sections_dataframe()
    )

    @classmethod
    def plot_absorbtion_cross_section(self) -> None:
        """Class method to produce graph of various
        interaction cross-sections against photon energy.

        Interaction cross-sections:
            1) Total Absorbtion Cross-Section
            2) FeK Absorbtion Cross-Section
            3) Non-FeK Absorbtion Cross-Section
            4) Klein-Nishina Scattering Cross-Section
        """

        # Filter to range of observable photon energies and
        # plot interaction cross-sections against photon energies on a loglog scale.
        self.INTERACTION_CROSS_SECTIONS.loc[
            (self.INTERACTION_CROSS_SECTIONS.index >= Photon.MIN_ENERGY)
            & (self.INTERACTION_CROSS_SECTIONS.index <= Photon.MAX_ENERGY)
        ].plot(
            y=[
                "Total Absorbtion Cross-Section",
                "FeK Absorbtion Cross-Section",
                "Non-FeK Absorbtion Cross-Section",
                "Klein-Nishina Scattering Cross-Section",
            ],
            loglog=True,
        )

        # Add Legend.
        plt.legend(loc="best")

        # Save figure.
        plt.savefig(
            "./graphs/interaction_cross_sections.png",
            dpi=300,
            bbox_inches="tight",
        )


class Photon:
    """Class representing Photon in the MCRT simulation.

    Class attributes:
        1) MIN_ENERGY
        2) MAX_ENERGY
        3) GAMMA
        4) FEK_BETA_TO_ALPHA_RATIO
        5) FLUORESCENCE_YIELD
        6) FEK_ALPHA_ENERGY
        7) FEK_BETA_ENERGY
        8) MAX_MOVEMENT_PER_ITERATION
        9) STEP_SIZE
        10) MAX_ITERATIONS
    """

    # Range of possible photon energies.
    MIN_ENERGY: float = 0.4  # (keV) Taken from Miller & Turner (2011), approximately the minimum observable energy with telescopes.
    MAX_ENERGY: float = 400.0  # (keV) Highest energy that can contribute observable energy bands via Compton down-scattering.

    # Exponent in photon flux power law.
    GAMMA: float = 2.38  # Taken from Miller et al. (2007)
    FEK_BETA_TO_ALPHA_RATIO: float = 0.120  # Taken from Han et al. (2007)
    FLUORESCENCE_YIELD: float = 0.358  # Taken from Han et al. (2007)
    FEK_ALPHA_ENERGY: float = 6.407  # Yaqoob et al. (2007)
    FEK_BETA_ENERGY: float = 7.034  # Yaqoob et al. (2007)

    # Photon movement parameters.
    MAX_MOVEMENT_PER_ITERATION: int = 50  # The maximum length over which scattering probabilities are evaluated within an iteration.
    STEP_SIZE: int = 1  # Steps over which scattering probabilities are evaluated.
    MAX_ITERATIONS: int = 15  # Number of iterations to perform on a photon in the box before it is discarded.

    _tiny_float: float = np.finfo(
        float
    ).eps  # Smallest float to avoid using log(0) in probability calcs.

    def __init__(self, box_length: int) -> None:
        """Initialisation method for instance of class Photon.

        Args:
            box_length (int): Non-dimensional length of box containing circumnuclear gas.
            Should be set to Gas.box_length.
        """

        # Initial photon position in center of the box.
        self._x: int = box_length // 2 - 1
        self._y: int = box_length // 2 - 1
        self._z: int = 0

        # Random initial direction of travel in spherical coords.
        self._theta: float = 2 * np.pi * random.uniform(0.0, 1.0)  # Polar angle
        self._phi: float = math.acos(random.uniform(0.0, 1.0))  # Azimuthal angle

        # Random initial photon energy and weighting factor
        # using artificially flattened power-law distribution.
        self._energy: float = self.MIN_ENERGY + (
            self.MAX_ENERGY - self.MIN_ENERGY
        ) * random.uniform(0.0, 1.0)

        self._weight: float = (
            (self.GAMMA - 1.0)
            * (self.MAX_ENERGY - self.MIN_ENERGY)
            * (self._energy ** (-self.GAMMA))
            / (
                self.MIN_ENERGY ** (1.0 - self.GAMMA)
                - self.MAX_ENERGY ** (1.0 - self.GAMMA)
            )
        )

        # Initial distance travelled by photon is zero.
        self._distance: float = 0.0

        # Initially set event flag as None.
        self._event: str = "None"

        # Store box length in __dict__ for easy access.
        self._box_length: int = box_length

    def _display_photon_attributes(self) -> None:
        """Method to print attributes of photon instance to the console.

        Not used in main() but useful for debugging/development.
        """

        # Iterate through __dict__ and print to console.
        print("Photon Params: ")
        for param, value in vars(self).items():
            param = param.lstrip("_")
            print(f"{param}: {value}")

    def output_photon_attributes(self) -> str:
        """Method to output attributes of photon instance as a single line of a .csv file.

        This will be used to store outputs of MCRT simulation in a single .csv file per process.

        Returns:
            str: String in the format of a single line in a .csv file.
        """

        return (
            ", ".join(
                [
                    str(value)
                    for param, value in vars(self).items()
                    if param != "_box_length"
                ]
            )
            + "\n"
        )

    def _bin_photon_energy(self) -> float:
        """Method to bin photon energies into buckets consistent
        with the available interaction cross-section data.

        Returns:
            float: Binned photon energy.
        """

        if self._energy < 10:
            return round(self._energy, 3)
        elif self._energy < 20:
            return round(self._energy, 1)
        else:
            return round(self._energy, 0)

    def _calculate_possible_coords(self) -> np.ndarray:
        """Create arrays of possible x, y, and z coordinates
        for a single iteration of the simulation.

        Returns:
            np.ndarray: Numpy arrays of possible x, y and z coords.
        """

        # Generate array of possible radial movements.
        steps = np.arange(0, self.MAX_MOVEMENT_PER_ITERATION, self.STEP_SIZE)

        # Project radial movements onto the x-axis and clip values to the edge of the box.
        possible_x = (
            self._x + math.sin(self._phi) * math.cos(self._theta) * steps
        ).astype(int, copy=False)
        possible_x.clip(0, self._box_length - 1, out=possible_x)

        # Project radial movements onto the y-axis and clip values to the edge of the box.
        possible_y = (
            self._y + math.sin(self._phi) * math.sin(self._theta) * steps
        ).astype(int, copy=False)
        possible_y.clip(0, self._box_length - 1, out=possible_y)

        # Project radial movements onto the z-axis and clip values to the edge of the box.
        possible_z = (self._z + math.cos(self._phi) * steps).astype(int, copy=False)
        possible_z.clip(0, self._box_length // 2 - 1, out=possible_z)

        return possible_x, possible_y, possible_z

    def _generate_random_interaction_log_probability(self) -> float:
        """Method to generate the negative log of a random number
        in the range (0, 1] from a uniform distribution.

        Returns:
            float: negative log of random number in range (0, 1]
        """

        # Generate random number from a uniform distribution.
        # An infinitessimal number is used as the lower bound to prevent
        # a potential log(0) error.
        random_interaction_probability = random.uniform(self._tiny_float, 1.0)

        return -math.log(random_interaction_probability)

    def _calculate_event_index(self, log_probability_along_path: np.ndarray) -> int:
        """Method to evaluate the index along the radial vector at
        which an interaction first occurs for a given interaction type.

        Args:
            log_probability_along_path (np.ndarray): A NumPy array of cumulative log probabilies
            along the radial vector for a given interaction type.

        Returns:
            int: Index along the radial vector at which an event first occurs.
            Will return endpoint index if no event occurs.
        """

        # Call _generate_random_interaction_log_probability
        random_interaction_log_probability = (
            self._generate_random_interaction_log_probability()
        )

        # If any events occur then return first index of event, else return endpoint index.
        if (log_probability_along_path >= random_interaction_log_probability).any():
            return (
                log_probability_along_path >= random_interaction_log_probability
            ).argmax()
        else:
            return log_probability_along_path.shape[0] - 1

    def _determine_event(
        self, KN_event_index: int, FeK_event_index: int, non_FeK_event_index: int
    ) -> int:
        """Method to compare event indices of various interaction types
        to determine which event, if any, occurs.

        Args:
            KN_event_index (int): Klein-Nishina scattering event index.
            FeK_event_index (int): Fe-K absorbtion event index.
            non_FeK_event_index (int): non-Fe-K absorbtion event index.

        Returns:
            int: Smallest event index.
        """

        # Find smallest event index and corresponding event,
        # defaulting to endpoint of radial vector and no event occuring.
        minimum_event_index = self.MAX_MOVEMENT_PER_ITERATION - 1
        event = "None"
        if FeK_event_index < minimum_event_index:
            minimum_event_index = FeK_event_index
            event = "FeK_Absorb"
        if KN_event_index < minimum_event_index:
            minimum_event_index = KN_event_index
            event = "KN_Scatter"
        if non_FeK_event_index < minimum_event_index:
            minimum_event_index = non_FeK_event_index
            event = "Non_FeK_Absorb"

        # Update photon event flag.
        self._event = event

        return minimum_event_index

    def _update_energy(self) -> None:
        """Method to update photon energy and weighting factor,
        depending on which event has occured.
        """

        # If photon is at the edge of the box then it
        # exits the box with no further attribute updates.
        if (
            (self._x in (0, self._box_length - 1))
            | (self._y in (0, self._box_length - 1))
            | (self._z == self._box_length // 2 - 1)
        ):
            self._event = "Box_Exit"
            return None

        # If photon hits the AGN accretion disk then it is
        # absorbed and removed from the simulation.
        if self._z == 0:
            self._event = "Disc_Absorb"
            self._weight = 0.0
            return None

        # If photon encounters non-FeK absorbtion then it is
        # absorbed and removed from the simulation.
        if self._event == "Non_FeK_Absorb":
            self._weight = 0.0
            return None

        # Klein-Nishina scattering.
        if self._event == "KN_Scatter":

            # Generate new random direction of travel.
            new_theta = 2 * math.pi * random.uniform(0.0, 1.0)
            new_phi = math.acos(random.uniform(-1.0, 1.0))

            # Determine cosine of scattering angle
            cosine_scattering_angle = math.cos(new_theta) * math.cos(
                self._theta
            ) + math.sin(new_theta) * math.sin(self._theta) * math.cos(
                new_phi - self._phi
            )

            # Calculate ratio of photon energy to electron rest mass energy.
            energy_to_rest_mass_ratio = self._energy / Constants.ELECTRON_REST_MASS

            # Calculate Klein-Nishina angular factor.
            KN_angular_factor = 1.0 / (
                1.0 + energy_to_rest_mass_ratio * (1.0 - cosine_scattering_angle)
            )

            # Update energy of photon post-scattering.
            self._energy = self._energy * KN_angular_factor

            # Calculate divisor for Klein-Nishina weight adjustment.
            weight_divisor = (
                3.0
                / (8.0 * energy_to_rest_mass_ratio)
                * (
                    (
                        1
                        - 2
                        * (1 + energy_to_rest_mass_ratio)
                        / energy_to_rest_mass_ratio ** 2
                    )
                    * math.log(1 + 2 * energy_to_rest_mass_ratio)
                    + 0.5
                    + 4.0 / energy_to_rest_mass_ratio
                    - 1 / (2 * (1 + 2 * energy_to_rest_mass_ratio) ** 2)
                )
            )

            # Update photon energy weighting to account for
            # angular dependence of Klein-Nishina scattering.
            self._weight = (
                self._weight
                * 0.75
                * KN_angular_factor ** 2
                * (
                    KN_angular_factor
                    + 1.0 / KN_angular_factor
                    - 1
                    + cosine_scattering_angle ** 2
                )
                / weight_divisor
            )

            # Update photon direction of travel post-scattering.
            self._theta = new_theta
            self._phi = new_phi

        # FeK absorbtion.
        if self._event == "FeK_Absorb":

            # Generate new random direction of travel.
            self._theta = 2 * np.pi * random.uniform(0.0, 1.0)
            self._phi = math.acos(random.uniform(-1.0, 1.0))

            # Randomly assign event to FeK Alpha or Beta absorbtion and update energy.
            if random.uniform(0.0, 1.0) < self.FEK_BETA_TO_ALPHA_RATIO:
                self._energy = self.FEK_BETA_ENERGY
            else:
                self._energy = self.FEK_ALPHA_ENERGY

            # Update photon energy weight to account for potential Auger electron emission.
            self._weight = self.FLUORESCENCE_YIELD * self._weight

        # If photon drops out of observable energy range then remove from simulation.
        if (self._energy < self.MIN_ENERGY) | (self._energy > self.MAX_ENERGY):
            self._event = "Unobservable_Absorb"
            self._weight = 0.0

        return None

    def iterate(self, gas_array: np.ndarray) -> None:
        """Method to perform a single iteration of the MCRT simulation for a photon.

        Args:
            gas_array (np.ndarray): 3-D NumPy array giving the density of
            circumnuclear gas at each point in the Cartesian box.
            Will be passed in via simulate method.
        """

        # Calculate possible x, y, & z coordinates.
        possible_x, possible_y, possible_z = self._calculate_possible_coords()

        # Evaluate cumulative density at each point along the generated radial vector.
        cumulative_density_along_path = np.cumsum(
            gas_array[possible_x, possible_y, possible_z]
        )

        # Calculate binned photon energy to look up corresponding interaction cross-sections.
        binned_energy = self._bin_photon_energy()

        # Calculate cumulative log probabilities along generated radial vector.
        KN_log_probability_along_path = (
            cumulative_density_along_path
            * Constants.INTERACTION_CROSS_SECTIONS.at[
                binned_energy, "Klein-Nishina Scattering Cross-Section"
            ]
        )

        FeK_log_probability_along_path = (
            cumulative_density_along_path
            * Constants.INTERACTION_CROSS_SECTIONS.at[
                binned_energy, "FeK Absorbtion Cross-Section"
            ]
        )

        non_FeK_log_probability_along_path = (
            cumulative_density_along_path
            * Constants.INTERACTION_CROSS_SECTIONS.at[
                binned_energy, "Non-FeK Absorbtion Cross-Section"
            ]
        )

        # Determine index of radial vector at which each event first occurs.
        KN_event_index = self._calculate_event_index(KN_log_probability_along_path)
        FeK_event_index = self._calculate_event_index(FeK_log_probability_along_path)
        non_FeK_event_index = self._calculate_event_index(
            non_FeK_log_probability_along_path
        )

        # Determine event that occurs first and return corresponding index.
        minimum_event_index = self._determine_event(
            KN_event_index, FeK_event_index, non_FeK_event_index
        )

        # Determine new coordinates.
        new_x = possible_x[minimum_event_index]
        new_y = possible_y[minimum_event_index]
        new_z = possible_z[minimum_event_index]

        # Update distance travelled.
        self._distance += math.sqrt(
            (new_x - self._x) ** 2 + (new_y - self._y) ** 2 + (new_z - self._z) ** 2
        )

        # Update coordinates.
        self._x = new_x
        self._y = new_y
        self._z = new_z

        # Update energy and weight of photon.
        self._update_energy()

    def _continue_simulation(self) -> bool:
        """Method to determine whether or not to continue with simulation for photon.

        Returns:
            bool: Boolean indicating whether or not to continue with simulation.
        """

        # Dictionary of continue conditions.
        continue_dict = {
            "Box_Exit": False,
            "Disc_Absorb": False,
            "Non_FeK_Absorb": False,
            "Unobservable_Absorb": False,
            "None": True,
            "KN_Scatter": True,
            "FeK_Absorb": True,
        }

        return continue_dict[self._event]

    def simulate(self, gas_array: np.ndarray):
        """Main method to perform MCRT simulation on a single photon.

        This method will repeatedly loop iterations of MCRT simulation
        until the simulation is exited or the max number of iterations is reached.

        Args:
            gas_array (np.ndarray): 3-D NumPy array giving the density of
            circumnuclear gas at each point in the Cartesian box.
            Should be set to Gas().generate_conical_gas_array().gas_array.
        """

        # Initialise iteration counter.
        iteration_counter = 0

        # Loop iterate method until the simulation encounters a break event.
        while self._continue_simulation():
            self.iterate(gas_array)

            # Increment iteration counter and break loop
            # if max iteration number is exceeded.
            iteration_counter += 1
            if iteration_counter == self.MAX_ITERATIONS:
                break

        return self


class Gas:
    """Class for circumnuclear gas distribution surrounding an AGN."""

    # Black hole mass.
    BLACK_HOLE_MASS: float = Constants.SOLAR_MASS * 2 * 10 ** 6

    # Gravitational radius.
    GRAVITATIONAL_RADIUS: float = (
        Constants.GRAVITATIONAL_CONSTANT
        * BLACK_HOLE_MASS
        / Constants.SPEED_OF_LIGHT ** 2
    )

    # Compton thick column density.
    COMPTON_THICK_COLUMN_DENSITY = (
        10 ** 28 / 10
    )  # Divide by 10 added to reduce number of scatterings per photon.

    def __init__(
        self,
        min_radius: Union[int, float] = 100.0,
        max_radius: Union[int, float] = 400.0,
        opening_phi: float = np.pi / 4,
        box_length: int = 100,
    ) -> None:

        # Gas attributes.
        self._min_radius = min_radius * self.GRAVITATIONAL_RADIUS
        self._max_radius = max_radius * self.GRAVITATIONAL_RADIUS
        self._opening_phi = opening_phi

        # Cartesian box attributes.
        self._box_length = box_length
        self._unit_cell_density = self.COMPTON_THICK_COLUMN_DENSITY

        # Gas array (Initialised as empty).
        self._gas_array = np.zeros(
            shape=(box_length, box_length, box_length // 2),
            dtype=float,
        )
        self._gas_array_type = "empty"

    @property
    def box_length(self) -> int:
        return self._box_length

    @property
    def gas_array(self) -> np.ndarray:
        return self._gas_array

    def generate_conical_gas_array(self):

        gas_array = np.zeros(
            shape=(self._box_length, self._box_length, self._box_length // 2),
            dtype=float,
        )

        scaled_min_radius = (self._min_radius / self._max_radius) * (
            self._box_length // 2
        )
        midpoint_position_adjustment = self._box_length // 2 - 1
        tan_opening_phi = math.tan(self._opening_phi)

        for x, y, z in product(
            range(self._box_length),
            range(self._box_length),
            range(self._box_length // 2),
        ):

            tan_phi = np.sqrt(
                (x - midpoint_position_adjustment) ** 2
                + (y - midpoint_position_adjustment) ** 2
            ) / (z + scaled_min_radius / tan_opening_phi)

            if tan_phi > tan_opening_phi:
                gas_array[x, y, z] = self._unit_cell_density

        gas_array[0, :, :] = 0.0
        gas_array[:, 0, :] = 0.0
        gas_array[:, :, 0] = 0.0
        gas_array[self._box_length - 1, :, :] = 0.0
        gas_array[:, self._box_length - 1, :] = 0.0
        gas_array[:, :, self._box_length // 2 - 1] = 0.0

        self._gas_array = gas_array
        self._gas_array_type = "conical"

        return self

    def plot_gas_array(self) -> None:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=50.0, azim=-60.0)

        ax.voxels(self._gas_array)

        plt.savefig(
            f"./graphs/{self._gas_array_type}_gas_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )


def main(process_number: int) -> None:

    gas = Gas()
    gas.generate_conical_gas_array()

    t1 = time()
    results = [Photon(gas.box_length).simulate(gas.gas_array) for i in range(5000000)]
    t2 = time()
    print(f"{t2-t1}s")

    with open(f"./results/results{process_number}.csv", "w") as results_file:

        results_file.write("x, y, z, theta, phi, energy, weight, distance, event\n")
        for result in results:
            results_file.write(result.output_photon_attributes())


if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        result = executor.map(main, [1, 2, 3, 4])
