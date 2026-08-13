import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


# Filename convention used by WaveExtractRWZ
# (wave_extract_rwz.cpp, DUMP_CMETRIC_DRVTS_SPHERE):
#
#   <basename>_r<radius>_<rank>_i<iter>.txt
#
# e.g. wave_cmetric_sphere_r100.00_009_i0000023.txt
#
# Note: the metadata (iteration, time, radius, rank) actually used by
# MetricData is always read from the file's own header, never from
# the filename -- this pattern is only used to *discover* files on
# disk.
CMETRIC_FNAME_RE = re.compile(
    r"^(?P<base>.+)_r(?P<radius>\d+\.\d+)_(?P<rank>\d+)_i(?P<iter>\d+)\.txt$"
)


class MetricData:
    """
    Cartesian metric data on 2-speres.
    
    Reader/container for cmetric files.
    
    HDF5 layout:

        /grid/
            k
            i
            j
            theta
            phi

        /<iteration>/<time>/<radius>/
            k
            i
            j
            theta
            phi
            alpha
            alpha_t
            ...
            gzz_z

    Data are internally organized as

        (iteration, time, radius) -> DataFrame
    """

    def __init__(self):
        self.slices = {}
        self.grid = {}
        self.fields = []
        self.metadata = {}

    # ==================================================================
    # Reading
    # ==================================================================

    def read_file(self, filename):
        """
        Read one metric file and add it to the container.
        """

        filename = Path(filename)

        metadata = {}
        columns = None

        # --------------------------------------------------------------
        # Read header
        # --------------------------------------------------------------

        with open(filename, "r") as f:

            for line in f:

                if not line.startswith("#"):
                    break

                line = line.strip()

                # Metadata, e.g.
                # # iter = 100
                # # time = 1.234e+00
                # # radius = 1.000e+02
                # # rank = 0

                m = re.match(r"#\s*(\w+)\s*=\s*(.*)", line)

                if m:

                    key, value = m.groups()

                    try:
                        value = float(value)

                        if value.is_integer():
                            value = int(value)

                    except ValueError:
                        pass

                    metadata[key] = value

                # Column header:
                #
                # # k:0 i:1 j:2 theta:3 phi:4 alpha:5 ...
                #
                elif line.startswith("# k:"):

                    entries = line[1:].split()

                    columns = [
                        entry.rsplit(":", 1)[0]
                        for entry in entries
                    ]

        if columns is None:
            raise ValueError(
                f"No column header found in {filename}"
            )

        # --------------------------------------------------------------
        # Read numerical data
        # --------------------------------------------------------------

        df = pd.read_csv(
            filename,
            comment="#",
            sep=r"\s+",
            header=None,
            names=columns
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        if "iter" not in metadata:
            raise ValueError(
                f"No iteration information in {filename}"
            )

        if "time" not in metadata:
            raise ValueError(
                f"No time information in {filename}"
            )

        if "radius" not in metadata:
            raise ValueError(
                f"No radius information in {filename}"
            )

        iteration = int(metadata["iter"])
        time = float(metadata["time"])
        radius = float(metadata["radius"])
        rank = int(metadata.get("rank", -1))

        df["rank"] = rank

        # --------------------------------------------------------------
        # Key
        # --------------------------------------------------------------

        key = (
            iteration,
            time,
            radius
        )

        # --------------------------------------------------------------
        # Merge rank files belonging to the same iteration/time/radius
        # --------------------------------------------------------------

        if key in self.slices:

            self.slices[key] = pd.concat(
                [
                    self.slices[key],
                    df
                ],
                ignore_index=True
            )

        else:

            self.slices[key] = df

        # --------------------------------------------------------------
        # Fields
        # --------------------------------------------------------------

        self.fields = [
            c for c in columns
            if c not in (
                "k",
                "i",
                "j",
                "theta",
                "phi"
            )
        ]

        # rank is metadata, not a physical field
        if "rank" in self.fields:
            self.fields.remove("rank")

        # --------------------------------------------------------------
        # Store metadata
        # --------------------------------------------------------------

        self.metadata[key] = {
            "iteration": iteration,
            "time": time,
            "radius": radius,
            "rank": rank,
        }

        return df

    # ==================================================================
    # Read multiple files
    # ==================================================================

    def read_files(self, filenames):
        """
        Read a list of files.

        Files belonging to the same
        (iteration, time, radius)
        are automatically merged.
        """

        for filename in filenames:
            self.read_file(filename)

        self._finalize()

    # ==================================================================
    # Discover files on disk
    # ==================================================================

    @staticmethod
    def find_files(directory, basename="wave_cmetric_sphere"):
        """
        Find cmetric files on disk following the naming convention

            <basename>_r<radius>_<rank>_i<iter>.txt

        used by WaveExtractRWZ, e.g.

            wave_cmetric_sphere_r100.00_009_i0000023.txt

        Returns a sorted list of matching file paths. This is purely
        for discovery -- the actual (iteration, time, radius, rank)
        values are always read from each file's own header.
        """

        directory = Path(directory)

        candidates = glob.glob(
            str(directory / f"{basename}_r*_*_i*.txt")
        )

        matches = [
            f for f in candidates
            if (
                m := CMETRIC_FNAME_RE.match(Path(f).name)
            ) and m.group("base") == basename
        ]

        return sorted(matches)

    def read_directory(self, directory, basename="wave_cmetric_sphere"):
        """
        Discover and read all cmetric files in a directory.

        Equivalent to:

            files = MetricData.find_files(directory, basename)
            self.read_files(files)

        Returns the list of files that were read.
        """

        files = self.find_files(directory, basename)

        if not files:
            raise FileNotFoundError(
                f"No '{basename}_r*_*_i*.txt' files found in {directory}"
            )

        self.read_files(files)

        return files

    # ==================================================================
    # Incremental timeslice
    # ==================================================================

    def add_timeslice(self, filenames):
        """
        Add another set of rank files.

        Useful when processing a simulation incrementally.
        """

        for filename in filenames:
            self.read_file(filename)

        self._finalize()

    # ==================================================================
    # Finalize
    # ==================================================================

    def _finalize(self):
        """
        Sort data and construct the global grid.
        """

        for key, df in self.slices.items():

            # Sort by global linear index
            if "k" in df:
                df.sort_values(
                    "k",
                    inplace=True
                )

                # Remove duplicate points, if any
                df.drop_duplicates(
                    subset="k",
                    keep="first",
                    inplace=True
                )

            df.reset_index(
                drop=True,
                inplace=True
            )

        # --------------------------------------------------------------
        # Construct grid from first available slice
        # --------------------------------------------------------------

        if self.slices:

            df = next(
                iter(self.slices.values())
            )

            for name in [
                "k",
                "i",
                "j",
                "theta",
                "phi"
            ]:

                if name in df:

                    self.grid[name] = (
                        df[name].to_numpy()
                    )

    # ==================================================================
    # List available data
    # ==================================================================

    def iterations(self):
        """Return sorted list of available iterations."""

        return sorted(
            set(
                key[0]
                for key in self.slices
            )
        )

    def times(self, iteration=None):
        """
        Return sorted times.

        If iteration is specified, return only times for that iteration.
        """

        if iteration is None:

            return sorted(
                set(
                    key[1]
                    for key in self.slices
                )
            )

        return sorted(
            set(
                key[1]
                for key in self.slices
                if key[0] == iteration
            )
        )

    def radii(self, iteration=None, time=None):
        """
        Return sorted radii.

        Optionally restrict by iteration and/or time.
        """

        values = []

        for it, t, r in self.slices:

            if iteration is not None and it != iteration:
                continue

            if time is not None and not np.isclose(t, time):
                continue

            values.append(r)

        return sorted(set(values))

    # ==================================================================
    # Query
    # ==================================================================

    def get_sphere(
        self,
        iteration,
        time,
        radius
    ):
        """
        Return all data on one sphere.
        """

        key = self._find_key(
            iteration,
            time,
            radius
        )

        if key is None:

            raise KeyError(
                f"No data for "
                f"iteration={iteration}, "
                f"time={time}, "
                f"radius={radius}"
            )

        return self.slices[key].copy()

    def get_field(
        self,
        field,
        iteration,
        time,
        radius
    ):
        """
        Return one field on a sphere.
        """

        df = self.get_sphere(
            iteration,
            time,
            radius
        )

        if field not in df.columns:

            raise KeyError(
                f"Unknown field '{field}'"
            )

        return df[field].to_numpy()

    def _find_key(
        self,
        iteration,
        time,
        radius,
        atol=1e-10
    ):
        """
        Find a key using numerical tolerance for time/radius.
        """

        for it, t, r in self.slices:

            if it != iteration:
                continue

            if (
                np.isclose(t, time, atol=atol)
                and
                np.isclose(r, radius, atol=atol)
            ):
                return (it, t, r)

        return None

    # ==================================================================
    # HDF5
    # ==================================================================

    def dump_hdf5(
        self,
        filename,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        dtype=None,
    ):
        """
        Write all data to HDF5.

        Layout:

        /
        ├── grid/
        │   ├── k
        │   ├── i
        │   ├── j
        │   ├── theta
        │   └── phi
        │
        ├── iteration_<N>/
        │   └── time_<T>/
        │       └── radius_<R>/
        │           ├── alpha
        │           ├── alpha_t
        │           ├── ...
        │           └── gzz_z
        │
        └── ...

        The grid is stored once because it is common to all
        iterations, times, and radii.

        Parameters
        ----------
        compression : str or None
            HDF5 filter applied to every dataset, e.g. "gzip" or
            "lzf". "gzip" compresses more but is slower; "lzf" is
            faster but compresses less and is not available outside
            HDF5/h5py (i.e. not readable by every third-party tool).
            Pass None to disable compression and write plain,
            uncompressed datasets (matches the original behaviour).
            Note that compression only pays off once datasets are
            reasonably large -- for very small spheres the per-dataset
            chunk/filter overhead can make compressed files *larger*
            than uncompressed ones. Also note that gzip/lzf/shuffle
            typically only buy ~1.2x on real (noisy) double-precision
            simulation data -- see `dtype` below for a much bigger win.
        compression_opts : int or None
            Compression level, 0 (fastest/largest) to 9
            (slowest/smallest). Only used with compression="gzip";
            ignored for "lzf" and None.
        shuffle : bool
            Apply the HDF5 byte-shuffle filter before compression.
            This reorders bytes across elements and typically improves
            the compression ratio for floating-point arrays like ours
            at essentially no extra cost. Ignored when compression is
            None.
        dtype : numpy dtype or None
            If given (e.g. `np.float32`), physical field data is cast
            to this dtype before writing, roughly halving file size
            for float32 on top of whatever `compression` gives. This
            is a real (lossy) precision reduction -- float32 keeps
            about 7 significant decimal digits, vs ~16 for float64 --
            so make sure that is acceptable for your downstream
            analysis. The coordinate grid (k, i, j, theta, phi) is
            always written at its original precision regardless of
            this setting, since it is tiny and precision there is
            usually more important. Default None keeps the original
            dtype (typically float64), matching the previous
            behaviour.
        """

        # compression_opts is only meaningful for gzip
        gzip_opts = compression_opts if compression == "gzip" else None

        def _create_dataset(group, name, data, cast=False, **attrs):
            """Create a dataset, compressed unless it is empty (HDF5
            cannot chunk/compress a zero-length dataset).

            `cast` controls whether `dtype` downcasting applies to
            this dataset -- it is applied to physical field data but
            never to the coordinate grid.
            """

            data = np.asarray(data)

            if cast and dtype is not None and np.issubdtype(
                    data.dtype, np.floating):
                data = data.astype(dtype)

            kwargs = {}
            if compression is not None and data.size > 0:
                kwargs["compression"] = compression
                kwargs["shuffle"] = shuffle
                if gzip_opts is not None:
                    kwargs["compression_opts"] = gzip_opts

            dataset = group.create_dataset(
                name,
                data=data,
                **kwargs,
            )

            for key, value in attrs.items():
                dataset.attrs[key] = value

            return dataset

        with h5py.File(filename, "w") as h5:

            # ==============================================================
            # File description
            # ==============================================================

            h5.attrs["description"] = (
                "Metric data extracted on coordinate spheres"
            )

            h5.attrs["data_type"] = (
                "3+1 Cartesian metric and derivatives"
            )

            h5.attrs["coordinate_system"] = (
                "spherical coordinates (theta, phi)"
            )

            h5.attrs["grid_description"] = (
                "Global sphere grid shared by all iterations, "
                "times, and radii"
            )

            h5.attrs["data_structure"] = (
                "iteration/time/radius/field"
            )

            h5.attrs["field_dtype"] = (
                np.dtype(dtype).name if dtype is not None
                else "original (not downcast)"
            )

            # ==============================================================
            # Global grid
            # ==============================================================

            grid_group = h5.create_group("grid")

            grid_group.attrs["description"] = (
                "Global coordinate grid on the extraction sphere"
            )

            grid_group.attrs["k_description"] = (
                "Linear index of the sphere grid point"
            )

            grid_group.attrs["i_description"] = (
                "Theta-grid index"
            )

            grid_group.attrs["j_description"] = (
                "Phi-grid index"
            )

            grid_group.attrs["theta_description"] = (
                "Polar angle theta"
            )

            grid_group.attrs["phi_description"] = (
                "Azimuthal angle phi"
            )

            for name, values in self.grid.items():

                _create_dataset(
                    grid_group,
                    name,
                    values,
                )

            # ==============================================================
            # Data
            # ==============================================================

            for (
                    iteration,
                    time,
                    radius
            ), df in sorted(self.slices.items()):

                # ----------------------------------------------------------
                # iteration_<N>
                # ----------------------------------------------------------

                iteration_group = h5.require_group(
                    self._group_name(
                        "iteration",
                        iteration
                    )
                )

                iteration_group.attrs["iteration"] = iteration

                iteration_group.attrs["description"] = (
                    f"Data corresponding to simulation iteration "
                    f"{iteration}"
                )

                # ----------------------------------------------------------
                # time_<T>
                # ----------------------------------------------------------

                time_group = iteration_group.require_group(
                    self._group_name(
                        "time",
                        time
                    )
                )

                time_group.attrs["iteration"] = iteration
                time_group.attrs["time"] = time

                time_group.attrs["description"] = (
                    f"Data at simulation time t = {time:.12e}"
                )

                # ----------------------------------------------------------
                # radius_<R>
                # ----------------------------------------------------------

                radius_group = time_group.require_group(
                    self._group_name(
                        "radius",
                        radius
                    )
                )

                radius_group.attrs["iteration"] = iteration
                radius_group.attrs["time"] = time
                radius_group.attrs["radius"] = radius

                radius_group.attrs["description"] = (
                    f"Metric data extracted on a sphere of "
                    f"radius r = {radius:.12e}"
                )

                # ----------------------------------------------------------
                # Physical fields
                # ----------------------------------------------------------
                
                for field in self.fields:
                    
                    if field in df:

                        _create_dataset(
                            radius_group,
                            field,
                            df[field].to_numpy(),
                            cast=True,
                            description=f"Metric field {field}",
                        )


    def read_hdf5(self, filename):
        """
        Read a MetricData HDF5 file.

        Expected layout:

        /grid/
            k
            i
            j
            theta
            phi

        /iteration_<N>/
            /time_<T>/
                /radius_<R>/
                    alpha
                    alpha_t
                    ...
                    gzz_z

        The HDF5 data are reconstructed into self.slices using

        (iteration, time, radius) -> DataFrame
        """

        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {filename}"
            )

        # Reset current data
        self.slices = {}
        self.grid = {}
        self.fields = []
        self.metadata = {}
        
        with h5py.File(filename, "r") as h5:

            # ==============================================================
            # Read global grid
            # ==============================================================

            if "grid" not in h5:
                raise ValueError(
                    "HDF5 file does not contain a 'grid' group"
                )

            grid_group = h5["grid"]

            for name in [
                    "k",
                    "i",
                    "j",
                    "theta",
                    "phi",
            ]:

                if name in grid_group:

                    self.grid[name] = (
                        grid_group[name][:]
                    )

            # ==============================================================
            # Read iterations
            # ==============================================================

            for iteration_name in h5:

                # Skip global groups
                if iteration_name == "grid":
                    continue

                # Expect a bare iteration number, e.g. "100"
                try:
                    iteration = int(iteration_name)
                except ValueError:
                    continue

                iteration_group = h5[iteration_name]

                # ==========================================================
                # Read times
                # ==========================================================

                for time_name in iteration_group:

                    # Expect a bare formatted float, e.g.
                    # "0.000000000000e+00"
                    try:
                        time = float(time_name)
                    except ValueError:
                        continue

                    time_group = iteration_group[time_name]

                    # ======================================================
                    # Read radii
                    # ======================================================

                    for radius_name in time_group:

                        # Expect a bare formatted float, e.g.
                        # "1.000000000000e+01"
                        try:
                            radius = float(radius_name)
                        except ValueError:
                            continue

                        radius_group = (
                            time_group[radius_name]
                        )

                        key = (
                            iteration,
                            time,
                            radius,
                        )

                        # ==================================================
                        # Construct DataFrame
                        # ==================================================

                        data = {}

                        # Global grid information
                        # is common to all fields.
                        for name in [
                                "k",
                                "i",
                                "j",
                                "theta",
                                "phi",
                        ]:

                            if name in self.grid:

                                data[name] = (
                                    self.grid[name].copy()
                                )

                        # ==================================================
                        # Read physical fields
                        # ==================================================

                        for field_name in radius_group:

                            dataset = radius_group[field_name]
                            
                            data[field_name] = (
                                dataset[:]
                            )

                            if field_name not in self.fields:
                                self.fields.append(
                                    field_name
                                )

                        # ==================================================
                        # Create DataFrame
                        # ==================================================

                        df = pd.DataFrame(data)

                        self.slices[key] = df

                        # ==================================================
                        # Metadata
                        # ==================================================

                        self.metadata[key] = {
                            "iteration": iteration,
                            "time": time,
                            "radius": radius,
                        }

            # ==============================================================
            # Sort fields consistently
            # ==============================================================

            self.fields = [
                field
                for field in self.fields
                if field not in [
                        "k",
                        "i",
                        "j",
                        "theta",
                        "phi",
                        "rank",
                ]
            ]

            print(
                f"Read HDF5 file: {filename}"
            )
            
            print(
                f"  iterations : {len(self.iterations())}"
            )

            print(
                f"  slices     : {len(self.slices)}"
            )

            print(
                f"  fields     : {len(self.fields)}"
            )
            
            if self.grid:
                print(
                    f"  grid points: {len(self.grid['k'])}"
                )
                        
    # ==================================================================
    # HDF5 helper
    # ==================================================================

    @staticmethod
    def _group_name(prefix, value):
        """
        Create an HDF5 group name.

        No prefix is used: iteration groups are named by the bare
        iteration number, and time/radius groups are named by the
        bare formatted float. This matches the cmetric HDF5
        convention (<iteration>/<time>/<radius>/<field>).

        Examples:
           100
           1.000000000000e+00
           1.000000000000e+02
        """

        if prefix == "iteration":
            return f"{int(value)}"

        return f"{value:.12e}"
    
    # ==================================================================
    # Visualization: 2D sphere
    # ==================================================================

    def plot_sphere(
        self,
        field,
        iteration,
        time,
        radius,
        cmap="viridis",
        figsize=(8, 5)
    ):
        """
        Plot a field on a sphere using theta/phi coordinates.
        """

        df = self.get_sphere(
            iteration,
            time,
            radius
        )

        theta = df["theta"].to_numpy()
        phi = df["phi"].to_numpy()
        values = df[field].to_numpy()

        fig, ax = plt.subplots(
            figsize=figsize
        )

        sc = ax.scatter(
            phi,
            theta,
            c=values,
            s=8,
            cmap=cmap
        )

        ax.set_xlabel(
            r"$\phi$"
        )

        ax.set_ylabel(
            r"$\theta$"
        )

        ax.set_title(
            f"{field}, "
            f"iteration={iteration}, "
            f"t={time:g}, "
            f"r={radius:g}"
        )

        fig.colorbar(
            sc,
            ax=ax,
            label=field
        )

        return fig, ax

    # ==================================================================
    # Visualization: fixed theta
    # ==================================================================

    def plot_theta(
        self,
        field,
        iteration,
        time,
        radius,
        theta,
        atol=1e-8
    ):
        """
        Plot a field along phi at approximately fixed theta.
        """

        df = self.get_sphere(
            iteration,
            time,
            radius
        )

        mask = (
            np.abs(
                df["theta"] - theta
            ) < atol
        )

        sub = df[
            mask
        ].sort_values(
            "phi"
        )

        fig, ax = plt.subplots()

        ax.plot(
            sub["phi"],
            sub[field],
            ".-"
        )

        ax.set_xlabel(
            r"$\phi$"
        )

        ax.set_ylabel(
            field
        )

        ax.set_title(
            f"{field}, "
            f"$\\theta={theta:g}$"
        )

        return fig, ax

    # ==================================================================
    # Visualization: fixed phi
    # ==================================================================

    def plot_phi(
        self,
        field,
        iteration,
        time,
        radius,
        phi,
        atol=1e-8
    ):
        """
        Plot a field along theta at approximately fixed phi.
        """

        df = self.get_sphere(
            iteration,
            time,
            radius
        )

        mask = (
            np.abs(
                df["phi"] - phi
            ) < atol
        )

        sub = df[
            mask
        ].sort_values(
            "theta"
        )

        fig, ax = plt.subplots()

        ax.plot(
            sub["theta"],
            sub[field],
            ".-"
        )

        ax.set_xlabel(
            r"$\theta$"
        )

        ax.set_ylabel(
            field
        )

        ax.set_title(
            f"{field}, "
            f"$\\phi={phi:g}$"
        )

        return fig, ax
