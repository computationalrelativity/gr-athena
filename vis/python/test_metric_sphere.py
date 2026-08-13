import os
import shutil
from pathlib import Path

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # no display required for tests

from metric_sphere import MetricData


# ======================================================================
# Synthetic data generation
# ======================================================================

FIELDS = [
    "alpha",
    "alpha_t",
    "alpha_x", "alpha_y", "alpha_z",

    "betax", "betay", "betaz",
    "betax_t", "betay_t", "betaz_t",
    "betax_x", "betay_x", "betaz_x",
    "betax_y", "betay_y", "betaz_y",
    "betax_z", "betay_z", "betaz_z",

    "gxx", "gxy", "gxz",
    "gyx", "gyy", "gyz",
    "gzx", "gzy", "gzz",

    "gxx_t", "gxy_t", "gxz_t",
    "gyx_t", "gyy_t", "gyz_t",
    "gzx_t", "gzy_t", "gzz_t",

    "gxx_x", "gxy_x", "gxz_x",
    "gyx_x", "gyy_x", "gyz_x",
    "gzx_x", "gzy_x", "gzz_x",

    "gxx_y", "gxy_y", "gxz_y",
    "gyx_y", "gyy_y", "gyz_y",
    "gzx_y", "gzy_y", "gzz_y",

    "gxx_z", "gxy_z", "gxz_z",
    "gyx_z", "gyy_z", "gyz_z",
    "gzx_z", "gzy_z", "gzz_z",
]


def synthetic_value(field, k, theta, phi, iteration, time, radius):
    """
    Deterministic synthetic value.

    Using a deterministic function makes it possible to verify that
    the values read from disk are actually correct.
    """

    field_number = FIELDS.index(field)

    return (
        1000.0 * field_number
        + 100.0 * iteration
        + 10.0 * time
        + radius
        + k
        + np.sin(theta)
        + np.cos(phi)
    )


def write_cmetric_file(
    filename,
    iteration,
    time,
    radius,
    rank,
    k_values,
    ntheta,
    nphi,
):
    """
    Write one synthetic rank file in exactly the format expected
    by MetricData.
    """

    filename = Path(filename)

    with open(filename, "w") as f:

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        f.write(f"# iter = {iteration}\n")
        f.write(f"# time = {time:.9e}\n")
        f.write(f"# radius = {radius:.9e}\n")
        f.write(f"# rank = {rank}\n")

        # --------------------------------------------------------------
        # Column header
        # --------------------------------------------------------------

        f.write(
            "# k:0 i:1 j:2 theta:3 phi:4"
        )

        for index, field in enumerate(FIELDS):
            f.write(
                f" {field}:{index + 5}"
            )

        f.write("\n")

        # --------------------------------------------------------------
        # Data
        # --------------------------------------------------------------

        for k in k_values:

            i = k // nphi
            j = k % nphi

            theta = np.pi * (i + 0.5) / ntheta
            phi = 2.0 * np.pi * j / nphi

            values = [
                synthetic_value(
                    field,
                    k,
                    theta,
                    phi,
                    iteration,
                    time,
                    radius,
                )
                for field in FIELDS
            ]

            f.write(
                f"{k:d} "
                f"{i:d} "
                f"{j:d} "
                f"{theta:.15e} "
                f"{phi:.15e}"
            )

            for value in values:
                f.write(
                    f" {value:.15e}"
                )

            f.write("\n")


def generate_test_files(directory):
    """
    Generate a small fake MPI dataset.

    Dataset:

        (iteration, time) = (100, 0.0), (200, 1.0)
        radii              = [10.0, 20.0]

    Iteration and time are paired one-to-one, as they are in a real
    simulation (each iteration corresponds to exactly one time) --
    this matters here because the real WaveExtractRWZ filename
    convention encodes the iteration but not the time, so two
    different times sharing an iteration would collide on disk.

    Sphere:

        ntheta = 4
        nphi   = 8
        total  = 32 points

    Four fake MPI ranks split the 32 points.
    """

    directory = Path(directory)

    ntheta = 4
    nphi = 8
    npoints = ntheta * nphi

    iteration_times = [(100, 0.0), (200, 1.0)]
    radii = [10.0, 20.0]

    nranks = 4

    files = []

    # Split k among fake MPI ranks
    rank_k = np.array_split(
        np.arange(npoints),
        nranks,
    )

    for iteration, time in iteration_times:

        for radius in radii:

            for rank in range(nranks):

                # Real WaveExtractRWZ naming convention, e.g.
                # wave_cmetric_sphere_r100.00_009_i0000023.txt
                filename = (
                    directory
                    / (
                        f"wave_cmetric_sphere"
                        f"_r{radius:.2f}"
                        f"_{rank:03d}"
                        f"_i{iteration:07d}.txt"
                    )
                )

                write_cmetric_file(
                    filename=filename,
                    iteration=iteration,
                    time=time,
                    radius=radius,
                    rank=rank,
                    k_values=rank_k[rank],
                    ntheta=ntheta,
                    nphi=nphi,
                )

                files.append(filename)

    return files


# ======================================================================
# Tests
# ======================================================================

def test_read_and_merge(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()

    data.read_files(files)

    # 2 (iteration, time) pairs x 2 radii
    assert len(data.slices) == 4

    # --------------------------------------------------------------
    # Check iterations
    # --------------------------------------------------------------

    assert data.iterations() == [100, 200]

    # --------------------------------------------------------------
    # Check times
    # --------------------------------------------------------------

    assert data.times(100) == [0.0]
    assert data.times(200) == [1.0]

    # --------------------------------------------------------------
    # Check radii
    # --------------------------------------------------------------

    assert data.radii(
        iteration=100,
        time=0.0,
    ) == [10.0, 20.0]

    # --------------------------------------------------------------
    # Check one merged sphere
    # --------------------------------------------------------------

    sphere = data.get_sphere(
        iteration=100,
        time=0.0,
        radius=10.0,
    )

    # 4 x 8 points
    assert len(sphere) == 32

    # All k values should be present
    assert np.array_equal(
        sphere["k"].to_numpy(),
        np.arange(32),
    )

    # All four ranks should have been merged
    assert set(sphere["rank"]) == {0, 1, 2, 3}

    print("read/merge: OK")


def test_grid(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    assert len(data.grid["k"]) == 32
    assert len(data.grid["theta"]) == 32
    assert len(data.grid["phi"]) == 32

    assert data.grid["k"][0] == 0
    assert data.grid["k"][-1] == 31

    print("grid: OK")


def test_fields(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    # Check that expected fields exist
    assert "alpha" in data.fields
    assert "gxx" in data.fields
    assert "gzz_z" in data.fields

    # rank should not be a physical field
    assert "rank" not in data.fields

    print("fields: OK")


def test_field_values(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    iteration = 200
    time = 1.0
    radius = 20.0

    sphere = data.get_sphere(
        iteration,
        time,
        radius,
    )

    # Test several points
    for row in sphere.iloc[[0, 10, 20, 31]].itertuples():

        k = int(row.k)
        theta = row.theta
        phi = row.phi

        expected = synthetic_value(
            "gxx",
            k,
            theta,
            phi,
            iteration,
            time,
            radius,
        )

        actual = row.gxx

        assert np.isclose(
            actual,
            expected,
        )

    print("field values: OK")


def test_get_field(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    values = data.get_field(
        "alpha",
        iteration=200,
        time=1.0,
        radius=10.0,
    )

    assert len(values) == 32
    assert np.all(np.isfinite(values))

    print("get_field: OK")


def test_find_files(tmp_path):

    generate_test_files(tmp_path)

    found = MetricData.find_files(tmp_path)

    # 2 (iteration, time) pairs x 2 radii x 4 ranks
    assert len(found) == 16

    for f in found:
        assert Path(f).name.startswith("wave_cmetric_sphere_r")
        assert Path(f).name.endswith(".txt")

    # A non-matching basename should find nothing
    assert MetricData.find_files(tmp_path, basename="not_a_match") == []

    print("find_files: OK")


def test_read_directory(tmp_path):

    generate_test_files(tmp_path)

    data = MetricData()
    files = data.read_directory(tmp_path)

    assert len(files) == 16
    assert len(data.slices) == 4

    assert data.iterations() == [100, 200]

    print("read_directory: OK")


def test_read_directory_missing(tmp_path):

    data = MetricData()

    try:
        data.read_directory(tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "expected FileNotFoundError for an empty directory"
        )

    print("read_directory (missing): OK")


def test_hdf5(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    h5file = tmp_path / "test.h5"

    data.dump_hdf5(h5file)

    assert h5file.exists()

    # --------------------------------------------------------------
    # Read the generated HDF5
    # --------------------------------------------------------------

    with h5py.File(h5file, "r") as h5:

        # Grid
        assert "grid" in h5

        assert "k" in h5["grid"]
        assert "theta" in h5["grid"]
        assert "phi" in h5["grid"]

        # Iterations
        assert "100" in h5
        assert "200" in h5

        # Times (iteration 100 pairs with time 0.0 only)
        it100 = h5["100"]

        assert "0.000000000000e+00" in it100

        # Radius
        t0 = it100["0.000000000000e+00"]

        assert "1.000000000000e+01" in t0
        assert "2.000000000000e+01" in t0

        # Field
        sphere = t0["1.000000000000e+01"]

        assert "alpha" in sphere
        assert "gxx" in sphere
        assert "gzz_z" in sphere

        # 32 points
        assert sphere["gxx"].shape == (32,)

    print("HDF5: OK")

def check_hdf5_roundtrip(data1, h5file):
    """
    Verify that writing to HDF5 and reading it back preserves
    the scientific data.

    This is a helper, not a pytest test itself (it takes plain
    arguments rather than fixtures) -- see test_hdf5_roundtrip()
    below for the actual pytest entry point, and demo() for the
    other caller.

    The MPI 'rank' column is intentionally excluded because it is
    file-level bookkeeping and is not stored in the HDF5 format.
    """

    # --------------------------------------------------------------
    # Read HDF5
    # --------------------------------------------------------------

    data2 = MetricData()
    data2.read_hdf5(h5file)

    # --------------------------------------------------------------
    # Compare iterations
    # --------------------------------------------------------------

    assert data1.iterations() == data2.iterations()

    # --------------------------------------------------------------
    # Compare times and radii
    # --------------------------------------------------------------

    for iteration in data1.iterations():

        assert data1.times(iteration) == data2.times(iteration)

        for time in data1.times(iteration):

            assert (
                data1.radii(iteration, time)
                == data2.radii(iteration, time)
            )

    # --------------------------------------------------------------
    # Compare global grid
    # --------------------------------------------------------------

    grid_names = [
        "k",
        "i",
        "j",
        "theta",
        "phi",
    ]

    for name in grid_names:

        assert name in data1.grid
        assert name in data2.grid

        np.testing.assert_allclose(
            data1.grid[name],
            data2.grid[name],
        )

    # --------------------------------------------------------------
    # Compare fields
    #
    # Do NOT compare list ordering. HDF5 does not necessarily
    # reproduce the order in which fields were originally read.
    # --------------------------------------------------------------

    assert set(data1.fields) == set(data2.fields)

    # --------------------------------------------------------------
    # Compare slices
    # --------------------------------------------------------------

    columns = grid_names + list(data1.fields)

    for key in data1.slices:

        assert key in data2.slices

        df1 = data1.slices[key]
        df2 = data2.slices[key]

        # Check that all expected columns exist
        for column in columns:

            assert column in df1.columns, (
                f"Missing column '{column}' in original data "
                f"for slice {key}"
            )

            assert column in df2.columns, (
                f"Missing column '{column}' in HDF5 data "
                f"for slice {key}"
            )

        # Compare actual values
        for column in columns:

            np.testing.assert_allclose(
                df1[column].to_numpy(),
                df2[column].to_numpy(),
                equal_nan=True,
                err_msg=(
                    f"Mismatch in field '{column}' "
                    f"for slice {key}"
                ),
            )

    print("HDF5 round-trip: OK")
    
def test_hdf5_roundtrip(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    h5file = tmp_path / "roundtrip.h5"
    data.dump_hdf5(h5file)

    check_hdf5_roundtrip(data, h5file)


def test_hdf5_compression(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    # Default (gzip + shuffle)
    h5_gzip = tmp_path / "gzip.h5"
    data.dump_hdf5(h5_gzip)

    with h5py.File(h5_gzip, "r") as h5:
        dset = h5["100"]["0.000000000000e+00"]["1.000000000000e+01"]["gxx"]
        assert dset.compression == "gzip"
        assert dset.shuffle is True

    # Uncompressed, for backward compatibility
    h5_none = tmp_path / "none.h5"
    data.dump_hdf5(h5_none, compression=None)

    with h5py.File(h5_none, "r") as h5:
        dset = h5["100"]["0.000000000000e+00"]["1.000000000000e+01"]["gxx"]
        assert dset.compression is None

    # Both must round-trip to identical data regardless of compression
    for h5file in (h5_gzip, h5_none):
        check_hdf5_roundtrip(data, h5file)

    print("HDF5 compression: OK")


def test_hdf5_dtype(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    # --------------------------------------------------------------
    # Default: no downcast, fields stay float64
    # --------------------------------------------------------------

    h5_default = tmp_path / "default.h5"
    data.dump_hdf5(h5_default)

    with h5py.File(h5_default, "r") as h5:
        dset = h5["100"]["0.000000000000e+00"]["1.000000000000e+01"]["gxx"]
        assert dset.dtype == np.float64
        assert h5.attrs["field_dtype"] == "original (not downcast)"

    # --------------------------------------------------------------
    # float32 downcast: fields shrink, grid stays float64
    # --------------------------------------------------------------

    h5_f32 = tmp_path / "f32.h5"
    data.dump_hdf5(h5_f32, dtype=np.float32)

    with h5py.File(h5_f32, "r") as h5:
        dset = h5["100"]["0.000000000000e+00"]["1.000000000000e+01"]["gxx"]
        assert dset.dtype == np.float32
        assert h5.attrs["field_dtype"] == "float32"

        # grid coordinates are never downcast
        assert h5["grid"]["theta"].dtype == np.float64
        assert h5["grid"]["phi"].dtype == np.float64

    assert (
        os.path.getsize(h5_f32) < os.path.getsize(h5_default)
    )

    # --------------------------------------------------------------
    # Values should round-trip to within float32 precision
    # --------------------------------------------------------------

    data2 = MetricData()
    data2.read_hdf5(h5_f32)

    original = data.get_field("gxx", 100, 0.0, 10.0)
    downcast = data2.get_field("gxx", 100, 0.0, 10.0)

    assert downcast.dtype == np.float32

    np.testing.assert_allclose(
        downcast,
        original,
        rtol=1e-6,
        atol=1e-6,
    )

    print("HDF5 dtype downcast: OK")


def test_plots(tmp_path):

    files = generate_test_files(tmp_path)

    data = MetricData()
    data.read_files(files)

    # --------------------------------------------------------------
    # 2D sphere
    # --------------------------------------------------------------

    fig, ax = data.plot_sphere(
        field="gxx",
        iteration=100,
        time=0.0,
        radius=10.0,
    )

    assert fig is not None
    assert ax is not None

    # --------------------------------------------------------------
    # Fixed theta
    # --------------------------------------------------------------

    theta = data.grid["theta"][4]

    fig, ax = data.plot_theta(
        field="gxx",
        iteration=100,
        time=0.0,
        radius=10.0,
        theta=theta,
    )

    assert fig is not None
    assert ax is not None

    # --------------------------------------------------------------
    # Fixed phi
    # --------------------------------------------------------------

    phi = data.grid["phi"][4]

    fig, ax = data.plot_phi(
        field="gxx",
        iteration=100,
        time=0.0,
        radius=10.0,
        phi=phi,
    )

    assert fig is not None
    assert ax is not None

    print("plots: OK")


# ======================================================================
# Demo
# ======================================================================

def demo(output_dir="test_cmetric_output"):
    """
    Run the complete MetricData demo.

    The output directory is deleted if it already exists and then
    recreated.
    """

    output_dir = Path(output_dir)

    # --------------------------------------------------------------
    # Clean output directory
    # --------------------------------------------------------------

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("MetricData demo")
    print("=" * 70)

    print()
    print(f"Output directory: {output_dir}")

    # --------------------------------------------------------------
    # Generate synthetic files
    # --------------------------------------------------------------

    print()
    print("Generating synthetic files...")

    files = generate_test_files(output_dir)

    print(
        f"Generated {len(files)} files"
    )

    # --------------------------------------------------------------
    # Read
    # --------------------------------------------------------------

    data = MetricData()

    data.read_files(files)

    # --------------------------------------------------------------
    # Overview
    # --------------------------------------------------------------

    print()
    print("Iterations:")
    print(data.iterations())

    for iteration in data.iterations():

        print(
            f"  iteration {iteration}: "
            f"times = {data.times(iteration)}"
        )

    print()
    print("Fields:")
    print(data.fields)

    print()
    print("Grid points:")
    print(len(data.grid["k"]))

    # --------------------------------------------------------------
    # Query
    # --------------------------------------------------------------

    iteration = 200
    time = 1.0
    radius = 20.0

    sphere = data.get_sphere(
        iteration,
        time,
        radius,
    )

    print()
    print(
        f"Sphere: "
        f"iteration={iteration}, "
        f"time={time}, "
        f"radius={radius}"
    )

    print(
        f"Number of points: {len(sphere)}"
    )

    print()

    print(
        sphere[
            [
                "k",
                "i",
                "j",
                "theta",
                "phi",
                "alpha",
                "gxx",
            ]
        ].head()
    )

    # --------------------------------------------------------------
    # Single field
    # --------------------------------------------------------------

    gxx = data.get_field(
        "gxx",
        iteration,
        time,
        radius,
    )

    print()
    print(
        f"gxx: min={gxx.min():.6e}, "
        f"max={gxx.max():.6e}"
    )

    # --------------------------------------------------------------
    # HDF5
    # --------------------------------------------------------------

    h5file = output_dir / "cmetric.h5"

    data.dump_hdf5(h5file)

    print()
    print(
        f"HDF5 written to: {h5file}"
    )

    # --------------------------------------------------------------
    # HDF5 round-trip test
    # --------------------------------------------------------------

    print()
    print("Testing HDF5 round-trip...")

    check_hdf5_roundtrip(
        data,
        h5file,
    )
    
    # --------------------------------------------------------------
    # Inspect HDF5
    # --------------------------------------------------------------

    with h5py.File(h5file, "r") as h5:

        print()
        print("HDF5 structure:")

        def show(name, obj):

            if isinstance(obj, h5py.Dataset):

                print(
                    f"  {name} "
                    f"{obj.shape}"
                )

            else:

                print(
                    f"  {name}/"
                )

        h5.visititems(show)

    # --------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------

    print()
    print("Creating plots...")

    fig, ax = data.plot_sphere(
        "gxx",
        iteration,
        time,
        radius,
    )

    fig.savefig(
        output_dir / "gxx_sphere.png",
        dpi=150,
    )

    plt.close(fig)

    fig, ax = data.plot_theta(
        "gxx",
        iteration,
        time,
        radius,
        theta=data.grid["theta"][4],
    )

    fig.savefig(
        output_dir / "gxx_theta.png",
        dpi=150,
    )

    plt.close(fig)

    fig, ax = data.plot_phi(
        "gxx",
        iteration,
        time,
        radius,
        phi=data.grid["phi"][4],
    )

    fig.savefig(
        output_dir / "gxx_phi.png",
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Plots written to: {output_dir}"
    )

    print()
    print("=" * 70)
    print("Demo completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    demo("test_metric_sphere_output")
