import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

sys.path.append('/home/mi58rip/gr-athena/vis/python')
import athena_read

# M to ms factor
M_to_ms = 4.92549095e-6 * 1e3

# rho to CGS units factor
rho_to_cgs = 6.177e17

# 1. Correct Cartesian geometry functions matching the athena_read.py API
# face_func requires 4 arguments: xmin, xmax, ratio, and number of points
f_func = lambda xmin, xmax, xrat, nx: np.linspace(xmin, xmax, nx)

# center_func requires 2 arguments: left face (xm) and right face (xp)
c_func = lambda xm, xp: 0.5 * (xm + xp)

# vol_func requires 6 arguments: the min and max faces for all 3 dimensions
v_func = lambda x1m, x1p, x2m, x2p, x3m, x3p: (x1p - x1m) * (x2p - x2m) * (x3p - x3m)

def plot_rho_equator(output_path, indices):
    """
    Plots the equatorial density (rho) from a given Athena++ output file.

    Parameters:
    - output_path: str, path to the output directory
    - indices: list of integers, the indices of the output files to plot
    """

    PLOT_DIR = f"{output_path}/plots/rho_eq_plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    for index in indices:
        #if index is one digit, it should fill as 0000i and so forth
        if len(str(index)) == 1:
            index_str = f"0000{index}"
        elif len(str(index)) == 2:
            index_str = f"000{index}"
        elif len(str(index)) == 3:
            index_str = f"00{index}"
        elif len(str(index)) == 4:
            index_str = f"0{index}"
        else:
            index_str = str(index)
        filepath = f"{output_path}/gr_rns.out3.{index_str}.athdf"

        with h5py.File(filepath, 'r') as f:
            time_val = f.attrs['Time']

        # 2. Extract and stitch SMR data, chooses the highest resolution level available
        data = athena_read.athdf(
            filepath, 
            quantities=['hydro.prim.rho'],  # Use 'hydro.prim.rho' if your specific fork requires it
            vol_func=v_func,
            face_func_1=f_func,
            face_func_2=f_func,
            face_func_3=f_func,
            center_func_1=c_func,
            center_func_2=c_func,
            center_func_3=c_func
        )

        # 3. Find the index closest to z = 0 using cell-centered (v) coordinates
        z_coords = data['x3v']
        z_idx = np.argmin(np.abs(z_coords - 0.0))

        # 4. Extract the 2D density array and face-centered (f) bounding coordinates
        rho_equator = data['hydro.prim.rho'][z_idx, :, :]
        x = data['x1f']
        y = data['x2f']

        # 5. Render the Plot
        fig, ax = plt.subplots(figsize=(6, 5))

        mesh = ax.pcolormesh(
            x, y, rho_equator, 
            cmap='magma', 
            norm=LogNorm(vmin=1e-10, vmax=rho_equator.max())
        )

        # Formatting
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(r'Rest Mass Density $\rho$', fontsize=12)

        ax.set_xlim(-20.0, 20.0)
        ax.set_ylim(-20.0, 20.0)
        ax.set_xlabel(r'$x$ (Code Units)', fontsize=12)
        ax.set_ylabel(r'$y$ (Code Units)', fontsize=12)
        ax.set_title(rf'Equatorial Density $\rho$ at $t={time_val*M_to_ms:.1f}$ ms', fontsize=14)

        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/rho_eq_t{time_val:.1f}.png', dpi=600)
        plt.close(fig)

def plot_vel_equator(output_path, indices):
    """
    Plots the equatorial velocity from a given Athena++ output file.

    Parameters:
    - output_path: str, path to the output directory
    - indices: list of integers, the indices of the output files to plot
    """

    PLOT_DIR = f"{output_path}/plots/vel_eq_plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    for index in indices:
        #if index is one digit, it should fill as 0000i and so forth
        if len(str(index)) == 1:
            index_str = f"0000{index}"
        elif len(str(index)) == 2:
            index_str = f"000{index}"
        elif len(str(index)) == 3:
            index_str = f"00{index}"
        elif len(str(index)) == 4:
            index_str = f"0{index}"
        else:
            index_str = str(index)
        filepath = f"{output_path}/gr_rns.out3.{index_str}.athdf"

        with h5py.File(filepath, 'r') as f:
            time_val = f.attrs['Time']

        data = athena_read.athdf(
                            filepath, 
                            quantities=['hydro.prim.util_u_1', 'hydro.prim.util_u_2'],
                            vol_func=v_func, face_func_1=f_func, face_func_2=f_func, face_func_3=f_func,
                            center_func_1=c_func, center_func_2=c_func, center_func_3=c_func
                        )

        # 3. Pull directly from Z-index 0 (since the grid is 2D)
        z_coords = data['x3v']
        z_idx = np.argmin(np.abs(z_coords - 0.0))

        ux = data['hydro.prim.util_u_1'][z_idx, :, :]
        uy = data['hydro.prim.util_u_2'][z_idx, :, :]

        # 4. Calculate Radial Velocity
        xv, yv = np.meshgrid(data['x1v'], data['x2v'])
        r = np.sqrt(xv**2 + yv**2)
        r[r == 0] = 1e-10  # Prevent division by zero at origin

        ur = (xv * ux + yv * uy) / r

        # 5. Render Plot
        fig, ax = plt.subplots(figsize=(6, 5))

        # Keep limits extremely tight to catch numerical noise at the surface
        # v_limit = 1e-4 
        mesh = ax.pcolormesh(
            data['x1f'], data['x2f'], ur, 
            cmap='RdBu_r',
            norm=Normalize(vmin=0.01, vmax=0.05)
        )

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(r'Radial Velocity $u_r$', fontsize=12)

        ax.set_xlim(-20.0, 20.0)
        ax.set_ylim(-20.0, 20.0)
        ax.set_xlabel(r'$x$ (Code Units)', fontsize=12)
        ax.set_ylabel(r'$y$ (Code Units)', fontsize=12)
        ax.set_title(rf'Equatorial Radial Velocity at $t={time_val*M_to_ms:.1f}$ ms', fontsize=14)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/vel_eq_t{time_val:.1f}.png', dpi=600)
        plt.close(fig)

def plot_rho_max_vs_time(output_path):
    PLOT_DIR = f"{output_path}/plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    data = np.loadtxt(rf"{output_path}/gr_rns.hst", comments='#')

    fig, ax = plt.subplots(figsize=(6, 4))

    times = data[:, 0]
    print("No of snapshots:", len(times))

    # Convert time to milliseconds
    times = times * M_to_ms

    max_rhos = data[:, 20]

    normalised_rho = (max_rhos) / max_rhos[0]


    ax.plot(times, normalised_rho - 1.0, color='navy')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel(r'$\rho_{\rm max} \left(t\right) / \rho_{\rm max}\left(0\right) - 1$', fontsize=12)
    ax.set_title('Fractional Change in Max Density', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    # ax.axhline(y=0, color='red', linestyle='--', label=r'$\Delta \rho_{\rm max} / \rho_{\rm max}\left(t=0\right) = 0$')
    # ax.legend()

    plt.tight_layout()
    plt.savefig(rf"{PLOT_DIR}/frac_change_rho_max.png", dpi=600)
    plt.close(fig)

def plot_mass_bar_vs_time(output_path):
    PLOT_DIR = f"{output_path}/plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    data = np.loadtxt(rf"{output_path}/gr_rns.hst", comments='#')

    fig, ax = plt.subplots(figsize=(6, 4))

    times = data[:, 0]
    print("No of snapshots:", len(times))

    # Convert time to milliseconds
    times = times * M_to_ms

    mass = data[:, 3]

    normalised_mass = (mass) / mass[0]
    ax.plot(times, normalised_mass - 1.0, color='navy')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel(r'$M_{\rm b} \left(t\right) / M_{\rm b}\left(0\right) - 1$', fontsize=12)
    ax.set_title('Fractional Change in Barionic Mass', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(rf"{PLOT_DIR}/frac_change_mass_bar.png", dpi=600)
    plt.close(fig)

def plot_rho_u_y_along_x(output_path, time_idx_1, time_idx_2):
    PLOT_DIR = f"{output_path}/plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 2. Setup Figure with Dual Y-Axes
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx() 

    # Format assumes standard GR-Athena++ naming: gr_rns.out3.00010.athdf
    filepath_1 = f"{output_path}/gr_rns.out3.{time_idx_1:05d}.athdf"
    filepath_2 = f"{output_path}/gr_rns.out3.{time_idx_2:05d}.athdf"

    # Extract data
    data_1 = athena_read.athdf(
        filepath_1, 
        quantities=['hydro.prim.rho', 'hydro.prim.util_u_2'],
        vol_func=v_func, face_func_1=f_func, face_func_2=f_func, face_func_3=f_func,
        center_func_1=c_func, center_func_2=c_func, center_func_3=c_func
    )

    data_2 = athena_read.athdf(
        filepath_2, 
        quantities=['hydro.prim.rho', 'hydro.prim.util_u_2'],
        vol_func=v_func, face_func_1=f_func, face_func_2=f_func, face_func_3=f_func,
        center_func_1=c_func, center_func_2=c_func, center_func_3=c_func
    )
    with h5py.File(filepath_1, 'r') as f:
        time_val_1 = f.attrs['Time']
    with h5py.File(filepath_2, 'r') as f:
        time_val_2 = f.attrs['Time']

    # Isolate the x-axis (where y=0 and z=0)
    z_idx = np.argmin(np.abs(data_1['x3v'] - 0.0))
    y_idx = np.argmin(np.abs(data_1['x2v'] - 0.0))
    
    x_1d = data_1['x1v']
    rho_1 = data_1['hydro.prim.rho'][z_idx, y_idx, :]
    uy_1 = data_1['hydro.prim.util_u_2'][z_idx, y_idx, :]

    rho_2 = data_2['hydro.prim.rho'][z_idx, y_idx, :]
    uy_2 = data_2['hydro.prim.util_u_2'][z_idx, y_idx, :]

    # Plot Density on the left axis (solid line)
    ax1.plot(x_1d, rho_1*rho_to_cgs, color='navy', linestyle='-', linewidth=2, label=rf'$\rho$ (t={time_val_1*M_to_ms:.1f} ms)')
    ax1.plot(x_1d, rho_2*rho_to_cgs, color='navy', linestyle='-.', linewidth=2, label=rf'$\rho$ (t={time_val_2*M_to_ms:.1f} ms)')
    
    # Plot Velocity on the right axis (dashed line)
    ax2.plot(x_1d, uy_1, color='red', linestyle='-', linewidth=2, label=rf'$u_y$ (t={time_val_1*M_to_ms:.1f} ms)')
    ax2.plot(x_1d, uy_2, color='red', linestyle='-.', linewidth=2, label=rf'$u_y$ (t={time_val_2*M_to_ms:.1f} ms)')

    # 3. Formatting Left Axis (Density)
    ax1.set_xlabel(r'$x$ (Code Units)', fontsize=12)
    ax1.set_ylabel(r'$\rho \left(\rm{g/cm^3}\right)$ ', fontsize=12, color='navy')
    ax1.set_yscale('log')
    # ax1.set_ylim(1e-12, 1e-3)  # Adjust based on your SFHo central density
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.grid(True, alpha=0.3)
    
    # 4. Formatting Right Axis (Velocity)
    ax2.set_ylabel(r'$u_y / c$', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=9, ncol=2)

    ax1.set_xlim(-40.0, 40.0)

    # add minor ticks to both axes
    ax1.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*1.0, numticks=30))
    ax2.minorticks_on()

    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=False)
    ax2.tick_params(axis='y', which='both', direction='in')

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/1d_profile_rho_uy_x_axis.png", dpi=600)
    plt.close()

def main():
    output_path = '/home/mi58rip/gr-athena/runs/rns_test_SFHo/outputs/test_SMR'
    # indices = [0, 136, 272, 407]

    # plot_rho_equator(output_path, indices)
    # plot_vel_equator(output_path, [407])
    # plot_rho_max_vs_time(output_path)
    plot_mass_bar_vs_time(output_path)
    # plot_rho_u_y_along_x(output_path, 0, 407)

if __name__ == "__main__":
    main()