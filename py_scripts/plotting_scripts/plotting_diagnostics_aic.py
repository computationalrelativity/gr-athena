import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('/home/mi58rip/gr-athena/vis/python')
import athena_read

# M to ms factor
M_to_ms = 4.92549095e-6 * 1e3

# rho to CGS units factor [g/cm^3]
rho_to_cgs = 6.177e17

# code units to km factor
code_to_km = 1.47662504

# pressure to CGS units factor [dyne/cm^2]
press_to_cgs = rho_to_cgs * (code_to_km / M_to_ms)**2  # P = rho * c^2, where c = code_to_km / M_to_ms

# 1. Correct Cartesian geometry functions matching the athena_read.py API
# face_func requires 4 arguments: xmin, xmax, ratio, and number of points
f_func = lambda xmin, xmax, xrat, nx: np.linspace(xmin, xmax, nx)

# center_func requires 2 arguments: left face (xm) and right face (xp)
c_func = lambda xm, xp: 0.5 * (xm + xp)

# vol_func requires 6 arguments: the min and max faces for all 3 dimensions
v_func = lambda x1m, x1p, x2m, x2p, x3m, x3p: (x1p - x1m) * (x2p - x2m) * (x3p - x3m)

def plot_rho_equator(output_path, indices, PLOT_DIR):
    """
    Plots the equatorial density (rho) from a given Athena++ output file.

    Parameters:
    - output_path: str, path to the output directory
    - indices: list of integers, the indices of the output files to plot
    - PLOT_DIR: str, path to the plot directory
    """

    PLOTS_DIR = f"{PLOT_DIR}/rho_eq_plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for index in indices:
        filepath = f"{output_path}/aic.out3.{index:05d}.athdf"

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
        # print(z_coords)
        z_idx = np.argmin(np.abs(z_coords - 0.0))

        # 4. Extract the 2D density array and face-centered (f) bounding coordinates
        rho_equator = data['hydro.prim.rho'][z_idx, :, :]
        x = data['x1f']
        y = data['x2f']

        # 5. Render the Plot
        fig, ax = plt.subplots(figsize=(6, 5))

        mesh = ax.pcolormesh(
            x*code_to_km, y*code_to_km, rho_equator, 
            cmap='magma', 
            norm=LogNorm(vmin=1e-10, vmax=rho_equator.max())
        )

        # Formatting
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(mesh, cax=cax)
        cbar.set_label(r'Rest Mass Density $\rho$', fontsize=12)

        # ax.set_xlim(-20.0, 20.0)
        # ax.set_ylim(-20.0, 20.0)
        ax.set_xlabel(r'$x$ (km)', fontsize=12)
        ax.set_ylabel(r'$y$ (km)', fontsize=12)
        ax.set_title(rf'$t={time_val*M_to_ms:.3f}$ ms', fontsize=14)

        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/rho_eq_t{time_val:.3f}.png', dpi=600)
        plt.close(fig)

def plot_vel_equator(output_path, indices, PLOT_DIR):
    """
    Plots the equatorial velocity from a given Athena++ output file.

    Parameters:
    - output_path: str, path to the output directory
    - indices: list of integers, the indices of the output files to plot
    - PLOT_DIR: str, path to the plot directory
    """

    PLOTS_DIR = f"{PLOT_DIR}/vel_eq_plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for index in indices:
        filepath = f"{output_path}/aic.out3.{index:05d}.athdf"

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
        # find the max positive and negative values of ur, and set the vmin and vmax to be symmetric around zero
        vmax = np.max(np.abs(ur))
        vmin = -vmax
        mesh = ax.pcolormesh(
            data['x1f']*code_to_km, data['x2f']*code_to_km, ur, 
            cmap='RdBu_r',
            norm=Normalize(vmin=vmin, vmax=vmax)
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        cbar = plt.colorbar(mesh, cax=cax)
        cbar.set_label(r'Radial Velocity $u_r$', fontsize=12)
        # control the height of the bar
        cbar.ax.set_ylim(-vmax, vmax)

        # ax.set_xlim(-20.0, 20.0)
        # ax.set_ylim(-20.0, 20.0)
        ax.set_xlabel(r'$x$ (km)', fontsize=12)
        ax.set_ylabel(r'$y$ (km)', fontsize=12)
        ax.set_title(rf'$t={time_val*M_to_ms:.3f}$ ms', fontsize=14)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/vel_eq_t{time_val:.3f}.png', dpi=600)
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

def plot_from_hst_vs_time(output_path, PLOT_DIR):

    data = np.loadtxt(rf"{output_path}/aic.hst", comments='#')
    times = data[:, 0]

    max_rhos = data[:, 23]
    mass = data[:, 3]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times*M_to_ms, max_rhos*rho_to_cgs, color='navy')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel(r'$\rho_{\rm max} $ (g/cm$^3$)', fontsize=12)
    ax.set_title('Max Density', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    # ax.set_ylim(-0.1e14, 1e14) 

    plt.tight_layout()
    plt.savefig(rf"{PLOT_DIR}/rho_max_vs_time.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times*M_to_ms, mass, color='navy')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel(r'$M_{\rm b}$', fontsize=12)
    ax.set_title('Barionic Mass', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(rf"{PLOT_DIR}/mass_bar_vs_time.png", dpi=600)
    plt.close(fig)

def plot_hydro_along_x(variables, output_path, time_idxs, PLOT_DIR):
    
    # 1. Configuration dictionary for variable-specific formatting
    var_format = {
        'hydro.prim.rho': {
            'ylabel': r'$\rho \left(\rm{g/cm^3}\right)$', 
            'scale': rho_to_cgs, 
            'log': True, 
            'filename': 'rho_x_axis.png',
            'label_sym': r'$\rho$'
        },
        'hydro.prim.p': {
            'ylabel': r'$P \left(\rm{dyn/cm^2}\right)$', 
            'scale': press_to_cgs,  # Ensure press_to_cgs is defined in your script
            'log': True, 
            'filename': 'pressure_x_axis.png',
            'label_sym': r'$P$'
        },
        'hydro.aux.T': {
            'ylabel': r'$T \left(\rm{MeV}\right)$', 
            'scale': 1.0, 
            'log': False, 
            'filename': 'temp_x_axis.png',
            'label_sym': r'$T$'
        },
        'passive_scalar.r_0': {
            'ylabel': r'$Y_e$', 
            'scale': 1.0, 
            'log': False, 
            'filename': 'ye_x_axis.png',
            'label_sym': r'$Y_e$'
        }
    }

    # 2. Dependency Resolution
    ye_var = 'passive_scalar.r_0'
    rho_var = 'hydro.prim.rho'
    has_ye = ye_var in variables
    
    # If Ye is requested, we MUST extract rho to plot the phase diagram, 
    # even if the user didn't ask for a spatial rho plot.
    extract_vars = list(variables)
    if has_ye and rho_var not in extract_vars:
        extract_vars.append(rho_var)

    # Segregate variables by their source file streams
    out3_vars = [v for v in extract_vars if v != ye_var]
    out4_vars = [v for v in extract_vars if v == ye_var]

    extracted_data = {var: [] for var in extract_vars}
    colors = plt.cm.inferno(np.linspace(0, 0.8, len(time_idxs)))

    # 3. Single-Pass Data Extraction
    for time_idx in time_idxs:
        file_3 = f"{output_path}/aic.out3.{time_idx:05d}.athdf"
        
        # Read out3 variables
        data_3 = athena_read.athdf(
            file_3, quantities=out3_vars,
            vol_func=v_func, face_func_1=f_func, face_func_2=f_func, face_func_3=f_func,
            center_func_1=c_func, center_func_2=c_func, center_func_3=c_func, raw=True
        )

        with h5py.File(file_3, 'r') as f:
            time_val = f.attrs['Time']

        # Read out4 variables (Ye) if required
        data_4 = None
        if out4_vars:
            file_4 = f"{output_path}/aic.out4.{time_idx:05d}.athdf"
            data_4 = athena_read.athdf(
                file_4, quantities=out4_vars,
                vol_func=v_func, face_func_1=f_func, face_func_2=f_func, face_func_3=f_func,
                center_func_1=c_func, center_func_2=c_func, center_func_3=c_func, raw=True
            )

        num_blocks = data_3[out3_vars[0]].shape[0]
        
        x_pts_tmp = []
        var_pts_tmp = {var: [] for var in extract_vars}

        # Iterate through MeshBlocks and filter for the x-axis
        for b in range(num_blocks):
            y_min, y_max = data_3['x2f'][b, 0], data_3['x2f'][b, -1]
            z_min, z_max = data_3['x3f'][b, 0], data_3['x3f'][b, -1]

            if (y_min <= 0.0 <= y_max) and (z_min <= 0.0 <= z_max):
                j = np.argmin(np.abs(data_3['x2v'][b, :] - 0.0))
                k = np.argmin(np.abs(data_3['x3v'][b, :] - 0.0))
                
                x_pts_tmp.extend(data_3['x1v'][b, :])
                
                # Extract from out3
                for var in out3_vars:
                    var_pts_tmp[var].extend(data_3[var][b, k, j, :])
                
                # Extract from out4
                for var in out4_vars:
                    var_pts_tmp[var].extend(data_4[var][b, k, j, :])

        # Sort spatially and store
        if len(x_pts_tmp) > 0:
            x_pts_tmp = np.array(x_pts_tmp)
            sort_idx = np.argsort(x_pts_tmp)
            x_pts_sorted = x_pts_tmp[sort_idx]

            for var in extract_vars:
                y_pts_sorted = np.array(var_pts_tmp[var])[sort_idx]
                extracted_data[var].append((time_val, x_pts_sorted, y_pts_sorted))

    # 4. Plot 1D Spatial Profiles (Only for explicitly requested variables)
    for var in variables:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        fmt = var_format.get(var, {
            'ylabel': var, 'scale': 1.0, 'log': False, 
            'filename': f'{var.replace(".", "_")}_x_axis.png', 'label_sym': var
        })

        for i, (time_val, x_pts, y_pts) in enumerate(extracted_data[var]):
            color = colors[i]
            
            ax.plot(x_pts * code_to_km, y_pts * fmt['scale'], 
                    color=color, linestyle='-', marker='.', markersize=4, linewidth=1.5, 
                    label=rf"{fmt['label_sym']} (t={time_val*M_to_ms:.1f} ms)")

        ax.set_xlabel(r'$x$ (km)', fontsize=12)
        ax.set_ylabel(fmt['ylabel'], fontsize=12, color='black')
        
        if fmt['log']:
            ax.set_yscale('log')
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*1.0, numticks=30))

        ax.tick_params(axis='y', labelcolor='black')
        ax.grid(True, alpha=0.3)

        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best', fontsize=9, ncol=2)

        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=False)
        ax.tick_params(axis='y', which='both', direction='in')

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{fmt['filename']}", dpi=600)
        plt.close()

    # 5. Specialized Diagnostic: Ye vs Rho Phase Space Plot
    if has_ye:
        fig_rho, ax_rho = plt.subplots(figsize=(8, 5))
        
        for i, time_idx in enumerate(time_idxs):
            color = colors[i]
            time_val = extracted_data[ye_var][i][0]
            
            # Fetch the pre-sorted (by x) arrays
            rho_pts = extracted_data[rho_var][i][2]
            ye_pts  = extracted_data[ye_var][i][2]
            
            # Re-sort them relative to density for a clean phase-space curve
            sort_rho = np.argsort(rho_pts)
            
            ax_rho.plot(rho_pts[sort_rho] * rho_to_cgs, ye_pts[sort_rho], 
                        color=color, linestyle='-', marker='.', markersize=4, 
                        label=rf"$t=${time_val*M_to_ms:.1f} ms")

        ax_rho.set_xlabel(r'$\rho \left(\rm{g/cm^3}\right)$', fontsize=12)
        ax_rho.set_ylabel(r'$Y_e$', fontsize=12)
        ax_rho.set_xscale('log')
        ax_rho.grid(True, alpha=0.3)
        ax_rho.legend(loc='best', fontsize=9, ncol=2)
        ax_rho.tick_params(axis='both', direction='in', top=True, right=True)
        
        fig_rho.tight_layout()
        fig_rho.savefig(f"{PLOT_DIR}/ye_vs_rho.png", dpi=600)
        plt.close(fig_rho)

def main():
    # output_path = '/home/mi58rip/gr-athena/runs/aic_SFHo_64_64_64/test3_3'
    # PLOT_DIR = '/home/mi58rip/gr-athena/plots/aic_SFHo_64_64_64_test3_3'

    output_path = '/home/mi58rip/gr-athena/runs/aic_LS220_64_64_64/testnone'
    PLOT_DIR = '/home/mi58rip/gr-athena/plots/aic_LS220_64_64_64_testnone'

    os.makedirs(PLOT_DIR, exist_ok=True)

    all_outputs = sorted([int(f.split('.')[2]) for f in os.listdir(output_path) if f.startswith('aic.out3.') and f.endswith('.athdf')])
    indices = [all_outputs[i] for i in np.linspace(0, len(all_outputs)-2, 4, dtype=int)]
    print(np.array(indices) * 100 / 203)
    
    plot_from_hst_vs_time(output_path, PLOT_DIR)

    #indices = [0, 4, 8, 12, 16, 20, 25]
    plot_hydro_along_x(["hydro.prim.rho", "hydro.prim.p", "hydro.aux.T", "passive_scalar.r_0"], output_path, indices, PLOT_DIR)
    #plot_hydro_along_x(["hydro.prim.rho", "passive_scalar.r_0"], output_path, indices, PLOT_DIR)
    

if __name__ == "__main__":
    main()
