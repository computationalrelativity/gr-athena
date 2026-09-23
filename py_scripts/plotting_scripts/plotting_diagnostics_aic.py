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

# mass to CGS
mass_to_cgs = 1.989e33  # solar mass in grams

# code units to km factor
code_to_km = 1.47662504

code_to_cm = code_to_km * 1e5

M_to_s = M_to_ms * 1e-3

c_cgs = code_to_cm / M_to_s

press_to_cgs = rho_to_cgs * (c_cgs)**2  

energy_to_cgs = mass_to_cgs * (c_cgs)**2

# 1. Correct Cartesian geometry functions matching the athena_read.py API
# face_func requires 4 arguments: xmin, xmax, ratio, and number of points
f_func = lambda xmin, xmax, xrat, nx: np.linspace(xmin, xmax, nx)

# center_func requires 2 arguments: left face (xm) and right face (xp)
c_func = lambda xm, xp: 0.5 * (xm + xp)

# vol_func requires 6 arguments: the min and max faces for all 3 dimensions
v_func = lambda x1m, x1p, x2m, x2p, x3m, x3p: (x1p - x1m) * (x2p - x2m) * (x3p - x3m)

def plot_from_hst_vs_time(variables, output_path, PLOT_DIR):

    data = np.loadtxt(rf"{output_path}/aic.hst", comments='#')
    times = data[:, 0]

    max_rhos = data[:, 23]
    mass = data[:, 3]
    mass_per_cell = data[:, 47]
    E_kin = data[:, 28]
    min_alpha = data[:, 45]

    if "rho" in variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times*M_to_ms, max_rhos*rho_to_cgs, color='navy')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(r'$\rho_{\rm max} $ (g/cm$^3$)', fontsize=12)
        ax.set_title('Max Density', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        # ax.set_yscale("log")
        # ax.set_ylim(-0.1e14, 1e14) 

        ax.set_xbound([80, 102])
        plt.tight_layout()
        plt.savefig(rf"{PLOT_DIR}/rho_max_vs_time.png", dpi=600)
        plt.close(fig)

    if "alpha" in variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times*M_to_ms, min_alpha, color='navy')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(r'$\alpha_{\rm min}$', fontsize=12)
        ax.set_title('Min Lapse', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(rf"{PLOT_DIR}/alpha_min_vs_time.png", dpi=600)
        plt.close(fig)
    
    if "mass" in variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times*M_to_ms, mass, color='navy')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(r'$M_{\rm b}$', fontsize=12)
        ax.set_title('Barionic Mass', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(rf"{PLOT_DIR}/mass_bar_vs_time.png", dpi=600)
        plt.close(fig)

    if "mass_per_cell" in variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times*M_to_ms, mass_per_cell, color='navy')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(r'$M_{\rm b}$ per cell', fontsize=12)
        ax.set_title('Barionic Mass per cell', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(rf"{PLOT_DIR}/mass_per_cell_vs_time.png", dpi=600)
        plt.close(fig)

    if "E_kin" in variables:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times*M_to_ms, E_kin*energy_to_cgs, color='navy')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(r'$E_{\rm kin} $ (erg)', fontsize=12)
        ax.set_title('Kinetic Energy', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(rf"{PLOT_DIR}/kinetic_energy_vs_time.png", dpi=600)
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
            'log': True, 
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

def plot_hydro_equator_2d(variables, output_path, time_idxs, PLOT_DIR):
    
    # 1. Configuration dictionary for 2D specific formatting
    var_format = {
        'hydro.prim.rho': {
            'cbar_label': r'$\rho \left(\rm{g/cm^3}\right)$', 
            'scale': rho_to_cgs, 
            'cmap': 'magma',
            'log': True, 
            'diverging': False,
            'folder': 'rho_equator',
            'title_sym': r'$\rho$'
        },
        'hydro.prim.p': {
            'cbar_label': r'$P \left(\rm{dyn/cm^2}\right)$', 
            'scale': press_to_cgs,
            'cmap': 'viridis',
            'log': True, 
            'diverging': False,
            'folder': 'pressure_equator',
            'title_sym': r'$P$'
        },
        'hydro.aux.T': {
            'cbar_label': r'$T \left(\rm{MeV}\right)$', 
            'scale': 1.0, 
            'cmap': 'inferno',
            'log': True, 
            'diverging': False,
            'folder': 'temp_equator',
            'title_sym': r'$T$'
        },
        'passive_scalar.r_0': {
            'cbar_label': r'$Y_e$', 
            'scale': 1.0, 
            'cmap': 'RdYlBu',
            'log': False, 
            'diverging': False,
            'folder': 'ye_equator',
            'title_sym': r'$Y_e$'
        },
        'velocity_r': {
            'cbar_label': r'Radial Velocity $u_r$', 
            'scale': 1.0, 
            'cmap': 'RdBu_r',  # Diverging colormap
            'log': False, 
            'diverging': True, # Triggers symmetric colorbar around 0
            'folder': 'vel_equator',
            'title_sym': r'$u_r$'
        }
    }

    # 2. Dependency Resolution
    ye_var = 'passive_scalar.r_0'
    
    # Determine exactly which variables need to be extracted from HDF5
    fetch_vars_3 = set(v for v in variables if v not in [ye_var, 'velocity_r'])
    fetch_vars_4 = set([ye_var]) if ye_var in variables else set()
    
    # If radial velocity is requested, we must fetch the underlying momentum/velocity components
    if 'velocity_r' in variables:
        fetch_vars_3.add('hydro.prim.util_u_1')
        fetch_vars_3.add('hydro.prim.util_u_2')
        
    fetch_vars_3 = list(fetch_vars_3)
    fetch_vars_4 = list(fetch_vars_4)

    # 3. Iterate ONLY over the explicitly requested time indices
    for time_idx in time_idxs:
        file_3 = f"{output_path}/aic.out3.{time_idx:05d}.athdf"
        
        # Read out3 variables
        data_3 = athena_read.athdf(
            file_3, quantities=fetch_vars_3,
            vol_func=v_func, face_func_1=f_func, face_func_2=f_func, face_func_3=f_func,
            center_func_1=c_func, center_func_2=c_func, center_func_3=c_func, raw=True
        )

        with h5py.File(file_3, 'r') as f:
            time_val = f.attrs['Time']

        # Read out4 variables (Ye) if required
        data_4 = None
        if fetch_vars_4:
            file_4 = f"{output_path}/aic.out4.{time_idx:05d}.athdf"
            data_4 = athena_read.athdf(
                file_4, quantities=fetch_vars_4,
                vol_func=v_func, face_func_1=f_func, face_func_2=f_func, face_func_3=f_func,
                center_func_1=c_func, center_func_2=c_func, center_func_3=c_func, raw=True
            )

        num_blocks = data_3['x1f'].shape[0]
        slices_2d = {var: [] for var in variables}
        
        # 4. Extract Equatorial Slices (Z=0)
        for b in range(num_blocks):
            z_min, z_max = data_3['x3f'][b, 0], data_3['x3f'][b, -1]
            
            # Filter: Only process MeshBlocks that straddle the equator (z=0)
            if z_min <= 0.0 <= z_max:
                k = np.argmin(np.abs(data_3['x3v'][b, :] - 0.0))
                
                # Face coordinates for rendering
                x_faces = data_3['x1f'][b, :] * code_to_km
                y_faces = data_3['x2f'][b, :] * code_to_km
                X, Y = np.meshgrid(x_faces, y_faces)
                
                for var in variables:
                    if var == ye_var:
                        slice_data = data_4[var][b, k, :, :]
                    elif var == 'velocity_r':
                        # Compute derived radial velocity for this block
                        ux = data_3['hydro.prim.util_u_1'][b, k, :, :]
                        uy = data_3['hydro.prim.util_u_2'][b, k, :, :]
                        
                        # Center coordinates for vector math
                        xv, yv = np.meshgrid(data_3['x1v'][b, :], data_3['x2v'][b, :])
                        r = np.sqrt(xv**2 + yv**2)
                        r[r == 0] = 1e-10  # Prevent division by zero
                        
                        slice_data = (xv * ux + yv * uy) / r
                    else:
                        slice_data = data_3[var][b, k, :, :]
                        
                    slices_2d[var].append((X, Y, slice_data))

        # 5. Render and Save Plots (Into variable-specific subdirectories)
        for var in variables:
            fmt = var_format.get(var, {
                'cbar_label': var, 'scale': 1.0, 'cmap': 'viridis', 'log': False, 
                'diverging': False, 'folder': f'{var.replace(".", "_")}_equator', 'title_sym': var
            })
            
            # Create specific subdirectory for this variable
            var_dir = f"{PLOT_DIR}/{fmt['folder']}"
            os.makedirs(var_dir, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(8, 7))
            
            # Calculate global min/max across all MeshBlocks
            all_data = np.concatenate([s[2].flatten() for s in slices_2d[var]]) * fmt['scale']
            
            if fmt['diverging']:
                # Symmetric limits centered exactly on 0
                vmax = max(np.nanmax(np.abs(all_data)), 1e-10) # 1e-10 prevents vmin=vmax=0 collapse
                norm = Normalize(vmin=-vmax, vmax=vmax)
            elif fmt['log']:
                all_data_positive = all_data[all_data > 0]
                vmin = np.nanmin(all_data_positive) if len(all_data_positive) > 0 else 1e-10
                vmax = np.nanmax(all_data)
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                vmin, vmax = np.nanmin(all_data), np.nanmax(all_data)
                norm = Normalize(vmin=vmin, vmax=vmax)

            # Render each MeshBlock natively
            mesh = None
            for X, Y, slice_data in slices_2d[var]:
                scaled_data = slice_data * fmt['scale']
                mesh = ax.pcolormesh(X, Y, scaled_data, cmap=fmt['cmap'], norm=norm, 
                                     shading='flat', edgecolors='none', rasterized=True)

            # Formatting
            ax.set_aspect('equal')
            ax.set_xlabel(r'$x$ (km)', fontsize=12)
            ax.set_ylabel(r'$y$ (km)', fontsize=12)
            ax.set_title(rf"{fmt['title_sym']} Equatorial Plane ($z=0$) | $t={time_val*M_to_ms:.2f}$ ms", fontsize=14)
            
            ax.set_xlim(-250, 250)
            ax.set_ylim(-250, 250)

            # Setup Colorbar
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(fmt['cbar_label'], fontsize=12)
            
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            
            plt.tight_layout()
            
            # Save inside the dynamically created subdirectory
            filename = f"{var_dir}/{fmt['folder']}_{time_idx:05d}.png"
            plt.savefig(filename, dpi=300) 
            plt.close(fig)

def check_min_max_grid_spacing_x_y_z(output_path, time_idxs):
    """
    Calculates the minimum and maximum grid spacings (dx, dy, dz) for Athena++ 
    AMR grids by extracting the face coordinates using athena_read.
    """
    for time_idx in time_idxs:
        file_path = f"{output_path}/aic.out3.{time_idx:05d}.athdf"
        
        try:
            # We don't need any hydro quantities, just the grid geometry.
            # raw=True ensures we bypass any coordinate transformation logic.
            data = athena_read.athdf(file_path, quantities=[], raw=True)
            with h5py.File(file_path, 'r') as f:
                time_val = f.attrs['Time']
            # athena_read returns face arrays of shape [num_blocks, num_faces_per_block]
            # Spacing is constant within a single block, so we just subtract the 
            # first face from the second face for all blocks simultaneously.
            dx_blocks = data['x1f'][:, 1] - data['x1f'][:, 0]
            dy_blocks = data['x2f'][:, 1] - data['x2f'][:, 0]
            dz_blocks = data['x3f'][:, 1] - data['x3f'][:, 0]
            
            # Find the global min and max spacing across all blocks
            dx_min, dx_max = np.min(dx_blocks), np.max(dx_blocks)
            dy_min, dy_max = np.min(dy_blocks), np.max(dy_blocks)
            dz_min, dz_max = np.min(dz_blocks), np.max(dz_blocks)
            
            # Extract active AMR levels for context
            levels = data['Levels']
            l_min, l_max = np.min(levels), np.max(levels)
            
            print(f"--- Time: {time_val*M_to_ms:1.5e} ms (Levels {l_min} to {l_max}) ---")
            print(f"  dx: min = {dx_min*code_to_km:1.5e} km, max = {dx_max*code_to_km:1.5e} km")
            print(f"  dy: min = {dy_min*code_to_km:1.5e} km, max = {dy_max*code_to_km:1.5e} km")
            print(f"  dz: min = {dz_min*code_to_km:1.5e} km, max = {dz_max*code_to_km:1.5e} km\n")
            
        except FileNotFoundError:
            print(f"Error: Could not find {file_path}\n")
        except Exception as e:
            print(f"Error processing {file_path}: {e}\n")

def mass_of_star(output_path, time_idxs, RHO_CUTOFF=1e11):
    """
    OBLATE_FACTOR: Accounts for rotational flattening. 
    1.0 = perfect sphere. ~0.75-0.85 is typical for rapidly rotating AIC cores.
    """
    
    for time_idx in time_idxs:
        filepath = f"{output_path}/aic.out3.{time_idx:05d}.athdf"
        with h5py.File(filepath, 'r') as f:
            time_val = f.attrs['Time']
            
        data = athena_read.athdf(filepath, quantities=['hydro.prim.rho'], raw=True)
        num_blocks = data['x1f'].shape[0]
        
        r_list, rho_list, dA_list = [], [], []
        
        # 1. Extract every cell's radius, density, and exact 2D area
        for b in range(num_blocks):
            z_min, z_max = data['x3f'][b, 0], data['x3f'][b, -1]
            
            if z_min <= 0.0 <= z_max:
                k = np.argmin(np.abs(data['x3v'][b, :] - 0.0))
                
                # Physical cell dimensions
                dx = (data['x1f'][b, 1:] - data['x1f'][b, :-1]) * code_to_cm
                dy = (data['x2f'][b, 1:] - data['x2f'][b, :-1]) * code_to_cm
                DX, DY = np.meshgrid(dx, dy)
                cell_area = DX * DY
                
                # Center coordinates
                xv, yv = np.meshgrid(data['x1v'][b, :], data['x2v'][b, :])
                r_slice = np.sqrt((xv * code_to_cm)**2 + (yv * code_to_cm)**2)
                
                rho_slice = data['hydro.prim.rho'][b, k, :, :] * rho_to_cgs
                
                r_list.append(r_slice.flatten())
                rho_list.append(rho_slice.flatten())
                dA_list.append(cell_area.flatten())

        r_flat = np.concatenate(r_list)
        rho_flat = np.concatenate(rho_list)
        dA_flat = np.concatenate(dA_list)

        mask = rho_flat >= RHO_CUTOFF
        r_core = r_flat[mask]
        rho_core = rho_flat[mask]
        dA_core = dA_flat[mask]

        if len(rho_core) == 0:
            print(f"Time {time_val*M_to_ms:.4f}: No matter found above {RHO_CUTOFF:.1e} g/cm^3")
            continue

        # 2. Area-Weighted Radial Binning (Eliminates grid noise)
        num_bins = 200
        r_bins = np.linspace(0, np.max(r_core), num_bins)
        dr = r_bins[1] - r_bins[0]
        r_mid = r_bins[:-1] + dr / 2

        # Sum the (density * area) in each bin, and the total area in each bin
        mass_2d, _ = np.histogram(r_core, bins=r_bins, weights=rho_core * dA_core)
        area_2d, _ = np.histogram(r_core, bins=r_bins, weights=dA_core)
        
        valid = area_2d > 0
        rho_1d = np.zeros_like(r_mid)
        rho_1d[valid] = mass_2d[valid] / area_2d[valid] # True average density of the ring

        # 3. Spherical Integration with Oblate Correction
        # dV = 4 * pi * r^2 * dr
        dV = 4.0 * np.pi * (r_mid**2) * dr
        mass_grams = np.sum(rho_1d * dV)
        mass_msun = mass_grams / mass_to_cgs

        print(f"Time {time_val*M_to_ms:.4f} | Core Mass (> {RHO_CUTOFF:.0e}): {mass_msun:.4f} M_sun")

def compare_rho_max_evolution(output_path_1, output_path_2, PLOT_DIR):
    data_1 = np.loadtxt(rf"{output_path_1}/aic.hst", comments='#')
    times_1 = data_1[:, 0]
    
    data_2 = np.loadtxt(rf"{output_path_2}/aic.hst", comments='#')
    times_2 = data_2[:, 0]

    max_rhos_1 = data_1[:, 23]
    max_rhos_2 = data_2[:, 23]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times_1*M_to_ms, max_rhos_1*rho_to_cgs, color='navy', label='SFHo')
    ax.plot(times_2*M_to_ms, max_rhos_2*rho_to_cgs, color='red', label='LS220')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel(r'$\rho_{\rm max} $ (g/cm$^3$)', fontsize=12)
    ax.set_title('Max Density', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    # ax.set_yscale("log")
    # ax.set_ylim(-0.1e14, 1e14) 

    ax.legend()
    ax.set_xbound([90, 105])
    plt.tight_layout()
    plt.savefig(rf"{PLOT_DIR}/compare_rho_max_vs_time.png", dpi=600)
    plt.close(fig)

def main():
    output_path = '/home/mi58rip/gr-athena/runs/aic_SFHo_64_64_64/test_rewrite_ID_AMR_M1'
    PLOT_DIR = '/home/mi58rip/gr-athena/plots/aic_SFHo_64_64_64_test_rewrite_ID_AMR_M1'

    # Create the output directory if it doesn't exist, if exists, then clean it
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    else:
        # Clean the directory
        for f in os.listdir(PLOT_DIR):
            file_path = os.path.join(PLOT_DIR, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    all_outputs = sorted([int(f.split('.')[2]) for f in os.listdir(output_path) if f.startswith('aic.out3.') and f.endswith('.athdf')])
    indices = all_outputs #[all_outputs[i] for i in np.linspace(0, len(all_outputs)-1, 18, dtype=int)]
    #print(np.array(indices))
    #print(np.array(indices) * 100 * M_to_ms)
    
    #plot_from_hst_vs_time(["rho", "alpha", "mass", "mass_per_cell", "E_kin"], output_path, PLOT_DIR)
    # plot_from_hst_vs_time(["rho"], output_path, PLOT_DIR)
    # plot_from_hst_vs_time(["mass_per_cell"], output_path, PLOT_DIR)

    # indices = [186, 192, 197]
    #check_min_max_grid_spacing_x_y_z(output_path, indices)

    #plot_hydro_along_x(["hydro.prim.rho", "hydro.prim.p", "hydro.aux.T", "passive_scalar.r_0"], output_path, indices, PLOT_DIR)
    plot_hydro_equator_2d(["hydro.prim.rho", "hydro.prim.p", "hydro.aux.T", "passive_scalar.r_0", "velocity_r"], output_path, indices, PLOT_DIR)

    #mass_of_star(output_path, indices, RHO_CUTOFF=1e11)

    # compare_rho_max_evolution('/home/mi58rip/gr-athena/runs/aic_SFHo_64_64_64/test_rewrite_ID_AMR', 
    #                           '/home/mi58rip/gr-athena/runs/aic_LS220_64_64_64/test_rewrite_ID_AMR', 
    #                           '/home/mi58rip/gr-athena/plots/aic')
    
if __name__ == "__main__":
    main()
