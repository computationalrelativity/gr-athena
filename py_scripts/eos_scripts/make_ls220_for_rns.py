import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

C_CGS = 2.99792458e10
MEV_FM3_TO_DYNE = 1.602176634e33                 
MEV_FM3_TO_G_CM3 = MEV_FM3_TO_DYNE / (C_CGS**2)  
FM3_TO_CM3 = 1e39
mn_CGS = 1.674927370796472e-24

LS220_table_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose.h5"

# 1. Load Data using CompOSE Conventions
with h5py.File(LS220_table_path, "r") as f:
    nb_fm3 = np.array(f['nb'])
    yq = np.array(f['yq'])
    t = np.array(f['t'])
    mn = f['mn'][()]  # Neutron mass in MeV
    Q1 = np.array(f['Q1'])  
    Q7 = np.array(f['Q7'])  

# Find indices for Ye = 0.5 and T = lowest (index 0)
idx_ye = np.argmin(np.abs(yq - 0.5))
idx_t = 0 

# Extract 1D slices for the Cold, Ye=0.5 White Dwarf core
Q1_1d = Q1[:, idx_ye, idx_t]
Q7_1d = Q7[:, idx_ye, idx_t]

# 2. Convert to Physical CGS units exactly as pycompose does
nb_1d = nb_fm3 * FM3_TO_CM3

# P = Q1 * nb
P_1d = Q1_1d * nb_fm3 * MEV_FM3_TO_DYNE

# Total Mass-Energy Density e = (1 + Q7) * mn * nb
e_1d = (1.0 + Q7_1d) * mn * nb_fm3 * MEV_FM3_TO_G_CM3

# Define a strict rest-mass density array for logarithmic interpolation targeting
rho_1d = nb_1d * mn_CGS

num_points = 2000
POLYTROPE_EXTENSION = True

if POLYTROPE_EXTENSION:
    # 3. Define True White Dwarf Polytrope parameters
    Gamma = 4.0 / 3.0
    rho_tr = 1e8 # Transition density in g/cm^3
    
    # Calculate K_P based on the transition density to maintain continuity
    P_tr = interp1d(rho_1d, P_1d)(rho_tr)
    K_P = P_tr / (rho_tr**Gamma)

    rho_min = rho_1d[0]
    rho_target = np.logspace(np.log10(rho_min), np.log10(rho_1d[-1]), num_points)
    
    P_target = np.zeros(num_points)
    nb_target = np.zeros(num_points)
    e_target = np.zeros(num_points)

    crust_mask = rho_target < rho_tr
    core_mask = ~crust_mask

    # Apply Crust Polytrope (safe mathematical vacuum for RNS)
    P_target[crust_mask] = K_P * (rho_target[crust_mask]**Gamma)
    nb_target[crust_mask] = rho_target[crust_mask] / mn_CGS
    # Thermodynamically consistent energy density for a polytrope: e = rho + P / ((Gamma-1)c^2)
    e_target[crust_mask] = rho_target[crust_mask] + P_target[crust_mask] / ((Gamma - 1.0) * C_CGS**2)

    # Interpolate Core (True LS220 Physics)
    P_target[core_mask] = interp1d(rho_1d, P_1d)(rho_target[core_mask])
    nb_target[core_mask] = interp1d(rho_1d, nb_1d)(rho_target[core_mask])
    e_target[core_mask] = interp1d(rho_1d, e_1d)(rho_target[core_mask])

    integrand = C_CGS**2 / (e_target * C_CGS**2 + P_target)
    H_target = cumulative_trapezoid(integrand, P_target, initial=0.0)

    H_target[0] = 1.0
    # P_target[0] = 1.0

else:
    rho_target = np.logspace(np.log10(rho_1d[0]), np.log10(rho_1d[-1]), num_points)

    P_target = interp1d(rho_1d, P_1d, bounds_error=False, fill_value="extrapolate")(rho_target)
    nb_target = interp1d(rho_1d, nb_1d, bounds_error=False, fill_value="extrapolate")(rho_target)
    e_target = interp1d(rho_1d, e_1d, bounds_error=False, fill_value="extrapolate")(rho_target)

    # Truncate: Force the lowest pressure point to exactly 0 to define the physical surface
    P_target -= P_target[0]

    # Integrate Enthalpy: H = int[ dP / (e * c^2 + P) ] * c^2
    integrand = C_CGS**2 / (e_target * C_CGS**2 + P_target)
    H_target = cumulative_trapezoid(integrand, P_target, initial=0.0)

    # Pseudo-Vacuum Anchor: RNS throws -inf on log10(0.0). Replace exact 0s with 1.0.
    # (In CGS, 1 dyne/cm^2 and 1 cm^2/s^2 are mathematically indistinguishable from 0 for the star)
    H_target[0] = 1.0
    P_target[0] = 1.0

# 5. Write directly to file
if POLYTROPE_EXTENSION:
    fname_out = f"/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose_{num_points}pt_poly_{rho_tr:.2e}_{rho_min:.2e}.rns"
else:
    fname_out = f"/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose_{num_points}pt.rns"
with open(fname_out, "w") as out:
    out.write(f"{num_points}\n")
    for i in range(num_points):
        # Format strictly follows: e_CGS, p_CGS, h_CGS, nb_CGS
        out.write(f"{e_target[i]:.15e} {P_target[i]:.15e} {H_target[i]:.15e} {nb_target[i]:.15e}\n")

print(f"Successfully generated {fname_out}")