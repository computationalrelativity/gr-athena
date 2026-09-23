<comment> # ===================================================================
# additional comments (such as e.g. compilation parameters go here)
# ...
# =============================================================================

<job> # =======================================================================
problem_id = aic
# =============================================================================

<z4c> # =======================================================================
lapse_harmonic  = 0.0
lapse_harmonicf = 1.0
lapse_oplog     = 2.0
lapse_advect    = 1.0
lapse_K         = 0.0

shift_Gamma       = 0.75     # T 2011: BNS takes 0.75
shift_eta         = 0.3      # T 2011
shift_alpha2Gamma = 0.0
shift_H           = 0.0
shift_advect      = 1.0

# Kreiss-Oliger dissipation parameter.
diss = 0.5 #0.1

# conformal factor flooring
chi_div_floor = 1e-5

# Constraint damping factors
damp_kappa1 = 0.02
damp_kappa2 = 0.0

# do not compute constraints outside this radius (relative to origin)
r_max_con = 256

#store_metric_drvts = true
communicate_aux_adm = true
#extended_aux_adm = true

# =============================================================================

<time> # ======================================================================
cfl_number   = 0.25
tlim	     = 500000
integrator   = rk3   # time integration algorithm

# spatial reconstruction - element of:
# {donate, lin_vl, lin_mc2, ppm, ceno3, weno5, weno5z, weno5d_si,
#  mp3, mp5, mp7, mp5_R}

xorder = weno5z
xorder_eps = 1e-40

xorder_use_fb = false             # control if fallback is used
xorder_use_fb_unphysical = false  # optionally verify (fluid) energy cond.
xorder_fb = lin_mc2

ncycle_out   = 1     # interval for stdout summary info
# =============================================================================

<hydro> # =====================================================================
rsolver               = llf

# Dummy variables
# gamma = 1.333333333333
# k_adi = 0.455

# EOS Table Path --------------------------------------------------------------
table                 = /home/mi58rip/gr-athena/eos_tables/SFHo.h5

# Tabulated Numerical Safeguards ----------------------------------------------
restrict_cs2          = true
warn_unrestricted_cs2 = false
max_cs_W              = 10
flux_table_limiter    = true
smooth_temperature    = true
recompute_temperature = true
n_max_factor          = 3

# PrimitiveSolver (C2P) Settings ----------------------------------------------
verbose               = false
c2p_acc               = 1e-12
max_iter              = 45
c2p_validate_density  = true
tighten_bracket       = true
use_toms_748          = false
max_W                 = 50.0

# Atmosphere & Floors ---------------------------------------------------------
dfloor                = 5e-15
dthreshold            = 1.00000001
tfloor                = 0.01
y0_atmosphere         = 0.5

# Hydro / Evolution Coupling --------------------------------------------------
use_split_grmhd_z4c   = false
flux_reconstruction   = false
# =============================================================================

<mesh> # ======================================================================
# no. cell in dir, min/max x, inner/outer BC
nx1    = 64
x1min  = -2048
x1max  = +2048
ix1_bc = gr_sommerfeld
ox1_bc = gr_sommerfeld

nx2    = 64
x2min  = -2048
x2max  = +2048
ix2_bc = gr_sommerfeld
ox2_bc = gr_sommerfeld

nx3    = 64
x3min  = -2048
x3max  = +2048
ix3_bc = gr_sommerfeld
ox3_bc = gr_sommerfeld

refinement = adaptive
numlevel = 12
#deref_count = 5

num_threads = 2

# impose bitant symmetry (use with ix3_bc=reflecting)
bitant = false #true
# =============================================================================

<meshblock> # =================================================================
nx1=16
nx2=16
nx3=16
# =============================================================================

<problem> # ===================================================================
filename  = rns_for_aic.par

use_ye_of_rho_table = true
Ye_rho_table     = /home/mi58rip/gr-athena/eos_tables/ye_of_rho_luis.dat

initial_Ye          = 0.5          # C/O White Dwarf electron fraction
rho_cut             = 1.0e3        # CGS density to truncate the WD atmosphere

# --- Magnetic Fields ---
bfield_type         = dipole       # "dipole" (ccsn) or "rns" (density-scaled)
B0_amp              = 0 #1.0e10       # Dipole max amplitude
B0_rad              = 10.0         # Dipole cutoff radius
# If bfield_type = "rns":
# pcut = 1.0e-5
# b_amp = 1.0e-3
# magindex = 1

# --- Deleptonization ---
M1_enabled              = false
deleptonization_method = Liebendoerfer  # "Liebendoerfer", "Simple", or "None"
update_conserved       = false
update_entropy         = true
E_nu_avg               = 10.0      # Average escaping neutrino energy (MeV)
rho_trap               = 2.0e12    # CGS density where neutrinos become trapped

# --- AMR ---
refinement_method   = MaxMassInCellTracker 
# delta_min_m         = 0.01         # Minimum cell mass to coarsen
delta_max_m         = 0.000125         # Maximum cell mass to trigger refinement

# --- Core Bounce Detection ---
detect_bounce            = true
bounce_detection_method  = maximum # "maximum", "local_maximum", or "entropy"
# D_max_steps_dec          = 3
# rat_D_max_threshold      = 0.001
bdm_rho_max              = 3.7e14  # CGS density trigger
equilibriate_post_bounce = true    # Set M1 equilibrium after bounce

pres_pert = 0.0
v_pert    = 0.0


id_floor_primitives = true

# Atmosphere settings
fatm = 1e-40 # set atmosphere rho to rho = fatm
fthr = 1     # set point to atmosphere if rho < fthr*fatm
# =============================================================================

<M1> # ========================================================================
ngroups  = 1
nspecies = 3

# element of:
# {approximate,
#  mixed,
#  exact_thin,
#  exact_thick,
#  exact_closure
# }
characteristics_variety = approximate

# element of:
# {LO,
#  HybridizeMinMod,
#  HybridizeMinModA,
#  HybridizeMinModC,
#  HybridizeMinModD}
flux_variety = HybridizeMinModC

# element of: {fluid, mixed, zero, none}
fiducial_velocity = fluid

# tolerances / ad-hoc fiddle parameters
fl_E = 1e-40
fl_J = 1e-40
fl_nG = 1e-40
eps_E = 1e-5
eps_J = 1e-10

# use mask for flux limiter control
flux_limiter_use_mask = false
# LO w/ nearest neighbours in flux limiter?
flux_limiter_nn = false
# Reduce order to lowest that is common over all flux components?
flux_limiter_multicomponent = false

# Use LO flux if: E < 0 \/ causality violated in a cell?
flux_lo_fallback_E = true
# Use LO flux if: nG < 0
flux_lo_fallback_nG = true

# fall-back checked on a per-species basis
flux_lo_fallback_species = true

# during fallback always force eql to take HO soln?
# WARNING: appears to introduce instability when true
flux_lo_fallback_eql_ho = false

# control whether flux lo is also performed on first substep?
# faster if false, but less robust
flux_lo_fallback_first_stage = true

# erase lo between stages or retain?
flux_lo_fallback_mask_reset_all_stages = true

# control whether to retain equilibrium dens.
retain_equilibrium = true

# used in EnforceCausality to prevent 0-div
eps_ec_fac = 1e-15

# Introduce a minimal diff. as F <- F_HO - Theta * (F_HO - F_LO)
min_flux_Theta = 0 # 1e-4

enforce_causality = true
enforce_finite = true

# control source couplings
couple_sources_ADM   = true
couple_sources_hydro = true
couple_sources_Y_e   = true

# [DEBUG] retain status of evolved points
ev_strat_status = false
# =============================================================================

<M1_closure> # ================================================================
verbose = false

# element of:
# {thin,
#  thick,
#  Minerbo, - Brent-Dekker (GSL based)
#  Kershaw
# }
variety = Minerbo

# element of:
# {none,       - Solver is noop
#  gsl_Brent,  - Brent-Dekker (GSL based)
#  gsl_Newton  - Newton (GSL based)
#  custom_NB   - Newton-Bisection (direct implementation)
#  custom_NAB  - Newton-Anderson-Bjorck (direct implementation)
#  custom_ONAB - Ostrowski - NAB fallback (direct implementation)
# }
method = custom_NAB

# dx of xi \in [0,1]
abs_tol = 1e-14

# magnitude of non-linear functional that is considered to be a root
fcn_tol = 1e-15

# if the solver is stagnating, should we propagate the solution or reset?
fail_on_npg = false

# during Newton iterate allow xi on [-delta,1+delta]
bnd_xi_delta = 0.1

eps_Z_o_E = 0   # If | Z_xi / E|_xi\in{0,1} | < eps, break-fast
fac_Z_o_E = 0   # Apply above if |xi-x_min,max| < fac
iter_max  = 32      # N.B.: 0.5 ^ 34 ~ 6e-11

# gsl_Newton specific
# If Newton fails, revert to Brent?
fallback_brent = true

fallback_thin = false  # on failures (i.e. bracket etc, revert to thin)

# MinerboN specific
use_Ostrowski = false  # Attempt 4th order Ostrowski?
use_Neighbor  = false  # take sc_xi(k,j,i-1) where possible as initial guess?

# MinerboP specific
iter_max_rst = 5     # maximum relaxation restarts allowed
w_opt_ini    = 1.0   # initial value of relaxation factor
fac_err_amp  = 1.11  # error amplification tolerance between iterates

# =============================================================================

<M1_solver> # =================================================================
verbose = false

# solver_REGIME is an element of:
# {do_nothing,                   - does... nothing, tada!
#  full_explicit,                - non-stiff regime
#  explicit_approximate_semi_implicit - approximate SI soln @ O(v/c)
#  semi_implicit_Hybrids         - GSL: FD approx. of J
#  semi_implicit_HybridsJ        - GSL: Full Jacobian
# }

# select evolution method to utilize in each regime
solver_non_stiff   = full_explicit
solver_stiff       = semi_implicit_custom_N
solver_scattering  = full_explicit
solver_equilibrium = semi_implicit_custom_N

# Given fixed grid-point: enforce common solver / source treatment?
# Should split this by species
solver_reduce_to_common = false

# Regardless of implicit (E, F_d) use explicit for nG?
solver_explicit_nG = false

# fallback to thick limit when tol not reached / no progress?
thick_tol = true
thick_npg = true

# Control whether / when equilibrium is set;
# If not set, above evolution choice is set with (S_sc_E, s_sp_F_d) = 0
#
# _enforce is for each step, _initial is only @ pmesh->time == 0
equilibrium_enforce = false
equilibrium_initial = false

# what to set for sources when reverting eql (true - full, false - nothing)
equilibrium_sources = false

# should be assume optical thickness at eql?
# may be more robust to set false here...
equilibrium_use_thick = false

# set eql directly for n, nG?
equilibrium_n_nG = false

# set eql directly for E, F_d? (too violent, evolve that instead)
equilibrium_E_F_d = false

# set sources for nG, E_F_d?
equilibrium_src_nG = true
equilibrium_src_E_F_d = false

# use differences or full expression for the srcs
equilibrium_use_diff_src = false

# control whether to evolve eql
equilibrium_evolve = false

# true: take Euler step; false: evolve fid. frame quantities
equilibrium_evolve_use_euler = false

# only enforce equilibrium above this density threshold (code units)
eql_rho_min = 1.619170040639225e-07 # 1.0e11 / 6.1760036e+17 [CGS->Code]

# only compute weak equilibrium above this density (code units)
tra_rho_min = 0 # 1e-8

# only enforce equilibrium above this temperature
eql_t_min = 0.1

# Solver-specific settings
eps_a_tol   = 1e-13
eps_r_tol   = 1e-6
iter_max  = 64

# seed initial guess?
use_Neighbor = false

# Additional limiters ---------------------------------------------------------
# flux fallback on unphysical fluid states
# Used with:
# M1/flux_lo_fallback_E = true
# M1/flux_lo_fallback_nG = true

# ignored if negative
flux_lo_fallback_tau_min = -10
flux_lo_fallback_Ye_min = -10.01
flux_lo_fallback_Ye_max = -10.6

# dampen decay in soln
limit_full_radiation = false
full_lim = -0.75

# source limiting (should help with stability / fidelity of results)
limit_src_fluid = true
limit_src_radiation = true

src_lim = 0.75
src_lim_Ye_min = 0.01
src_lim_Ye_max = 0.6

# not really a limiter but one way of eql enforcement (use wr instead)
src_lim_thick = -20.0

# Candidate state slope limiter / LO fallback:
# Inactive if negative; if used limit only E,nG otherwise too diffusive
fb_rat_sl_E = -0.25
fb_rat_sl_F_d = -0.25
fb_rat_sl_nG = -0.25
# =============================================================================

# =============================================================================
<bns_nurates>
# =============================================================================

# N.B. max quadrature points is BS_N_MAX/2 = 10 due to 2*n indexing in bns_nurates
n_quad_points_beta_nucleon_scat = 6
n_quad_points_pair_bremsstrahlung_lepton_scat = -1

use_abs_em = true
use_pair   = true
use_brem   = true
use_iso    = true
use_inelastic_scatt = false
use_WM_ab = false
use_WM_sc = false
use_NN_medium_corr = false

neglect_blocking = false
use_decay = false
use_BRT_brem = false

use_dU = false
dU = 0.0        # (MWE) 18.92714728;  Nucleon interaction potential difference (Un-Up) [MeV]

## const double mp_eff = 278.87162217;       # Proton effective mass [MeV]
## const double mn_eff = 280.16495513;       # Neutron effective mass [MeV]
## const double dm = mn_eff - mp_eff;        # Nucleon effective mass difference [MeV]

use_dm_eff = false
effective_mass_diff = 0.0

use_equilibrium_distribution = true

<M1_opacities> # ==============================================================
# element of: {bnsnurates, weakrates, photon, fake, zero, none}
variety = weakrates

tau_trap = 1.0
tau_delta = 1.0
max_correction_factor = 1.75

# Use fiducial frame for opac. correction? shouldn't matter..
correction_uses_fiducial_frame = true

# only allow corrections >= 1
correction_adjust_upward = false

# For NUX:
# correct emissivity first then derive kap?
correct_emissivity_nux = true

# correct kap first then derive emissivity?
correct_opacity_nux = false

# For WEAKRATES
#  M1_opacities/correct_emissivity_nux=true \
#  M1_opacities/correct_opacity_nux=false \

# For BNSNURATES
#  M1_opacities/correct_emissivity_nux=false \
#  M1_opacities/correct_opacity_nux=true \

# control whether trapped neutrinos get correction factor
correct_trapped = true

# control whether we recompute opacities for trapped regime
recompute_opacities_trapped = false

# control whether we recompute opacities for interpolated regime
recompute_opacities_interpolated = false

# flooring / table limit strategy ---------------------------------------------

# the following (code units) are cuts under which opacities are not computed
cut_dfloor = 1e-10
cut_rho_floor = 1.6191700468788603e-10
cut_tfloor = 0.1

# limits on various tabulated quantities (code units)
# likely values under-cut (above) are not usually encountered

# val / 6.1760036e+17 [CGS->Code]
min_rho = 1e-13
max_rho = 100

min_t = 0.1
max_t = 200

# can help to set lower limit slightly below table to allow WE
# to find solutions near bound.
min_ye = 1e-4  # 0.01
max_ye = 0.6

# only applied during equilibrium calculations
min_eql_yl = 1e-6

# optimally override with table limits:
limits_from_table = false

# take eos floors for the following:
min_rho_usefloor = false
min_t_usefloor = false

# apply limits internally within weakrates?
apply_table_limits_internally = true

# if limits have been enforced, do we proceed with opac calc?
enforced_limits_fail = false

# equilibrium etc -------------------------------------------------------------

# if we hit the table bound control whether we instead just solve a 1d problem
# with ye fixed
equilibrium_ye_bnd_reduce = true

# on failures, control which fallback is attempted
equilibrium_fallback_thin = false
equilibrium_fallback_zero = false

# use tabulated particle fraction?
tabulated_particle_fractions = true

# Hard coded vs table derived constants
tabulated_degeneracy_parameter = true

# flag equilibrium on kirchoff-law applied opacities?
flag_equilibrium_raw = false

# if we cannot resolve dynamics at a point then flag this
# uses eql_rho_min threshold
flag_equilibrium = true

# tau < (dt * this_fac) / opacity_tau_trap
flag_equilibrium_dt_factor = 1.0

# flag on a per-species basis
flag_equilibrium_species=true

# for the equilibrium density, use thin density limit as initial guess?
# warning, may not be in suitable basin of cvgce
flag_equilibrium_thin_guess=false

# switch off eql for heavy leptons?
flag_equilibrium_no_nux=false

# take tau_{a,e} = min(tau_e, tau_a)?
flag_equilibrium_nue_equals_nua=false

# nearest-neighbour equilibrium smear
# If a point is at eql then this many nearest neighbours are set to be also:
flag_equilibrium_nn = 2

# after opacity calculation perform sanity check
validate_opacities = true

# invalid opacities -> set zero radiation quantities?
zero_invalid_radmat = true

# on failure in opacity / equilibirum calculation replace hydro state
# with nearest neighbour average with extrema values clipped
use_averaging_fix = true

# for each point, use nn average
# N.B. this may smooth excessively
use_averages = false

# give more information when failing
verbose_warn_weak = true

# fix-tricks ------------------------------------------------------------------
# - Is base-point a local, isolated extrema (allow fac_min, fac_max tol.)
# - If so, consider spurious, replace with average based on values near median.

# enable the above for eql densities
fix_nn_densities=false
# enable the above for the opacities
fix_nn_opacities=false

fix_num_passes=3
fix_num_neighbors=1
fix_fac_median=2.0
fix_exclude_first_extrema=true
fix_keep_base_point=false
fix_fac_min=0.5
fix_fac_max=1.5

# fake specific ---------------------------------------------------------------
fake_avg_atomic_mass = 1.0
fake_avg_baryon_mass = 1.0

# here indices X_(ix_g,ix_s) can vary based on ngroups & nspecies
fake_eta_0_0   = 1.0
fake_kap_a_0_0 = 1.0
fake_kap_s_0_0 = 1.0

# photon specific -------------------------------------------------------------
photon_rad_constant = 2.471313401078565e-13 # Msun^-2/MeV^4
photon_kap_a = 1.0
photon_kap_s = 1.0
# =============================================================================

<refinement1>
x1min = -1024
x1max =  1024
x2min = -1024
x2max =  1024
x3min = -1024
x3max =  1024
level = 1

<refinement2>
x1min = -512
x1max =  512
x2min = -512
x2max =  512
x3min = -512
x3max =  512
level = 2

<refinement3>
x1min = -256
x1max =  256
x2min = -256
x2max =  256
x3min = -256
x3max =  256
level = 3

<refinement4>
x1min = -128
x1max =  128
x2min = -128
x2max =  128
x3min = -128
x3max =  128
level = 4

<refinement5>
x1min = -64
x1max =  64
x2min = -64
x2max =  64
x3min = -64
x3max =  64
level = 5             
# =============================================================================

# <task_triggers>  # ============================================================
# control frequency of task execution; certain constraints are augmented
# further automatically by choices in output block

# missing / zero values do not get activated
# adjust_mesh_dt = true

# dt_tracker_extrema = 1
# dt_Z4c_Weyl     = 0.5
#dt_Z4c_AHF     = 1
# dt_Z4c_RWZ      = 0.5
#dt_Z4c_CCE_dump_freq = 6
# =============================================================================

# <psi4_extraction>  # ==========================================================
# # control psi4 extraction

# filename  = wave_psi4

# nlev = 30  # Controls # of points on spherical grid
# lmax = 4   # Extracted set: {l <= lmax && |m|<=l}

# # specify extraction radii
# num_radii = 4
# radius_0  = 100
# radius_1  = 200
# radius_2  = 400
# radius_3  = 60
# # =============================================================================

#<surface1> # ==================================================================
# If 0 then triggered after each system step
#dt = 0.5 # = dt_Z4c_Weyl , like waves

# Set to negative value to ignore a limit
#start_time = -1
#stop_time  = -2

# Control whether global time-step can be adjusted
#adjust_mesh_dt = true

# Control whether data is dumped
#dump_data = true

# element of:
# { spherical
# }
#surface = spherical

# element of:
# { uniform - uniform theta & phi
# }
#sampling = uniform

# element of:
# { Lagrange,  [2(NGHOST)-1]
# }
#interpolator = Lagrange

# surface "spherical" specific (either arr or scalar-singleton):
#radii     = [100, 200, 400, 60]
#nth       = [64,  64,  64, 64] # [128,  128,  128, 128]
#nph       = [128,  128,  128, 128] # [256,  256,  256, 256]

# element can be:
# { geom.Z4c
#   geom.ADM,
#   geom.aux,
#   geom.weyl,
#   hydro.cons,
#   hydro.prim,
#   hydro.aux,
#   field.aux,
#   passive_scalars.cons,
#   passive_scalars.prim,
#   B,
#   M1.geom.sc_alpha,
#   M1.geom.sp_beta_u,
#   M1.geom.sp_g_dd,
#   M1.geom.sp_K_dd,
#   M1.lab,
#   M1.rad,
#   M1.radmat,
#   M1.radmat.sc_avg_nrg_00,
#   M1.radmat.sc_avg_nrg_01,
#   M1.radmat.sc_avg_nrg_02,
#   M1.geom.sc_sqrt_det_g,
#   tracer.vel,
#   tracer.rho,
#   tracer.ye,
#   tracer.aux.T,
#   tracer.aux.U_d_0,
#   tracer.aux.HU_d_0,
#   tracer.aux.SPB,
# }

#variables = [
#  geom.ADM,
#  geom.aux,
#  geom.weyl
#]
# =============================================================================

# <rwz_extraction>  # ==========================================================
# # control rwz metric extraction

# method_areal_radius = average_schw
# lmax = 4

# ntheta = 64
# nphi   = 128

# subtract_background = false
# method_integrals    = gausslegendre  # riemann

# # specify extraction radii
# num_radii = 4
# radius_0  = 100
# radius_1  = 200
# radius_2  = 400
# radius_3  = 60
# # =============================================================================

# <cce> # ============================================================
# output_dir = .
# num_theta        = 82
# num_phi          = 164
# num_r_inshell    = 28
# num_l_modes      = 15
# num_radial_modes = 11
# num_radii        = 0 #4
# # 200
# rin_0            = 198
# rout_0           = 202
# # 400
# rin_1            = 398
# rout_1           = 402
# # 600
# rin_2            = 598
# rout_2           = 602
# # 800
# rin_3            = 798
# rout_3           = 802
# # =============================================================================

# <trackers_extrema>  # =========================================================
# # control trackers based on field extrema

# filename = tra.ext

# N_tracker = 1
# filename = tra.ext

# control_field = Z4c.chi

# ini_1_x1 = 0               # initial offset
# ini_1_x2 = 0
# ini_1_x3 = 0
# minima_1 = true              # for update_strategy = 1
# ref_level_1 = 1
# ref_type_1 = 1              # 0: to the point, 1: sph. region
# ref_zone_radius_1 = 15

# ini_2_x1 = 0               # initial offset
# ini_2_x2 = 0
# ini_2_x3 = 0
# minima_2 = true              # for update_strategy = 1
# ref_level_2 = 2
# ref_type_2 = 1              # 0: to the point, 1: sph. region
# ref_zone_radius_2 = 0.1

# update_strategy = 1         # 0: based on dt, 1: based on quad interp.
# # safety factor avoids taking too large step (multiple of local dx)
# update_max_step_factor = 10
# # =============================================================================

# dumped quantities ===========================================================
<output1>
file_type = rst
dt        = 500

<output2>
file_type   = hst
dt          = 5
data_format = %.18g

<output3>
file_type   = hdf5
xdmf        = false
variable    = hydro
ghost_zones = false
dt          = 100
x3_slice    = 0.0

<output4>
file_type   = hdf5
xdmf        = false
variable    = passive_scalars
ghost_zones = false
dt          = 100
x3_slice    = 0.0

<output5>
file_type   = hdf5
xdmf        = false
variable    = geom
ghost_zones = false
dt          = 100
x3_slice    = 0.0

<output6>
file_type   = hdf5
xdmf        = false
variable    = M1.rad
ghost_zones = false
dt          = 100
x3_slice    = 0.0

#
# :D
#
