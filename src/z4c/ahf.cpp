//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file ahf.cpp
//  \brief implementation of the apparent horizon finder class
//         Fast flow algorithm of Gundlach:1997us and Alcubierre:1998rq

#include <unistd.h>

#include <algorithm>  // std::fill
#include <cmath>      // NAN
#include <cstdio>
#include <cstring>  // std::memcpy
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/spherical_harmonics.hpp"
#include "../utils/tensor_symmetry.hpp"
#include "ahf.hpp"
#include "puncture_tracker.hpp"
#include "z4c.hpp"

//----------------------------------------------------------------------------------------
//! \fn AHF::AHF(Mesh * pmesh, ParameterInput * pin, int idx_ahf)
//  \brief class for apparent horizon finder
AHF::AHF(Mesh* pmesh, ParameterInput* pin, int idx_ahf)
    : pmesh(pmesh), pin(pin), idx_ahf(idx_ahf)
{
  ReadOptions(pin);
  PrepareArrays();
  SetupIO();
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::ReadOptions(ParameterInput * pin)
//  \brief read all configuration from ParameterInput into opt struct and state
void AHF::ReadOptions(ParameterInput* pin)
{
  const std::string n_str = std::to_string(idx_ahf);
  auto parkey             = [&n_str](const char* base)
  { return std::string(base) + n_str; };

  // Grid and quadrature weights
  const int ntheta_val = pin->GetOrAddInteger("ahf", "ntheta", 14);
  const int nphi_val   = pin->GetOrAddInteger("ahf", "nphi", 28);
  std::string quadrature =
    pin->GetOrAddString("ahf", "quadrature", "gausslegendre");
  if (quadrature == "sums")
    quadrature = "midpoint";
  grid_.Initialize(ntheta_val, nphi_val, quadrature);

  opt.lmax = pin->GetOrAddInteger("ahf", "lmax", 12);

  opt.flow_iterations =
    pin->GetOrAddInteger("ahf", parkey("flow_iterations_"), 50);

  opt.flow_alpha_beta_const =
    pin->GetOrAddReal("ahf", parkey("flow_alpha_beta_const_"), 1.0);

  // Flow function (weight applied to H before spectral projection):
  //   "H"  : rho = H
  //   "Hu" : rho = H * u                              (default)
  //   "F3" : rho = H * 2 r^2 |grad F| / [(g^ij - s^i s^j)(gbar_ij - grad_i r
  //          grad_j r)], the Gundlach (1998) area-normalized flow weight
  {
    std::string ff =
      pin->GetOrAddString("ahf", parkey("flow_function_"), "Hu");
    if (ff == "H")
      opt.flow_function = FlowFunction::H;
    else if (ff == "Hu")
      opt.flow_function = FlowFunction::Hu;
    else if (ff == "F3")
      opt.flow_function = FlowFunction::F3;
    else
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF::ReadOptions" << std::endl;
      msg << "Unknown flow_function_" << n_str << " '" << ff
          << "' (expected: H | Hu | F3)";
      throw std::runtime_error(msg.str().c_str());
    }
  }

  opt.hmean_tol    = pin->GetOrAddReal("ahf", parkey("hmean_tol_"), 100.);
  opt.mass_tol     = pin->GetOrAddReal("ahf", parkey("mass_tol_"), 1e-3);
  opt.spec_tol     = pin->GetOrAddReal("ahf", parkey("spec_tol_"), 1e-5);
  opt.hrms_tol     = pin->GetOrAddReal("ahf", parkey("hrms_tol_"), 1e-1);
  opt.hrms_rel_tol = pin->GetOrAddReal("ahf", parkey("hrms_rel_tol_"), 1e-3);

  // Stagnation detection: abort FastFlowLoop when hrms plateaus while still
  // above the absolute hrms tol.
  opt.stagnation_detect =
    pin->GetOrAddBoolean("ahf", parkey("stagnation_detect_"), true);
  opt.stagnation_window =
    pin->GetOrAddInteger("ahf", parkey("stagnation_window_"), 8);
  opt.stagnation_improvement_frac =
    pin->GetOrAddReal("ahf", parkey("stagnation_improvement_frac_"), 1.0e-1);
  opt.stagnation_warmup =
    pin->GetOrAddInteger("ahf", parkey("stagnation_warmup_"), 5);

  // mode_ramp: continuation in lmax. Iteration starts on the truncated
  // manifold modes l <= mode_ramp_lmin and progressively unlocks
  // mode_ramp_modes_per_step modes every mode_ramp_iters_per_step iterations
  // until lmax is reached.
  // Set mode_ramp_lmin = -1 to disable.
  opt.mode_ramp_lmin =
    pin->GetOrAddInteger("ahf", parkey("mode_ramp_lmin_"), -1);
  opt.mode_ramp_iters_per_step =
    pin->GetOrAddInteger("ahf", parkey("mode_ramp_iters_per_step_"), 2);
  opt.mode_ramp_modes_per_step =
    pin->GetOrAddInteger("ahf", parkey("mode_ramp_modes_per_step_"), 2);

  if (opt.mode_ramp_lmin == -1)
    opt.mode_ramp_lmin = opt.lmax;

  // Auto-retry options. On failed Find(), retry with adjusted initial_radius
  // selected by exit-code (see ExitCode enum / Find() retry loop).
  opt.auto_retry   = pin->GetOrAddBoolean("ahf", parkey("auto_retry_"), true);
  opt.max_retries  = pin->GetOrAddInteger("ahf", parkey("max_retries_"), 5);
  opt.retry_shrink = pin->GetOrAddReal("ahf", parkey("retry_shrink_"), 0.5);
  opt.retry_grow   = pin->GetOrAddReal("ahf", parkey("retry_grow_"), 2.0);

  // Validate
  if (!(opt.retry_shrink > 0.0 && opt.retry_shrink < 1.0))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in AHF::ReadOptions" << std::endl;
    msg << "retry_shrink_" << n_str << " = " << opt.retry_shrink
        << " is out of range (0, 1)";
    throw std::runtime_error(msg.str().c_str());
  }
  if (!(opt.retry_grow > 1.0))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in AHF::ReadOptions" << std::endl;
    msg << "retry_grow_" << n_str << " = " << opt.retry_grow
        << " is out of range (1, +inf)";
    throw std::runtime_error(msg.str().c_str());
  }
  // Soft-clamp max_retries
  if (opt.max_retries < 0)
    opt.max_retries = 0;
  if (opt.max_retries > 32)
    opt.max_retries = 32;

  // Adaptive step-size (line-search) options
  {
    std::string sr =
      pin->GetOrAddString("ahf", parkey("step_rule_"), "monotone");
    if (sr == "fixed")
      opt.step_rule = StepRule::fixed;
    else if (sr == "monotone")
      opt.step_rule = StepRule::monotone;
    else if (sr == "bb1")
      opt.step_rule = StepRule::bb1;
    else if (sr == "bb2")
      opt.step_rule = StepRule::bb2;
    else
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF::ReadOptions" << std::endl;
      msg << "Unknown step_rule '" << sr
          << "' (expected: fixed | monotone | bb1 | bb2)";
      throw std::runtime_error(msg.str().c_str());
    }
  }
  opt.alpha_min    = pin->GetOrAddReal("ahf", parkey("alpha_min_"), 0.1);
  opt.alpha_max    = pin->GetOrAddReal("ahf", parkey("alpha_max_"), 4.0);
  opt.alpha_grow   = pin->GetOrAddReal("ahf", parkey("alpha_grow_"), 1.1);
  opt.alpha_shrink = pin->GetOrAddReal("ahf", parkey("alpha_shrink_"), 0.5);

  opt.verbose         = pin->GetOrAddBoolean("ahf", "verbose", false);
  opt.mpi_root        = pin->GetOrAddInteger("ahf", "mpi_root", 0);
  opt.merger_distance = pin->GetOrAddReal("ahf", "merger_distance", 0.1);
  opt.bitant          = pin->GetOrAddBoolean("mesh", "bitant", false);

  // Initial guess
  opt.initial_radius =
    pin->GetOrAddReal("ahf", parkey("initial_radius_"), 1.0);
  rr_min = -1.0;

  opt.expand_guess = pin->GetOrAddReal("ahf", "expand_guess", 1.0);

  opt.propagate_iter_coefficients =
    pin->GetOrAddBoolean("ahf", "propagate_iter_coefficients", true);

  // Center
  center[0] = pin->GetOrAddReal("ahf", parkey("center_x_"), 0.0);
  center[1] = pin->GetOrAddReal("ahf", parkey("center_y_"), 0.0);
  center[2] = pin->GetOrAddReal("ahf", parkey("center_z_"), 0.0);

  opt.use_puncture = pin->GetOrAddInteger("ahf", parkey("use_puncture_"), -1);

  if (opt.use_puncture >= 0)
  {
    // Center is determined on the fly during the initial guess
    // to follow the chosen puncture
    const int npunct = static_cast<int>(pmesh->pz4c_tracker.size());
    if (opt.use_puncture >= npunct)
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF constructor" << std::endl;
      msg << " : punc = " << opt.use_puncture << " > npunct = " << npunct;
      throw std::runtime_error(msg.str().c_str());
    }
  }
  opt.use_puncture_massweighted_center = pin->GetOrAddBoolean(
    "ahf", parkey("use_puncture_massweighted_center_"), 0);

  opt.use_extrema = pin->GetOrAddInteger("ahf", parkey("use_extrema_"), -1);

  if (opt.use_extrema >= 0)
  {
    const int N_tracker = pmesh->ptracker_extrema->N_tracker;
    if (opt.use_extrema >= N_tracker)
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF constructor" << std::endl;
      msg << " : extrema = " << opt.use_extrema
          << " > N_tracker = " << N_tracker;
      throw std::runtime_error(msg.str().c_str());
    }
  }

  opt.start_time = pin->GetOrAddReal(
    "ahf", parkey("start_time_"), std::numeric_limits<double>::max());

  opt.stop_time = pin->GetOrAddReal("ahf", parkey("stop_time_"), -1.0);

  opt.wait_until_punc_are_close =
    pin->GetOrAddBoolean("ahf", parkey("wait_until_punc_are_close_"), 0);

  // Initialize last & found
  last_a0 = pin->GetOrAddReal("ahf", parkey("last_a0_"), -1);

  ah_found = pin->GetOrAddBoolean("ahf", parkey("ah_found_a0_"), false);

  time_first_found =
    pin->GetOrAddReal("ahf", parkey("time_first_found_"), -1.0);

  // Output filenames
  opt.ofname_summary = pin->GetString("job", "problem_id") + ".";
  opt.ofname_summary += pin->GetOrAddString(
    "ahf", parkey("horizon_file_summary_"), "horizon_summary_" + n_str);
  opt.ofname_summary += ".txt";

  opt.ofname_shape = pin->GetString("job", "problem_id") + ".";
  opt.ofname_shape += pin->GetOrAddString(
    "ahf", parkey("horizon_file_shape_"), "horizon_shape_" + n_str);
  opt.ofname_shape += ".txt";

  opt.ofname_shear = pin->GetString("job", "problem_id") + ".";
  opt.ofname_shear += pin->GetOrAddString(
    "ahf", parkey("horizon_file_shear_"), "horizon_shear_" + n_str);
  opt.ofname_shear += ".txt";

  if (opt.verbose)
  {
    opt.ofname_verbose = pin->GetString("job", "problem_id") + ".";
    opt.ofname_verbose += pin->GetOrAddString(
      "ahf", parkey("horizon_verbose_"), "horizon_verbose_" + n_str);
    opt.ofname_verbose += ".txt";
  }

  // Expansion fix method
  std::string expfix_str =
    pin->GetOrAddString("ahf", "expansion_fix", "cure_divu");
  if (expfix_str == "do_nothing")
    opt.expansion_fix = ExpansionFix::do_nothing;
  else if (expfix_str == "cure_divu")
    opt.expansion_fix = ExpansionFix::cure_divu;
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in AHF" << std::endl
        << "unknown expansion_fix: " << expfix_str << std::endl;
    ATHENA_ERROR(msg);
  }

  // Warn if AHF will run but storage.aux ghost zones won't be communicated
  {
    const Real dt_ahf = pin->GetOrAddReal("task_triggers", "dt_Z4c_AHF", 0.0);
    if (dt_ahf > 0.0 &&
        !pin->GetOrAddBoolean("z4c", "communicate_aux_adm", false))
    {
      if ((Globals::my_rank == 0) && (idx_ahf == 0))
      {
        std::printf(
          "### WARNING [AHF]: z4c/communicate_aux_adm is false.\n"
          "  AHF interpolates storage.aux (metric derivatives) near "
          "MeshBlock\n"
          "  boundaries where ghost-zone values are uninitialized without\n"
          "  communication. Results may be inaccurate.\n");
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::PrepareArrays()
//  \brief allocate spectral, harmonic, and field arrays; compute Ylm tables
void AHF::PrepareArrays()
{
  // Compute spherical harmonic tables on the grid
  ylm_.Initialize(
    opt.lmax, grid_.ntheta, grid_.nphi, grid_.th_grid, grid_.ph_grid);

  // Coefficients
  a0.NewAthenaArray(opt.lmax + 1);
  ac.NewAthenaArray(ylm_.lmpoints);
  as.NewAthenaArray(ylm_.lmpoints);

  // Full last-found coefficients (for initial guess)
  last_a0_full.NewAthenaArray(opt.lmax + 1);
  last_ac.NewAthenaArray(ylm_.lmpoints);
  last_as.NewAthenaArray(ylm_.lmpoints);
  last_a0_full.ZeroClear();
  last_ac.ZeroClear();
  last_as.ZeroClear();

  // Fields on the sphere
  rr.NewAthenaArray(grid_.ntheta, grid_.nphi);
  rr_dth.NewAthenaArray(grid_.ntheta, grid_.nphi);
  rr_dph.NewAthenaArray(grid_.ntheta, grid_.nphi);

  g.NewAthenaTensor(grid_.ntheta, grid_.nphi);
  dg.NewAthenaTensor(grid_.ntheta, grid_.nphi);
  K.NewAthenaTensor(grid_.ntheta, grid_.nphi);

  // Array computed in surface integrals
  rho.NewAthenaArray(grid_.ntheta, grid_.nphi);

  // --- Shear tensor storage ------------------------------------------------
  sigma_dd.NewAthenaTensor(grid_.ntheta, grid_.nphi);
  sigma_uu.NewAthenaTensor(grid_.ntheta, grid_.nphi);
  shear2.NewAthenaArray(grid_.ntheta, grid_.nphi);
  shear_re.NewAthenaArray(grid_.ntheta, grid_.nphi);
  shear_im.NewAthenaArray(grid_.ntheta, grid_.nphi);

  // --- Spin -2 harmonic table + coefficient storage -------------------------
  PrepareSWSH2Table();
  const int n_s2 = gra::sph_harm::lmpoints_complex(opt.lmax);
  c2_re.NewAthenaArray(n_s2);
  c2_im.NewAthenaArray(n_s2);
  c2_re.ZeroClear();
  c2_im.ZeroClear();

  // Initialize horizon properties to NAN
  for (int v = 0; v < hnvar; ++v)
  {
    ah_prop[v] = NAN;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::PrepareSWSH2Table()
//  \brief Precompute the spin-weight -2 spherical harmonics _{-2}Y_lm on the
//  AHF's (theta,phi) grid, for l = 2..lmax, m = -l..l. Packing follows
//  gra::sph_harm::lmindex_complex/lmpoints_complex so indices agree with
//  ComplexHarmonicTable elsewhere in the code. Spin-2 harmonics vanish
//  identically for l < 2, so those modes are simply never populated/used.
void AHF::PrepareSWSH2Table()
{
  const int lmax   = opt.lmax;
  const int n_s2   = gra::sph_harm::lmpoints_complex(lmax);
  const int lmin_s = 2;

  swsh2_re.NewAthenaArray(grid_.ntheta, grid_.nphi, std::max(n_s2, 1));
  swsh2_im.NewAthenaArray(grid_.ntheta, grid_.nphi, std::max(n_s2, 1));
  swsh2_re.ZeroClear();
  swsh2_im.ZeroClear();

  if (lmax < lmin_s)
    return;

#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < grid_.ntheta; ++i)
  {
    for (int j = 0; j < grid_.nphi; ++j)
    {
      const Real theta = grid_.th_grid(i);
      const Real phi   = grid_.ph_grid(j);
      for (int l = lmin_s; l <= lmax; ++l)
        for (int m = -l; m <= l; ++m)
        {
          const int lm = gra::sph_harm::lmindex_complex(l, m);
          Real YR, YI;
          gra::sph_harm::sYlm(-2, l, m, theta, phi, &YR, &YI);
          swsh2_re(i, j, lm) = YR;
          swsh2_im(i, j, lm) = YI;
        }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::SetupIO()
//  \brief open output files, write column headers
void AHF::SetupIO()
{
  if (Globals::my_rank == opt.mpi_root)
  {
    // Summary file
    bool new_file = true;
    if (access(opt.ofname_summary.c_str(), F_OK) == 0)
    {
      new_file = false;
    }
    pofile_summary = fopen(opt.ofname_summary.c_str(), "a");
    if (pofile_summary == nullptr)
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF constructor" << std::endl;
      msg << "Could not open file '" << opt.ofname_summary << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    if (new_file)
    {
      fprintf(pofile_summary,
              "# 1:iter 2:time 3:mass 4:mass_irr 5:Sx 6:Sy 7:Sz 8:S 9:chi "
              "10:area 11:hrms 12:hmean 13:meanradius 14:minradius "
              "15:shear_rms 16:gw_flux "
              "17:exit_code 18:num_iters 19:spec_resid\n");
      fflush(pofile_summary);
    }

    // Shear file: write a one-time explanatory header (file itself is
    // opened/appended per-Write() call, same as the shape file).
    bool shear_new_file = true;
    if (access(opt.ofname_shear.c_str(), F_OK) == 0)
    {
      shear_new_file = false;
    }
    if (shear_new_file)
    {
      FILE* pf_shear_hdr = fopen(opt.ofname_shear.c_str(), "a");
      if (pf_shear_hdr == nullptr)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in AHF constructor" << std::endl;
        msg << "Could not open file '" << opt.ofname_shear
            << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pf_shear_hdr,
              "# col1: shear_rms = sqrt(<sigma_ij sigma^ij>_area)\n"
              "# col2: gw_flux = (1/16pi) * oint sigma_ij sigma^ij dA "
              "(instantaneous GW flux through the horizon)\n"
              "# then Re(c_lm) Im(c_lm) pairs for l=2..lmax, m=-l..l, where\n"
              "# sigma(theta,phi) = sigma_ab m^a m^b = sum_lm c_lm "
              "_{-2}Y_lm(theta,phi)\n");
      fflush(pf_shear_hdr);
      fclose(pf_shear_hdr);
    }

    if (opt.verbose)
    {
      pofile_verbose = fopen(opt.ofname_verbose.c_str(), "a");
      if (pofile_verbose == nullptr)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in AHF constructor" << std::endl;
        msg << "Could not open file '" << opt.ofname_verbose
            << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
    }
  }
}

AHF::~AHF()
{
  // Close files
  // (pofile_shape / pofile_shear are opened and closed per Write() call,
  //  same pattern as before -- nothing persistent to close here for them.)
  if (Globals::my_rank == opt.mpi_root)
  {
    fclose(pofile_summary);
    if (opt.verbose)
    {
      fclose(pofile_verbose);
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::Write(int iter, Real time)
// \brief output summary and shape file, for each horizon
void AHF::Write(int iter, Real time)
{
  if (Globals::my_rank == opt.mpi_root)
  {
    std::string i_str = std::to_string(iter);
    if ((time < opt.start_time) || (time > opt.stop_time))
      return;
    if (opt.wait_until_punc_are_close && !(PuncAreClose()))
      return;

    // Summary file
    fprintf(pofile_summary, "%d %g ", iter, time);
    fprintf(pofile_summary,
            "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e "
            "%.15e %.15e %.15e %.15e",
            ah_prop[hmass],
            ah_prop[hmass_irr],
            ah_prop[hSx],
            ah_prop[hSy],
            ah_prop[hSz],
            ah_prop[hS],
            ah_prop[hchi],
            ah_prop[harea],
            ah_prop[hhrms],
            ah_prop[hhmean],
            ah_prop[hmeanradius],
            ah_prop[hminradius],
            ah_prop[hshearrms],
            ah_prop[hgwflux]);
    fprintf(pofile_summary,
            " %d %d %.15e",
            static_cast<int>(last_exit),
            fastflow_iter + 1,
            spec_resid_last);
    fprintf(pofile_summary, "\n");
    fflush(pofile_summary);

    if (ah_found)
    {
      // Shape file (coefficients)
      pofile_shape = fopen(opt.ofname_shape.c_str(), "a");
      if (pofile_shape == nullptr)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in AHF constructor" << std::endl;
        msg << "Could not open file '" << opt.ofname_shape << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile_shape, "# iter = %d, Time = %g\n", iter, time);
      for (int l = 0; l <= opt.lmax; l++)
        fprintf(pofile_shape, "%e ", a0(l));
      for (int l = 1; l <= opt.lmax; l++)
      {
        for (int m = 1; m <= l; m++)
        {
          int l1 = ylm_.lmindex(l, m);
          fprintf(pofile_shape, "%e ", ac(l1));
          fprintf(pofile_shape, "%e ", as(l1));
        }
      }
      fprintf(pofile_shape, "\n");
      fclose(pofile_shape);

      // Shear file (area-rms + complex spin-2 coefficients c_lm, l=2..lmax)
      pofile_shear = fopen(opt.ofname_shear.c_str(), "a");
      if (pofile_shear == nullptr)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in AHF constructor" << std::endl;
        msg << "Could not open file '" << opt.ofname_shear
            << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile_shear, "# iter = %d, Time = %g\n", iter, time);
      fprintf(pofile_shear, "%.15e %.15e ", ah_prop[hshearrms], ah_prop[hgwflux]);
      for (int l = 2; l <= opt.lmax; l++)
      {
        for (int m = -l; m <= l; m++)
        {
          const int lm = gra::sph_harm::lmindex_complex(l, m);
          fprintf(pofile_shear, "%.15e %.15e ", c2_re(lm), c2_im(lm));
        }
      }
      fprintf(pofile_shear, "\n");
      fclose(pofile_shear);
    }
  }

  // This is needed on all ranks.
  if (ah_found && (time_first_found < 0))
  {
    std::string parname{ "time_first_found_" + std::to_string(idx_ahf) };
    time_first_found = time;
    pin->SetReal("ahf", parname, time_first_found);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::MetricInterp()
// \brief interpolate metric on the surface using pre-built interpolator pools
void AHF::MetricInterp()
{
  using InterpType = LagrangeInterpND<metric_interp_order, 3>;

  // Select the interpolator pool matching Z4c centering
  std::vector<InterpType>& pool =
    SW_CCX_VC(grid_.interp_pool_cc, grid_.interp_pool_vc);

  const Real zc = center[2];

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < grid_.ntheta; ++i)
  {
    for (int j = 0; j < grid_.nphi; ++j)
    {
      const Real costh = grid_.cos_theta(i);
      if (!grid_.IsOwned(i, j))
        continue;

      MeshBlock* pmb = grid_.mask_mb(i, j);
      Z4c* pz4c      = pmb->pz4c;

      AT_N_sym adm_g_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
      AT_N_sym adm_K_dd(pz4c->storage.adm, Z4c::I_ADM_Kxx);

      InterpType& interp = pool[grid_.mask_interp_idx(i, j)];

      // Bitant: check the raw (unreflected) z coordinate
      const Real z_raw      = zc + rr(i, j) * costh;
      const bool bitant_sym = (opt.bitant && z_raw < 0);

      // With bitant wrt z=0, pick a (-) sign every time a z component is
      // encountered.
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          const int bsign = BitantSign(bitant_sym, a, b);
          g(a, b, i, j)   = interp.eval(&(adm_g_dd(a, b, 0, 0, 0))) * bsign;
          K(a, b, i, j)   = interp.eval(&(adm_K_dd(a, b, 0, 0, 0))) * bsign;
          for (int c = 0; c < NDIM; ++c)
          {
            dg(c, a, b, i, j) =
              interp.eval(&(pz4c->aux.dg_ddd(c, a, b, 0, 0, 0))) *
              BitantSign(bitant_sym, a, b, c);
          }
        }

    }  // phi loop
  }  // theta loop
}
//----------------------------------------------------------------------------------------
//! \fn bool AHF::LevelSetGradient(...)
//  \brief Compute Cartesian coords and level-set function derivatives dFdi,
//  dFdidj via chain rule.  Reads precomputed Jacobian from grid_.con_J/con_J2
//  and spherical harmonic derivatives from ylm_.  Returns false if the surface
//  radius is below min_surface_radius (caller should break).
bool AHF::LevelSetGradient(int i,
                           int j,
                           ATP_N_vec& dFdi,
                           ATP_N_sym& dFdidj,
                           Real& xp,
                           Real& yp,
                           Real& zp)
{
  using namespace gra::sph_harm::ix_D;
  const AT_N_T2& con_J        = grid_.con_J;
  const AT_N_VS2& con_J2      = grid_.con_J2;
  const AthenaArray<Real>& Y0 = ylm_.Y0;
  const AthenaArray<Real>& Yc = ylm_.Yc;
  const AthenaArray<Real>& Ys = ylm_.Ys;

  // Cartesian coordinates of the surface point (relative to center)
  xp = rr(i, j) * grid_.sin_theta(i) * grid_.cos_phi(j);
  yp = rr(i, j) * grid_.sin_theta(i) * grid_.sin_phi(j);
  zp = rr(i, j) * grid_.cos_theta(i);

  const Real rp = std::sqrt(xp * xp + yp * yp + zp * zp);
  if (rp < min_surface_radius)
    return false;

  // Chain rule: dF/dx^a = dr/dx^a - sum_{lm} c_{lm} * (dY/dth * dth/dx^a
  //                                                    + dY/dph * dph/dx^a)
  // con_J(A, a) = d(sph coord A)/d(cart coord a), A: 0=r, 1=th, 2=ph
  for (int a = 0; a < NDIM; ++a)
  {
    dFdi(a) = con_J(0, a, i, j);
    for (int l = 0; l <= opt.lmax; l++)
      dFdi(a) -= a0(l) * con_J(1, a, i, j) * Y0(D10, i, j, l);
    for (int l = 1; l <= opt.lmax; l++)
      for (int m = 1; m <= l; m++)
      {
        const int l1 = ylm_.lmindex(l, m);
        dFdi(a) -= ac(l1) * (con_J(1, a, i, j) * Yc(D10, i, j, l1) +
                             con_J(2, a, i, j) * Yc(D01, i, j, l1)) +
                   as(l1) * (con_J(1, a, i, j) * Ys(D10, i, j, l1) +
                             con_J(2, a, i, j) * Ys(D01, i, j, l1));
      }
  }

  // Second chain rule (symmetric, upper triangle + copy)
  for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    {
      dFdidj(a, b) = con_J2(0, a, b, i, j);
      for (int l = 0; l <= opt.lmax; l++)
        dFdidj(a, b) -=
          a0(l) * (con_J2(1, a, b, i, j) * Y0(D10, i, j, l) +
                   con_J(1, a, i, j) * con_J(1, b, i, j) * Y0(D20, i, j, l));
      for (int l = 1; l <= opt.lmax; l++)
        for (int m = 1; m <= l; m++)
        {
          const int l1 = ylm_.lmindex(l, m);
          dFdidj(a, b) -=
            ac(l1) *
              (con_J2(1, a, b, i, j) * Yc(D10, i, j, l1) +
               con_J(1, a, i, j) * (con_J(1, b, i, j) * Yc(D20, i, j, l1) +
                                    con_J(2, b, i, j) * Yc(D11, i, j, l1)) +
               con_J2(2, a, b, i, j) * Yc(D01, i, j, l1) +
               con_J(2, a, i, j) * (con_J(1, b, i, j) * Yc(D11, i, j, l1) +
                                    con_J(2, b, i, j) * Yc(D02, i, j, l1))) +
            as(l1) *
              (con_J2(1, a, b, i, j) * Ys(D10, i, j, l1) +
               con_J(1, a, i, j) * (con_J(1, b, i, j) * Ys(D20, i, j, l1) +
                                    con_J(2, b, i, j) * Ys(D11, i, j, l1)) +
               con_J2(2, a, b, i, j) * Ys(D01, i, j, l1) +
               con_J(2, a, i, j) * (con_J(1, b, i, j) * Ys(D11, i, j, l1) +
                                    con_J(2, b, i, j) * Ys(D02, i, j, l1)));
        }
      dFdidj(b, a) = dFdidj(a, b);
    }

  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::ExpansionAndNormal(...)
//  \brief Compute the expansion H and outward unit normal R from the metric,
//  extrinsic curvature, metric derivatives, and level-set derivatives at
//  surface point (i,j). Also hands back the covariant Hessian nnF, inverse
//  3-metric ginv, and raised gradient dFdi_u so that ShearTensor() can reuse
//  them without recomputing.
void AHF::ExpansionAndNormal(int i,
                             int j,
                             const ATP_N_vec& dFdi,
                             const ATP_N_sym& dFdidj,
                             ATP_N_vec& R,
                             Real& H,
                             Real& u,
                             ATP_N_sym& nnF_out,
                             ATP_N_sym& ginv_out,
                             ATP_N_vec& dFdi_u_out)
{
  using namespace LinearAlgebra;

  // Determinant of 3-metric
  Real detg    = Det3Metric(g(0, 0, i, j),
                            g(0, 1, i, j),
                            g(0, 2, i, j),
                            g(1, 1, i, j),
                            g(1, 2, i, j),
                            g(2, 2, i, j));
  Real oo_detg = 1.0 / detg;

  // Inverse metric
  ATP_N_sym ginv;
  Inv3Metric(oo_detg,
             g(0, 0, i, j),
             g(0, 1, i, j),
             g(0, 2, i, j),
             g(1, 1, i, j),
             g(1, 2, i, j),
             g(2, 2, i, j),
             &ginv(0, 0),
             &ginv(0, 1),
             &ginv(0, 2),
             &ginv(1, 1),
             &ginv(1, 2),
             &ginv(2, 2));

  // Trace of K
  Real TrK = TraceRank2(oo_detg,
                        g(0, 0, i, j),
                        g(0, 1, i, j),
                        g(0, 2, i, j),
                        g(1, 1, i, j),
                        g(1, 2, i, j),
                        g(2, 2, i, j),
                        K(0, 0, i, j),
                        K(0, 1, i, j),
                        K(0, 2, i, j),
                        K(1, 1, i, j),
                        K(1, 2, i, j),
                        K(2, 2, i, j));

  // Raise index: dFdi_u^a = g^{ab} dFdi_b
  ATP_N_vec dFdi_u;
  for (int a = 0; a < NDIM; ++a)
  {
    dFdi_u(a) = 0;
    for (int b = 0; b < NDIM; ++b)
      dFdi_u(a) += ginv(a, b) * dFdi(b);
  }

  // Norm of gradient: |nabla F|^2
  Real norm = 0;
  for (int a = 0; a < NDIM; ++a)
    norm += dFdi_u(a) * dFdi(a);

  u = (norm > 0) ? std::sqrt(norm) : 0.0;

  // Covariant Hessian: nabla_a nabla_b F = d_a d_b F - Gamma^c_{ab} d_c F
  ATP_N_sym nnF;
  for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    {
      nnF(a, b) = dFdidj(a, b);
      for (int d = 0; d < NDIM; ++d)
        nnF(a, b) -=
          0.5 * dFdi_u(d) *
          (dg(a, b, d, i, j) + dg(b, a, d, i, j) - dg(d, a, b, i, j));
      nnF(b, a) = nnF(a, b);
    }

  // Contract symmetric tensors for expansion
  Real d2F = 0.0, dFdadFdbKab = 0.0, dFdadFdbFdadb = 0.0;
  for (int a = 0; a < NDIM; ++a)
    for (int b = 0; b < NDIM; ++b)
    {
      d2F += ginv(a, b) * nnF(a, b);
      Real ff = dFdi_u(a) * dFdi_u(b);
      dFdadFdbKab += ff * K(a, b, i, j);
      dFdadFdbFdadb += ff * nnF(a, b);
    }

  // Expansion: Theta = div(s) = (1/u) nabla^2 F + (1/u^3) dF^a dF^b K_ab
  //                            - (1/u^3) dF^a dF^b nabla_a nabla_b F - K
  Real divu = (opt.expansion_fix == ExpansionFix::cure_divu)
              ? ((norm > 0) ? 1.0 / u : 0.0)
              : 1.0 / u;

  // Assemble
  H = d2F * divu + dFdadFdbKab * (divu * divu) -
      dFdadFdbFdadb * (divu * divu * divu) - TrK;

  // Outward unit normal: s^a = dF^a / |nabla F|
  for (int a = 0; a < NDIM; ++a)
    R(a) = dFdi_u(a) * divu;

  // Hand back the working set for ShearTensor()
  nnF_out    = nnF;
  ginv_out   = ginv;
  dFdi_u_out = dFdi_u;
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::ShearTensor(...)
//  \brief Compute the horizon shear tensor sigma_ij = B_ij - 1/2 q_ij B, with
//  B_ij = q_i^k q_j^l (D_k s_l - K_kl), and its complex spin-weight -2
//  projection sigma_ab m^a m^b, m = (v - i w)/sqrt(2), where (v,w) are the
//  orthonormal (theta,phi)-tangent dyad Gram-Schmidt'd from the Strahlkoerper
//  parametrization -- consistent with the (u,v,w) tetrad convention used in
//  Z4c::Z4cWeyl for Psi4 (uvec=normal, vvec~theta, wvec~phi, m=(v-iw)/sqrt2).
//
//  Uses the identity q_i^k q_j^l D_k s_l = (1/u) q_i^k q_j^l nabla_k nabla_l F
//  (the normalization-derivative term is annihilated by the projector), so no
//  extra derivatives beyond nnF (already built in ExpansionAndNormal) are
//  needed. As a byproduct, tr_g(B_ij) reproduces H exactly.
void AHF::ShearTensor(int i,
                      int j,
                      const ATP_N_vec& dFdi,
                      const ATP_N_vec& dFdi_u,
                      const ATP_N_sym& nnF,
                      const ATP_N_sym& ginv,
                      Real u,
                      Real& shear2_out,
                      Real& sre,
                      Real& sim)
{
  // s_a (down), s^a (up)
  ATP_N_vec s_d, s_u;
  for (int a = 0; a < NDIM; ++a)
  {
    s_d(a) = dFdi(a) / u;
    s_u(a) = dFdi_u(a) / u;
  }

  // T_kl = nnF(k,l)/u - K_kl
  ATP_N_sym T;
  for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    {
      T(a, b) = nnF(a, b) / u - K(a, b, i, j);
      T(b, a) = T(a, b);
    }

  // sT_l = s^k T_kl ; ssT = s^k s^l T_kl
  ATP_N_vec sT;
  Real ssT = 0.0;
  for (int l = 0; l < NDIM; ++l)
  {
    sT(l) = 0.0;
    for (int k = 0; k < NDIM; ++k)
      sT(l) += s_u(k) * T(k, l);
  }
  for (int l = 0; l < NDIM; ++l)
    ssT += s_u(l) * sT(l);

  // B_ij = q_i^k q_j^l T_kl = T_ij - s_i sT_j - s_j sT_i + s_i s_j ssT
  ATP_N_sym B;
  for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    {
      B(a, b) = T(a, b) - s_d(a) * sT(b) - s_d(b) * sT(a) +
                s_d(a) * s_d(b) * ssT;
      B(b, a) = B(a, b);
    }

  // Trace: Btrace = g^{ab} B_ab  (== H from ExpansionAndNormal, by
  // construction; kept as an independent local computation here so
  // ShearTensor() is self-contained given (nnF, ginv, u, dFdi, dFdi_u, K)).
  Real Btrace = 0.0;
  for (int a = 0; a < NDIM; ++a)
    for (int b = 0; b < NDIM; ++b)
      Btrace += ginv(a, b) * B(a, b);

  // q_ij = g_ij - s_i s_j ; sigma_ij = B_ij - 1/2 q_ij Btrace
  for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    {
      const Real q_ab = g(a, b, i, j) - s_d(a) * s_d(b);
      sigma_dd(a, b, i, j) = B(a, b) - 0.5 * q_ab * Btrace;
      sigma_dd(b, a, i, j) = sigma_dd(a, b, i, j);
    }

  // sigma^{ij} = g^{ik} g^{jl} sigma_kl ; sigma_ij sigma^ij
  shear2_out = 0.0;
  for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    {
      Real s_ab_up = 0.0;
      for (int k = 0; k < NDIM; ++k)
        for (int l = 0; l < NDIM; ++l)
          s_ab_up += ginv(a, k) * ginv(b, l) * sigma_dd(k, l, i, j);
      sigma_uu(a, b, i, j) = s_ab_up;
      sigma_uu(b, a, i, j) = s_ab_up;
    }
  for (int a = 0; a < NDIM; ++a)
    for (int b = 0; b < NDIM; ++b)
      shear2_out += sigma_dd(a, b, i, j) * sigma_uu(a, b, i, j);

  // --- orthonormal (theta,phi) tangent dyad from the parametrization -----
  const Real costh = grid_.cos_theta(i), sinth = grid_.sin_theta(i);
  const Real cosph = grid_.cos_phi(j),   sinph = grid_.sin_phi(j);

  const Real n[3]    = { sinth * cosph, sinth * sinph, costh };
  const Real n_th[3] = { costh * cosph, costh * sinph, -sinth };
  const Real n_ph[3] = { -sinth * sinph, sinth * cosph, 0.0 };

  Real e_th[3], e_ph[3];
  for (int a = 0; a < NDIM; ++a)
  {
    e_th[a] = rr_dth(i, j) * n[a] + rr(i, j) * n_th[a];
    e_ph[a] = rr_dph(i, j) * n[a] + rr(i, j) * n_ph[a];
  }

  auto inner = [&](const Real* X, const Real* Y)
  {
    Real r = 0.0;
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        r += g(a, b, i, j) * X[a] * Y[b];
    return r;
  };

  const Real norm_th = std::sqrt(inner(e_th, e_th));
  Real v[3];  // ~ E_theta (unit tangent)
  for (int a = 0; a < NDIM; ++a)
    v[a] = e_th[a] / norm_th;

  const Real proj = inner(v, e_ph);
  Real e_ph_perp[3];
  for (int a = 0; a < NDIM; ++a)
    e_ph_perp[a] = e_ph[a] - proj * v[a];
  const Real norm_ph = std::sqrt(inner(e_ph_perp, e_ph_perp));
  Real w[3];  // ~ E_phi (unit tangent)
  for (int a = 0; a < NDIM; ++a)
    w[a] = e_ph_perp[a] / norm_ph;

  // sigma = sigma_ab m^a m^b, m = (v - i w)/sqrt(2), same sign convention as
  // Z4c::Z4cWeyl's Tr = v.v - w.w, Ti = -(v.w + w.v):
  //   sigma = [ (sigma_vv - sigma_ww) - 2 i sigma_vw ] / 2
  // and sigma is trace-free in the induced 2-metric => sigma_ww = -sigma_vv.
  Real s_vv = 0.0, s_vw = 0.0;
  for (int a = 0; a < NDIM; ++a)
    for (int b = 0; b < NDIM; ++b)
    {
      s_vv += sigma_dd(a, b, i, j) * v[a] * v[b];
      s_vw += sigma_dd(a, b, i, j) * v[a] * w[b];
    }
  sre = s_vv;   // Re[sigma] = (sigma_vv - sigma_ww)/2 = sigma_vv
  sim = -s_vw;  // Im[sigma] = -sigma_vw
}

//----------------------------------------------------------------------------------------
//! \fn Real AHF::FlowFunctionRho(...)
//  \brief Evaluate the fast-flow driving function rho = weight(theta,phi)*H
//  at surface point (i,j), per opt.flow_function (Gundlach 1998,
//  gr-qc/9809004 eq. 8-9):
//    H  : weight = 1
//    Hu : weight = u = |grad F|                                   (default)
//    F3 : weight = 2 r^2 |grad F| /
//           [ (g^ij - s^i s^j)(gbar_ij - grad_i r grad_j r) ]
//  For F3, gbar is the flat background metric of (r,theta,phi); in the
//  Cartesian components used throughout this file, gbar_ij = delta_ij and
//  grad_i r = n_i = (x-xc,y-yc,z-zc)_i / r is the flat radial unit covector,
//  so (gbar_ij - grad_i r grad_j r) = delta_ij - n_i n_j is the flat-space
//  angular projector, and n_i reduces to (sin(th)cos(ph), sin(th)sin(ph),
//  cos(th)) since the surface point is already center-relative (see
//  LevelSetGradient). Note s^i = dFdi_u/u is unit w.r.t. the PHYSICAL metric
//  g, not w.r.t. the flat metric, so sum_a(s^a)^2 != 1 in general and must
//  be computed explicitly when contracting against the flat delta_ij.
Real AHF::FlowFunctionRho(int i,
                          int j,
                          Real H,
                          Real u,
                          const ATP_N_vec& dFdi_u,
                          const ATP_N_sym& ginv)
{
  switch (opt.flow_function)
  {
    case FlowFunction::H:
      return H;

    case FlowFunction::Hu:
      return H * u;

    case FlowFunction::F3:
    default:
    {
      const Real r = rr(i, j);

      // Flat radial unit covector (center-relative, see LevelSetGradient)
      const Real n[3] = { grid_.sin_theta(i) * grid_.cos_phi(j),
                          grid_.sin_theta(i) * grid_.sin_phi(j),
                          grid_.cos_theta(i) };

      // s^i = dFdi_u(i)/u is normalized w.r.t. the PHYSICAL metric g
      // (g_ij s^i s^j = 1); it is NOT Euclidean-unit, so sum_a (s^a)^2 must
      // be computed explicitly rather than assumed to equal 1.
      Real trace_ginv = 0.0, nGn = 0.0, sn = 0.0, ss_flat = 0.0;
      for (int a = 0; a < NDIM; ++a)
      {
        const Real sa = dFdi_u(a) / u;
        trace_ginv += ginv(a, a);
        sn += sa * n[a];
        ss_flat += sa * sa;
        for (int b = 0; b < NDIM; ++b)
          nGn += ginv(a, b) * n[a] * n[b];
      }

      // D = (g^ij - s^i s^j)(delta_ij - n_i n_j)
      //   = trace(g^ij) - n^T g^{-1} n - sum_a(s^a)^2 + (s.n)^2
      const Real D = trace_ginv - nGn - ss_flat + SQR(sn);

      if (!(std::isfinite(D)) || std::fabs(D) < 1.0e-14)
        return H * u;  // degenerate fallback: behave like Hu

      const Real weight = 2.0 * SQR(r) * u / D;
      return H * weight;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real AHF::SurfaceElement(...)
//  \brief Compute the determinant of the induced 2-metric on the horizon
//  surface at point (i,j).  Returns det(h) (clamped >= 0).
Real AHF::SurfaceElement(int i, int j)
{
  return gra::grids::theta_phi::SurfaceElement2D(rr(i, j),
                                                 rr_dth(i, j),
                                                 rr_dph(i, j),
                                                 grid_.sin_theta(i),
                                                 grid_.cos_theta(i),
                                                 grid_.sin_phi(j),
                                                 grid_.cos_phi(j),
                                                 g(0, 0, i, j),
                                                 g(0, 1, i, j),
                                                 g(0, 2, i, j),
                                                 g(1, 1, i, j),
                                                 g(1, 2, i, j),
                                                 g(2, 2, i, j));
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::SpinIntegrand(...)
//  \brief Compute the spin angular momentum integrand at a surface point.
void AHF::SpinIntegrand(Real xp,
                        Real yp,
                        Real zp,
                        const ATP_N_vec& R,
                        int i,
                        int j,
                        Real& Sx,
                        Real& Sy,
                        Real& Sz)
{
  // Flat-space coordinate rotational Killing vectors
  ATP_N_vec phix;
  phix(0) = 0;
  phix(1) = -zp;
  phix(2) = yp;

  ATP_N_vec phiy;
  phiy(0) = zp;
  phiy(1) = 0;
  phiy(2) = -xp;

  ATP_N_vec phiz;
  phiz(0) = -yp;
  phiz(1) = xp;
  phiz(2) = 0;

  Sx = 0.0;
  Sy = 0.0;
  Sz = 0.0;
  for (int a = 0; a < NDIM; ++a)
    for (int b = 0; b < NDIM; ++b)
    {
      Real RbKab = R(b) * K(a, b, i, j);
      Sx += phix(a) * RbKab;
      Sy += phiy(a) * RbKab;
      Sz += phiz(a) * RbKab;
    }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::SurfaceIntegrals()
//  \brief Compute expansion, surface element and spin integrand on surface.
//  Needs metric and extrinsic curvature interpolated on the surface.
//  Performs local sums only; MPI reduce is batched in FastFlowLoop().
void AHF::SurfaceIntegrals()
{
  for (int v = 0; v < invar; v++)
    integrals[v] = 0.0;
  rho.ZeroClear();

  const int n_s2 = gra::sph_harm::lmpoints_complex(opt.lmax);
  c2_re.ZeroClear();
  c2_im.ZeroClear();

  Real sum_area = 0.0, sum_coarea = 0.0, sum_hrms = 0.0, sum_hmean = 0.0;
  Real sum_Sx = 0.0, sum_Sy = 0.0, sum_Sz = 0.0, sum_shear2 = 0.0;

#pragma omp parallel for schedule(dynamic) reduction(+ : sum_area, \
                                                      sum_coarea,  \
                                                      sum_hrms,    \
                                                      sum_hmean,   \
                                                      sum_Sx,      \
                                                      sum_Sy,      \
                                                      sum_Sz,      \
                                                      sum_shear2)
  for (int i = 0; i < grid_.ntheta; i++)
  {
    ATP_N_vec dFdi;
    ATP_N_sym dFdidj;
    ATP_N_vec R;
    ATP_N_sym nnF, ginv;
    ATP_N_vec dFdi_u;

    // Thread-private accumulators for the spin-2 projection
    std::vector<Real> t_c2_re(n_s2, 0.0), t_c2_im(n_s2, 0.0);

    for (int j = 0; j < grid_.nphi; j++)
    {
      if (!grid_.IsOwned(i, j))
        continue;

      // Level-set derivatives (Jacobian + Ylm chain rule)
      Real xp, yp, zp;
      if (!LevelSetGradient(i, j, dFdi, dFdidj, xp, yp, zp))
        break;

      // Expansion and outward unit normal (also returns nnF, ginv, dFdi_u
      // for reuse by ShearTensor and FlowFunctionRho)
      Real H, u;
      ExpansionAndNormal(i, j, dFdi, dFdidj, R, H, u, nnF, ginv, dFdi_u);
      rho(i, j) = FlowFunctionRho(i, j, H, u, dFdi_u, ginv);

      // Shear tensor sigma_ij, sigma_ij sigma^ij, and its complex spin-2
      // dyad projection sigma_ab m^a m^b at this surface point
      Real sh2, sre, sim;
      ShearTensor(i, j, dFdi, dFdi_u, nnF, ginv, u, sh2, sre, sim);
      shear2(i, j)   = sh2;
      shear_re(i, j) = sre;
      shear_im(i, j) = sim;

      // Surface area element
      Real deth = SurfaceElement(i, j);

      // Spin angular momentum integrand
      Real Sx, Sy, Sz;
      SpinIntegrand(xp, yp, zp, R, i, j, Sx, Sy, Sz);

      // Accumulate weighted integrals
      const Real wght  = grid_.weights(i, j);
      const Real sinth = grid_.sin_theta(i);
      const Real da    = wght * std::sqrt(deth) / sinth;

      sum_area += da;
      sum_coarea += wght * SQR(rr(i, j));
      sum_hrms += da * SQR(H);
      sum_hmean += da * H;
      sum_Sx += da * Sx;
      sum_Sy += da * Sy;
      sum_Sz += da * Sz;
      sum_shear2 += da * sh2;

      // Project sigma(theta,phi) onto conj(_{-2}Y_lm) using the pure
      // angular measure (grid_.weights alone, no sqrt(deth)/sinth factor --
      // this is a decomposition on the round parameter sphere, following
      // the same convention as ComplexHarmonicTable::ProjectScalar: the
      // conjugate Y* flips the sign of the imaginary part).
      for (int l = 2; l <= opt.lmax; ++l)
      {
        for (int m = -l; m <= l; ++m)
        {
          const int lm  = gra::sph_harm::lmindex_complex(l, m);
          const Real YR = swsh2_re(i, j, lm);
          const Real YI = swsh2_im(i, j, lm);
          t_c2_re[lm] += wght * (sre * YR + sim * YI);
          t_c2_im[lm] += wght * (sim * YR - sre * YI);
        }
      }

    }  // phi loop

#pragma omp critical
    {
      for (int lm = 0; lm < n_s2; ++lm)
      {
        c2_re(lm) += t_c2_re[lm];
        c2_im(lm) += t_c2_im[lm];
      }
    }
  }  // theta loop

  integrals[iarea]   = sum_area;
  integrals[icoarea] = sum_coarea;
  integrals[ihrms]   = sum_hrms;
  integrals[ihmean]  = sum_hmean;
  integrals[iSx]     = sum_Sx;
  integrals[iSy]     = sum_Sy;
  integrals[iSz]     = sum_Sz;
  integrals[ishear2] = sum_shear2;
}

//----------------------------------------------------------------------------------------
// File-local helper: map exist code (for log)
namespace
{
const char* ExitCodeName(AHF::ExitCode c)
{
  switch (c)
  {
    case AHF::ExitCode::success:
      return "success";
    case AHF::ExitCode::not_finite:
      return "not_finite";
    case AHF::ExitCode::hmean_diverged:
      return "hmean_diverged";
    case AHF::ExitCode::meanradius_neg:
      return "meanradius_neg";
    case AHF::ExitCode::mass_collapse:
      return "mass_collapse";
    case AHF::ExitCode::max_iters:
      return "max_iters";
    case AHF::ExitCode::stagnated:
      return "stagnated";
  }
  return "?";
}
}  // namespace

//----------------------------------------------------------------------------------------
// \!fn void AHF::Find(int iter, Real time)
// \brief Search for the horizons; on failure, optionally retry with adjusted
//        initial radius (see opt.auto_retry / max_retries / retry_shrink /
//        retry_grow). Direction (shrink / grow / alternate) is selected based
//        on the previous ExitCode
void AHF::Find(int iter, Real time)
{
  if ((time < opt.start_time) || (time > opt.stop_time))
    return;
  if (opt.wait_until_punc_are_close && !(PuncAreClose()))
    return;
  if (opt.verbose && (Globals::my_rank == opt.mpi_root))
  {
    fprintf(pofile_verbose, "time=%.4f, cycle=%d\n", time, iter);
  }

  // Retry bookkeeping
  const Real saved_initial_radius = opt.initial_radius;
  Real factor_total               = 1.0;
  int alternate_idx               = 0;

  // Compile-time alternate pattern: shrink, grow, shrink (cycles)
  static constexpr int alt_dir[3] = { -1, +1, -1 };  // taken modulo
  static constexpr int alt_n      = std::size(alt_dir);

  const int max_attempts = (opt.auto_retry ? opt.max_retries : 0) + 1;

  for (int attempt = 0; attempt < max_attempts; ++attempt)
  {
    const bool cold = (attempt > 0);

    if (attempt > 0)
    {
      // Pick direction from prior attempt's exit code.
      Real factor_step = 1.0;
      switch (last_exit)
      {
        case ExitCode::not_finite:
        case ExitCode::hmean_diverged:
          factor_step = opt.retry_shrink;
          break;
        case ExitCode::mass_collapse:
          factor_step = opt.retry_grow;
          break;
        case ExitCode::meanradius_neg:
        case ExitCode::max_iters:
        case ExitCode::stagnated:
        {
          const int dir = alt_dir[alternate_idx % alt_n];
          factor_step   = (dir < 0) ? opt.retry_shrink : opt.retry_grow;
          ++alternate_idx;
          break;
        }
        case ExitCode::success:
        default:
          break;  // unreachable: success exits the loop
      }
      factor_total *= factor_step;
      opt.initial_radius = saved_initial_radius * factor_total;

      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose,
                "--- AHF[%d] retry %d/%d (last=%s, radius=%.6e) ---\n",
                idx_ahf,
                attempt,
                opt.max_retries,
                ExitCodeName(last_exit),
                opt.initial_radius);
      }
    }

    InitialGuess(cold);
    FastFlowLoop();

    if (last_exit == ExitCode::success)
      break;
  }

  // Restore parameter (we mutate opt.initial_radius across attempts only).
  opt.initial_radius = saved_initial_radius;

  // Retain `last_a0` in restart: this serves as primary ini. guess.
  if (ah_found)
  {
    std::string parname;
    parname = "last_a0_" + std::to_string(idx_ahf);

    pin->SetReal("ahf", parname, last_a0);

    parname = "ah_found_a0_" + std::to_string(idx_ahf);
    pin->SetBoolean("ahf", parname, ah_found);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::FastFlowLoop()
// \brief Fast Flow loop for horizon n
void AHF::RecomputeABfac(Real alpha, Real beta, int lmax, Real* ABfac) const
{
  const Real A = alpha / (lmax * (lmax + 1)) + beta;
  const Real B = beta / alpha;
  for (int l = 0; l <= lmax; l++)
    ABfac[l] = A / (1.0 + B * l * (l + 1));
}

void AHF::FastFlowLoop()
{
  ah_found        = false;
  spec_resid_last = -1.0;

  // Set default status
  last_exit = ExitCode::max_iters;

  Real meanradius      = a0(0) / SQRT_4PI;
  Real mass            = 0;
  Real mass_prev       = 0;
  Real area            = 0;
  Real hrms            = 0;
  Real hrms_prev       = -1.0;  // no prior hrms yet
  Real hrms_best       = std::numeric_limits<Real>::infinity();
  int iters_no_improve = 0;
  Real hmean           = 0;
  Real Sx              = 0;
  Real Sy              = 0;
  Real Sz              = 0;
  Real S               = 0;
  bool failed          = false;

  if (opt.verbose && (Globals::my_rank == opt.mpi_root))
  {
    fprintf(pofile_verbose, "\nSearching for horizon %d\n", idx_ahf);
    fprintf(pofile_verbose,
            "center = (%f, %f, %f)\n",
            center[0],
            center[1],
            center[2]);
    fprintf(pofile_verbose, "r_mean = %f\n", meanradius);
    fprintf(
      pofile_verbose,
      " iter      area          mass_irr       meanradius       "
      "minradius        hmean            hrms             Sx            "
      "  Sy             "
      " Sz             S              chi            spec_resid       alpha"
      "         lmax_act\n");
  }

  // Adaptive step size: alpha can change between iterations (line search).
  // Here beta = alpha/2 as in the original Gundlach formulation.
  // ABfac[l] = A / (1 + B l(l+1)) with A,B derived from (alpha,beta) is
  // recomputed via RecomputeABfac whenever alpha changes.
  Real alpha = opt.flow_alpha_beta_const;

  const int nspec0 = opt.lmax + 1;
  const int ntotal = nspec0 + 2 * ylm_.lmpoints;

  std::vector<Real> ABfac_vec(nspec0);
  Real* ABfac = ABfac_vec.data();

  RecomputeABfac(alpha, 0.5 * alpha, opt.lmax, ABfac);

  // Caches for line search (BB requires previous iterate + previous bare
  // gradient; monotone needs only the previous residual norm).
  Real r_prev = -1.0;  // previous bare residual norm sqrt(||g||^2)

  const bool need_bb_cache =
    (opt.step_rule == StepRule::bb1 || opt.step_rule == StepRule::bb2);

  AA a0_prev, ac_prev, as_prev;
  std::vector<Real> g0_prev, gc_prev, gs_prev;
  if (need_bb_cache)
  {
    a0_prev.NewAthenaArray(opt.lmax + 1);
    ac_prev.NewAthenaArray(ylm_.lmpoints);
    as_prev.NewAthenaArray(ylm_.lmpoints);
    g0_prev.assign(nspec0, 0.0);
    gc_prev.assign(ylm_.lmpoints, 0.0);
    gs_prev.assign(ylm_.lmpoints, 0.0);
  }

  // Combined buffer: integrals[invar] + spec_buf[ntotal] + shear c2[2*n_s2]
  const int n_s2           = gra::sph_harm::lmpoints_complex(opt.lmax);
  const int combined_size  = invar + ntotal + 2 * n_s2;
  std::vector<Real> combined_buf(combined_size);
  Real* cb_integrals = combined_buf.data();
  Real* cb_spec_buf  = combined_buf.data() + invar;
  Real* cb_shear_buf = cb_spec_buf + ntotal;

  // mode_ramp: continuation in lmax.
  if (opt.mode_ramp_lmin < opt.lmax)
  {
    for (int l = opt.mode_ramp_lmin + 1; l <= opt.lmax; ++l)
      a0(l) = 0.0;
    for (int l = opt.mode_ramp_lmin + 1; l <= opt.lmax; ++l)
    {
      for (int m = 1; m <= l; ++m)
      {
        const int l1 = ylm_.lmindex(l, m);
        ac(l1)       = 0.0;
        as(l1)       = 0.0;
      }
    }
  }

  auto lmax_active_at = [&](int k) -> int
  {
    const int steps = k / opt.mode_ramp_iters_per_step;
    return std::min(opt.mode_ramp_lmin + steps * opt.mode_ramp_modes_per_step,
                    opt.lmax);
  };

  int lmax_active      = lmax_active_at(0);
  int lmax_active_prev = -1;

  for (int k = 0; k < opt.flow_iterations; k++)
  {
    fastflow_iter = k;

    // Update mode_ramp state for this iteration
    lmax_active_prev            = lmax_active;
    lmax_active                 = lmax_active_at(k);
    const bool ramp_in_progress = (lmax_active < opt.lmax);
    const bool ramp_transition  = (k > 0 && lmax_active != lmax_active_prev);
    const bool ramp_just_finished =
      (lmax_active_prev < opt.lmax && lmax_active >= opt.lmax);

    // Compute radius r = a_lm Y_lm
    ylm_.Synthesize(
      a0, ac, as, grid_.ntheta, grid_.nphi, rr, rr_dth, rr_dph, rr_min);

    // Fill x_cart with sphere coordinates (bitant-reflected)
    grid_.FillCartesianCoords(center, rr, opt.bitant);

    // Compute Jacobian d(r,th,ph)/d(x,y,z)
    grid_.ComputeConJacobian(rr, min_surface_radius);

    // Build interpolator pools
    grid_.Prepare(pmesh, SW_CCX_VC(true, false), SW_CCX_VC(false, true));

    // Zero metric arrays and interpolate on surface
    g.ZeroClear();
    dg.ZeroClear();
    K.ZeroClear();
    MetricInterp();

    // Compute local sums for surface integrals (no MPI reduce)
    SurfaceIntegrals();

    // Compute local sums for spectral projection (optimistic, before reduce)
    std::fill(cb_spec_buf, cb_spec_buf + ntotal, 0.0);
    Real* spec0 = cb_spec_buf;
    Real* specc = cb_spec_buf + nspec0;
    Real* specs = specc + ylm_.lmpoints;

    ylm_.Project(grid_.weights,
                 rho,
                 grid_.ntheta,
                 grid_.nphi,
                 spec0,
                 specc,
                 specs,
                 [this](int i, int j) { return grid_.IsOwned(i, j); });

    // Pack integrals + local shear spin-2 coefficients into the combined
    // buffer (folded into the same reduce as the spectral update, avoiding
    // an extra communication round-trip)
    std::memcpy(cb_integrals, integrals, invar * sizeof(Real));
    std::memcpy(cb_shear_buf, c2_re.data(), n_s2 * sizeof(Real));
    std::memcpy(cb_shear_buf + n_s2, c2_im.data(), n_s2 * sizeof(Real));

    // Single batched MPI_Allreduce for integrals, spectral sums, and shear
    // spin-2 coefficients
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,
                  combined_buf.data(),
                  combined_size,
                  MPI_ATHENA_REAL,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif

    // Unpack reduced integrals and shear coefficients
    std::memcpy(integrals, cb_integrals, invar * sizeof(Real));
    std::memcpy(c2_re.data(), cb_shear_buf, n_s2 * sizeof(Real));
    std::memcpy(c2_im.data(), cb_shear_buf + n_s2, n_s2 * sizeof(Real));

    // mode_ramp: zero gradient components for l > lmax_active.
    if (lmax_active < opt.lmax)
    {
      for (int l = lmax_active + 1; l <= opt.lmax; ++l)
        spec0[l] = 0.0;
      for (int l = lmax_active + 1; l <= opt.lmax; ++l)
      {
        for (int m = 1; m <= l; ++m)
        {
          const int l1 = ylm_.lmindex(l, m);
          specc[l1]    = 0.0;
          specs[l1]    = 0.0;
        }
      }
    }

    area  = integrals[iarea];
    hrms  = std::sqrt(integrals[ihrms] / area);
    hmean = integrals[ihmean];
    Sx    = integrals[iSx] / (8 * PI);
    Sy    = integrals[iSy] / (8 * PI);
    Sz    = integrals[iSz] / (8 * PI);
    S     = std::sqrt(SQR(Sx) + SQR(Sy) + SQR(Sz));

    meanradius = a0(0) / SQRT_4PI;

    // Bare spectral residual: alpha-independent (uses raw projections, not
    // ABfac-weighted) so it remains a meaningful descent indicator when alpha
    // varies across iterations.
    Real gnorm2 = 0.0, anorm2 = 0.0;
    for (int l = 0; l <= opt.lmax; l++)
    {
      gnorm2 += spec0[l] * spec0[l];
      anorm2 += a0(l) * a0(l);
    }
    for (int l = 1; l <= opt.lmax; l++)
    {
      for (int m = 1; m <= l; m++)
      {
        const int l1 = ylm_.lmindex(l, m);
        gnorm2 += specc[l1] * specc[l1] + specs[l1] * specs[l1];
        anorm2 += ac(l1) * ac(l1) + as(l1) * as(l1);
      }
    }
    const Real gnorm = std::sqrt(gnorm2);
    const Real spec_resid =
      gnorm / std::max(std::sqrt(anorm2), min_surface_radius);
    spec_resid_last = spec_resid;

    // Adaptive alpha update (line-search rules). Performed BEFORE the spectral
    // update so the new alpha shapes the step about to be taken.
    if (k >= 1 && opt.step_rule != StepRule::fixed)
    {
      if (opt.step_rule == StepRule::monotone)
      {
        // Grow alpha on descent, shrink on overshoot.
        if (gnorm < r_prev)
          alpha = std::min(alpha * opt.alpha_grow, opt.alpha_max);
        else
          alpha = std::max(alpha * opt.alpha_shrink, opt.alpha_min);
        RecomputeABfac(alpha, 0.5 * alpha, opt.lmax, ABfac);
      }
      else if (need_bb_cache && !ramp_transition)
      {
        // Barzilai-Borwein step from (a_k - a_{k-1}) and (g_k - g_{k-1}).
        Real ss = 0.0, sy = 0.0, yy = 0.0;
        for (int l = 0; l <= opt.lmax; l++)
        {
          const Real ds = a0(l) - a0_prev(l);
          const Real dy = spec0[l] - g0_prev[l];
          ss += ds * ds;
          sy += ds * dy;
          yy += dy * dy;
        }
        for (int l = 1; l <= opt.lmax; l++)
        {
          for (int m = 1; m <= l; m++)
          {
            const int l1   = ylm_.lmindex(l, m);
            const Real dsc = ac(l1) - ac_prev(l1);
            const Real dyc = specc[l1] - gc_prev[l1];
            const Real dss = as(l1) - as_prev(l1);
            const Real dys = specs[l1] - gs_prev[l1];
            ss += dsc * dsc + dss * dss;
            sy += dsc * dyc + dss * dys;
            yy += dyc * dyc + dys * dys;
          }
        }
        Real alpha_bb = alpha;
        if (opt.step_rule == StepRule::bb1)
        {
          if (std::isfinite(sy) && std::fabs(sy) > 0.0)
            alpha_bb = ss / sy;
        }
        else  // bb2 (more aggressive)
        {
          if (std::isfinite(yy) && yy > 0.0)
            alpha_bb = sy / yy;
        }
        if (std::isfinite(alpha_bb) && alpha_bb > 0.0)
        {
          alpha = std::min(std::max(alpha_bb, opt.alpha_min), opt.alpha_max);
          RecomputeABfac(alpha, 0.5 * alpha, opt.lmax, ABfac);
        }
      }
    }
    r_prev = gnorm;

    // Cache current iterate + bare gradient for next BB step
    if (need_bb_cache)
    {
      for (int l = 0; l <= opt.lmax; l++)
      {
        a0_prev(l) = a0(l);
        g0_prev[l] = spec0[l];
      }
      for (int l1 = 0; l1 < ylm_.lmpoints; l1++)
      {
        ac_prev(l1) = ac(l1);
        as_prev(l1) = as(l1);
        gc_prev[l1] = specc[l1];
        gs_prev[l1] = specs[l1];
      }
    }

    // Check we get a finite result
    if (!(std::isfinite(area)))
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed, Area not finite\n");
        fflush(pofile_verbose);
      }
      last_exit = ExitCode::not_finite;
      failed    = true;
      break;
    }

    if (!(std::isfinite(hmean)))
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed, hmean not finite\n");
        fflush(pofile_verbose);
      }
      last_exit = ExitCode::not_finite;
      failed    = true;
      break;
    }

    // Irreducible mass
    mass_prev = mass;
    mass      = std::sqrt(area / (16.0 * PI));

    if (opt.verbose && (Globals::my_rank == opt.mpi_root))
    {
      const Real mass_C_iter = std::sqrt(SQR(mass) + 0.25 * SQR(S / mass));
      const Real chi_iter =
        (mass_C_iter > min_mass) ? S / SQR(mass_C_iter) : 0.0;
      fprintf(pofile_verbose,
              "%3d %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e "
              "%15.7e %15.7e %15.7e %15.7e %15.7e %15d\n",
              k,
              area,
              mass,
              meanradius,
              rr_min,
              hmean,
              hrms,
              Sx,
              Sy,
              Sz,
              S,
              chi_iter,
              spec_resid,
              alpha,
              lmax_active);
      fflush(pofile_verbose);
    }

    // Divergence catch: |hmean| blowing up past hmean_tol indicates the
    // surface has run away.
    if (std::fabs(hmean) > opt.hmean_tol)
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed, hmean > %f\n", opt.hmean_tol);
        fflush(pofile_verbose);
      }
      last_exit = ExitCode::hmean_diverged;
      failed    = true;
      break;
    }

    if (meanradius < 0.)
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed, meanradius < 0\n");
        fflush(pofile_verbose);
      }
      last_exit = ExitCode::meanradius_neg;
      failed    = true;
      break;
    }

    // Check to prevent horizon radius blow up and mass = 0
    if (mass < min_mass)
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed mass < min_mass\n");
        fflush(pofile_verbose);
      }
      last_exit = ExitCode::mass_collapse;
      failed    = true;
      break;
    }

    // End flow criteria:
    // - Require k >= 1 so that mass_prev was set from a previous iteration
    // - Mass must satisfy tol
    // - hrms*mass < hrms_tol
    // - |hrms - hrms_prev| < hrms_rel_tol * hrms_prev
    // - spec_resid: spectral residual on the projected update is small
    const bool hrms_abs_ok = (hrms * mass < opt.hrms_tol);
    const bool hrms_rel_ok =
      (k >= 2) && (hrms_prev > 0.0) &&
      (std::fabs(hrms - hrms_prev) < opt.hrms_rel_tol * hrms_prev);

    if ((k >= 1) && (std::fabs(mass_prev - mass) < opt.mass_tol) &&
        hrms_abs_ok && hrms_rel_ok && (spec_resid < opt.spec_tol) &&
        (lmax_active >= opt.lmax))
    {
      ah_found  = true;
      last_exit = ExitCode::success;
      break;
    }

    // Stagnation detection: hrms failing to improve over a window while still
    // above the absolute hrms tolerate. Suppressed while mode_ramp is active;
    // reset stagnation tracking at the moment ramp completes so the window
    // starts fresh in the final phase.
    {
      if (ramp_just_finished)
      {
        hrms_best        = std::numeric_limits<Real>::infinity();
        iters_no_improve = 0;
      }

      if (!ramp_in_progress)
      {
        const Real impr_thresh =
          hrms_best * (1.0 - opt.stagnation_improvement_frac);
        if (hrms < impr_thresh)
        {
          hrms_best        = hrms;
          iters_no_improve = 0;
        }
        else
        {
          ++iters_no_improve;
        }

        const bool past_warmup = (k >= opt.stagnation_warmup);
        const bool below_abs   = (hrms * mass < opt.hrms_tol);

        if (opt.stagnation_detect && past_warmup && !below_abs &&
            iters_no_improve >= opt.stagnation_window)
        {
          if (opt.verbose && (Globals::my_rank == opt.mpi_root))
          {
            fprintf(pofile_verbose,
                    "Stagnated, hrms %.3e (best %.3e) for %d iters; "
                    "abs gate hrms*mass=%.3e >= hrms_tol=%.3e\n",
                    hrms,
                    hrms_best,
                    iters_no_improve,
                    hrms * mass,
                    opt.hrms_tol);
            fflush(pofile_verbose);
          }
          last_exit = ExitCode::stagnated;
          failed    = true;
          break;
        }
      }
    }

    hrms_prev = hrms;

    // Apply reduced spectral update (optimistic projection was done above)
    for (int l = 0; l <= opt.lmax; l++)
      a0(l) -= ABfac[l] * spec0[l];

    for (int l = 1; l <= opt.lmax; l++)
    {
      for (int m = 1; m <= l; m++)
      {
        int l1 = ylm_.lmindex(l, m);
        ac(l1) -= ABfac[l] * specc[l1];
        as(l1) -= ABfac[l] * specs[l1];
      }
    }

    // Release pools (AHF rebuilds every iteration)
    grid_.TearDown();
  }

  // Ensure pools are released after early-exit breaks
  grid_.TearDown();

  if (ah_found)
  {
    last_a0 = a0(0);

    // Retain for potential use as next initial guess
    for (int l = 0; l <= opt.lmax; ++l)
    {
      last_a0_full(l) = a0(l);
    }
    for (int k = 0; k < ylm_.lmpoints; ++k)
    {
      last_ac(k) = ac(k);
      last_as(k) = as(k);
    }

    ah_prop[harea]       = area;
    ah_prop[hcoarea]     = integrals[icoarea];
    ah_prop[hhrms]       = hrms;
    ah_prop[hhmean]      = hmean;
    ah_prop[hmeanradius] = meanradius;
    ah_prop[hminradius]  = rr_min;
    ah_prop[hSx]         = Sx;
    ah_prop[hSy]         = Sy;
    ah_prop[hSz]         = Sz;
    ah_prop[hS]          = S;
    ah_prop[hshearrms]   = std::sqrt(integrals[ishear2] / area);
    // Instantaneous GW flux through the horizon (Ashtekar-Krishnan
    // dynamical-horizon flux law, gravitational-shear term):
    //   dE/dt = 1/(16 pi) * oint sigma_ij sigma^ij dA
    ah_prop[hgwflux]     = integrals[ishear2] / (16.0 * M_PI);
    // Christodoulou mass
    ah_prop[hmass]     = std::sqrt(SQR(mass) + 0.25 * SQR(S / mass));
    ah_prop[hmass_irr] = mass;
    ah_prop[hchi] =
      (ah_prop[hmass] > min_mass) ? S / SQR(ah_prop[hmass]) : 0.0;
  }

  if (opt.verbose && (Globals::my_rank == opt.mpi_root))
  {
    if (ah_found)
    {
      fprintf(pofile_verbose, "Found horizon %d\n", idx_ahf);
      fprintf(pofile_verbose, " mass = %f\n", ah_prop[hmass]);
      fprintf(pofile_verbose, " mass_irr = %f\n", mass);
      fprintf(pofile_verbose, " meanradius = %f\n", meanradius);
      fprintf(pofile_verbose, " minradius = %f\n", rr_min);
      fprintf(pofile_verbose, " hrms = %f\n", hrms);
      fprintf(pofile_verbose, " hmean = %f\n", hmean);
      fprintf(pofile_verbose, " Sx = %f\n", Sx);
      fprintf(pofile_verbose, " Sy = %f\n", Sy);
      fprintf(pofile_verbose, " Sz = %f\n", Sz);
      fprintf(pofile_verbose, " S  = %f\n", S);
      fprintf(pofile_verbose, " chi = %f\n", ah_prop[hchi]);
      fprintf(pofile_verbose, " shear_rms = %f\n", ah_prop[hshearrms]);
      fprintf(pofile_verbose, " gw_flux = %e\n", ah_prop[hgwflux]);
    }
    else if (!failed && !ah_found)
    {
      fprintf(pofile_verbose,
              "Failed, reached max iterations %d\n",
              opt.flow_iterations);
    }
    fflush(pofile_verbose);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::UpdateFlowSpectralComponents(const int n)
// \brief find new spectral components with fast flow

void AHF::UpdateFlowSpectralComponents()
{
  const Real alpha = opt.flow_alpha_beta_const;
  const Real beta  = 0.5 * opt.flow_alpha_beta_const;
  const Real A     = alpha / (opt.lmax * (opt.lmax + 1)) + beta;
  const Real B     = beta / alpha;

  const int nspec0 = opt.lmax + 1;
  const int ntotal = nspec0 + 2 * ylm_.lmpoints;

  std::vector<Real> ABfac_vec(nspec0);
  std::vector<Real> spec_buf_vec(ntotal, 0.0);  // zero-initialized
  Real* ABfac    = ABfac_vec.data();
  Real* spec_buf = spec_buf_vec.data();

  Real* spec0 = spec_buf;
  Real* specc = spec_buf + nspec0;
  Real* specs = specc + ylm_.lmpoints;

  for (int l = 0; l <= opt.lmax; l++)
  {
    ABfac[l] = A / (1.0 + B * l * (l + 1));
  }

  // Local sums via ylm_.Project
  ylm_.Project(grid_.weights,
               rho,
               grid_.ntheta,
               grid_.nphi,
               spec0,
               specc,
               specs,
               [this](int i, int j) { return grid_.IsOwned(i, j); });

#ifdef MPI_PARALLEL
  MPI_Allreduce(
    MPI_IN_PLACE, spec_buf, ntotal, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Update the coefs
  for (int l = 0; l <= opt.lmax; l++)
  {
    a0(l) -= ABfac[l] * spec0[l];
  }

  for (int l = 1; l <= opt.lmax; l++)
  {
    for (int m = 1; m <= l; m++)
    {
      int l1 = ylm_.lmindex(l, m);
      ac(l1) -= ABfac[l] * specc[l1];
      as(l1) -= ABfac[l] * specs[l1];
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::InitialGuess(bool cold)
// \brief initial guess for spectral coefs of horizon n.
//        If `cold` is true, ignore the warm-start cache.
void AHF::InitialGuess(bool cold)
{
  // ---- Phase A: center update --------------------------------------------
  // Center updates are applied in order so that later options (extrema /
  // mass-weighted) can override an earlier one (puncture) when combined.
  if (opt.use_puncture >= 0)
  {
    center[0] = pmesh->pz4c_tracker[opt.use_puncture]->GetPos(0);
    center[1] = pmesh->pz4c_tracker[opt.use_puncture]->GetPos(1);
    center[2] = pmesh->pz4c_tracker[opt.use_puncture]->GetPos(2);
  }

  if (opt.use_extrema >= 0)
  {
    center[0] = pmesh->ptracker_extrema->c_x1(opt.use_extrema);
    center[1] = pmesh->ptracker_extrema->c_x2(opt.use_extrema);
    center[2] = pmesh->ptracker_extrema->c_x3(opt.use_extrema);
  }

  if (opt.use_puncture_massweighted_center)
  {
    Real pos[3];
    PuncWeightedMassCentralPoint(&pos[0], &pos[1], &pos[2]);
    center[0] = pos[0];
    center[1] = pos[1];
    center[2] = pos[2];
  }

  // ---- Phase B: coefficient guess ----------------------------------------
  a0.ZeroClear();
  ac.ZeroClear();
  as.ZeroClear();

  if (!cold && ah_found && last_a0 > 0)
  {
    if (opt.propagate_iter_coefficients)
    {
      // Seed with the full last-found spectral shape.
      for (int l = 0; l <= opt.lmax; ++l)
      {
        a0(l) = last_a0_full(l);
      }
      for (int k = 0; k < ylm_.lmpoints; ++k)
      {
        ac(k) = last_ac(k);
        as(k) = last_as(k);
      }
    }
    else
    {
      a0(0) = last_a0;
    }

    // expand_guess scales only the mean-radius mode (the enclosing sphere);
    // higher-l deviations are left untouched.
    a0(0) *= opt.expand_guess;
    return;
  }

  // No prior find: fall back to a config-driven guess.
  if (opt.use_puncture >= 0)
  {
    // For single BH in isotropic coordinates: horizon radius = m/2, but
    // ensure a0(0) comfortably surrounds all punctures, i.e. a bit larger
    // than half the distance between any of the punctures.
    const Real mass      = pmesh->pz4c_tracker[opt.use_puncture]->GetMass();
    const Real largedist = PuncMaxDistance(opt.use_puncture);
    a0(0) = SQRT_4PI * std::max(0.5 * mass, std::min(mass, 0.5 * largedist));
  }
  else
  {
    a0(0) = SQRT_4PI * opt.initial_radius;
  }
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::PuncMaxDistance()
// \brief Max Euclidean distance between punctures

Real AHF::PuncMaxDistance()
{
  const int npunct = static_cast<int>(pmesh->pz4c_tracker.size());
  Real maxdist     = 0.0;
  for (int pix = 0; pix < npunct; ++pix)
  {
    Real xp = pmesh->pz4c_tracker[pix]->GetPos(0);
    Real yp = pmesh->pz4c_tracker[pix]->GetPos(1);
    Real zp = pmesh->pz4c_tracker[pix]->GetPos(2);
    for (int p = pix + 1; p < npunct; ++p)
    {
      Real x = pmesh->pz4c_tracker[p]->GetPos(0);
      Real y = pmesh->pz4c_tracker[p]->GetPos(1);
      Real z = pmesh->pz4c_tracker[p]->GetPos(2);
      maxdist =
        std::max(maxdist, std::sqrt(SQR(x - xp) + SQR(y - yp) + SQR(z - zp)));
    }
  }
  return maxdist;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::PuncMaxDistance(const int pix)
// \brief Max Euclidean distance from puncture pix to other punctures

Real AHF::PuncMaxDistance(const int pix)
{
  const int npunct = static_cast<int>(pmesh->pz4c_tracker.size());
  Real xp          = pmesh->pz4c_tracker[pix]->GetPos(0);
  Real yp          = pmesh->pz4c_tracker[pix]->GetPos(1);
  Real zp          = pmesh->pz4c_tracker[pix]->GetPos(2);
  Real maxdist     = 0.0;
  for (int p = 0; p < npunct; ++p)
  {
    if (p == pix)
      continue;
    Real x = pmesh->pz4c_tracker[p]->GetPos(0);
    Real y = pmesh->pz4c_tracker[p]->GetPos(1);
    Real z = pmesh->pz4c_tracker[p]->GetPos(2);
    maxdist =
      std::max(maxdist, std::sqrt(SQR(x - xp) + SQR(y - yp) + SQR(z - zp)));
  }
  return maxdist;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::PuncSumMasses()
// \brief Return sum of puncture's intial masses

Real AHF::PuncSumMasses()
{
  const int npunct = static_cast<int>(pmesh->pz4c_tracker.size());
  Real mass        = 0.0;
  for (int p = 0; p < npunct; ++p)
  {
    mass += pmesh->pz4c_tracker[p]->GetMass();
  }
  return mass;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc)
// \brief Return mss-weighted center of puncture positions

void AHF::PuncWeightedMassCentralPoint(Real* xc, Real* yc, Real* zc)
{
  const int npunct = static_cast<int>(pmesh->pz4c_tracker.size());
  Real sumx        = 0.0;  // sum of m_i*x_i
  Real sumy        = 0.0;
  Real sumz        = 0.0;
  Real divsum      = 0.0;  // sum of m_i to later divide by
  for (int p = 0; p < npunct; p++)
  {
    Real x = pmesh->pz4c_tracker[p]->GetPos(0);
    Real y = pmesh->pz4c_tracker[p]->GetPos(1);
    Real z = pmesh->pz4c_tracker[p]->GetPos(2);
    Real m = pmesh->pz4c_tracker[p]->GetMass();
    sumx += m * x;
    sumy += m * y;
    sumz += m * z;
    divsum += m;
  }
  divsum = 1.0 / divsum;
  *xc    = sumx * divsum;
  *yc    = sumy * divsum;
  *zc    = sumz * divsum;
}

//----------------------------------------------------------------------------------------
// \!fn int AHF::PuncAreClose()
// \brief Check when the maximal distance between all punctures is below
// threshold

bool AHF::PuncAreClose()
{
  Real const mass    = PuncSumMasses();
  Real const maxdist = PuncMaxDistance();
  return (maxdist < opt.merger_distance * mass);
}
