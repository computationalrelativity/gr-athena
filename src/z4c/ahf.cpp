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
#include <sstream>
#include <stdexcept>

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
  const int ntheta_val = pin->GetOrAddInteger("ahf", "ntheta", 5);
  const int nphi_val   = pin->GetOrAddInteger("ahf", "nphi", 10);
  std::string quadrature =
    pin->GetOrAddString("ahf", "quadrature", "gausslegendre");
  if (quadrature == "sums")
    quadrature = "midpoint";
  grid_.Initialize(ntheta_val, nphi_val, quadrature);

  opt.lmax = pin->GetOrAddInteger("ahf", "lmax", 4);

  opt.flow_iterations =
    pin->GetOrAddInteger("ahf", parkey("flow_iterations_"), 100);

  opt.flow_alpha_beta_const =
    pin->GetOrAddReal("ahf", parkey("flow_alpha_beta_const_"), 1.0);

  opt.hmean_tol = pin->GetOrAddReal("ahf", parkey("hmean_tol_"), 100.);

  opt.mass_tol = pin->GetOrAddReal("ahf", parkey("mass_tol_"), 1e-2);

  opt.spec_tol = pin->GetOrAddReal("ahf", parkey("spec_tol_"), 1e-5);

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

  // Initialize horizon properties to NAN
  for (int v = 0; v < hnvar; ++v)
  {
    ah_prop[v] = NAN;
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
              "# 1:iter 2:time 3:mass 4:Sx 5:Sy 6:Sz 7:S 8:area 9:hrms "
              "10:hmean 11:meanradius 12:minradius 13:converged "
              "14:num_iters 15:spec_resid\n");
      fflush(pofile_summary);
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
            "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e",
            ah_prop[hmass],
            ah_prop[hSx],
            ah_prop[hSy],
            ah_prop[hSz],
            ah_prop[hS],
            ah_prop[harea],
            ah_prop[hhrms],
            ah_prop[hhmean],
            ah_prop[hmeanradius],
            ah_prop[hminradius]);
    fprintf(pofile_summary,
            " %d %d %.15e",
            (ah_found ? 1 : 0),
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
//  surface point (i,j).
void AHF::ExpansionAndNormal(int i,
                             int j,
                             const ATP_N_vec& dFdi,
                             const ATP_N_sym& dFdidj,
                             ATP_N_vec& R,
                             Real& H,
                             Real& u)
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

  Real sum_area = 0.0, sum_coarea = 0.0, sum_hrms = 0.0, sum_hmean = 0.0;
  Real sum_Sx = 0.0, sum_Sy = 0.0, sum_Sz = 0.0;

#pragma omp parallel for schedule(dynamic) reduction( \
    + : sum_area, sum_coarea, sum_hrms, sum_hmean, sum_Sx, sum_Sy, sum_Sz)
  for (int i = 0; i < grid_.ntheta; i++)
  {
    ATP_N_vec dFdi;
    ATP_N_sym dFdidj;
    ATP_N_vec R;

    for (int j = 0; j < grid_.nphi; j++)
    {
      if (!grid_.IsOwned(i, j))
        continue;

      // Level-set derivatives (Jacobian + Ylm chain rule)
      Real xp, yp, zp;
      if (!LevelSetGradient(i, j, dFdi, dFdidj, xp, yp, zp))
        break;

      // Expansion and outward unit normal
      Real H, u;
      ExpansionAndNormal(i, j, dFdi, dFdidj, R, H, u);
      rho(i, j) = H * u;

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

    }  // phi loop
  }  // theta loop

  integrals[iarea]   = sum_area;
  integrals[icoarea] = sum_coarea;
  integrals[ihrms]   = sum_hrms;
  integrals[ihmean]  = sum_hmean;
  integrals[iSx]     = sum_Sx;
  integrals[iSy]     = sum_Sy;
  integrals[iSz]     = sum_Sz;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::Find(int iter, Real time)
// \brief Search for the horizons
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
  InitialGuess();
  FastFlowLoop();

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

  Real meanradius = a0(0) / SQRT_4PI;
  Real mass       = 0;
  Real mass_prev  = 0;
  Real area       = 0;
  Real hrms       = 0;
  Real hmean      = 0;
  Real Sx         = 0;
  Real Sy         = 0;
  Real Sz         = 0;
  Real S          = 0;
  bool failed     = false;

  if (opt.verbose && (Globals::my_rank == opt.mpi_root))
  {
    fprintf(pofile_verbose, "\nSearching for horizon %d\n", idx_ahf);
    fprintf(pofile_verbose,
            "center = (%f, %f, %f)\n",
            center[0],
            center[1],
            center[2]);
    fprintf(pofile_verbose, "r_mean = %f\n", meanradius);
    fprintf(pofile_verbose,
            " iter      area            mass         meanradius       "
            "minradius        hmean            Sx              Sy             "
            " Sz             S              spec_resid       alpha\n");
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

  // Combined buffer: integrals[invar] + spec_buf[ntotal]
  const int combined_size = invar + ntotal;
  std::vector<Real> combined_buf(combined_size);
  Real* cb_integrals = combined_buf.data();
  Real* cb_spec_buf  = combined_buf.data() + invar;

  for (int k = 0; k < opt.flow_iterations; k++)
  {
    fastflow_iter = k;

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

    // Pack integrals into the combined buffer
    std::memcpy(cb_integrals, integrals, invar * sizeof(Real));

    // Single batched MPI_Allreduce for both integrals and spectral sums
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,
                  combined_buf.data(),
                  combined_size,
                  MPI_ATHENA_REAL,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#endif

    // Unpack reduced integrals
    std::memcpy(integrals, cb_integrals, invar * sizeof(Real));

    area  = integrals[iarea];
    hrms  = integrals[ihrms] / area;
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
      else if (need_bb_cache)
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
      failed = true;
      break;
    }

    if (!(std::isfinite(hmean)))
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed, hmean not finite\n");
        fflush(pofile_verbose);
      }
      failed = true;
      break;
    }

    // Irreducible mass
    mass_prev = mass;
    mass      = std::sqrt(area / (16.0 * PI));

    if (opt.verbose && (Globals::my_rank == opt.mpi_root))
    {
      fprintf(pofile_verbose,
              "%3d %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e "
              "%15.7e %15.7e %15.7e\n",
              k,
              area,
              mass,
              meanradius,
              rr_min,
              hmean,
              Sx,
              Sy,
              Sz,
              S,
              spec_resid,
              alpha);
      fflush(pofile_verbose);
    }

    if (std::fabs(hmean) > opt.hmean_tol)
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed, hmean > %f\n", opt.hmean_tol);
        fflush(pofile_verbose);
      }
      failed = true;
      break;
    }

    if (meanradius < 0.)
    {
      if (opt.verbose && (Globals::my_rank == opt.mpi_root))
      {
        fprintf(pofile_verbose, "Failed, meanradius < 0\n");
        fflush(pofile_verbose);
      }
      failed = true;
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
      failed = true;
      break;
    }

    // End flow criteria:
    // - Require k >= 1 so that mass_prev was set from a previous iteration
    // - Mass must satisfy tol
    // - hmean tol must be satisfied
    if ((k >= 1) && (std::fabs(mass_prev - mass) < opt.mass_tol) &&
        (std::fabs(hmean) < opt.hmean_tol) && (spec_resid < opt.spec_tol))
    {
      ah_found = true;
      break;
    }
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
    // Christodoulou mass
    ah_prop[hmass] = std::sqrt(SQR(mass) + 0.25 * SQR(S / mass));
  }

  if (opt.verbose && (Globals::my_rank == opt.mpi_root))
  {
    if (ah_found)
    {
      fprintf(pofile_verbose, "Found horizon %d\n", idx_ahf);
      fprintf(pofile_verbose, " mass_irr = %f\n", mass);
      fprintf(pofile_verbose, " meanradius = %f\n", meanradius);
      fprintf(pofile_verbose, " minradius = %f\n", rr_min);
      fprintf(pofile_verbose, " hrms = %f\n", hrms);
      fprintf(pofile_verbose, " hmean = %f\n", hmean);
      fprintf(pofile_verbose, " Sx = %f\n", Sx);
      fprintf(pofile_verbose, " Sy = %f\n", Sy);
      fprintf(pofile_verbose, " Sz = %f\n", Sz);
      fprintf(pofile_verbose, " S  = %f\n", S);
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
// \!fn void AHF::InitialGuess()
// \brief initial guess for spectral coefs of horizon n
void AHF::InitialGuess()
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

  if (ah_found && last_a0 > 0)
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
