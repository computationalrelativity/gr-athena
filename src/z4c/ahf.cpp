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

#include <cmath>  // NAN
#include <cstdio>
#include <sstream>
#include <stdexcept>

#ifdef MPI_PARALLEL
#include <mpi.h>
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

  opt.verbose         = pin->GetOrAddBoolean("ahf", "verbose", false);
  opt.mpi_root        = pin->GetOrAddInteger("ahf", "mpi_root", 0);
  opt.merger_distance = pin->GetOrAddReal("ahf", "merger_distance", 0.1);
  opt.bitant          = pin->GetOrAddBoolean("mesh", "bitant", false);

  // Initial guess
  opt.initial_radius =
    pin->GetOrAddReal("ahf", parkey("initial_radius_"), 1.0);
  rr_min = -1.0;

  opt.expand_guess = pin->GetOrAddReal("ahf", "expand_guess", 1.0);

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
              "10:hmean 11:meanradius 12:minradius\n");
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

  for (int i = 0; i < grid_.ntheta; ++i)
  {
    const Real costh = grid_.cos_theta(i);

    for (int j = 0; j < grid_.nphi; ++j)
    {
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
//  Performs local sums and MPI reduce.
void AHF::SurfaceIntegrals()
{
  for (int v = 0; v < invar; v++)
    integrals[v] = 0.0;
  rho.ZeroClear();

  ATP_N_vec dFdi;
  ATP_N_sym dFdidj;
  ATP_N_vec R;

  for (int i = 0; i < grid_.ntheta; i++)
  {
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

      integrals[iarea] += da;
      integrals[icoarea] += wght * SQR(rr(i, j));
      integrals[ihrms] += da * SQR(H);
      integrals[ihmean] += da * H;
      integrals[iSx] += da * Sx;
      integrals[iSy] += da * Sy;
      integrals[iSz] += da * Sz;

    }  // phi loop
  }  // theta loop

#ifdef MPI_PARALLEL
  MPI_Allreduce(
    MPI_IN_PLACE, integrals, invar, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
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
void AHF::FastFlowLoop()
{
  ah_found = false;

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
            " Sz             S\n");
  }

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

    SurfaceIntegrals();

    area  = integrals[iarea];
    hrms  = integrals[ihrms] / area;
    hmean = integrals[ihmean];
    Sx    = integrals[iSx] / (8 * PI);
    Sy    = integrals[iSy] / (8 * PI);
    Sz    = integrals[iSz] / (8 * PI);
    S     = std::sqrt(SQR(Sx) + SQR(Sy) + SQR(Sz));

    meanradius = a0(0) / SQRT_4PI;

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
      fprintf(
        pofile_verbose,
        "%3d %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e\n",
        k,
        area,
        mass,
        meanradius,
        rr_min,
        hmean,
        Sx,
        Sy,
        Sz,
        S);
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

    // End flow when mass difference is small
    if (std::fabs(mass_prev - mass) < opt.mass_tol)
    {
      ah_found = true;
      break;
    }

    // Find new spectral components
    UpdateFlowSpectralComponents();

    // Release pools (AHF rebuilds every iteration)
    grid_.TearDown();
  }

  // Ensure pools are released after early-exit breaks
  grid_.TearDown();

  if (ah_found)
  {
    last_a0 = a0(0);

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
  // Reset Coefficients to Zero
  a0.ZeroClear();
  ac.ZeroClear();
  as.ZeroClear();

  if (opt.use_puncture >= 0)
  {
    // Update the center to the puncture position
    center[0] = pmesh->pz4c_tracker[opt.use_puncture]->GetPos(0);
    center[1] = pmesh->pz4c_tracker[opt.use_puncture]->GetPos(1);
    center[2] = pmesh->pz4c_tracker[opt.use_puncture]->GetPos(2);
    // Update a0
    // For single BH in isotropic coordinates: horizon radius=m/2
    // but make sure it can surround all punctures comfortably, i.e.
    // make radius a bit larger than half the distance between any of the
    // punctures
    Real mass      = pmesh->pz4c_tracker[opt.use_puncture]->GetMass();
    Real largedist = PuncMaxDistance(opt.use_puncture);
    if (ah_found && last_a0 > 0)
    {
      a0(0) = last_a0 * opt.expand_guess;
    }
    else
    {
      a0(0) = std::max(0.5 * mass, std::min(mass, 0.5 * largedist));
      a0(0) *= SQRT_4PI;
    }
    return;
  }

  if (opt.use_extrema >= 0)
  {
    // Update the center to the extrema
    center[0] = pmesh->ptracker_extrema->c_x1(opt.use_extrema);
    center[1] = pmesh->ptracker_extrema->c_x2(opt.use_extrema);
    center[2] = pmesh->ptracker_extrema->c_x3(opt.use_extrema);
  }

  if (opt.use_puncture_massweighted_center)
  {
    // Update the center based on the mass-weighted distance
    Real pos[3];
    PuncWeightedMassCentralPoint(&pos[0], &pos[1], &pos[2]);
    center[0] = pos[0];
    center[1] = pos[1];
    center[2] = pos[2];
  }

  // Take a0 either from previous or from input value
  if (ah_found && last_a0 > 0)
  {
    a0(0) = last_a0 * opt.expand_guess;
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
