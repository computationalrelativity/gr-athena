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
      if (Globals::my_rank == 0)
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
  // Points for sph harm l>=1
  lmpoints =
    (opt.lmax + 1) *
    (opt.lmax + 1);  // = (lmax+1)^2, matches lmindex(l,m) = l*(lmax+1) + m

  // Coefficients
  a0.NewAthenaArray(opt.lmax + 1);
  ac.NewAthenaArray(lmpoints);
  as.NewAthenaArray(lmpoints);

  // Legendre polynomials (scratch, rewritten per theta in
  // ComputeSphericalHarmonics)
  P_all.NewAthenaArray(3, opt.lmax + 1, opt.lmax + 1);
  P.InitWithShallowSlice(P_all, 3, 0, 1);
  dPdth.InitWithShallowSlice(P_all, 3, 1, 1);
  dPdth2.InitWithShallowSlice(P_all, 3, 2, 1);

  // m=0 spherical harmonics: Y0_all(3, ntheta, nphi, lmax+1)
  Y0_all.NewAthenaArray(3, grid_.ntheta, grid_.nphi, opt.lmax + 1);
  Y0.InitWithShallowSlice(Y0_all, 4, 0, 1);
  dY0dth.InitWithShallowSlice(Y0_all, 4, 1, 1);
  dY0dth2.InitWithShallowSlice(Y0_all, 4, 2, 1);

  // m>0 cosine harmonics: Yc_all(NDERIV, ntheta, nphi, lmpoints)
  Yc_all.NewAthenaArray(NDERIV, grid_.ntheta, grid_.nphi, lmpoints);
  Yc.InitWithShallowSlice(Yc_all, 4, D00, 1);
  dYcdth.InitWithShallowSlice(Yc_all, 4, D10, 1);
  dYcdph.InitWithShallowSlice(Yc_all, 4, D01, 1);
  dYcdth2.InitWithShallowSlice(Yc_all, 4, D20, 1);
  dYcdthdph.InitWithShallowSlice(Yc_all, 4, D11, 1);
  dYcdph2.InitWithShallowSlice(Yc_all, 4, D02, 1);

  // m>0 sine harmonics: Ys_all(NDERIV, ntheta, nphi, lmpoints)
  Ys_all.NewAthenaArray(NDERIV, grid_.ntheta, grid_.nphi, lmpoints);
  Ys.InitWithShallowSlice(Ys_all, 4, D00, 1);
  dYsdth.InitWithShallowSlice(Ys_all, 4, D10, 1);
  dYsdph.InitWithShallowSlice(Ys_all, 4, D01, 1);
  dYsdth2.InitWithShallowSlice(Ys_all, 4, D20, 1);
  dYsdthdph.InitWithShallowSlice(Ys_all, 4, D11, 1);
  dYsdph2.InitWithShallowSlice(Ys_all, 4, D02, 1);

  ComputeSphericalHarmonics();

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
      msg << "Could not open file '" << pofile_summary << "' for writing!";
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
        msg << "Could not open file '" << pofile_verbose << "' for writing!";
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
        msg << "Could not open file '" << pofile_shape << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile_shape, "# iter = %d, Time = %g\n", iter, time);
      for (int l = 0; l <= opt.lmax; l++)
        fprintf(pofile_shape, "%e ", a0(l));
      for (int l = 1; l <= opt.lmax; l++)
      {
        for (int m = 1; m <= l; m++)
        {
          int l1 = lmindex(l, m);
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
    const Real theta = grid_.th_grid(i);
    const Real costh = std::cos(theta);

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
          int bitant_z_fac = 1;
          if (bitant_sym)
          {
            if (a == 2)
              bitant_z_fac *= -1;
            if (b == 2)
              bitant_z_fac *= -1;
          }
          g(a, b, i, j) =
            interp.eval(&(adm_g_dd(a, b, 0, 0, 0))) * bitant_z_fac;
          K(a, b, i, j) =
            interp.eval(&(adm_K_dd(a, b, 0, 0, 0))) * bitant_z_fac;
          for (int c = 0; c < NDIM; ++c)
          {
            if (bitant_sym)
            {
              if (c == 2)
                bitant_z_fac *= -1;
            }
            dg(c, a, b, i, j) =
              interp.eval(&(pz4c->aux.dg_ddd(c, a, b, 0, 0, 0))) *
              bitant_z_fac;
          }
        }

    }  // phi loop
  }  // theta loop
}
//----------------------------------------------------------------------------------------
//! \fn void AHF::SurfaceIntegrals(const int n)
//  \brief compute expansion, surface element and spin integrand on surface n
// Needs metric and extr. curv. interpolated on the surface
// Performs local sums and MPI reduce
void AHF::SurfaceIntegrals()
{
  using namespace LinearAlgebra;

  // Derivatives of (r,theta,phi) w.r.t (x,y,z)
  ATP_N_vec drdi;
  ATP_N_vec dthetadi;
  ATP_N_vec dphidi;

  ATP_N_sym drdidj;
  ATP_N_sym dthetadidj;
  ATP_N_sym dphididj;

  // Derivatives of F
  ATP_N_vec dFdi;
  ATP_N_vec dFdi_u;  // upper index
  ATP_N_sym dFdidj;

  // Inverse metric
  ATP_N_sym ginv;

  // Normal
  ATP_N_vec R;

  // dx^adth , dx^a/dph
  ATP_N_vec dXdth;
  ATP_N_vec dXdph;

  // Flat-space coordinate rotational KV
  ATP_N_vec phix;
  ATP_N_vec phiy;
  ATP_N_vec phiz;

  ATP_N_sym nnF;

  // Initialize integrals
  for (int v = 0; v < invar; v++)
  {
    integrals[v] = 0.0;
  }

  rho.ZeroClear();

  // Loop over surface points
  for (int i = 0; i < grid_.ntheta; i++)
  {
    Real const theta = grid_.th_grid(i);
    Real const sinth = std::sin(theta);
    Real const costh = std::cos(theta);

    for (int j = 0; j < grid_.nphi; j++)
    {
      if (!grid_.IsOwned(i, j))
        continue;

      Real const phi   = grid_.ph_grid(j);
      Real const sinph = std::sin(phi);
      Real const cosph = std::cos(phi);

      // Calculate the expansion
      // -----------------------

      // Determinant of 3-metric
      Real detg    = Det3Metric(g(0, 0, i, j),
                             g(0, 1, i, j),
                             g(0, 2, i, j),
                             g(1, 1, i, j),
                             g(1, 2, i, j),
                             g(2, 2, i, j));
      Real oo_detg = 1.0 / detg;

      // Inverse metric
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

      // Local coordinates of the surface (re-used below)
      Real const xp = rr(i, j) * sinth * cosph;
      Real const yp = rr(i, j) * sinth * sinph;
      Real const zp = rr(i, j) * costh;

      Real const rp   = std::sqrt(xp * xp + yp * yp + zp * zp);
      Real const rhop = std::sqrt(xp * xp + yp * yp);

      if (rp < min_surface_radius)
      {
        // Do not stop the code, just AHF failing
        // break the loop and catch the nans in AHF later.
        break;
      }

      Real const _divrp    = 1.0 / rp;
      Real const _divrp3   = SQR(_divrp) * _divrp;
      Real const _divrp4   = SQR(_divrp) * SQR(_divrp);
      Real const _divrhop  = 1.0 / rhop;
      Real const _divrhop2 = SQR(_divrhop);
      Real const _divrhop3 = _divrhop2 * _divrhop;
      Real const _divrhop4 = SQR(_divrhop2);
      Real const xp2       = SQR(xp);
      Real const yp2       = SQR(yp);
      Real const zp2       = SQR(zp);

      // First derivatives of (r,theta,phi) with respect to (x,y,z)
      drdi(0) = xp * _divrp;
      drdi(1) = yp * _divrp;
      drdi(2) = zp * _divrp;

      dthetadi(0) = zp * xp * (SQR(_divrp) * _divrhop);
      dthetadi(1) = zp * yp * (SQR(_divrp) * _divrhop);
      dthetadi(2) = -rhop * SQR(_divrp);

      dphidi(0) = -yp * _divrhop2;
      dphidi(1) = xp * _divrhop2;
      dphidi(2) = 0.0;

      // Second derivatives of (r,theta,phi) with respect to (x,y,z)
      drdidj(0, 0) = _divrp - xp2 * _divrp3;
      drdidj(0, 1) = -xp * yp * _divrp3;
      drdidj(0, 2) = -xp * zp * _divrp3;
      drdidj(1, 1) = _divrp - yp2 * _divrp3;
      drdidj(1, 2) = -yp * zp * _divrp3;
      drdidj(2, 2) = _divrp - zp2 * _divrp3;

      dthetadidj(0, 0) = zp *
                         (-2.0 * SQR(xp2) - xp2 * yp2 + SQR(yp2) + zp2 * yp2) *
                         (_divrp4 * _divrhop3);
      dthetadidj(0, 1) =
        -xp * yp * zp * (3.0 * xp2 + 3.0 * yp2 + zp2) * (_divrp4 * _divrhop3);
      dthetadidj(0, 2) = xp * (xp2 + yp2 - zp2) * (_divrp4 * _divrhop);
      dthetadidj(1, 1) = zp *
                         (-2.0 * SQR(yp2) - yp2 * xp2 + SQR(xp2) + zp2 * xp2) *
                         (_divrp4 * _divrhop3);
      dthetadidj(1, 2) = yp * (xp2 + yp2 - zp2) * (_divrp4 * _divrhop);
      dthetadidj(2, 2) = 2.0 * zp * rhop * _divrp4;

      dphididj(0, 0) = 2.0 * yp * xp * _divrhop4;
      dphididj(0, 1) = (yp2 - xp2) * _divrhop4;
      dphididj(0, 2) = 0.0;
      dphididj(1, 1) = -2.0 * yp * xp * _divrhop4;
      dphididj(1, 2) = 0.0;
      dphididj(2, 2) = 0.0;

      // Compute first derivatives of F
      for (int a = 0; a < NDIM; ++a)
      {
        dFdi(a) = drdi(a);
        for (int l = 0; l <= opt.lmax; l++)
          dFdi(a) -= a0(l) * dthetadi(a) * dY0dth(i, j, l);
        for (int l = 1; l <= opt.lmax; l++)
          for (int m = 1; m <= l; m++)
          {
            int l1 = lmindex(l, m);
            dFdi(a) -= ac(l1) * (dthetadi(a) * dYcdth(i, j, l1) +
                                 dphidi(a) * dYcdph(i, j, l1)) +
                       as(l1) * (dthetadi(a) * dYsdth(i, j, l1) +
                                 dphidi(a) * dYsdph(i, j, l1));
          }
      }

      // Compute second derivatives of F (symmetric, upper triangle + copy)
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          dFdidj(a, b) = drdidj(a, b);
          for (int l = 0; l <= opt.lmax; l++)
            dFdidj(a, b) -=
              a0(l) * (dthetadidj(a, b) * dY0dth(i, j, l) +
                       dthetadi(a) * dthetadi(b) * dY0dth2(i, j, l));
          for (int l = 1; l <= opt.lmax; l++)
            for (int m = 1; m <= l; m++)
            {
              int l1 = lmindex(l, m);
              dFdidj(a, b) -=
                ac(l1) * (dthetadidj(a, b) * dYcdth(i, j, l1) +
                          dthetadi(a) * (dthetadi(b) * dYcdth2(i, j, l1) +
                                         dphidi(b) * dYcdthdph(i, j, l1)) +
                          dphididj(a, b) * dYcdph(i, j, l1) +
                          dphidi(a) * (dthetadi(b) * dYcdthdph(i, j, l1) +
                                       dphidi(b) * dYcdph2(i, j, l1))) +
                as(l1) * (dthetadidj(a, b) * dYsdth(i, j, l1) +
                          dthetadi(a) * (dthetadi(b) * dYsdth2(i, j, l1) +
                                         dphidi(b) * dYsdthdph(i, j, l1)) +
                          dphididj(a, b) * dYsdph(i, j, l1) +
                          dphidi(a) * (dthetadi(b) * dYsdthdph(i, j, l1) +
                                       dphidi(b) * dYsdph2(i, j, l1)));
            }
          dFdidj(b, a) = dFdidj(a, b);
        }

      // Compute dFdi with the index up
      for (int a = 0; a < NDIM; ++a)
      {
        dFdi_u(a) = 0;
        for (int b = 0; b < NDIM; ++b)
        {
          dFdi_u(a) += ginv(a, b) * dFdi(b);
        }
      }

      // Compute norm of dFdi
      Real norm = 0;
      for (int a = 0; a < NDIM; ++a)
      {
        norm += dFdi_u(a) * dFdi(a);
      }

      Real u = (norm > 0) ? std::sqrt(norm) : 0.0;

      // Compute nabla_a nabla_b F = d_a d_b F - Gamma^c_{ab} d_c F
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

      // Expansion & rho = H * u * sigma (sigma=1)
      Real divu = (opt.expansion_fix == ExpansionFix::cure_divu)
                  ? ((norm > 0) ? 1.0 / u : 0.0)
                  : 1.0 / u;
      Real H    = d2F * divu + dFdadFdbKab * (divu * divu) -
               dFdadFdbFdadb * (divu * divu * divu) - TrK;

      rho(i, j) = H * u;

      // Normal vector
      for (int a = 0; a < NDIM; ++a)
      {
        R(a) = dFdi_u(a) * divu;
      }

      // Surface Element
      // ---------------

      // Derivatives of (x,y,z) vs (thetas, phi)

      // dr/dtheta, dr/dphi
      Real const drdt = rr_dth(i, j);
      Real const drdp = rr_dph(i, j);

      // Derivatives of (x,y,z) with respect to theta
      dXdth(0) = (drdt * sinth + rr(i, j) * costh) * cosph;
      dXdth(1) = (drdt * sinth + rr(i, j) * costh) * sinph;
      dXdth(2) = drdt * costh - rr(i, j) * sinth;

      // Derivatives of (x,y,z) with respect to phi
      dXdph(0) = (drdp * cosph - rr(i, j) * sinph) * sinth;
      dXdph(1) = (drdp * sinph + rr(i, j) * cosph) * sinth;
      dXdph(2) = drdp * costh;

      // Induced metric on the horizon
      Real h11 = 0.0, h12 = 0.0, h22 = 0.0;
      for (int a = 0; a < NDIM; ++a)
        for (int b = 0; b < NDIM; ++b)
        {
          Real gab = g(a, b, i, j);
          h11 += dXdth(a) * dXdth(b) * gab;
          h12 += dXdth(a) * dXdph(b) * gab;
          h22 += dXdph(a) * dXdph(b) * gab;
        }

      // Determinant of the induced metric
      Real deth = h11 * h22 - h12 * h12;
      if (deth < 0.)
        deth = 0.0;

      // Spin integrand
      // --------------

      // Flat-space coordinate rotational KV
      phix(0) = 0;
      phix(1) = -zp;  // -(z-zc);
      phix(2) = yp;   // (y-yc);
      phiy(0) = zp;   // (z-zc);
      phiy(1) = 0;
      phiy(2) = -xp;  // -(x-xc);
      phiz(0) = -yp;  // -(y-yc);
      phiz(1) = xp;   // (x-xc);
      phiz(2) = 0;

      // Integrand of spin: S_d = (1/8pi) oint phi^a_d s^b K_ab dA
      Real intSx = 0.0, intSy = 0.0, intSz = 0.0;
      for (int a = 0; a < NDIM; ++a)
        for (int b = 0; b < NDIM; ++b)
        {
          Real RbKab = R(b) * K(a, b, i, j);
          intSx += phix(a) * RbKab;
          intSy += phiy(a) * RbKab;
          intSz += phiz(a) * RbKab;
        }

      // Local sums
      // ----------

      const Real wght = grid_.weights(i, j);
      const Real da   = wght * std::sqrt(deth) / sinth;

      integrals[iarea] += da;
      integrals[icoarea] += wght * SQR(rr(i, j));
      integrals[ihrms] += da * SQR(H);
      integrals[ihmean] += da * H;
      integrals[iSx] += da * intSx;
      integrals[iSy] += da * intSy;
      integrals[iSz] += da * intSz;

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
    RadiiFromSphericalHarmonics();

    // Fill x_cart with sphere coordinates (bitant-reflected)
    {
      const Real xc = center[0];
      const Real yc = center[1];
      const Real zc = center[2];
      for (int i = 0; i < grid_.ntheta; ++i)
      {
        const Real sinth = std::sin(grid_.th_grid(i));
        const Real costh = std::cos(grid_.th_grid(i));
        for (int j = 0; j < grid_.nphi; ++j)
        {
          const Real sinph      = std::sin(grid_.ph_grid(j));
          const Real cosph      = std::cos(grid_.ph_grid(j));
          grid_.x_cart(0, i, j) = xc + rr(i, j) * sinth * cosph;
          grid_.x_cart(1, i, j) = yc + rr(i, j) * sinth * sinph;
          Real z                = zc + rr(i, j) * costh;
          if (opt.bitant)
            z = std::abs(z);
          grid_.x_cart(2, i, j) = z;
        }
      }
    }

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
  const int ntotal = nspec0 + 2 * lmpoints;

  Real* ABfac    = new Real[nspec0];
  Real* spec_buf = new Real[ntotal]();  // zero-initialized

  Real* spec0 = spec_buf;
  Real* specc = spec_buf + nspec0;
  Real* specs = specc + lmpoints;

  for (int l = 0; l <= opt.lmax; l++)
  {
    ABfac[l] = A / (1.0 + B * l * (l + 1));
  }

  // Local sums
  for (int i = 0; i < grid_.ntheta; i++)
  {
    for (int j = 0; j < grid_.nphi; j++)
    {
      if (!grid_.IsOwned(i, j))
        continue;
      const Real drho = grid_.weights(i, j) * rho(i, j);

      for (int l = 0; l <= opt.lmax; l++)
        spec0[l] += drho * Y0(i, j, l);

      for (int l = 1; l <= opt.lmax; l++)
      {
        for (int m = 1; m <= l; m++)
        {
          int l1 = lmindex(l, m);
          specc[l1] += drho * Yc(i, j, l1);
          specs[l1] += drho * Ys(i, j, l1);
        }
      }

    }  // phi loop
  }  // theta loop

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
      int l1 = lmindex(l, m);
      ac(l1) -= ABfac[l] * specc[l1];
      as(l1) -= ABfac[l] * specs[l1];
    }
  }

  delete[] ABfac;
  delete[] spec_buf;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::RadiiFromSphericalHarmonics()
// \brief compute the radius of the surface
void AHF::RadiiFromSphericalHarmonics()
{
  rr.ZeroClear();
  rr_dth.ZeroClear();
  rr_dph.ZeroClear();

  rr_min = std::numeric_limits<Real>::infinity();
  for (int i = 0; i < grid_.ntheta; i++)
  {
    for (int j = 0; j < grid_.nphi; j++)
    {
      for (int l = 0; l <= opt.lmax; l++)
      {
        rr(i, j) += a0(l) * Y0(i, j, l);
        rr_dth(i, j) += a0(l) * dY0dth(i, j, l);
      }

      for (int l = 1; l <= opt.lmax; l++)
      {
        for (int m = 1; m <= l; m++)
        {
          int l1 = lmindex(l, m);
          rr(i, j) += ac(l1) * Yc(i, j, l1) + as(l1) * Ys(i, j, l1);
          rr_dth(i, j) +=
            ac(l1) * dYcdth(i, j, l1) + as(l1) * dYsdth(i, j, l1);
          rr_dph(i, j) +=
            ac(l1) * dYcdph(i, j, l1) + as(l1) * dYsdph(i, j, l1);
        }
      }
      rr_min = std::min(rr_min, rr(i, j));
    }  // phi loop
  }  // theta loop
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
// \!fn void AHF::ComputeSphericalHarmonics()
// \brief compute spherical harmonics for grid of size ntheta*nphi.
// Results are used for all horizons.
void AHF::ComputeSphericalHarmonics()
{
  Y0_all.ZeroClear();
  Yc_all.ZeroClear();
  Ys_all.ZeroClear();

  for (int i = 0; i < grid_.ntheta; ++i)
  {
    const Real theta = grid_.th_grid(i);

    gra::sph_harm::NPlm(theta, opt.lmax, P, dPdth, dPdth2);

    for (int j = 0; j < grid_.nphi; ++j)
    {
      const Real phi = grid_.ph_grid(j);

      // l=0 spherical harmonics and drvts
      for (int l = 0; l <= opt.lmax; l++)
      {
        Y0(i, j, l)      = P(l, 0);
        dY0dth(i, j, l)  = dPdth(l, 0);
        dY0dth2(i, j, l) = dPdth2(l, 0);
      }

      // l>=1 spherical harmonics and drvts
      for (int l = 1; l <= opt.lmax; l++)
      {
        for (int m = 1; m <= l; m++)
        {
          const int l1 = lmindex(l, m);

          const Real cosmph = std::cos(m * phi);
          const Real sinmph = std::sin(m * phi);

          // spherical harmonics
          Yc(i, j, l1) = SQRT2 * P(l, m) * cosmph;
          Ys(i, j, l1) = SQRT2 * P(l, m) * sinmph;

          // first drvts
          dYcdth(i, j, l1) = SQRT2 * dPdth(l, m) * cosmph;
          dYsdth(i, j, l1) = SQRT2 * dPdth(l, m) * sinmph;
          dYcdph(i, j, l1) = -SQRT2 * P(l, m) * m * sinmph;
          dYsdph(i, j, l1) = SQRT2 * P(l, m) * m * cosmph;

          // second drvts
          dYcdth2(i, j, l1)   = SQRT2 * dPdth2(l, m) * cosmph;
          dYcdthdph(i, j, l1) = -SQRT2 * dPdth(l, m) * m * sinmph;
          dYsdth2(i, j, l1)   = SQRT2 * dPdth2(l, m) * sinmph;
          dYsdthdph(i, j, l1) = SQRT2 * dPdth(l, m) * m * cosmph;
          dYcdph2(i, j, l1)   = -SQRT2 * P(l, m) * m * m * cosmph;
          dYsdph2(i, j, l1)   = -SQRT2 * P(l, m) * m * m * sinmph;
        }
      }
    }  // phi loop
  }  // theta loop
}

//----------------------------------------------------------------------------------------
// \!fn int AHF::lmindex(const int l, const int m)
// \brief multipolar single index (l,m) -> index
int AHF::lmindex(const int l, const int m) const
{
  return l * (opt.lmax + 1) + m;
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
