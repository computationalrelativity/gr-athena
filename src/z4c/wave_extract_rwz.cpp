//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file wave_extract_rwz.cpp
//  \brief Implementation of metric-based extraction of Regge-Wheeler-Zerilli
//  functions.
//  Supports bitant symmetry

#include <unistd.h>

#include <cmath>  // NAN
#include <cstdio>
#include <cstring>  // std::memcpy
#include <iomanip>
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
#include "../utils/linear_algebra.hpp"  // Det, Inv3Metric
#include "../utils/spherical_harmonics.hpp"
#include "../utils/tensor_symmetry.hpp"
#include "wave_extract_rwz.hpp"
#include "z4c.hpp"
#include "z4c_macro.hpp"

using namespace gra::grids::theta_phi::ix_DRT;

char const* const
  WaveExtractRWZ::ArealRadiusMethod[WaveExtractRWZ::NOptRadius] = {
    "areal", "areal_simple", "average_schw", "schw_gthth", "schw_gphph",
  };

//----------------------------------------------------------------------------------------
//! \fn
//  \brief class for RWZ waveform extraction
WaveExtractRWZ::WaveExtractRWZ(Mesh* pmesh, ParameterInput* pin, int n)
    : pmesh(pmesh)
{
  ReadOptions(pin, n);
  PrepareArrays();
}

//----------------------------------------------------------------------------------------
//! \fn void WaveExtractRWZ::ReadOptions(ParameterInput* pin, int n)
//  \brief read all configuration from ParameterInput into opt struct
void WaveExtractRWZ::ReadOptions(ParameterInput* pin, int n)
{
  opt.bitant  = pin->GetOrAddBoolean("mesh", "bitant", false);
  opt.verbose = pin->GetOrAddBoolean("rwz_extraction", "verbose", false);
  opt.subtract_background =
    pin->GetOrAddBoolean("rwz_extraction", "subtract_background", false);

  Nrad              = n;
  std::string n_str = std::to_string(n);

  opt.lmax = pin->GetOrAddInteger("rwz_extraction", "lmax", 2);
  if ((opt.lmax > 8) || (opt.lmax < 2))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ setup" << std::endl
        << "lmax must be in [2,8] " << opt.lmax << std::endl;
    ATHENA_ERROR(msg);
  }

  // Set method to compute areal radius
  std::string radius_method =
    pin->GetOrAddString("rwz_extraction", "method_areal_radius", "areal");
  bool found_radius_method = false;
  for (int i = 0; i < NOptRadius; ++i)
  {
    if (radius_method == ArealRadiusMethod[i])
    {
      opt.method_areal_radius = i;
      found_radius_method     = true;
      break;
    }
  }
  if (!found_radius_method)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ setup" << std::endl
        << "unknown method_areal_radius " << radius_method << std::endl;
    ATHENA_ERROR(msg);
  }

  // Get extraction radii
  std::string parname = "radius_" + n_str;
  opt.Radius          = pin->GetOrAddReal("rwz_extraction", parname, 10.0);

  // Center of the sphere
  parname = "center_x_";
  parname += n_str;
  opt.center[0] = pin->GetOrAddReal("rwz_extraction", parname, 0.0);
  parname       = "center_y_";
  parname += n_str;
  opt.center[1] = pin->GetOrAddReal("rwz_extraction", parname, 0.0);
  parname       = "center_z_";
  parname += n_str;
  opt.center[2] = pin->GetOrAddReal("rwz_extraction", parname, 0.0);

  // (theta,phi) coordinate points
  {
    const int ntheta = pin->GetOrAddInteger("rwz_extraction", "ntheta", 32);
    const int nphi   = pin->GetOrAddInteger("rwz_extraction", "nphi", 64);

    std::string integral_method =
      pin->GetOrAddString("rwz_extraction", "method_integrals", "riemann");
    // Canonical name mapping
    if (integral_method == "riemann")
      integral_method = "midpoint";

    grid_.Initialize(ntheta, nphi, integral_method);
  }

  // Output configuration
  opt.mpi_root = pin->GetOrAddInteger("rwz_extraction", "mpi_root", 0);
  opt.outprec  = pin->GetOrAddInteger("rwz_extraction", "output_digits", 16);
  opt.extra_output =
    pin->GetOrAddBoolean("rwz_extraction", "extra_output", false);

  // Baseline output filenames
  ofbname[Iof_diagnostic] =
    pin->GetOrAddString("rwz_extraction", "filename_diagnostic", "diagnostic");
  ofbname[Iof_adm] =
    pin->GetOrAddString("rwz_extraction", "filename_adm", "adm");
  ofbname[Iof_hlm] =
    pin->GetOrAddString("rwz_extraction", "filename_hlm", "wave_rwz");
  ofbname[Iof_Psie] =
    pin->GetOrAddString("rwz_extraction", "filename_psie", "wave_psie");
  ofbname[Iof_Psio] =
    pin->GetOrAddString("rwz_extraction", "filename_psio", "wave_psio");

  ofbname[Iof_Psie_dyn] = pin->GetOrAddString(
    "rwz_extraction", "filename_psie_dyn", "wave_psie_dyn");
  ofbname[Iof_Psio_dyn] = pin->GetOrAddString(
    "rwz_extraction", "filename_psio_dyn", "wave_psio_dyn");
  ofbname[Iof_Qplus] =
    pin->GetOrAddString("rwz_extraction", "filename_Qplus", "wave_Qplus");
  ofbname[Iof_Qstar] =
    pin->GetOrAddString("rwz_extraction", "filename_Qstar", "wave_Qstar");

  if (opt.extra_output)
  {
    ofbname[Iof_H1_dot] =
      pin->GetOrAddString("rwz_extraction", "filename_H1_dot", "wave_H1dot");
    ofbname[Iof_H0_dr] =
      pin->GetOrAddString("rwz_extraction", "filename_H0_dr", "wave_H0dr");
    ofbname[Iof_H0] =
      pin->GetOrAddString("rwz_extraction", "filename_H0", "wave_H0");
    ofbname[Iof_H1] =
      pin->GetOrAddString("rwz_extraction", "filename_H1", "wave_H1");
    ofbname[Iof_H] =
      pin->GetOrAddString("rwz_extraction", "filename_H", "wave_H");
    ofbname[Iof_H_dr] =
      pin->GetOrAddString("rwz_extraction", "filename_H_dr", "wave_Hdr");

    ofbname[Iof_Psie_dr] = pin->GetOrAddString(
      "rwz_extraction", "filename_psie_dr", "wave_psie_dr");
    ofbname[Iof_Psio_dr] = pin->GetOrAddString(
      "rwz_extraction", "filename_psio_dr", "wave_psio_dr");
    ofbname[Iof_Qplus_dr] = pin->GetOrAddString(
      "rwz_extraction", "filename_Qplus_dr", "wave_Qplus_dr");
    ofbname[Iof_Qstar_dr] = pin->GetOrAddString(
      "rwz_extraction", "filename_Qstar_dr", "wave_Qstar_dr");
  }

  // Warn if RWZ will run but storage.aux ghost zones won't be communicated
  {
    const Real dt_rwz = pin->GetOrAddReal("task_triggers", "dt_Z4c_RWZ", 0.0);
    if (dt_rwz > 0.0 &&
        !pin->GetOrAddBoolean("z4c", "communicate_aux_adm", false))
    {
      if ((Globals::my_rank == 0) && (n == 0))
      {
        std::printf(
          "### WARNING [WaveExtractRWZ]: z4c/communicate_aux_adm is false.\n"
          "  RWZ interpolates storage.aux (metric derivatives) near "
          "MeshBlock\n"
          "  boundaries where ghost-zone values are uninitialized without\n"
          "  communication. Results may be inaccurate.\n");
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void WaveExtractRWZ::PrepareArrays()
//  \brief allocate metric, multipole, and master-function arrays; compute Ylm
void WaveExtractRWZ::PrepareArrays()
{
  // 3+1 metric (& derivatives) on the sphere
  gamma.Allocate(grid_.ntheta, grid_.nphi);
  beta_d.Allocate(grid_.ntheta, grid_.nphi);
  beta_u.Allocate(grid_.ntheta, grid_.nphi);
  beta2.Allocate(grid_.ntheta, grid_.nphi);
  alpha.Allocate(grid_.ntheta, grid_.nphi);

  // Spherical harmonics (complex, l = 2 ... lmax)
  ylm_.Initialize(grid_, opt.lmax);
  lmpoints = ylm_.lmpoints;

  // Allocate memory for reducing the multipoles
  // NVMultipoles complex multipole with lm indexes
  integrals_multipoles = new Real[2 * NVMultipoles * lmpoints];

  // Even-parity Multipoles & dvrts
  h00.Allocate(lmpoints);
  h01.Allocate(lmpoints);
  h11.Allocate(lmpoints);
  h0.Allocate(lmpoints);
  h1.Allocate(lmpoints);
  G.Allocate(lmpoints);
  K.Allocate(lmpoints);

  // Odd-parity Multipoles & dvrts
  H0.Allocate(lmpoints);
  H1.Allocate(lmpoints);
  H.Allocate(lmpoints);

  // Gauge-invariant
  kappa_dd.NewAthenaTensor(lmpoints, 2);
  kappa_d.NewAthenaTensor(lmpoints, 2);
  kappa.NewAthenaArray(lmpoints, 2);
  Tr_kappa_dd.NewAthenaArray(lmpoints, 2);

  // Master functions multipoles
  Psie.NewAthenaArray(lmpoints, 2);
  Psio.NewAthenaArray(lmpoints, 2);
  Psie_dyn.NewAthenaArray(lmpoints, 2);
  Psio_dyn.NewAthenaArray(lmpoints, 2);
  Psie_sch.NewAthenaArray(lmpoints, 2);
  Psio_sch.NewAthenaArray(lmpoints, 2);
  Qplus.NewAthenaArray(lmpoints, 2);
  Qstar.NewAthenaArray(lmpoints, 2);

  Psie_dr.NewAthenaArray(lmpoints, 2);
  Psio_dr.NewAthenaArray(lmpoints, 2);
  Qplus_dr.NewAthenaArray(lmpoints, 2);
  Qstar_dr.NewAthenaArray(lmpoints, 2);
}

//----------------------------------------------------------------------------------------

WaveExtractRWZ::~WaveExtractRWZ()
{
  for (int i = 0; i < Iof_Num; ++i)
    if (ofile[i])
      fclose(ofile[i]);
  delete[] integrals_multipoles;
}

//----------------------------------------------------------------------------------------
//! \fn std::string WaveExtractRWZ::OutputFileName(std::string base)
//  \brief compute filenames from a basename and adding extraction radius index
std::string WaveExtractRWZ::OutputFileName(std::string base)
{
  std::stringstream strObj3;
  strObj3 << std::setfill('0') << std::setw(5) << std::fixed
          << std::setprecision(2) << opt.Radius;
  std::string fname = base + "_r" + strObj3.str() + ".txt";
  return fname;
}

//----------------------------------------------------------------------------------------
//! \fn FILE* WaveExtractRWZ::OpenOutputFile(int iof)
//  \brief Open output file for a given Iof index.  If the file already exists
//         on disk (e.g. after a restart) it is opened for appending; otherwise
//         a new file is created and a header line is written.  The handle is
//         cached in ofile[iof] and kept open until the destructor.
FILE* WaveExtractRWZ::OpenOutputFile(int iof)
{
  const std::string ofname = OutputFileName(ofbname[iof]);
  const bool exists        = (access(ofname.c_str(), F_OK) == 0);
  FILE* f                  = fopen(ofname.c_str(), exists ? "a" : "w");
  if (!f)
  {
    char buf[512];
    snprintf(buf,
             sizeof(buf),
             "### FATAL ERROR in WaveExtractRWZ: "
             "Could not open file '%s' for writing!",
             ofname.c_str());
    throw std::runtime_error(buf);
  }

  // Write column header for newly created files
  if (!exists)
  {
    if (iof == Iof_diagnostic)
    {
      fprintf(f,
              "# 1:iter 2:time 3:SchwarzschildRradius 4:SchwarzschildMass "
              "5:SchwarzschildRadius_dot 6:NormDeltaGamma "
              "7:NormTracekappaAB\n");
    }
    else if (iof == Iof_adm)
    {
      fprintf(f,
              "# 1:iter 2:time 3:ADM_M 4:ADM_Px 5:ADM_Py 6:ADM_Pz "
              "7:ADM_Jx 8:ADM_Jy 9:ADM_Jz\n");
    }
    else
    {
      fprintf(f, "# 1:iter 2:time");
      int idx = 3;
      for (int l = 2; l <= opt.lmax; ++l)
        for (int m = -l; m <= l; ++m)
        {
          fprintf(
            f, " %d:l=%d-m=%d-Re %d:l=%d-m=%d-Im", idx, l, m, idx + 1, l, m);
          idx += 2;
        }
      fprintf(f, "\n");
    }
    fflush(f);
  }

  ofile[iof] = f;
  return f;
}

//----------------------------------------------------------------------------------------
// \!fn WaveExtractRWZ::Write(int iter, Real time)
// \brief write output at given time and for given radius
//        output frequency is controlled by the RWZ trigger (see main.hpp)
void WaveExtractRWZ::Write(int iter, Real time)
{
  if (Globals::my_rank != opt.mpi_root)
    return;

  const int P = opt.outprec;

  // -- Diagnostic file ------------------------------------------------------

  {
    FILE* f = ofile[Iof_diagnostic] ? ofile[Iof_diagnostic]
                                    : OpenOutputFile(Iof_diagnostic);
    fprintf(f,
            "%d %.*e %.*e %.*e %.*e %.*e %.*e\n",
            iter,
            P,
            time,
            P,
            Schwarzschild_radius,
            P,
            Schwarzschild_mass,
            P,
            dot_rsch,
            P,
            norm_Delta_Gamma,
            P,
            norm_Tr_kappa_dd);
    fflush(f);
  }

  // -- ADM file -------------------------------------------------------------

  {
    FILE* f = ofile[Iof_adm] ? ofile[Iof_adm] : OpenOutputFile(Iof_adm);
    fprintf(f, "%d %.*e", iter, P, time);
    for (int i = 0; i < NVADM; ++i)
      fprintf(f, " %.*e", P, integrals_adm[i]);
    fprintf(f, "\n");
    fflush(f);
  }

  // -- Multipole files ------------------------------------------------------
  // data[] maps enum values Iof_Psie..Iof_Qstar_dr (minus 2) to arrays.
  // Iof_hlm is special-cased (computed on-the-fly from Psie/Psio).

  std::vector<AthenaArray<Real>*> data;
  data.reserve(opt.extra_output ? 16 : 6);
  data.push_back(&Psie);
  data.push_back(&Psio);
  data.push_back(&Psie_dyn);
  data.push_back(&Psio_dyn);
  data.push_back(&Qplus);
  data.push_back(&Qstar);
  if (opt.extra_output)
  {
    data.push_back(&H1[D01]);
    data.push_back(&H0[D10]);
    data.push_back(&H0[D00]);
    data.push_back(&H1[D00]);
    data.push_back(&H[D00]);
    data.push_back(&H[D10]);
    data.push_back(&Psie_dr);
    data.push_back(&Psio_dr);
    data.push_back(&Qplus_dr);
    data.push_back(&Qstar_dr);
  }

  for (int i = Iof_adm + 1; i < Iof_Num; ++i)
  {
    // Skip extra output files when extra_output is disabled
    if (!opt.extra_output && i >= Iof_H1_dot && i <= Iof_Qstar_dr)
      continue;

    FILE* f = ofile[i] ? ofile[i] : OpenOutputFile(i);

    if (i == Iof_hlm)
    {
      // hlm is computed on the fly from Psie/Psio (no persistent storage)
      fprintf(f, "%d %.*e", iter, P, time);
      for (int l = 2; l <= opt.lmax; ++l)
      {
        const Real NRWZ = RWZnorm(l) / opt.Radius;
        for (int m = -l; m <= l; ++m)
        {
          const int lm     = ylm_.lmindex(l, m);
          const Real hlm_R = NRWZ * (Psie(lm, Re) - Psio(lm, Im));
          const Real hlm_I = NRWZ * (Psie(lm, Im) + Psio(lm, Re));
          fprintf(f, " %.*e %.*e", P, hlm_R, P, hlm_I);
        }
      }
      fprintf(f, "\n");
    }
    else
    {
      fprintf(f, "%d %.*e", iter, P, time);
      for (int l = 2; l <= opt.lmax; ++l)
      {
        for (int m = -l; m <= l; ++m)
        {
          const int lm = ylm_.lmindex(l, m);
          fprintf(f,
                  " %.*e %.*e",
                  P,
                  (*data[i - 2])(lm, Re),
                  P,
                  (*data[i - 2])(lm, Im));
        }
      }
      fprintf(f, "\n");
    }

    fflush(f);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::ExtractAll(...)
// \brief Run the full RWZ extraction pipeline for every radius, batching MPI
//        reductions so the total number of Allreduce calls is 2 (background +
//        multipoles) regardless of how many extraction radii are active.
void WaveExtractRWZ::ExtractAll(std::vector<WaveExtractRWZ*>& rwz_vec,
                                int iter,
                                Real time)
{
  const int nrad = static_cast<int>(rwz_vec.size());
  if (nrad == 0)
    return;

  // Phase 1: local accumulation of background integrals for all radii
  for (auto prwz : rwz_vec)
  {
    prwz->MetricToSphere();
    prwz->BackgroundAccumulate();
  }

  // Phase 2: single batched MPI_Allreduce for all background integrals
#ifdef MPI_PARALLEL
  {
    const int nbg = rwz_vec[0]->n_background_integrals();
    std::vector<Real> buf(nrad * nbg);
    for (int r = 0; r < nrad; ++r)
      std::memcpy(
        &buf[r * nbg], rwz_vec[r]->background_integrals(), nbg * sizeof(Real));
    MPI_Allreduce(MPI_IN_PLACE,
                  buf.data(),
                  nrad * nbg,
                  MPI_ATHENA_REAL,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    for (int r = 0; r < nrad; ++r)
      std::memcpy(
        rwz_vec[r]->background_integrals(), &buf[r * nbg], nbg * sizeof(Real));
  }
#endif

  // Phase 3: finalize background, then local accumulation of multipoles
  for (auto prwz : rwz_vec)
  {
    prwz->BackgroundFinalize();
    prwz->MultipoleAccumulate();
  }

  // Phase 4: single batched MPI_Allreduce for all multipole integrals
#ifdef MPI_PARALLEL
  {
    const int nmp = rwz_vec[0]->n_multipole_integrals();
    std::vector<Real> buf(nrad * nmp);
    for (int r = 0; r < nrad; ++r)
      std::memcpy(
        &buf[r * nmp], rwz_vec[r]->multipole_integrals(), nmp * sizeof(Real));
    MPI_Allreduce(MPI_IN_PLACE,
                  buf.data(),
                  nrad * nmp,
                  MPI_ATHENA_REAL,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    for (int r = 0; r < nrad; ++r)
      std::memcpy(
        rwz_vec[r]->multipole_integrals(), &buf[r * nmp], nmp * sizeof(Real));
  }
#endif

  // Phase 5+6: finalize multipoles and write output
  for (auto prwz : rwz_vec)
  {
    prwz->MultipoleFinalize();
    prwz->Write(iter, time);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MetricToSphere()
// \brief Lazily prepare the extraction sphere, then interpolate the ADM metric
//        and its derivatives onto the sphere in spherical coordinates.
void WaveExtractRWZ::MetricToSphere()
{
  grid_.PrepareFixedSphere(pmesh,
                           opt.center,
                           opt.Radius,
                           opt.bitant,
                           SW_CCX_VC(true, false),
                           SW_CCX_VC(false, true));
  InterpMetricToSphere();
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::InterpMetricToSphere(MeshBlock * pmb)
// \brief interpolate the ADM metric and its drvts on the sphere and
//        transform Cartesian to spherical coordinates.
//        ADM integrals (local sums) are also computed here.
//
// This assumes there is a special storage with the spatial drvts of
// ADM metric and lapse and shift
// These derivatives are computed via PrepareZ4cDerivatives (storage.aux)

// TODO 2nd drvts are missing, but they can be taken from the interpolation.

void WaveExtractRWZ::InterpMetricToSphere()
{
  // Zero ADM integral accumulators (local partial sums, reduced in
  // BackgroundReduce)
  for (int i = 0; i < NVADM; i++)
    integrals_adm[i] = 0.0;

  using InterpType = LagrangeInterpND<metric_interp_order, 3, true>;

  // Center of the sphere
  const Real xc = opt.center[0];
  const Real yc = opt.center[1];
  const Real zc = opt.center[2];

  const Real r     = opt.Radius;
  const Real r2    = SQR(opt.Radius);
  const Real div_r = 1.0 / (opt.Radius);

  // Precompute both Jacobians on the grid for this extraction radius
  grid_.ComputeFixedRadiusJacobians(r);

  // Pool selection: Z4c lives on CC or CX -> use cc pool; Z4c on VC -> use vc
  std::vector<InterpType>& pool =
    SW_CCX_VC(grid_.interp_pool_cc, grid_.interp_pool_vc);

  Real sum_adm_M = 0.0, sum_adm_Px = 0.0, sum_adm_Py = 0.0;
  Real sum_adm_Pz = 0.0, sum_adm_Jx = 0.0, sum_adm_Jy = 0.0;
  Real sum_adm_Jz = 0.0;

#pragma omp parallel for collapse(2) schedule(dynamic) \
  reduction(+ : sum_adm_M,                             \
              sum_adm_Px,                              \
              sum_adm_Py,                              \
              sum_adm_Pz,                              \
              sum_adm_Jx,                              \
              sum_adm_Jy,                              \
              sum_adm_Jz)
  for (int i = 0; i < grid_.ntheta; i++)
  {
    for (int j = 0; j < grid_.nphi; j++)
    {
      const Real theta  = grid_.th_grid(i);
      const Real sinth  = std::sin(theta);
      const Real costh  = std::cos(theta);
      const Real sinth2 = SQR(sinth);
      const Real costh2 = SQR(costh);
      if (!grid_.IsOwned(i, j))
        continue;

      // Pointwise tensors: Cartesian components, bundled with radial/time
      // drvts D00=value, D10=d_r, D20=d^2_r, D01=d_t, D11=d_r d_t
      using gra::grids::theta_phi::DTensorFieldPoint;
      DTensorFieldPoint<Real, TensorSymm::SYM2, 3, 2> Cgamma;   // gamma_{ab}
      DTensorFieldPoint<Real, TensorSymm::NONE, 3, 1> Cbeta_u;  // beta^a
      DTensorFieldPoint<Real, TensorSymm::NONE, 3, 0> Calpha;   // alpha

      // Standalone Cartesian intermediates (no radial/time derivative
      // variants)
      ATP_N_sym Cgamma_uu;
      ATP_N_sym CK_dd;
      ATP_N_VS2 CK_der_ddd;

      ATP_N_S2S2 Cgamma_der2_dddd;
      ATP_N_VS2 Cgamma_der_ddd;
      ATP_N_T2 Cbeta_der_du;
      ATP_N_S2V Cbeta_der2_ddu;
      ATP_N_vec Calpha_der_d;
      ATP_N_sym Calpha_der2_dd;

      ATP_N_VS2 Cgamma_derdot_dddd;

      MeshBlock* pmb = grid_.mask_mb(i, j);
      Z4c* pz4c      = pmb->pz4c;

      // 3+1 metric
      AT_N_sym adm_g_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
      AT_N_vec adm_beta_u(pz4c->storage.adm, Z4c::I_ADM_betax);
      AT_N_sca adm_alpha(pz4c->storage.adm, Z4c::I_ADM_alpha);

      AT_N_VS2 adm_dg_ddd(pz4c->storage.aux, Z4c::I_AUX_dgxx_x);
      AT_N_T2 adm_dbeta_du(pz4c->storage.aux, Z4c::I_AUX_dbetax_x);
      AT_N_vec adm_dalpha_d(pz4c->storage.aux, Z4c::I_AUX_dalpha_x);

      AT_N_vec adm_beta_dot_u(pz4c->storage.rhs, Z4c::I_Z4c_betax);
      AT_N_sca adm_alpha_dot(pz4c->storage.rhs, Z4c::I_Z4c_alpha);

      AT_N_sym adm_K_dd(pz4c->storage.adm, Z4c::I_ADM_Kxx);

      InterpType& pinterp3 = pool[grid_.mask_interp_idx(i, j)];

      const Real phi    = grid_.ph_grid(j);
      const Real sinph  = std::sin(phi);
      const Real cosph  = std::cos(phi);
      const Real sinph2 = SQR(sinph);
      const Real cosph2 = SQR(cosph);

      // Global coordinates of the surface
      const Real x = xc + r * sinth * cosph;
      const Real y = yc + r * sinth * sinph;
      Real z       = zc + r * costh;

      // Normal vector
      Real n[3];
      n[0] = (x - xc) * div_r;
      n[1] = (y - yc) * div_r;
      n[2] = (z - zc) * div_r;

      // Impose bitant symmetry below
      bool bitant_sym = (opt.bitant && z < 0) ? true : false;

      // Interpolate Cartesian components at point (theta_i,phi_j)
      // ----------------------------------------------------------

      // 3-metric
      // With bitant wrt z=0, pick a (-) sign every time a z component is
      // encountered.
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          const int bsign = BitantSign(bitant_sym, a, b);
          Cgamma(a, b)    = pinterp3.eval(&(adm_g_dd(a, b, 0, 0, 0))) * bsign;
          CK_dd(a, b)     = pinterp3.eval(&(adm_K_dd(a, b, 0, 0, 0))) * bsign;
        }

      // Shift (up)
      for (int a = 0; a < NDIM; ++a)
      {
        const int bsign = BitantSign(bitant_sym, a);
        Cbeta_u(a)      = pinterp3.eval(&(adm_beta_u(a, 0, 0, 0))) * bsign;
        Cbeta_u(D01, a) = pinterp3.eval(&(adm_beta_dot_u(a, 0, 0, 0))) * bsign;
      }

      // lapse
      Calpha()    = pinterp3.eval(&(adm_alpha(0, 0, 0)));
      Calpha(D01) = pinterp3.eval(&(adm_alpha_dot(0, 0, 0)));

      // 3-metric spatial drvts
      // NB g_ddd(c,a,b) means d/dx^c g_{ab}
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
          for (int c = 0; c < NDIM; ++c)
            for (int d = c; d < NDIM; ++d)
            {
              const int bsign_abc = BitantSign(bitant_sym, a, b, c);
              Cgamma_der_ddd(c, a, b) =
                pinterp3.eval(&(adm_dg_ddd(c, a, b, 0, 0, 0))) * bsign_abc;
              const int bsign_abcd = BitantSign(bitant_sym, a, b, c, d);
              Cgamma_der2_dddd(d, c, a, b) =
                InterpDeriv<Real>(
                  pinterp3, d, &(adm_dg_ddd(c, a, b, 0, 0, 0))) *
                bsign_abcd;
            }

      // shift (up) spatial drvts
      for (int a = 0; a < NDIM; ++a)
        for (int b = 0; b < NDIM; ++b)
        {
          const int bsign = BitantSign(bitant_sym, a, b);
          Cbeta_der_du(a, b) =
            pinterp3.eval(&(adm_dbeta_du(a, b, 0, 0, 0))) * bsign;
        }

      // shift (up) second spatial drvts
      for (int a = 0; a < NDIM; ++a)
        for (int b = 0; b < NDIM; ++b)
          for (int c = 0; c < NDIM; ++c)
          {
            const int bsign = BitantSign(bitant_sym, a, b, c);
            Cbeta_der2_ddu(c, a, b) =
              InterpDeriv<Real>(pinterp3, c, &(adm_dbeta_du(a, b, 0, 0, 0))) *
              bsign;
          }

      // lapse spatial drvts
      for (int a = 0; a < NDIM; ++a)
      {
        const int bsign = BitantSign(bitant_sym, a);
        Calpha_der_d(a) = pinterp3.eval(&(adm_dalpha_d(a, 0, 0, 0))) * bsign;
      }

      // lapse 2nd spatial drvts
      for (int a = 0; a < NDIM; ++a)
        for (int c = 0; c < NDIM; ++c)
        {
          const int bsign = BitantSign(bitant_sym, a, c);
          Calpha_der2_dd(c, a) =
            InterpDeriv<Real>(pinterp3, c, &(adm_dalpha_d(a, 0, 0, 0))) *
            bsign;
        }

      // Auxiliary Cartesian tensor components
      // ----------------------------------------

      // Time drvt of the 3-metric
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          Real Lie_beta_ab = 0.0;
          for (int c = 0; c < NDIM; ++c)
          {
            Lie_beta_ab += Cgamma(a, c) * Cbeta_der_du(b, c) +
                           Cgamma(b, c) * Cbeta_der_du(a, c) +
                           Cbeta_u(c) * Cgamma_der_ddd(c, a, b);
          }
          Cgamma(D01, a, b) = -2.0 * Calpha() * CK_dd(a, b) + Lie_beta_ab;
        }

      // Spatial drvt of Kij for dr_dot_gij
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
          for (int c = 0; c < NDIM; ++c)
          {
            const int bsign = BitantSign(bitant_sym, a, b, c);
            CK_der_ddd(c, a, b) =
              InterpDeriv<Real>(pinterp3, c, &(adm_K_dd(a, b, 0, 0, 0))) *
              bsign;
          }

      // Mixed drvt of the 3-metric
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          for (int c = 0; c < NDIM; ++c)
          {
            Real der_Lie_beta_cab = 0.0;
            for (int d = 0; d < NDIM; ++d)
            {
              der_Lie_beta_cab +=
                Cgamma_der_ddd(c, a, d) * Cbeta_der_du(b, d) +
                Cgamma(a, d) * Cbeta_der2_ddu(c, b, d) +
                Cgamma_der_ddd(c, b, d) * Cbeta_der_du(a, d) +
                Cgamma(b, d) * Cbeta_der2_ddu(c, a, d) +
                Cbeta_der_du(c, d) * Cgamma_der_ddd(d, a, b) +
                Cbeta_u(d) * Cgamma_der2_dddd(c, d, a, b);
            }

            Cgamma_derdot_dddd(c, a, b) =
              -2.0 * (Calpha() * CK_der_ddd(c, a, b) +
                      Calpha_der_d(c) * CK_dd(a, b)) +
              der_Lie_beta_cab;
          }
        }

      // Determinant of 3-metric
      const Real det      = LinearAlgebra::Det3Metric(Cgamma(0, 0),
                                                 Cgamma(0, 1),
                                                 Cgamma(0, 2),
                                                 Cgamma(1, 1),
                                                 Cgamma(1, 2),
                                                 Cgamma(2, 2));
      const Real sqrt_det = (det > 0.0) ? std::sqrt(det) : 1.0;
      const Real div_det  = (det > 0.0) ? 1.0 / det : 1.0;

      // Inverse 3-metric
      LinearAlgebra::Inv3Metric(div_det,
                                Cgamma(0, 0),
                                Cgamma(0, 1),
                                Cgamma(0, 2),
                                Cgamma(1, 1),
                                Cgamma(1, 2),
                                Cgamma(2, 2),
                                &Cgamma_uu(0, 0),
                                &Cgamma_uu(0, 1),
                                &Cgamma_uu(0, 2),
                                &Cgamma_uu(1, 1),
                                &Cgamma_uu(1, 2),
                                &Cgamma_uu(2, 2));

      // Trace of K_ab
      Real TrK = 0.0;
      for (int a = 0; a < NDIM; ++a)
        for (int b = 0; b < NDIM; ++b)
          TrK += Cgamma_uu(a, b) * CK_dd(a, b);

      // Radial drvts of Cartesian gamma_{ij}
      for (int a = 0; a < NDIM; ++a)
        for (int b = 0; b < NDIM; ++b)
        {
          Cgamma(D10, a, b) = 0.0;
          Cgamma(D20, a, b) = 0.0;
          Cgamma(D11, a, b) = 0.0;
          for (int c = 0; c < NDIM; ++c)
          {
            Cgamma(D10, a, b) += n[c] * Cgamma_der_ddd(c, a, b);
            Cgamma(D11, a, b) += n[c] * Cgamma_derdot_dddd(c, a, b);
            for (int d = 0; d < NDIM; ++d)
            {
              Cgamma(D20, a, b) += n[d] * n[c] * Cgamma_der2_dddd(d, c, a, b);
            }
          }
        }

      // Radial drvts of beta^i
      for (int a = 0; a < NDIM; ++a)
      {
        Cbeta_u(D10, a) = 0.0;
        Cbeta_u(D20, a) = 0.0;
        for (int b = 0; b < NDIM; ++b)
        {
          Cbeta_u(D10, a) += n[b] * Cbeta_der_du(b, a);
          for (int c = 0; c < NDIM; ++c)
          {
            Cbeta_u(D20, a) += n[c] * n[b] * Cbeta_der2_ddu(c, b, a);
          }
        }
      }

      // lapse & time drvts
      alpha(i, j)      = Calpha();
      alpha(D01, i, j) = Calpha(D01);

      // Radial drvts of lapse
      alpha(D10, i, j) = 0.0;
      alpha(D20, i, j) = 0.0;
      for (int a = 0; a < NDIM; ++a)
      {
        alpha(D10, i, j) += n[a] * Calpha_der_d(a);
        for (int b = 0; b < NDIM; ++b)
        {
          alpha(D20, i, j) += n[a] * n[b] * Calpha_der2_dd(a, b);
        }
      }

      // ADM integrals (based on Cartesian expressions)
      // These integrals will be reduced together with the background integrals
      // and will be normalized by 1/(4 Pi)
      // ----------------------------------------------------------------------

      const Real vol = r2 * grid_.weights(i, j);  // TODO: check r^2 : needed?

      Real iMadm = 0.0;
      for (int a = 0; a < NDIM; ++a)
      {
        for (int b = 0; b < NDIM; ++b)
        {
          Real deltag_n_ab = 0.0;
          for (int c = 0; c < NDIM; ++c)
          {
            deltag_n_ab +=
              (Cgamma_der_ddd(b, a, c) - Cgamma_der_ddd(c, a, b)) * n[c];
          }
          iMadm += Cgamma_uu(a, b) * deltag_n_ab;
        }
      }

      sum_adm_M += 0.25 * iMadm * sqrt_det * vol;

      Real iPadm[NDIM];
      for (int a = 0; a < NDIM; ++a)
      {
        iPadm[a] = 0.0;
        for (int b = 0; b < NDIM; ++b)
          for (int c = 0; c < NDIM; ++c)
            for (int d = 0; d < NDIM; ++d)
            {
              const Real delta_dc = (d == c) ? 1.0 : 0.0;
              const Real delta_ad = (a == d) ? 1.0 : 0.0;
              iPadm[a] += (Cgamma_uu(b, c) * CK_dd(d, b) - delta_dc * TrK) *
                          delta_ad * n[c];
            }
      }

      sum_adm_Px += 0.5 * iPadm[0] * sqrt_det * vol;
      sum_adm_Py += 0.5 * iPadm[1] * sqrt_det * vol;
      sum_adm_Pz += 0.5 * iPadm[2] * sqrt_det * vol;

      Real iJadm[NDIM];
      for (int a = 0; a < NDIM; ++a)
      {
        iJadm[a] = 0.0;
        for (int b = 0; b < NDIM; ++b)
        {
          for (int c = 0; c < NDIM; ++c)
          {
            const Real epsilon_abc = LinearAlgebra::LeviCivitaSymbol(a, b, c);
            for (int d = 0; d < NDIM; ++d)
            {
              const Real delta_cd = (d == c) ? 1.0 : 0.0;
              Real Kd_c           = 0.0;
              for (int e = 0; e < NDIM; ++e)
              {
                Kd_c += Cgamma_uu(e, d) * CK_dd(c, e);
              }
              iJadm[a] += (Kd_c - delta_cd * TrK) * epsilon_abc *
                          grid_.x_cart(b, i, j) * n[d];
            }
          }
        }
      }

      sum_adm_Jx += 0.5 * iJadm[0] * sqrt_det * vol;
      sum_adm_Jy += 0.5 * iJadm[1] * sqrt_det * vol;
      sum_adm_Jz += 0.5 * iJadm[2] * sqrt_det * vol;

      // =================================================================
      // Transform tensors to spherical coordinates using grid Jacobians
      // =================================================================
      //
      // For a fixed-radius sphere the Jacobian columns scale as r^{n_A}
      // with n = {0, 1, 1}, giving:
      //   d_r cov_J(a, A) = p[A] * cov_J(a, A),  p = {0, 1/r, 1/r}
      //   d^2_r cov_J = 0  (since n(n-1) = 0 for n in {0,1})
      //
      // Contravariant Jacobian rows scale as r^{m_A}, m = {0, -1, -1}:
      //   d_r  con_J(A, a) = q[A]  * con_J(A, a),  q  = {0, -1/r, -1/r}
      //   d^2_r con_J(A, a) = q2[A] * con_J(A, a),  q2 = {0, 2/r^2, 2/r^2}
      //
      // Product rules for radial derivatives of transformed quantities:
      //   gamma_{AB} = Sum_{cd} J^c_A J^d_B gamma_{cd}
      //   d_r gamma_{AB} = (p_A + p_B) S[D00] + S[D10]
      //   d^2_r gamma_{AB} = 2 p_A p_B S[D00] + 2(p_A + p_B) S[D10] + S[D20]
      //   d_t d_r gamma_{AB} = (p_A + p_B) S[D01] + S[D11]
      //   where S[Dkl] = Sum_{cd} J^c_A J^d_B Cgamma(Dkl, c, d)

      // ---- Step 4: gamma_{AB} and all derivatives via cov_J contraction ----
      const Real p[3] = { 0.0, div_r, div_r };

      for (int A = 0; A < 3; ++A)
        for (int B = A; B < 3; ++B)
        {
          Real S00 = 0.0, S01 = 0.0, S10 = 0.0, S20 = 0.0, S11 = 0.0;
          for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d)
            {
              const Real JJ =
                grid_.cov_J(c, A, i, j) * grid_.cov_J(d, B, i, j);
              S00 += JJ * Cgamma(c, d);
              S01 += JJ * Cgamma(D01, c, d);
              S10 += JJ * Cgamma(D10, c, d);
              S20 += JJ * Cgamma(D20, c, d);
              S11 += JJ * Cgamma(D11, c, d);
            }
          const Real pAB         = p[A] + p[B];
          const Real ppAB        = p[A] * p[B];
          gamma(A, B, i, j)      = S00;
          gamma(D01, A, B, i, j) = S01;
          gamma(D10, A, B, i, j) = pAB * S00 + S10;
          gamma(D20, A, B, i, j) = 2.0 * ppAB * S00 + 2.0 * pAB * S10 + S20;
          gamma(D11, A, B, i, j) = pAB * S01 + S11;
        }

      // ---- Step 5: beta^A and derivatives via con_J contraction ----
      //   beta^A = Sum_c con_J(A,c) beta^c
      //   d_r beta^A = q[A] U[D00] + U[D10]
      //   d^2_r beta^A = q2[A] U[D00] + 2 q[A] U[D10] + U[D20]
      //   where U[Dkl] = Sum_c con_J(A,c) Cbeta_u(Dkl, c)
      const Real q[3]  = { 0.0, -div_r, -div_r };
      const Real q2[3] = { 0.0, 2.0 * div_r * div_r, 2.0 * div_r * div_r };

      for (int A = 0; A < 3; ++A)
      {
        Real U00 = 0.0, U01 = 0.0, U10 = 0.0, U20 = 0.0;
        for (int c = 0; c < 3; ++c)
        {
          const Real J = grid_.con_J(A, c, i, j);
          U00 += J * Cbeta_u(c);
          U01 += J * Cbeta_u(D01, c);
          U10 += J * Cbeta_u(D10, c);
          U20 += J * Cbeta_u(D20, c);
        }
        beta_u(A, i, j)      = U00;
        beta_u(D01, A, i, j) = U01;
        beta_u(D10, A, i, j) = q[A] * U00 + U10;
        beta_u(D20, A, i, j) = q2[A] * U00 + 2.0 * q[A] * U10 + U20;
      }

      // ---- Step 6: beta_A by spherical lowering, then beta^2 ----
      //   beta_A = gamma_{AB} beta^B
      //   Product rules for d_r and d_t derivatives.
      for (int A = 0; A < 3; ++A)
      {
        Real bd00 = 0.0, bd01 = 0.0, bd10 = 0.0, bd20 = 0.0;
        for (int B = 0; B < 3; ++B)
        {
          bd00 += gamma(A, B, i, j) * beta_u(B, i, j);
          bd01 += gamma(D01, A, B, i, j) * beta_u(B, i, j) +
                  gamma(A, B, i, j) * beta_u(D01, B, i, j);
          bd10 += gamma(D10, A, B, i, j) * beta_u(B, i, j) +
                  gamma(A, B, i, j) * beta_u(D10, B, i, j);
          bd20 += gamma(D20, A, B, i, j) * beta_u(B, i, j) +
                  2.0 * gamma(D10, A, B, i, j) * beta_u(D10, B, i, j) +
                  gamma(A, B, i, j) * beta_u(D20, B, i, j);
        }
        beta_d(A, i, j)      = bd00;
        beta_d(D01, A, i, j) = bd01;
        beta_d(D10, A, i, j) = bd10;
        beta_d(D20, A, i, j) = bd20;
      }

      // beta^2 = beta^A beta_A and derivatives (product rule)
      {
        Real b2_00 = 0.0, b2_01 = 0.0, b2_10 = 0.0, b2_20 = 0.0;
        for (int A = 0; A < 3; ++A)
        {
          b2_00 += beta_u(A, i, j) * beta_d(A, i, j);
          b2_01 += beta_u(D01, A, i, j) * beta_d(A, i, j) +
                   beta_u(A, i, j) * beta_d(D01, A, i, j);
          b2_10 += beta_u(D10, A, i, j) * beta_d(A, i, j) +
                   beta_u(A, i, j) * beta_d(D10, A, i, j);
          b2_20 += beta_u(D20, A, i, j) * beta_d(A, i, j) +
                   2.0 * beta_u(D10, A, i, j) * beta_d(D10, A, i, j) +
                   beta_u(A, i, j) * beta_d(D20, A, i, j);
        }
        beta2(i, j)      = b2_00;
        beta2(D01, i, j) = b2_01;
        beta2(D10, i, j) = b2_10;
        beta2(D20, i, j) = b2_20;
      }

    }  // phi loop
  }  // theta loop

  integrals_adm[I_ADM_M]  = sum_adm_M;
  integrals_adm[I_ADM_Px] = sum_adm_Px;
  integrals_adm[I_ADM_Py] = sum_adm_Py;
  integrals_adm[I_ADM_Pz] = sum_adm_Pz;
  integrals_adm[I_ADM_Jx] = sum_adm_Jx;
  integrals_adm[I_ADM_Jy] = sum_adm_Jy;
  integrals_adm[I_ADM_Jz] = sum_adm_Jz;
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::BackgroundAccumulate()
// \brief compute local sums for the background spherical metric, areal radius,
//        mass, etc.  Must be followed by BackgroundFinalize() after MPI
//        reduce.
void WaveExtractRWZ::BackgroundAccumulate()
{
  // Zeros the integrals
  for (int i = 0; i < NVBackground; i++)
    integrals_background[i] = 0.0;

  // Local sums for ADM integrals are computed in InterpMetricToSphere(),
  // we reduce them here with the background quantities
  integrals_background[I_adm_M]  = integrals_adm[I_ADM_M];
  integrals_background[I_adm_Px] = integrals_adm[I_ADM_Px];
  integrals_background[I_adm_Py] = integrals_adm[I_ADM_Py];
  integrals_background[I_adm_Pz] = integrals_adm[I_ADM_Pz];
  integrals_background[I_adm_Jx] = integrals_adm[I_ADM_Jx];
  integrals_background[I_adm_Jy] = integrals_adm[I_ADM_Jy];
  integrals_background[I_adm_Jz] = integrals_adm[I_ADM_Jz];

  // Scalar reduction variables for background integrals
  Real s_rsch2 = 0.0, s_drsch_dri = 0.0, s_d2rsch_dri2 = 0.0;
  Real s_dot_rsch2 = 0.0, s_drsch_dri_dot = 0.0;
  Real s_g00 = 0.0, s_g0R = 0.0, s_gRR = 0.0;
  Real s_dR_g00 = 0.0, s_dR_g0R = 0.0, s_dR_gRR = 0.0;
  Real s_dR2_g00 = 0.0, s_dR2_g0R = 0.0, s_dR2_gRR = 0.0;
  Real s_dot_g00 = 0.0, s_dot_g0R = 0.0, s_dot_gRR = 0.0;
  Real s_gRt = 0.0, s_gtt = 0.0, s_gpp = 0.0, s_dR_gtt = 0.0;

#pragma omp parallel for collapse(2) schedule(dynamic) \
  reduction(+ : s_rsch2,                               \
              s_drsch_dri,                             \
              s_d2rsch_dri2,                           \
              s_dot_rsch2,                             \
              s_drsch_dri_dot,                         \
              s_g00,                                   \
              s_g0R,                                   \
              s_gRR,                                   \
              s_dR_g00,                                \
              s_dR_g0R,                                \
              s_dR_gRR,                                \
              s_dR2_g00,                               \
              s_dR2_g0R,                               \
              s_dR2_gRR,                               \
              s_dot_g00,                               \
              s_dot_g0R,                               \
              s_dot_gRR,                               \
              s_gRt,                                   \
              s_gtt,                                   \
              s_gpp,                                   \
              s_dR_gtt)
  for (int i = 0; i < grid_.ntheta; i++)
  {
    for (int j = 0; j < grid_.nphi; j++)
    {
      const Real theta      = grid_.th_grid(i);
      const Real sinth      = std::sin(theta);
      const Real costh      = std::cos(theta);
      const Real sinth2     = SQR(sinth);
      const Real costh2     = SQR(costh);
      const Real div_sinth  = 1.0 / sinth;
      const Real div_sinth2 = SQR(div_sinth);
      if (!grid_.IsOwned(i, j))
        continue;

      const Real phi = grid_.ph_grid(j);
      const Real vol = grid_.weights(i, j);

      Real int_r2            = 0.0;
      Real int_drsch_dri     = 0.0;
      Real int_d2rsch_dri2   = 0.0;
      Real int_r2dot         = 0.0;
      Real int_drsch_dri_dot = 0.0;  // TODO

      // NB These integrals will be all normalized with 1/(4 Pi)
      // Note some integrals are not in dOmega = sinth dtheta dphi but in
      // dtheta dphi, we divide here by sinth: This should be safe for both set
      // of nodes Riemann and Gauss-Legendre

      const Real dthdph = vol * div_sinth;

      if (opt.method_areal_radius == areal)
      {
        const Real aux_r2 = std::sqrt(gamma(1, 1, i, j) * gamma(2, 2, i, j) -
                                      SQR(gamma(1, 2, i, j)));
        const Real div_aux_r2 = 1.0 / aux_r2;
        const Real aux_r2_d   = gamma(D10, 1, 1, i, j) * gamma(2, 2, i, j) +
                              gamma(1, 1, i, j) * gamma(D10, 2, 2, i, j) -
                              2.0 * gamma(1, 2, i, j) * gamma(D10, 1, 2, i, j);
        const Real aux_r2_d2 =
          gamma(D20, 1, 1, i, j) * gamma(2, 2, i, j) +
          2.0 * gamma(D10, 1, 1, i, j) * gamma(D10, 2, 2, i, j) +
          gamma(1, 1, i, j) * gamma(D20, 2, 2, i, j) -
          2.0 * SQR(gamma(D10, 1, 2, i, j)) -
          2.0 * gamma(1, 2, i, j) * gamma(D20, 1, 2, i, j);
        const Real aux_r2_dot =
          gamma(D01, 1, 1, i, j) * gamma(2, 2, i, j) +
          gamma(1, 1, i, j) * gamma(D01, 2, 2, i, j) -
          2.0 * gamma(1, 2, i, j) * gamma(D01, 1, 2, i, j);
        const Real aux_r2_d_dot =
          gamma(D11, 1, 1, i, j) * gamma(2, 2, i, j) +
          gamma(D10, 1, 1, i, j) * gamma(D01, 2, 2, i, j) +
          gamma(D01, 1, 1, i, j) * gamma(D10, 2, 2, i, j) +
          gamma(1, 1, i, j) * gamma(D11, 2, 2, i, j) -
          2.0 * gamma(D10, 1, 2, i, j) * gamma(D01, 1, 2, i, j) -
          2.0 * gamma(1, 2, i, j) * gamma(D11, 1, 2, i, j);

        int_r2          = aux_r2 * dthdph;
        int_drsch_dri   = 0.25 * aux_r2_d * div_aux_r2 * dthdph;
        int_d2rsch_dri2 = (0.25 * aux_r2_d2 * div_aux_r2 -
                           0.125 * SQR(aux_r2_d) * std::pow(div_aux_r2, 3)) *
                          dthdph;

        int_r2dot = 0.25 * aux_r2_dot * div_aux_r2 * dthdph;
        int_drsch_dri_dot =
          (0.25 * aux_r2_d_dot * div_aux_r2 -
           0.125 * aux_r2_d * aux_r2_dot * std::pow(div_aux_r2, 3)) *
          dthdph;
      }
      else if (opt.method_areal_radius == areal_simple)
      {
        const Real aux_r2 = std::sqrt(gamma(1, 1, i, j) * gamma(2, 2, i, j));
        const Real div_aux_r2 = 1.0 / aux_r2;
        const Real aux_r2_d   = gamma(D10, 1, 1, i, j) * gamma(2, 2, i, j) +
                              gamma(1, 1, i, j) * gamma(D10, 2, 2, i, j);
        const Real aux_r2_d2 =
          gamma(D20, 1, 1, i, j) * gamma(2, 2, i, j) +
          2.0 * gamma(D10, 1, 1, i, j) * gamma(D10, 2, 2, i, j) +
          gamma(1, 1, i, j) * gamma(D20, 2, 2, i, j);
        const Real aux_r2_dot = gamma(D01, 1, 1, i, j) * gamma(2, 2, i, j) +
                                gamma(1, 1, i, j) * gamma(D01, 2, 2, i, j);
        const Real aux_r2_d_dot =
          gamma(D11, 1, 1, i, j) * gamma(2, 2, i, j) +
          gamma(D10, 1, 1, i, j) * gamma(D01, 2, 2, i, j) +
          gamma(D01, 1, 1, i, j) * gamma(D10, 2, 2, i, j) +
          gamma(1, 1, i, j) * gamma(D11, 2, 2, i, j);

        int_r2          = aux_r2 * dthdph;
        int_drsch_dri   = 0.25 * aux_r2_d * div_aux_r2 * dthdph;
        int_d2rsch_dri2 = (0.25 * aux_r2_d2 * div_aux_r2 -
                           0.125 * SQR(aux_r2_d) * std::pow(div_aux_r2, 3)) *
                          dthdph;

        int_r2dot = 0.25 * aux_r2_dot * div_aux_r2 * dthdph;
        int_drsch_dri_dot =
          (0.25 * aux_r2_d_dot * div_aux_r2 -
           0.125 * aux_r2_d * aux_r2_dot * std::pow(div_aux_r2, 3)) *
          dthdph;
      }
      else if (opt.method_areal_radius == average_schw)
      {
        int_r2 =
          0.5 * (gamma(1, 1, i, j) + gamma(2, 2, i, j) * div_sinth2) * vol;
        int_drsch_dri =
          0.25 *
          (gamma(D10, 1, 1, i, j) + gamma(D10, 2, 2, i, j) * div_sinth2) * vol;
        int_d2rsch_dri2 =
          0.25 *
          (gamma(D20, 1, 1, i, j) + gamma(D20, 2, 2, i, j) * div_sinth2) * vol;

        int_r2dot =
          0.25 *
          (gamma(D01, 1, 1, i, j) + gamma(D01, 2, 2, i, j) * div_sinth2) * vol;
        int_drsch_dri_dot =
          0.25 *
          (gamma(D11, 1, 1, i, j) + gamma(D11, 2, 2, i, j) * div_sinth2) * vol;
      }
      else if (opt.method_areal_radius == schw_gthth)
      {
        int_r2          = gamma(1, 1, i, j) * vol;
        int_drsch_dri   = 0.5 * gamma(D10, 1, 1, i, j) * vol;
        int_d2rsch_dri2 = 0.5 * gamma(D20, 1, 1, i, j) * vol;

        int_r2dot         = 0.5 * gamma(D01, 1, 1, i, j) * vol;
        int_drsch_dri_dot = 0.5 * gamma(D11, 1, 1, i, j) * vol;
      }
      else if (opt.method_areal_radius == schw_gphph)
      {
        int_r2          = gamma(2, 2, i, j) * div_sinth2 * vol;
        int_drsch_dri   = 0.5 * gamma(D10, 2, 2, i, j) * div_sinth2 * vol;
        int_d2rsch_dri2 = 0.5 * gamma(D20, 2, 2, i, j) * div_sinth2 * vol;

        int_r2dot         = 0.5 * gamma(D01, 2, 2, i, j) * div_sinth2 * vol;
        int_drsch_dri_dot = 0.5 * gamma(D11, 2, 2, i, j) * div_sinth2 * vol;
      }

      // Local sums
      // ----------

      // Schwarzschild radius & Jacobians
      // Integral weights have been take care above
      s_rsch2 += int_r2;
      s_drsch_dri += int_drsch_dri;
      s_d2rsch_dri2 += int_d2rsch_dri2;

      s_dot_rsch2 += int_r2dot;
      s_drsch_dri_dot += int_drsch_dri_dot;

      // 2-metric & drvts
      s_g00 -= vol * (SQR(alpha(i, j)) - beta2(i, j));
      s_g0R += vol * beta_d(0, i, j);
      s_gRR += vol * gamma(0, 0, i, j);

      s_dR_g00 -=
        vol * (2.0 * alpha(i, j) * alpha(D10, i, j) - beta2(D10, i, j));
      s_dR_g0R += vol * beta_d(D10, 0, i, j);
      s_dR_gRR += vol * gamma(D10, 0, 0, i, j);

      s_dR2_g00 -= vol * (2.0 * alpha(i, j) * alpha(D20, i, j) +
                          2.0 * SQR(alpha(D10, i, j)) - beta2(D20, i, j));
      s_dR2_g0R += vol * beta_d(D20, 0, i, j);
      s_dR2_gRR += vol * gamma(D20, 0, 0, i, j);

      s_dot_g00 -=
        vol * (2.0 * alpha(i, j) * alpha(D01, i, j) - beta2(D01, i, j));
      s_dot_g0R += vol * beta_d(D01, 0, i, j);
      s_dot_gRR += vol * gamma(D01, 0, 0, i, j);

      s_gRt += vol * gamma(0, 1, i, j);
      s_gtt += vol * gamma(1, 1, i, j);
      s_gpp += vol * gamma(2, 2, i, j);

      s_dR_gtt += vol * gamma(D10, 1, 1, i, j);

    }  // for j
  }  // for i

  integrals_background[Irsch2] += s_rsch2;
  integrals_background[Idrsch_dri] += s_drsch_dri;
  integrals_background[Id2rsch_dri2] += s_d2rsch_dri2;
  integrals_background[Idot_rsch2] += s_dot_rsch2;
  integrals_background[Idrsch_dri_dot] += s_drsch_dri_dot;
  integrals_background[Ig00] += s_g00;
  integrals_background[Ig0R] += s_g0R;
  integrals_background[IgRR] += s_gRR;
  integrals_background[IdR_g00] += s_dR_g00;
  integrals_background[IdR_g0R] += s_dR_g0R;
  integrals_background[IdR_gRR] += s_dR_gRR;
  integrals_background[IdR2_g00] += s_dR2_g00;
  integrals_background[IdR2_g0R] += s_dR2_g0R;
  integrals_background[IdR2_gRR] += s_dR2_gRR;
  integrals_background[Idot_g00] += s_dot_g00;
  integrals_background[Idot_g0R] += s_dot_g0R;
  integrals_background[Idot_gRR] += s_dot_gRR;
  integrals_background[IgRt] += s_gRt;
  integrals_background[Igtt] += s_gtt;
  integrals_background[Igpp] += s_gpp;
  integrals_background[IdR_gtt] += s_dR_gtt;
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::BackgroundFinalize()
// \brief finalize the background computation after MPI reduce:
//        normalization, Schwarzschild transformation, Christoffel symbols,
//        etc.
void WaveExtractRWZ::BackgroundFinalize()
{
  // Normalization
  const Real div_4PI = 1.0 / (4.0 * PI);
  for (int i = 0; i < NVBackground; i++)
    integrals_background[i] *= div_4PI;

  // Store back full ADM integrals
  integrals_adm[I_ADM_M]  = integrals_background[I_adm_M];
  integrals_adm[I_ADM_Px] = integrals_background[I_adm_Px];
  integrals_adm[I_ADM_Py] = integrals_background[I_adm_Py];
  integrals_adm[I_ADM_Pz] = integrals_background[I_adm_Pz];
  integrals_adm[I_ADM_Jx] = integrals_background[I_adm_Jx];
  integrals_adm[I_ADM_Jy] = integrals_background[I_adm_Jy];
  integrals_adm[I_ADM_Jz] = integrals_background[I_adm_Jz];

  // Check
  const Real rsch2 = integrals_background[Irsch2];
  if (!(std::isfinite(rsch2)) || (rsch2 <= 1e-20))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ::BackgroundFinalize" << std::endl
        << "Squared Schwarzschild radius is not finite or negative " << rsch2
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // All the data is here, time to finalize the background computation
  // -----------------------------------------------------------------

  rsch                = std::sqrt(rsch2);
  const Real div_rsch = 1.0 / rsch;

  // Idot_rsch2 is always d/dt(r^2) = d/dt(Integral)
  //  -> rdot = 1/(2r) d/dt(Integral)
  dot_rsch  = div_rsch * integrals_background[Idot_rsch2];
  dot2_rsch = 0.0;  // TODO we do not have 2nd drvts ATM

  drsch_dri   = integrals_background[Idrsch_dri];
  d2rsch_dri2 = integrals_background[Id2rsch_dri2];

  drsch_dri_dot = integrals_background[Idrsch_dri_dot];

  const Real g00 = integrals_background[Ig00];
  const Real g0R = integrals_background[Ig0R];
  const Real gRR = integrals_background[IgRR];
  const Real gRt = integrals_background[IgRt];
  const Real gtt = integrals_background[Igtt];
  const Real gpp = integrals_background[Igpp];

  const Real dR_g00 = integrals_background[IdR_g00];
  const Real dR_g0R = integrals_background[IdR_g0R];
  const Real dR_gRR = integrals_background[IdR_gRR];
  const Real dR_gtt = integrals_background[IdR_gtt];

  const Real dR2_g00 = integrals_background[IdR2_g00];
  const Real dR2_g0R = integrals_background[IdR2_g0R];
  const Real dR2_gRR = integrals_background[IdR2_gRR];

  const Real dot_g00 = integrals_background[Idot_g00];
  const Real dot_g0R = integrals_background[Idot_g0R];
  const Real dot_gRR = integrals_background[Idot_gRR];

  // Subtract background metric ?

  if (opt.subtract_background)
  {
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < grid_.ntheta; i++)
    {
      for (int j = 0; j < grid_.nphi; j++)
      {
        const Real theta  = grid_.th_grid(i);
        const Real sinth  = std::sin(theta);
        const Real sinth2 = SQR(sinth);

        if (!grid_.IsOwned(i, j))
          continue;

        gamma(0, 0, i, j) -= gRR;
        gamma(1, 1, i, j) -= gtt;
        gamma(2, 2, i, j) -= gtt * sinth2;
        gamma(D10, 0, 0, i, j) -= dR_gRR;
        gamma(D10, 1, 1, i, j) -= dR_gtt;
        gamma(D10, 2, 2, i, j) -= dR_gtt * sinth2;
      }
    }
  }

  // Update all quantities using the transformation of the metric components
  // from the isotropic radius R to the Schwarzschild radius r
  // Note variables are overwritten here, order matters!

  // dr_schwarzschild/dr_isotropic & Time derivatives of Schwarzschild radius
  // Some integrals must be corrected for extra terms

  // if (opt.method_areal_radius==average_schw) {

  drsch_dri *= div_rsch;
  dri_drsch = 1.0 / drsch_dri;

  d2rsch_dri2 = -div_rsch * (SQR(drsch_dri) - d2rsch_dri2);
  d2ri_drsch2 = -std::pow(dri_drsch, 3) * d2rsch_dri2;

  d3rsch_dri3 = -3.0 * div_rsch * drsch_dri *
                d2rsch_dri2;  // + Term d3r_g_ab that ~ next 1/R order
  d3ri_drsch3 = -std::pow(dri_drsch, 4) * d3rsch_dri3 +
                3.0 * SQR(d2rsch_dri2) * std::pow(dri_drsch, 5);

  drsch_dri_dot = div_rsch * (-dot_rsch * drsch_dri + drsch_dri_dot);
  dri_drsch_dot = -SQR(dri_drsch) * drsch_dri_dot;

  // time derivatives of g0r and grr
  const Real dot_g0r = dri_drsch_dot * g0R + dri_drsch * dot_g0R;
  const Real dot_grr =
    2.0 * dri_drsch * dri_drsch_dot * gRR + SQR(dri_drsch) * dot_gRR;

  const Real dr2_g00 = d2ri_drsch2 * dR_g00 + SQR(dri_drsch) * dR2_g00;
  const Real dr2_g0r = d3ri_drsch3 * g0R +
                       3.0 * d2ri_drsch2 * dri_drsch * dR_g0R +
                       std::pow(dri_drsch, 3) * dR2_g0R;
  const Real dr2_grr =
    2.0 * gRR * (SQR(d2ri_drsch2) + dri_drsch * d3ri_drsch3) +
    5.0 * SQR(dri_drsch) * d2ri_drsch2 * dR_gRR +
    std::pow(dri_drsch, 4) * dR2_gRR;

  const Real dr_g00 = dri_drsch * dR_g00;
  const Real dr_g0r = d2ri_drsch2 * g0R + SQR(dri_drsch) * dR_g0R;
  const Real dr_grr =
    2.0 * dri_drsch * d2ri_drsch2 * gRR + std::pow(dri_drsch, 3) * dR_gRR;

  const Real g0r = dri_drsch * g0R;
  const Real grr = SQR(dri_drsch) * gRR;

  // Inverse metric & Christoffel's symbols of the background metric

  // Determinant
  const Real detg      = LinearAlgebra::Det2Metric(g00, g0r, grr);
  const Real div_detg  = (std::fabs(detg) < eps_det) ? 1.0 : 1.0 / detg;
  const Real div_detg2 = SQR(div_detg);

  // Inverse matrix
  Real g00_uu, g0r_uu, grr_uu;
  LinearAlgebra::Inv2Metric(div_detg, g00, g0r, grr, g00_uu, g0r_uu, grr_uu);

  // Derivatives of the inverse metric
  const Real dr_div_detg =
    -div_detg2 * (dr_g00 * grr + g00 * dr_grr - 2.0 * g0r * dr_g0r);
  const Real dr_g00_uu = dr_grr * div_detg + grr * dr_div_detg;
  const Real dr_g0r_uu = -dr_g0r * div_detg - g0r * dr_div_detg;
  const Real dr_grr_uu = dr_g00 * div_detg + g00 * dr_div_detg;

  const Real dot_div_detg =
    -div_detg2 * (dot_g00 * grr + g00 * dot_grr - 2.0 * g0r * dot_g0r);
  const Real dot_g00_uu = dot_grr * div_detg + grr * dot_div_detg;
  const Real dot_g0r_uu = -dot_g0r * div_detg - g0r * dot_div_detg;
  const Real dot_grr_uu = dot_g00 * div_detg + g00 * dot_div_detg;

  const Real dr2_div_detg =
    -div_detg2 * (dr2_g00 * grr + 2.0 * dr_g00 * dr_grr + g00 * dr2_grr -
                  2.0 * SQR(dr_g0r) - 2.0 * g0r * dr2_g0r) +
    2.0 * SQR(dr_div_detg) / div_detg;
  const Real dr2_g00_uu =
    dr2_grr * div_detg + 2.0 * dr_grr * dr_div_detg + grr * dr2_div_detg;
  const Real dr2_g0r_uu =
    -dr2_g0r * div_detg - 2.0 * dr_g0r * dr_div_detg - g0r * dr2_div_detg;
  const Real dr2_grr_uu =
    dr2_g00 * div_detg + 2.0 * dr_g00 * dr_div_detg + g00 * dr2_div_detg;

  // G stands for Gamma, these are the Christoffel symbols.
  // Assuming time independence:
  const Real G000 = -0.5 * g0r_uu * dr_g00;
  const Real Gr00 = -0.5 * grr_uu * dr_g00;
  const Real G00r = 0.5 * g00_uu * dr_g00;
  const Real Gr0r = 0.5 * g0r_uu * dr_g00;
  const Real G0rr = g00_uu * dr_g0r + 0.5 * g0r_uu * dr_grr;
  const Real Grrr = g0r_uu * dr_g0r + 0.5 * grr_uu * dr_grr;

  // Generic (time-dependent)
  // *_dyn stands for "dynamic": it uses time derivatives of the bakgrund
  // metric
  const Real G000_dyn =
    0.5 * g00_uu * dot_g00 + g0r_uu * dot_g0r - 0.5 * g0r_uu * dr_g00;
  const Real Gr00_dyn =
    grr_uu * dot_g0r - 0.5 * grr_uu * dr_g00 + 0.5 * g0r_uu * dot_g00;
  const Real G00r_dyn = 0.5 * g00_uu * dr_g00 + 0.5 * g0r_uu * dot_grr;
  const Real Gr0r_dyn = 0.5 * g0r_uu * dr_g00 + 0.5 * grr_uu * dot_grr;
  const Real G0rr_dyn =
    g00_uu * dr_g0r - 0.5 * g00_uu * dot_grr + 0.5 * g0r_uu * dr_grr;
  const Real Grrr_dyn =
    g0r_uu * dr_g0r - 0.5 * g0r_uu * dot_grr + 0.5 * grr_uu * dr_grr;

  // Compute differences
  const Real Delta_G000 = G000 - G000_dyn;
  const Real Delta_Gr00 = Gr00 - Gr00_dyn;
  const Real Delta_G00r = G00r - G00r_dyn;
  const Real Delta_Gr0r = Gr0r - Gr0r_dyn;
  const Real Delta_G0rr = G0rr - G0rr_dyn;
  const Real Delta_Grrr = Grrr - Grrr_dyn;

  norm_Delta_Gamma = SQR(Delta_G000) + SQR(Delta_G00r) + SQR(Delta_G0rr);
  norm_Delta_Gamma += SQR(Delta_Gr00) + SQR(Delta_Gr0r) + SQR(Delta_Grrr);
  norm_Delta_Gamma = std::sqrt(norm_Delta_Gamma);

  // The background is now in the correct coordinates.
  // Store everything.

  Schwarzschild_radius = rsch;
  Schwarzschild_mass   = 0.5 * rsch * (1 - grr_uu);

  g_dd(0, 0) = g00;
  g_dd(0, 1) = g0r;
  g_dd(1, 1) = grr;

  g_dr_dd(0, 0) = dr_g00;
  g_dr_dd(0, 1) = dr_g0r;
  g_dr_dd(1, 1) = dr_grr;

  g_dr2_dd(0, 0) = dr2_g00;
  g_dr2_dd(0, 1) = dr2_g0r;
  g_dr2_dd(1, 1) = dr2_grr;

  g_dot_dd(0, 0) = dot_g00;
  g_dot_dd(0, 1) = dot_g0r;
  g_dot_dd(1, 1) = dot_grr;

  g_uu(0, 0) = g00_uu;
  g_uu(0, 1) = g0r_uu;
  g_uu(1, 1) = grr_uu;

  g_dr_uu(0, 0) = dr_g00_uu;
  g_dr_uu(0, 1) = dr_g0r_uu;
  g_dr_uu(1, 1) = dr_grr_uu;

  g_dr2_uu(0, 0) = dr2_g00_uu;
  g_dr2_uu(0, 1) = dr2_g0r_uu;
  g_dr2_uu(1, 1) = dr2_grr_uu;

  g_dot_uu(0, 0) = dot_g00_uu;  // TODO
  g_dot_uu(0, 1) = dot_g0r_uu;
  g_dot_uu(1, 1) = dot_grr_uu;  // TODO

  Gamma_udd(0, 0, 0) = G000;
  Gamma_udd(0, 0, 1) = G00r;
  Gamma_udd(0, 1, 1) = G0rr;
  Gamma_udd(1, 0, 0) = Gr00;
  Gamma_udd(1, 0, 1) = Gr0r;
  Gamma_udd(1, 1, 1) = Grrr;

  Gamma_dyn_udd(0, 0, 0) = G000_dyn;
  Gamma_dyn_udd(0, 0, 1) = G00r_dyn;  // N.B. SYM2
  Gamma_dyn_udd(0, 1, 1) = G0rr_dyn;
  Gamma_dyn_udd(1, 0, 0) = Gr00_dyn;
  Gamma_dyn_udd(1, 0, 1) = Gr0r_dyn;  // N.B. SYM2
  Gamma_dyn_udd(1, 1, 1) = Grrr_dyn;

  // Transform to Schw coordinates other quantities used for multipoles
  // projections:

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < grid_.ntheta; i++)
  {
    for (int j = 0; j < grid_.nphi; j++)
    {
      if (!grid_.IsOwned(i, j))
        continue;

      // alpha radial derivatives
      alpha(D20, i, j) =
        d2ri_drsch2 * alpha(D10, i, j) + SQR(dri_drsch) * alpha(D20, i, j);
      alpha(D10, i, j) *= dri_drsch;

      // dr2_beta_th and dr2_beta_ph
      for (int a = 1; a < 3; a++)
        beta_d(D20, a, i, j) = d2ri_drsch2 * beta_d(D10, a, i, j) +
                               SQR(dri_drsch) * beta_d(D20, a, i, j);

      // dr_beta_i
      beta_d(D10, 0, i, j) =
        d2ri_drsch2 * beta_d(0, i, j) + SQR(dri_drsch) * beta_d(D10, 0, i, j);
      for (int a = 1; a < 3; a++)
        beta_d(D10, a, i, j) *= dri_drsch;

      // dot_beta_r
      beta_d(D01, 0, i, j) =
        dri_drsch_dot * beta_d(0, i, j) + dri_drsch * beta_d(D01, 0, i, j);

      // beta_r
      beta_d(0, i, j) *= dri_drsch;

      // dr2_gamma_thth, dr2_gamma_thph, dr2_gamma_phph
      for (int a = 1; a < 3; a++)
        for (int b = 1; b < 3; b++)
        {
          gamma(D20, a, b, i, j) = d2ri_drsch2 * gamma(D10, a, b, i, j) +
                                   SQR(dri_drsch) * gamma(D20, a, b, i, j);
        }

      // dr_dot_gamma_thth, dr_dot_gamma_thph, dr_dot_gamma_phph
      for (int a = 1; a < 3; a++)
        for (int b = 1; b < 3; b++)
        {
          gamma(D11, a, b, i, j) *= dri_drsch;
        }

      // dr_dot_gamma_rth, dr_dot_gamma_rph
      for (int a = 1; a < 3; a++)
        gamma(D11, 0, a, i, j) =
          dri_drsch_dot * dri_drsch * gamma(D10, 0, a, i, j) +
          SQR(dri_drsch) * gamma(D11, 0, a, i, j) +
          d2ri_drsch2 * gamma(D01, 0, a, i, j);  // Lacks term dr_dri_drsch_dot

      // dr_gamma_ij
      gamma(D10, 0, 0, i, j) =
        2.0 * dri_drsch * d2ri_drsch2 * gamma(0, 0, i, j) +
        std::pow(dri_drsch, 3) * gamma(D10, 0, 0, i, j);
      for (int a = 1; a < 3; a++)
        gamma(D10, 0, a, i, j) = d2ri_drsch2 * gamma(0, a, i, j) +
                                 SQR(dri_drsch) * gamma(D10, 0, a, i, j);
      for (int a = 1; a < 3; a++)
        for (int b = 1; b < 3; b++)
        {
          gamma(D10, a, b, i, j) *= dri_drsch;
        }

      // dot_gamma_ri
      gamma(D01, 0, 0, i, j) =
        2.0 * dri_drsch * dri_drsch_dot * gamma(0, 0, i, j) +
        SQR(dri_drsch) * gamma(D01, 0, 0, i, j);

      for (int a = 1; a < 3; a++)
        gamma(D01, 0, a, i, j) = dri_drsch_dot * gamma(0, a, i, j) +
                                 dri_drsch * gamma(D01, 0, a, i, j);

      // gamma_ri
      gamma(0, 0, i, j) *= SQR(dri_drsch);
      for (int a = 1; a < 3; a++)
        gamma(0, a, i, j) *= dri_drsch;

      // beta2 is a scalar, so only derivatives transform
      beta2(D20, i, j) =
        d2ri_drsch2 * beta2(D10, i, j) + SQR(dri_drsch) * beta2(D20, i, j);
      beta2(D10, i, j) *= dri_drsch;

    }  // for j
  }  // for i
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MultipoleAccumulate()
// \brief compute multipole local sums (projections).
//        Must be followed by MultipoleFinalize() after MPI reduce.
void WaveExtractRWZ::MultipoleAccumulate()
{
  const int lmpoints_x2 = lmpoints * 2;
  const Real div_r      = 1.0 / Schwarzschild_radius;

  // Zeros the integrals
  for (int i = 0; i < NVMultipoles * 2 * lmpoints; i++)
    integrals_multipoles[i] = 0.0;

  // Shorthand for buffer pointer to a given multipole slot
  auto buf = [&](int slot) -> Real*
  { return &integrals_multipoles[lmpoints_x2 * slot]; };

  // ===== Scalar (0,0) sector ================================================

  // Ih00: -(alpha^2 - beta^2) Y*
  ylm_.ProjectScalar(buf(Ih00),
                     grid_,
                     [&](int i, int j) -> Real
                     { return -(SQR(alpha(i, j)) - beta2(i, j)); });
  if (opt.subtract_background)
  {
    ylm_.ProjectScalar(buf(Ih00),
                       grid_,
                       [&](int i, int j) -> Real
                       { return -(integrals_background[Ig00]); });
  }

  // Ih01, Ih11
  ylm_.ProjectScalar(buf(Ih01), grid_, beta_d, D00, 0);
  ylm_.ProjectScalar(buf(Ih11), grid_, gamma, D00, 0, 0);

  // Ih00_dr, Ih01_dr, Ih11_dr
  ylm_.ProjectScalar(
    buf(Ih00_dr),
    grid_,
    [&](int i, int j) -> Real
    { return -(2.0 * alpha(i, j) * alpha(D10, i, j) - beta2(D10, i, j)); });
  ylm_.ProjectScalar(buf(Ih01_dr), grid_, beta_d, D10, 0);
  ylm_.ProjectScalar(buf(Ih11_dr), grid_, gamma, D10, 0, 0);

  // Ih00_dot, Ih01_dot, Ih11_dot
  ylm_.ProjectScalar(
    buf(Ih00_dot),
    grid_,
    [&](int i, int j) -> Real
    { return -(2.0 * alpha(i, j) * alpha(D01, i, j) - beta2(D01, i, j)); });
  ylm_.ProjectScalar(buf(Ih01_dot), grid_, beta_d, D01, 0);
  ylm_.ProjectScalar(buf(Ih11_dot), grid_, gamma, D01, 0, 0);

  // ===== Vector (0,A) sector - paired even+odd, 1/lam baked in ==============
  // beta_d angular components -> (h0, H0) pairs
  ylm_.ProjectVectorPair(buf(Ih0), buf(IH0), grid_, beta_d, D00);
  ylm_.ProjectVectorPair(buf(Ih0_dr), buf(IH0_dr), grid_, beta_d, D10);
  ylm_.ProjectVectorPair(buf(Ih0_dot), buf(IH0_dot), grid_, beta_d, D01);
  ylm_.ProjectVectorPair(nullptr, buf(IH0_dr2), grid_, beta_d, D20);

  // gamma(r,A) components -> (h1, H1) pairs
  ylm_.ProjectVectorPair(buf(Ih1), buf(IH1), grid_, gamma, D00, 0);
  ylm_.ProjectVectorPair(buf(Ih1_dr), buf(IH1_dr), grid_, gamma, D10, 0);
  ylm_.ProjectVectorPair(buf(Ih1_dot), buf(IH1_dot), grid_, gamma, D01, 0);
  ylm_.ProjectVectorPair(nullptr, buf(IH1_dr_dot), grid_, gamma, D11, 0);

  // ===== Tensor (A,B) sector - paired even+odd, 1/(lam(lam-2)) baked in =====
  // G (even) + H (odd) from gamma_{AB}; r-power product rule on even side
  ylm_.ProjectTensorPair(buf(IG), buf(IH), grid_, gamma, D00, div_r);
  ylm_.ProjectTensorPair(buf(IG_dr), buf(IH_dr), grid_, gamma, D10, div_r);
  ylm_.ProjectTensorPair(buf(IG_dot), buf(IH_dot), grid_, gamma, D01, div_r);
  ylm_.ProjectTensorPair(buf(IG_dr2), buf(IH_dr2), grid_, gamma, D20, div_r);
  ylm_.ProjectTensorPair(buf(IG_dr_dot), nullptr, grid_, gamma, D11, div_r);

  // ===== Trace scalar (A,B) sector - K with 0.5*lam*G correction ============
  // ProjectTrace reads the already-filled G buffer (local partial sums)
  // and adds 0.5*lam*G per-mode.  R-power product rule handled internally.
  ylm_.ProjectTrace(buf(IK), buf(IG), grid_, gamma, D00, div_r);
  ylm_.ProjectTrace(buf(IK_dr), buf(IG_dr), grid_, gamma, D10, div_r);
  ylm_.ProjectTrace(buf(IK_dot), buf(IG_dot), grid_, gamma, D01, div_r);
  ylm_.ProjectTrace(buf(IK_dr2), buf(IG_dr2), grid_, gamma, D20, div_r);
  ylm_.ProjectTrace(buf(IK_dr_dot), buf(IG_dr_dot), grid_, gamma, D11, div_r);
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MultipoleFinalize()
// \brief unpack reduced multipole integrals, compute gauge-invariant
// quantities
//        and master functions.  Must be called after MPI reduce of
//        integrals_multipoles.
void WaveExtractRWZ::MultipoleFinalize()
{
  const int lmpoints_x2 = lmpoints * 2;

  // Shorthand for buffer pointer to a given multipole slot
  auto buf = [&](int slot) -> Real*
  { return &integrals_multipoles[lmpoints_x2 * slot]; };

  // ===== Unpack - plain copy, all prefactors already applied ================

  auto unpack = [&](AA& dest, int slot)
  {
    const Real* src = buf(slot);
    for (int l = 2; l <= opt.lmax; ++l)
      for (int m = -l; m <= l; ++m)
      {
        const int lm = ylm_.lmindex(l, m);
        for (int c = 0; c < RealImag; ++c)
          dest(lm, c) = src[2 * lm + c];
      }
  };

  // even scalar
  unpack(h00[D00], Ih00);
  unpack(h01[D00], Ih01);
  unpack(h11[D00], Ih11);
  unpack(h00[D10], Ih00_dr);
  unpack(h01[D10], Ih01_dr);
  unpack(h11[D10], Ih11_dr);
  unpack(h00[D01], Ih00_dot);
  unpack(h01[D01], Ih01_dot);
  unpack(h11[D01], Ih11_dot);

  // even vector
  unpack(h0[D00], Ih0);
  unpack(h1[D00], Ih1);
  unpack(h0[D10], Ih0_dr);
  unpack(h1[D10], Ih1_dr);
  unpack(h0[D01], Ih0_dot);
  unpack(h1[D01], Ih1_dot);

  // even tensor + trace
  unpack(G[D00], IG);
  unpack(K[D00], IK);
  unpack(G[D10], IG_dr);
  unpack(K[D10], IK_dr);
  unpack(G[D01], IG_dot);
  unpack(K[D01], IK_dot);
  unpack(G[D20], IG_dr2);
  unpack(K[D20], IK_dr2);
  unpack(G[D11], IG_dr_dot);
  unpack(K[D11], IK_dr_dot);

  // odd vector
  unpack(H0[D00], IH0);
  unpack(H0[D10], IH0_dr);
  unpack(H0[D20], IH0_dr2);
  unpack(H0[D01], IH0_dot);
  unpack(H1[D00], IH1);
  unpack(H1[D10], IH1_dr);
  unpack(H1[D01], IH1_dot);
  unpack(H1[D11], IH1_dr_dot);

  // odd tensor
  unpack(H[D00], IH);
  unpack(H[D10], IH_dr);
  unpack(H[D20], IH_dr2);
  unpack(H[D01], IH_dot);

  // Compute the various gauge-invariant functions
  MultipolesGaugeInvariant();
  MasterFuns();
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MasterFuns()
// \brief compute even and odd parity master functions
void WaveExtractRWZ::MasterFuns()
{
  const Real M     = Schwarzschild_mass;
  const Real r     = Schwarzschild_radius;
  const Real r2    = SQR(r);
  const Real rdot  = dot_rsch;
  const Real div_r = 1.0 / r;

  const Real S = (1.0 - 2.0 * M * div_r);
  const Real abs_detg =
    std::fabs(LinearAlgebra::Det2Metric(g_dd(0, 0), g_dd(0, 1), g_dd(1, 1)));
  const Real div_sqrtdetg = 1.0 / (std::sqrt(abs_detg));

  for (int l = 2; l <= opt.lmax; ++l)
  {
    const Real lambda       = l * (l + 1);
    const Real div_lambda_2 = 1.0 / (lambda - 2.0);
    const Real fac_Psie     = (2.0 * r) / ((lambda - 2) * r + 6.0 * M);
    const Real r_div_lambda = r / lambda;
    const Real Lam          = (l - 1) * (l + 2) + 6 * M * div_r;
    const Real Qpnorm       = std::sqrt(2 * (l - 1) * (l + 2) / lambda) / Lam;
    const Real dr_fac_Psie  = fac_Psie * 6.0 * M / (r2 * Lam);

    for (int m = -l; m <= l; ++m)
    {
      const int lm = ylm_.lmindex(l, m);
      for (int c = 0; c < RealImag; ++c)
      {
        // Even parity in Schwarschild coordinates
        // -----------------------------------------

        const Real Qplus_ =
          Qpnorm * (lambda * S * (r2 * G(D10, lm, c) - 2 * h1(lm, c)) +
                    2.0 * r * S * (S * h11(lm, c) - r * K(D10, lm, c)) +
                    r * Lam * K(lm, c));

        Psie_sch(lm, c) = Qplus_ / (std::sqrt(2.0 * lambda * (lambda - 2.0)));
        Qplus(lm, c)    = Qplus_;

        Qplus_dr(lm, c) =
          6.0 * M * SQR(div_r) * Qplus(lm, c) / Lam +
          Qpnorm *
            (lambda * 2.0 * M * SQR(div_r) *
               (r2 * G(D10, lm, c) - 2.0 * h1(lm, c)) +
             lambda * S *
               (2.0 * r * G(D10, lm, c) + r2 * G(D20, lm, c) -
                2.0 * h1(D10, lm, c)) +
             2.0 * (S * h11(lm, c) - r * K(D10, lm, c)) +
             2.0 * r * S *
               (2.0 * M * SQR(div_r) * h11(lm, c) + S * h11(D10, lm, c) -
                K(D10, lm, c) - r * K(D20, lm, c)) +
             (lambda - 2.0) * K(lm, c) + r * Lam * K(D10, lm, c));

        // Odd parity in Schwarzschild coordinates
        // -----------------------------------------

        const Real Psio_sch_ =
          r * (H1(D01, lm, c) - H0(D10, lm, c) + 2.0 * div_r * H0(lm, c)) *
          div_lambda_2;
        Psio_sch(lm, c) = Psio_sch_;

        const Real Qstar_ =
          div_r * S * (H1(lm, c) - 0.5 * H(D10, lm, c) + H(lm, c) * div_r);
        Qstar(lm, c) = Qstar_;

        Qstar_dr(lm, c) =
          SQR(div_r) * (1.0 - 2.0 * S) *
            (H1(lm, c) - 0.5 * H(D10, lm, c) + H(lm, c) * div_r) +
          div_r * S *
            (H1(D10, lm, c) - 0.5 * H(D20, lm, c) + div_r * H(D10, lm, c) -
             SQR(div_r) * H(lm, c));

        // Even parity in general coordinates (static)
        // -----------------------------------------

        const Real term1_K = K(lm, c);
        const Real term1_hG =
          (-2.0 * div_r) *
          (g_uu(1, 1) * (h1(lm, c) - 0.5 * r2 * G(D10, lm, c)) +
           g_uu(0, 1) * (h0(lm, c) - 0.5 * r2 * G(D01, lm, c)));

        const Real term_hAB = SQR(g_uu(0, 1)) * h00(lm, c) +
                              2.0 * g_uu(0, 1) * g_uu(1, 1) * h01(lm, c) +
                              SQR(g_uu(1, 1)) * h11(lm, c);
        const Real term2_K =
          -r * (g_uu(1, 1) * K(D10, lm, c) + g_uu(0, 1) * K(D01, lm, c));

        Real coef_h0 =
          -r * std::pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) +
          2.0 * r * g_uu(1, 1) *
            (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) + g_dr_uu(0, 1)) +
          g_uu(0, 1) * g_uu(1, 1) *
            (-2.0 + 2.0 * r * g_uu(0, 0) * g_dr_dd(0, 0) +
             r * g_uu(1, 1) * g_dr_dd(1, 1));

        coef_h0 *= div_r;

        Real coef_h1 =
          2.0 * SQR(g_uu(1, 1)) * (-1.0 + r * g_uu(0, 1) * g_dr_dd(0, 1)) +
          r * std::pow(g_uu(1, 1), 3) * g_dr_dd(1, 1) +
          r * g_uu(1, 1) * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) +
          2.0 * r * g_uu(1, 1) * g_dr_uu(1, 1);

        coef_h1 *= div_r;

        Real coef_G_dr =
          2.0 * g_uu(1, 1) * (-1.0 + r * g_uu(0, 1) * g_dr_dd(0, 1)) +
          r * SQR(g_uu(1, 1)) * g_dr_dd(1, 1) +
          r * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) + 2.0 * r * g_dr_uu(1, 1);

        coef_G_dr *= -0.5 * r * g_uu(1, 1);

        Real coef_G_dot =
          r * std::pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) -
          2 * r * g_uu(1, 1) *
            (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) + g_dr_uu(0, 1)) -
          g_uu(0, 1) * g_uu(1, 1) *
            (-2.0 + 2.0 * r * g_uu(0, 0) * g_dr_dd(0, 0) +
             r * g_uu(1, 1) * g_dr_dd(1, 1));

        coef_G_dot *= 0.5 * r;

        Psie(lm, c) = term1_K;
        Psie(lm, c) += term1_hG;
        Psie(lm, c) +=
          fac_Psie *
          (term_hAB + coef_h0 * h0(lm, c) + coef_h1 * h1(lm, c) +
           coef_G_dr * G(D10, lm, c) + coef_G_dot * G(D01, lm, c) + term2_K);
        Psie(lm, c) *= r_div_lambda;

        const Real dr_term1_hG =
          -div_r * term1_hG -
          2.0 * div_r *
            (g_dr_uu(1, 1) * (h1(lm, c) - 0.5 * r2 * G(D10, lm, c)) +
             g_dr_uu(0, 1) * (h0(lm, c) - 0.5 * r2 * G(D01, lm, c)) +
             g_uu(1, 1) * (h1(D10, lm, c) - r * G(D10, lm, c) -
                           0.5 * r2 * G(D20, lm, c)) +
             g_uu(0, 1) * (h0(D10, lm, c) - r * G(D01, lm, c) -
                           0.5 * r2 * G(D11, lm, c)));
        const Real dr_term_hAB =
          SQR(g_uu(0, 1)) * h00(D10, lm, c) +
          2.0 * g_uu(0, 1) * g_dr_uu(0, 1) * h00(lm, c) +
          2.0 * g_uu(0, 1) * g_uu(1, 1) * h01(D10, lm, c) +
          2.0 * (g_dr_uu(0, 1) * g_uu(1, 1) + g_uu(0, 1) * g_dr_uu(1, 1)) *
            h01(lm, c) +
          SQR(g_uu(1, 1)) * h11(D10, lm, c) +
          2.0 * g_uu(1, 1) * g_dr_uu(1, 1) * h11(lm, c);

        const Real dr_coef_h0 =
          -std::pow(g_uu(0, 1), 3) * g_dr2_dd(0, 0) -
          3.0 * SQR(g_uu(0, 1)) * g_dr_uu(0, 1) * g_dr_dd(0, 0) +
          2.0 * g_dr_uu(1, 1) *
            (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) + g_dr_uu(0, 1)) +
          2.0 * g_uu(1, 1) *
            ((g_uu(0, 0) * g_dr_uu(1, 1) + g_dr_uu(0, 0) * g_uu(1, 1)) *
               g_dr_dd(0, 1) +
             g_uu(0, 0) * g_uu(1, 1) * g_dr2_dd(0, 1) + g_dr2_uu(0, 1)) +
          (g_dr_uu(0, 1) * g_uu(1, 1) + g_uu(0, 1) * g_dr_uu(1, 1)) *
            (-2.0 / r + 2.0 * g_uu(0, 0) * g_dr_dd(0, 0) +
             g_uu(1, 1) * g_dr_dd(1, 1)) +
          g_uu(0, 1) * g_uu(1, 1) *
            (2.0 / r2 + 2.0 * g_dr_uu(0, 0) * g_dr_dd(0, 0) +
             2.0 * g_uu(0, 0) * g_dr2_dd(0, 0) + g_uu(1, 1) * g_dr2_dd(1, 1) +
             g_dr_uu(1, 1) * g_dr_dd(1, 1));

        const Real dr_coef_h1 =
          4.0 * g_uu(1, 1) * g_dr_uu(1, 1) *
            (-div_r + g_uu(0, 1) * g_dr_dd(0, 1)) +
          2.0 * SQR(g_uu(1, 1)) *
            (1.0 / r2 + g_dr_uu(0, 1) * g_dr_dd(0, 1) +
             g_uu(0, 1) * g_dr2_dd(0, 1)) +
          3.0 * SQR(g_uu(1, 1)) * g_dr_uu(1, 1) * g_dr_dd(1, 1) +
          std::pow(g_uu(1, 1), 3) * g_dr2_dd(1, 1) +
          g_dr_uu(1, 1) * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) +
          g_uu(1, 1) * (2.0 * g_uu(0, 1) * g_dr_uu(0, 1) * g_dr_dd(0, 0) +
                        // BD: verify Psi^(e)_dr, h_1 coefficient
                        //     SQR bracket placement
                        // ??: SQR(g_uu(0, 1) * g_dr2_dd(0, 0))
                        // ->: SQR(g_uu(0, 1)) * g_dr2_dd(0, 0)
                        SQR(g_uu(0, 1)) * g_dr2_dd(0, 0)) +
          // BD: verify Psi^(e)_dr, h_1 coefficient
          //     d/dr(2r g^{11} \partial_r g^{11})
          // ??: 2.0 * g_dr_uu(1, 1) * g_dr_dd(1, 1) + 2.0 * g_uu(1, 1) *
          // g_dr2_dd(1, 1)
          // ->: 2.0 * SQR(g_dr_uu(1, 1)) + 2.0 * g_uu(1, 1) * g_dr2_uu(1, 1)
          2.0 * SQR(g_dr_uu(1, 1)) + 2.0 * g_uu(1, 1) * g_dr2_uu(1, 1);

        const Real dr_coef_G_dr =
          coef_G_dr * (div_r + g_dr_uu(1, 1) / g_uu(1, 1)) -
          0.5 * r * g_uu(1, 1) *
            (2.0 * g_dr_uu(1, 1) * (-1.0 + r * g_uu(0, 1) * g_dr_dd(0, 1)) +
             2.0 * g_uu(1, 1) *
               (g_uu(0, 1) * g_dr_dd(0, 1) +
                r * g_dr_uu(0, 1) * g_dr_dd(0, 1) +
                r * g_uu(0, 1) * g_dr2_dd(0, 1)) +
             SQR(g_uu(1, 1)) * g_dr_dd(1, 1) +
             SQR(g_uu(0, 1)) * g_dr_dd(0, 0) +
             r * 2.0 * g_uu(1, 1) * g_dr_uu(1, 1) * g_dr_dd(1, 1) +
             r * SQR(g_uu(1, 1)) * g_dr2_dd(1, 1) +
             r * 2.0 * g_uu(0, 1) * g_dr_uu(0, 1) * g_dr_dd(0, 0) +
             r * SQR(g_uu(0, 1)) * g_dr2_dd(0, 0) + 2.0 * g_dr_uu(1, 1) +
             2.0 * r * g_dr2_uu(1, 1));

        const Real dr_coef_G_dot =
          coef_G_dot * div_r +
          0.5 * r *
            (std::pow(g_uu(0, 1), 3) * (g_dr_dd(0, 0) + r * g_dr2_dd(0, 0)) +
             3.0 * r * SQR(g_uu(0, 1)) * g_dr_uu(0, 1) * g_dr_dd(0, 0) -
             2.0 * (g_uu(1, 1) + r * g_dr_uu(1, 1)) *
               (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) + g_dr_uu(0, 1)) -
             2.0 * r * g_uu(1, 1) *
               (g_dr_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) +
                g_uu(0, 0) * g_dr_uu(1, 1) * g_dr_dd(0, 1) +
                g_uu(0, 0) * g_uu(1, 1) * g_dr2_dd(0, 1) +
                // BD: verify: Psi^(e)_dr, Gdot coefficient
                //   missing \partial^2_r g^{01} from
                //   d/dr(-2r g^{11} \partial_r g^{01})
                // ??: (term missing)
                // ->: g_dr2_uu(0, 1)
                g_dr2_uu(0, 1)) -
             (g_dr_uu(0, 1) * g_uu(1, 1) + g_uu(0, 1) * g_dr_uu(1, 1)) *
               (-2.0 + 2.0 * r * g_uu(0, 0) * g_dr_dd(0, 0) +
                r * g_uu(1, 1) * g_dr_dd(1, 1)) -
             g_uu(0, 1) * g_uu(1, 1) *
               (2.0 * g_uu(0, 0) * g_dr_dd(0, 0) +
                2.0 * r * g_dr_uu(0, 0) * g_dr_dd(0, 0) +
                2.0 * r * g_uu(0, 0) * g_dr2_dd(0, 0) +
                g_uu(1, 1) * g_dr_dd(1, 1) +
                r * g_dr_uu(1, 1) * g_dr_dd(1, 1) +
                r * g_uu(1, 1) * g_dr2_dd(1, 1)));

        const Real dr_term2_K =
          term2_K * div_r -
          r * (g_dr_uu(1, 1) * K(D10, lm, c) + g_uu(1, 1) * K(D20, lm, c) +
               g_dr_uu(0, 1) * K(D01, lm, c) + g_uu(0, 1) * K(D11, lm, c));

        Psie_dr(lm, c) =
          div_r * Psie(lm, c) +
          r_div_lambda *
            (K(D10, lm, c) + dr_term1_hG +
             dr_fac_Psie * (term_hAB + coef_h0 * h0(lm, c) +
                            coef_h1 * h1(lm, c) + coef_G_dr * G(D10, lm, c) +
                            coef_G_dot * G(D01, lm, c) + term2_K) +
             fac_Psie *
               (dr_term_hAB + coef_h0 * h0(D10, lm, c) +
                dr_coef_h0 * h0(lm, c) + coef_h1 * h1(D10, lm, c) +
                dr_coef_h1 * h1(lm, c) + coef_G_dr * G(D20, lm, c) +
                dr_coef_G_dr * G(D10, lm, c) + coef_G_dot * G(D11, lm, c) +
                dr_coef_G_dot * G(D01, lm, c) + dr_term2_K));

        // Odd parity in general coordinates (static)
        // -----------------------------------------

        Psio(lm, c) = Psio_sch(lm, c) * div_sqrtdetg;

        const Real dr_absdetg =
          // BD: verify Psi^(o)_dr, \partial_r|det g|:
          //     det g < 0 so \partial_r|det g| = -\partial_r(det g)
          // ??: std::fabs(\partial_r det g)
          // ->: -(\partial_r det g)
          -(g_dr_dd(0, 0) * g_dd(1, 1) + g_dd(0, 0) * g_dr_dd(1, 1) -
            2.0 * g_dd(0, 1) * g_dr_dd(0, 1));
        Psio_dr(lm, c) =
          (div_sqrtdetg - 0.5 * r * std::pow(div_sqrtdetg, 3) * dr_absdetg) *
            div_lambda_2 *
            (H1(D01, lm, c) - H0(D10, lm, c) + 2.0 * div_r * H0(lm, c)) +
          r * div_sqrtdetg * div_lambda_2 *
            (H1(D11, lm, c) - H0(D20, lm, c) + 2.0 * div_r * H0(D10, lm, c) -
             2.0 * SQR(div_r) * H0(lm, c));

        // Even parity in general coordinates (dynamic)
        // -----------------------------------------

        const Real term2_hG =
          (-2.0 * rdot * div_r) *
          (g_uu(0, 0) * (h0(lm, c) - 0.5 * r2 * G(D01, lm, c)) +
           g_uu(0, 1) * (h1(lm, c) - 0.5 * r2 * G(D10, lm, c)));

        const Real term_hAB_rdot =
          rdot * h00(lm, c) *
            (rdot * SQR(g_uu(0, 0)) + 2.0 * g_uu(0, 0) * g_uu(0, 1)) +
          2.0 * rdot * h01(lm, c) *
            (rdot * g_uu(0, 0) * g_uu(0, 1) + SQR(g_uu(0, 1)) +
             g_uu(0, 0) * g_uu(1, 1)) +
          rdot * h11(lm, c) *
            (rdot * SQR(g_uu(0, 1)) + 2.0 * g_uu(0, 1) * g_uu(1, 1));

        Real coef_h0_t =
          2.0 * r * std::pow(g_uu(0, 1), 3) * g_dot_dd(0, 1) +
          2.0 * r * g_uu(0, 1) *
            g_dot_uu(
              0, 1)  // CHECK g_dot_uu(0,1) in the notes, but eq full of typos
          + (-r * g_uu(0, 0) * SQR(g_uu(1, 1)) +
             2.0 * r * g_uu(1, 1) * SQR(g_uu(0, 1))) *
              g_dot_dd(1, 1) +
          r * SQR(g_uu(0, 1)) * g_uu(0, 0) * g_dot_dd(0, 0);

        coef_h0_t *= div_r;

        Real coef_h0_rdot =
          r * rdot * pow(g_uu(0, 0), 3) * g_dot_dd(0, 0) +
          2.0 * r * g_uu(1, 1) * g_dr_uu(0, 0) +
          2.0 * r * g_uu(0, 1) *
            (g_dr_uu(0, 1) + g_dot_uu(0, 0) + rdot * g_dr_uu(0, 0)) +
          r * pow(g_uu(0, 1), 3) *
            (2.0 * g_dot_dd(1, 1) + rdot * g_dr_dd(1, 1)) +
          2.0 * SQR(g_uu(0, 1)) * (-2.0 + r * g_uu(1, 1) * g_dr_dd(1, 1)) +
          SQR(g_uu(0, 0)) *
            (-2.0 * SQR(rdot) +
             r * rdot * g_uu(0, 1) * (g_dr_dd(0, 0) + 2.0 * g_dot_dd(0, 1)) +
             2.0 * r *
               (g_uu(0, 1) * g_dot_dd(0, 0) + g_uu(1, 1) * g_dr_dd(0, 0))) +
          2.0 * r * g_uu(0, 0) * g_dot_uu(0, 1) -
          2.0 * g_uu(0, 0) * g_uu(1, 1) +
          rdot * g_uu(0, 0) *
            (2.0 * r * g_dot_uu(0, 0) - 6.0 * g_uu(0, 1) +
             r * SQR(g_uu(0, 1)) * (2.0 * g_dr_dd(0, 1) + g_dot_dd(1, 1))) +
          // BD: verify Psi^(e)_dyn, h_0 rdot-coefficient
          //     covariant metric derivative index
          // ??: g_uu(1, 1) * g_dr_dd(0, 1)
          // ->: g_uu(1, 1) * g_dr_dd(1, 1)
          4.0 * r * g_uu(0, 0) * g_uu(0, 1) *
            (g_uu(0, 1) * g_dot_dd(0, 1) + g_uu(1, 1) * g_dr_dd(1, 1));

        coef_h0_rdot *= rdot * div_r;

        Real coef_h1_t = SQR(g_uu(0, 1)) * g_dot_dd(0, 0) +
                         2.0 * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(0, 1) +
                         SQR(g_uu(1, 1)) * g_dot_dd(1, 1);

        coef_h1_t *= g_uu(0, 1);  // TODO CHECK notes for factor *div_r

        Real coef_h1_rdot =
          -2.0 * SQR(rdot) * g_uu(0, 0) * g_uu(0, 1) +
          2.0 * r * pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) +
          2.0 * r * SQR(g_uu(0, 1)) *
            (g_uu(0, 0) * g_dot_dd(0, 0) + 2.0 * g_uu(1, 1) * g_dr_dd(0, 1)) +
          2.0 * r * g_uu(1, 1) *
            (g_dr_uu(0, 1) + g_uu(0, 0) * g_uu(1, 1) * g_dot_dd(1, 1)) +
          2.0 * g_uu(0, 1) *
            // BD: verify Psi^(e)_dyn, h_1 rdot-coefficient
            //     inverse metric time derivative index
            // ??: g_dot_uu(0, 1)
            // ->: g_dot_uu(0, 0)
            (r * (g_dr_uu(1, 1) + g_dot_uu(0, 0)) +
             g_uu(1, 1) * (-3.0 + 2.0 * r * g_uu(0, 0) * g_dot_dd(0, 1)) +
             r * SQR(g_uu(1, 1)) * g_dr_dd(1, 1)) +
          r * rdot * SQR(g_uu(0, 0)) *
            (g_uu(0, 1) * g_dot_dd(0, 0) -
             g_uu(1, 1) * (g_dr_dd(0, 0) - 2.0 * g_dot_dd(0, 1))) +
          2.0 * rdot * g_uu(0, 0) *
            (r * g_dot_uu(0, 1) + r * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) -
             g_uu(1, 1) + r * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(1, 1)) +
          2.0 * r * rdot * g_uu(0, 1) * g_dr_uu(0, 1) +
          r * rdot * pow(g_uu(0, 1), 3) *
            (2.0 * g_dr_dd(0, 1) - g_dot_dd(1, 1)) +
          rdot * SQR(g_uu(0, 1)) * (-4.0 + r * g_uu(1, 1) * g_dr_dd(1, 1));

        coef_h1_rdot *= rdot * div_r;

        Real coef_G_dr_t = 2.0 * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(0, 1) +
                           SQR(g_uu(0, 1)) * g_dot_dd(0, 0) +
                           SQR(g_uu(1, 1)) * g_dot_dd(1, 1);

        coef_G_dr_t *= -0.5 * r2 * g_uu(0, 1);

        Real coef_G_dr_rdot =
          -2.0 * SQR(rdot) * g_uu(0, 0) * g_uu(0, 1) +
          2.0 * r * pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) +
          2.0 * r * SQR(g_uu(0, 1)) *
            (g_uu(0, 0) * g_dot_dd(0, 0) + 2.0 * g_uu(1, 1) * g_dr_dd(0, 1)) +
          2.0 * r * g_uu(1, 1) *
            (g_dr_uu(0, 1) + g_uu(0, 0) * g_uu(1, 1) * g_dot_dd(1, 1)) +
          // BD: verify Psi^(e)_dyn, \partial_r G rdot-coefficient
          //     inverse metric time derivative index
          // ??: g_dot_uu(0, 1)
          // ->: g_dot_uu(0, 0)
          2.0 * r * g_uu(0, 1) * (g_dr_uu(1, 1) + g_dot_uu(0, 0)) +
          2.0 * g_uu(0, 1) * g_uu(1, 1) *
            (-3.0 + 2.0 * r * g_uu(0, 0) * g_dot_dd(0, 1)) +
          2.0 * r * g_uu(0, 1) * SQR(g_uu(1, 1)) * g_dr_dd(1, 1) +
          r * rdot * SQR(g_uu(0, 0)) *
            (g_uu(0, 1) * g_dot_dd(0, 0) -
             g_uu(1, 1) * (g_dr_dd(0, 0) - 2.0 * g_dot_dd(0, 1))) +
          2.0 * rdot * g_uu(0, 0) *
            (r * g_dot_uu(0, 1) + r * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) -
             g_uu(1, 1) + r * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(1, 1)) +
          2.0 * r * rdot * g_uu(0, 1) * g_dr_uu(0, 1) +
          r * rdot * pow(g_uu(0, 1), 3) *
            (2.0 * g_dr_dd(0, 1) - g_dot_dd(1, 1)) +
          rdot * SQR(g_uu(0, 1)) * (-4.0 + r * g_uu(1, 1) * g_dr_dd(1, 1));

        coef_G_dr_rdot *= -0.5 * r * rdot;

        Real coef_G_dot_t =
          // BD: verify Psi^(e)_dyn, Gdot t-coefficient
          //     spurious r factor
          // ??: -2.0 * r * std::pow(g_uu(0, 1), 3) * g_dot_dd(0, 1)
          // ->: -2.0 * std::pow(g_uu(0, 1), 3) * g_dot_dd(0, 1)
          -2.0 * std::pow(g_uu(0, 1), 3) * g_dot_dd(0, 1) -
          2.0 * g_uu(0, 1) * g_dot_uu(0, 1)  // CHECK in the notes
          - SQR(g_uu(0, 1)) * g_uu(0, 0) * g_dot_dd(0, 0) +
          g_uu(1, 1) * (g_uu(1, 1) * g_uu(0, 0) - 2.0 * SQR(g_uu(0, 1))) *
            g_dot_dd(1, 1);

        coef_G_dot_t *= 0.5 * r2;  // TODO CHECK again, notes

        Real coef_G_dot_rdot =
          r * rdot * pow(g_uu(0, 0), 3) * g_dot_dd(0, 0) +
          2.0 * r * g_uu(1, 1) * g_dr_uu(0, 0) +
          2.0 * r * g_uu(0, 1) *
            (g_dr_uu(0, 1) + g_dot_uu(0, 0) + rdot * g_dr_uu(0, 0)) +
          r * pow(g_uu(0, 1), 3) *
            (2.0 * g_dot_dd(1, 1) + rdot * g_dr_dd(1, 1)) +
          2.0 * SQR(g_uu(0, 1)) * (-2.0 + r * g_uu(1, 1) * g_dr_dd(1, 1)) +
          SQR(g_uu(0, 0)) *
            (-2.0 * SQR(rdot) +
             r * rdot * g_uu(0, 1) * (g_dr_dd(0, 0) + 2.0 * g_dot_dd(0, 1)) +
             2.0 * r *
               (g_uu(0, 1) * g_dot_dd(0, 0) + g_uu(1, 1) * g_dr_dd(0, 0))) +
          2.0 * r * g_uu(0, 0) * g_dot_uu(0, 1) -
          2.0 * g_uu(0, 0) * g_uu(1, 1) +
          rdot * g_uu(0, 0) *
            (2.0 * r * g_dot_uu(0, 0) - 6.0 * g_uu(0, 1) +
             r * SQR(g_uu(0, 1)) * (2.0 * g_dr_dd(0, 1) + g_dot_dd(1, 1))) +
          // BD: verify Psi^(e)_dyn, Gdot rdot-coefficient
          //     covariant metric derivative index
          // ??: g_uu(1, 1) * g_dr_dd(0, 1)
          // ->: g_uu(1, 1) * g_dr_dd(1, 1)
          4.0 * r * g_uu(0, 0) * g_uu(0, 1) *
            (g_uu(0, 1) * g_dot_dd(0, 1) + g_uu(1, 1) * g_dr_dd(1, 1));

        coef_G_dot_rdot *= -0.5 * r * rdot;

        const Real term2_K_rdot =
          -r * rdot *
          (g_uu(0, 0) * K(D01, lm, c) + g_uu(0, 1) * K(D10, lm, c));

        Psie_dyn(lm, c) = term1_K;
        Psie_dyn(lm, c) += term1_hG + term2_hG;
        Psie_dyn(lm, c) +=
          fac_Psie *
          (term_hAB + term_hAB_rdot +
           (coef_h0 + coef_h0_t + coef_h0_rdot) * h0(lm, c) +
           (coef_h1 + coef_h1_t + coef_h1_rdot) * h1(lm, c) +
           (coef_G_dr + coef_G_dr_t + coef_G_dr_rdot) * G(D10, lm, c) +
           (coef_G_dot + coef_G_dot_t + coef_G_dot_rdot) * G(D01, lm, c) +
           term2_K + term2_K_rdot);
        Psie_dyn(lm, c) *= r_div_lambda;

        // Odd parity in general coordinates (dynamic)
        // -----------------------------------------

        Psio_dyn(lm, c) = (r * (H1(D01, lm, c) - H0(D10, lm, c)) +
                           2.0 * (H0(lm, c) - H1(lm, c) * rdot));
        Psio_dyn(lm, c) *= div_lambda_2 * div_sqrtdetg;

      }  // real& imag
    }  // for m
  }  // for l
}

//----------------------------------------------------------------------------------------
// \!fn void WaveExtractRWZ::MultipolesGaugeInvariant()
// \brief compute gauge invariant multipoles from most general expression
// (time-dependent)
void WaveExtractRWZ::MultipolesGaugeInvariant()
{
  kappa_dd.ZeroClear();
  kappa_d.ZeroClear();
  kappa.ZeroClear();
  Tr_kappa_dd.ZeroClear();

  const Real r     = rsch;
  const Real r2    = SQR(rsch);
  const Real rdot  = dot_rsch;
  const Real div_r = 1.0 / rsch;

  // TODO This is a placeholder, these second drvt of the G multipole are not
  // computed ATM!
  const Real G_dt2 = 0.0;

  norm_Tr_kappa_dd = 0.0;

  for (int l = 2; l <= opt.lmax; ++l)
  {
    for (int m = -l; m <= l; ++m)
    {
      const int lm = ylm_.lmindex(l, m);
      for (int c = 0; c < RealImag; ++c)
      {
        // even-parity
        // -----------------------------------------

        // gauge-invariant tensor kappa_{AB}
        // NOTE: G_dt2 is a placeholder (set to 0), so kappa_dd(0,0,...)
        // is approximate. See TODO above.

        kappa_dd(0, 0, lm, c) = h00(lm, c);
        kappa_dd(0, 0, lm, c) +=
          -2.0 * h0(D01, lm, c) + 2.0 * (Gamma_dyn_udd(0, 0, 0) * h0(lm, c) +
                                         Gamma_dyn_udd(1, 0, 0) * h1(lm, c));
        kappa_dd(0, 0, lm, c) +=
          2.0 * r * rdot * G(D01, lm, c) +
          r2 * (G_dt2 - Gamma_dyn_udd(0, 0, 0) * G(D01, lm, c) -
                Gamma_dyn_udd(1, 0, 0) * G(D10, lm, c));

        kappa_dd(0, 1, lm, c) = h01(lm, c);
        kappa_dd(0, 1, lm, c) += -h1(D01, lm, c) - h0(D10, lm, c) +
                                 2.0 * (Gamma_dyn_udd(0, 0, 1) * h0(lm, c) +
                                        Gamma_dyn_udd(1, 0, 1) * h1(lm, c));
        kappa_dd(0, 1, lm, c) +=
          r * rdot * G(D10, lm, c) + r * G(D01, lm, c) +
          r2 * (G(D11, lm, c) - Gamma_dyn_udd(0, 0, 1) * G(D01, lm, c) -
                Gamma_dyn_udd(1, 0, 1) * G(D10, lm, c));

        kappa_dd(1, 1, lm, c) = h11(lm, c);
        kappa_dd(1, 1, lm, c) +=
          -2.0 * h1(D10, lm, c) + 2.0 * (Gamma_dyn_udd(0, 1, 1) * h0(lm, c) +
                                         Gamma_dyn_udd(1, 1, 1) * h1(lm, c));
        kappa_dd(1, 1, lm, c) +=
          2.0 * r * G(D10, lm, c) +
          r2 * (G(D20, lm, c) - Gamma_dyn_udd(0, 1, 1) * G(D01, lm, c) -
                Gamma_dyn_udd(1, 1, 1) * G(D10, lm, c));

        // gauge-invariant scalar kappa
        kappa(lm, c) = K(lm, c);
        kappa(lm, c) -=
          (g_uu(0, 0) * rdot * (2.0 * h0(lm, c) - r2 * G(D01, lm, c)) +
           g_uu(0, 1) * (rdot * (2.0 * h1(lm, c) - r2 * G(D10, lm, c)) +
                         2.0 * h0(lm, c) - r2 * G(D01, lm, c)) +
           g_uu(1, 1) * (2.0 * h1(lm, c) - r2 * G(D10, lm, c))) *
          div_r;

        // odd-parity
        // -----------------------------------------

        // BD: verify
        //     kappa_A = H_A - 1/2 \nabla_A H + H (\nabla_A r)/r
        // ??: kappa_d(0, lm, c) = H0 - H(D01) + H * 2.0 * rdot * div_r
        // ->: kappa_d(0, lm, c) = H0 - 0.5 * H(D01) + H * rdot * div_r
        // ??: kappa_d(1, lm, c) = H1 - H(D10) + H * 2.0 * div_r
        // ->: kappa_d(1, lm, c) = H1 - 0.5 * H(D10) + H * div_r
        kappa_d(0, lm, c) =
          H0(lm, c) - 0.5 * H(D01, lm, c) + H(lm, c) * rdot * div_r;
        kappa_d(1, lm, c) = H1(lm, c) - 0.5 * H(D10, lm, c) + H(lm, c) * div_r;

        // Trace constraint: Tr(kappa_AB) = g^{AB} kappa_{AB}
        Tr_kappa_dd(lm, c) = LinearAlgebra::TraceRank2(g_uu, kappa_dd, lm, c);
        norm_Tr_kappa_dd += SQR(Tr_kappa_dd(lm, c));

      }  // real&imag

    }  // for m
  }  // for l

  norm_Tr_kappa_dd = std::sqrt(norm_Tr_kappa_dd);
}