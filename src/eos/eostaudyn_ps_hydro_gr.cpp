//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eostaudyn_ps_hydro_gr.cpp
//  \brief Implements functions for going between primitive and conserved variables in
//  general-relativistic hydrodynamics, as well as for computing wavespeeds.

// includes -------------------------------------------------------------------

// C++ headers
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <sstream>

// Athena++ headers
#include "eos.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../utils/linear_algebra.hpp"     // Det. & friends

//#ifdef Z4C_AHF
#include "../z4c/ahf.hpp"
//#endif

// BD: TODO - a lot of the det / inv calculations could be stream-lined
//            and refactored for speed...

// ----------------------------------------------------------------------------

namespace {

// Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> eos;
// Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> ps{&eos};

typedef Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> PS;

// Declarations
static void PrimitiveToConservedSingle(
  AA &prim,
  AA &prim_scalar,
  AA &cons,
  AA &cons_scalar,
  const AT_N_sym & adm_gamma_dd_,
  int k, int j, int i,
  PS& ps);

static void SetPrimAtmo(
  AA &temperature,
  AA &prim,
  AA &prim_scalar,
  const int k, const int j, const int i,
  PS & ps);

}

//----------------------------------------------------------------------------------------
// Constructor
// Inputs:
//   pmb: pointer to MeshBlock
//   pin: pointer to runtime inputs

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) : ps{&eos}
{
  pmy_block_ = pmb;
  density_floor_ = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)));
  temperature_floor_ = pin->GetOrAddReal("hydro", "tfloor", std::sqrt(1024*(FLT_MIN)));
  scalar_floor_ = pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*FLT_MIN));

  // control PrimitiveSolver tolerances / iterates
  ps.SetRootfinderTol(pin->GetOrAddReal("hydro", "c2p_acc", 1e-15));
  ps.SetRootfinderMaxIter(pin->GetOrAddInteger("hydro", "max_iter", 30));
  ps.SetValidateDensity(pin->GetOrAddBoolean("hydro", "c2p_validate_density", true));
  ps.SetValidateDensity(pin->GetOrAddBoolean("hydro", "use_toms_748", false));

  // BD: TODO - clean up
  int ncells1 = pmb->block_size.nx1 + 2*NGHOST;
  g_.NewAthenaArray(NMETRIC, ncells1);
  g_inv_.NewAthenaArray(NMETRIC, ncells1);
  int ncells2 = (pmb->block_size.nx2 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  int ncells3 = (pmb->block_size.nx3 > 1) ? pmb->block_size.nx3 + 2*NGHOST : 1;
  fixed_.NewAthenaArray(ncells3, ncells2, ncells1);

  // Set up the EOS
#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
  std::string table = pin->GetString("hydro", "table");
  eos.SetCodeUnitSystem(&Primitive::GeometricSolar);
  eos.ReadTableFromFile(table);
  Real mb = eos.GetBaryonMass();

#elif defined(USE_IDEAL_GAS)
  // Baryon mass
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos.SetBaryonMass(mb);

#elif defined(USE_PIECEWISE_POLY)
  int n = pin->GetInteger("hydro", "n");
  Real gammas[n];
  for (int i = 0; i < n; i++)
    gammas[i] = pin->GetReal("hydro", "gamma" + std::to_string(i));
  Real rhos[n];
  rhos[0] = 0; // needed in initialization but ignored in the function
  for (int i = 1; i < n; i++)
    rhos[i] = pin->GetReal("hydro", "rho" + std::to_string(i));
  Real P0 = pin->GetReal("hydro", "P0");
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos.InitializeFromData(rhos, gammas, P0, mb, n);
  Real gamma_th = pin->GetOrAddReal("hydro", "gamma_th", 5.0/3.0);
  eos.SetThermalGamma(gamma_th);
#endif

  // Set the number density floor.
  eos.SetDensityFloor(density_floor_/mb);
  Real threshold = pin->GetOrAddReal("hydro", "dthreshold", 1.0);
  eos.SetThreshold(threshold);
  // Set the temperature floor.
  eos.SetTemperatureFloor(temperature_floor_);
  for (int i = 0; i < eos.GetNSpecies(); i++) {
    std::stringstream ss;
    ss << "y" << i << "_atmosphere";
    Real atmosphere = pin->GetOrAddReal("hydro", ss.str(), 0.5);
    eos.SetSpeciesAtmosphere(atmosphere, i);
  }


  // If we're working with an ideal gas, we need to fix the adiabatic constant.
  // MJ: is this needed
#ifdef USE_IDEAL_GAS
  gamma_ = pin->GetOrAddReal("hydro", "gamma", 2.0);
  eos.SetGamma(gamma_);
#elif defined(USE_HYBRID_EOS)
  gamma_ = pin->GetOrAddReal("hydro", "gamma_th", 2.0);
  eos.SetThermalGamma(gamma_);
#else
  // If we're not using a gamma-law EOS, we should not ever reference gamma.
  // Make sure that's the case by setting it to NaN.
  gamma_ = std::numeric_limits<double>::quiet_NaN();
#endif

  dbg_err_tol_abs = pin->GetOrAddReal("hydro", "dbg_err_tol_abs", 0);
  dbg_err_tol_rel = pin->GetOrAddReal("hydro", "dbg_err_tol_rel", 0);
  dbg_report_all  = pin->GetOrAddBoolean("hydro", "dbg_report_all", false);

}


// Initialize cold eos object based on parameters
void InitColdEOS(Primitive::ColdEOS<Primitive::COLDEOS_POLICY> *eos,
                 ParameterInput *pin) {

#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
  std::string table = pin->GetString("hydro", "table");

  // read in species names
  std::string species_names[NSCALARS];
  for (int i = 0; i < NSCALARS; i++) {
    species_names[i] = pin->GetOrAddString("hydro", "species" + std::to_string(i), "e");
  }

  eos->ReadColdSliceFromFile(table, species_names);
  eos->SetCodeUnitSystem(&Primitive::GeometricSolar);

#elif defined(USE_IDEAL_GAS)
  Real k_adi = pin->GetReal("hydro", "k_adi");
  eos->SetK(k_adi);
  Real gamma_adi = pin->GetReal("hydro","gamma");
  eos->SetGamma(gamma_adi);

#elif defined(USE_PIECEWISE_POLY)
  int n = pin->GetInteger("hydro", "n");
  Real gammas[n];
  for (int i = 0; i < n; i++)
    gammas[i] = pin->GetReal("hydro", "gamma" + std::to_string(i));
  Real rhos[n];
  rhos[0] = 0; // needed in initialization but ignored in the function
  for (int i = 1; i < n; i++)
    rhos[i] = pin->GetReal("hydro", "rho" + std::to_string(i));
  Real P0 = pin->GetReal("hydro", "P0");
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos->InitializeFromData(rhos, gammas, P0, mb, n);
  Real gamma_th = pin->GetOrAddReal("hydro", "gamma_th", 5.0/3.0);
  eos->SetThermalGamma(gamma_th);

  Real temperature_floor_ = pin->GetOrAddReal("hydro", "tfloor", std::sqrt(1024*(FLT_MIN)));
  eos->SetTemperature(temperature_floor_);
#endif

  Real density_floor = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)));
  eos->SetDensityFloor(density_floor);
}


//----------------------------------------------------------------------------------------
// Variable inverter
// Inputs:
//   cons: conserved quantities
//   prim_old: primitive quantities from previous half timestep
//   bb: face-centered magnetic field
//   pco: pointer to Coordinates
//   il,iu,jl,ju,kl,ku: index bounds of region to be updated
// Outputs:
//   prim: primitives
//   bb_cc: cell-centered magnetic field
// Notes:
//   follows Noble et al. 2006, ApJ 641 626 (N)
//       writing wgas_rel for W = \gamma^2 w
//       writing d for D
//       writing q for Q
//       writing qq for \tilde{Q}
//       writing uu for \tilde{u}
//       writing vv for v
//   implements formulas assuming no magnetic field

void EquationOfState::ConservedToPrimitive(
  AA &cons, const AA &prim_old,
  AA &prim, AA &cons_scalar,
  AA &prim_scalar, AA &bb_cc, Coordinates *pco,
  int il, int iu, int jl, int ju, int kl, int ku,
  int coarse_flag,
  bool skip_physical)
{
  MeshBlock* pmb = pmy_block_;
  Hydro * ph     = pmb->phydro;

  geom_sliced_cc gsc;

  AT_N_sca & alpha_    = gsc.alpha_;
  AT_N_sym & gamma_dd_ = gsc.gamma_dd_;
  AT_N_sym & gamma_uu_    = gsc.gamma_uu_;
  AT_N_sca & det_gamma_   = gsc.det_gamma_;


  // sanitize loop limits (coarse / fine auto-switched)
  int IL = il; int IU = iu;
  int JL = jl; int JU = ju;
  int KL = kl; int KU = ku;
  SanitizeLoopLimits(IL, IU, JL, JU, KL, KU, coarse_flag, pco);

  for (int k = KL; k <= KU; ++k)
  for (int j = JL; j <= JU; ++j)
  {
    GeometryToSlicedCC(gsc, k, j, IL, IU, coarse_flag, pco);

    // do actual variable conversion ------------------------------------------
    #pragma omp simd
    for (int i = IL; i <= IU; ++i)
    {
      if (skip_physical &&
          (pmb->is <= i) && (i <= pmb->ie) &&
          (pmb->js <= j) && (j <= pmb->je) &&
          (pmb->ks <= k) && (k <= pmb->ke))
      {
        continue;
      }

      // Check if the state is admissible; if not we reset to atmo.
      bool is_admissible = IsAdmissiblePoint(cons, prim, det_gamma_, k, j, i);

      // Deal with alpha based excision (if relevant)
      is_admissible = is_admissible && (alpha_(i) >
                                        ph->opt_excision.alpha_threshold);

      if(pmb->phydro->opt_excision.horizon_based)
      {
        Real horizon_radius;
        for (auto pah_f : pmy_block_->pmy_mesh->pah_finder)
        {
          if (not pah_f->ah_found)
            continue;
          horizon_radius = pah_f->rr_min;
          horizon_radius *= ph->opt_excision.horizon_factor;
          const Real r_2 = SQR(pco->x1v(i) - pah_f->center[0]) +
                            SQR(pco->x2v(j) - pah_f->center[1]) +
                            SQR(pco->x3v(k) - pah_f->center[2]);
          is_admissible = is_admissible && (r_2 > SQR(horizon_radius));
        }
      }

      ph->mask_reset_u(k,j,i) = !is_admissible;

      if (ph->mask_reset_u(k,j,i))
      {
        SetPrimAtmo(ph->temperature, prim, prim_scalar, k, j, i, ps);
        SetEuclideanCC(gamma_dd_, i);
        PrimitiveToConservedSingle(prim,
                                   prim_scalar,
                                   cons,
                                   cons_scalar,
                                   gamma_dd_,
                                   k, j, i,
                                   ps);

        if (pmb->phydro->c2p_status(k,j,i) == 0)
          pmb->phydro->c2p_status(k,j,i) = static_cast<int>(Primitive::Error::EXCISED);
        continue;
      }

      // Deal with PrimitiveSolver interface
      Real g3d[NSPMETRIC] = {gamma_dd_(0,0,i),
                             gamma_dd_(0,1,i),
                             gamma_dd_(0,2,i),
                             gamma_dd_(1,1,i),
                             gamma_dd_(1,2,i),
                             gamma_dd_(2,2,i)};
      const Real detgamma = det_gamma_(i);
      const Real sqrt_detgamma = std::sqrt(detgamma);
      const Real oo_sqrt_detgamma = OO(sqrt_detgamma);
      Real g3u[NSPMETRIC] = {gamma_uu_(0,0,i),
                             gamma_uu_(0,1,i),
                             gamma_uu_(0,2,i),
                             gamma_uu_(1,1,i),
                             gamma_uu_(1,2,i),
                             gamma_uu_(2,2,i)};

      // Extract and undensitize the conserved variables.
      Real cons_pt[NCONS] = {0.0};
      Real cons_old_pt[NCONS] = {0.0}; // Redundancy in case things go bad.
      cons_pt[IDN] = cons_old_pt[IDN] = cons(IDN, k, j, i) * oo_sqrt_detgamma;
      cons_pt[IM1] = cons_old_pt[IM1] = cons(IM1, k, j, i) * oo_sqrt_detgamma;
      cons_pt[IM2] = cons_old_pt[IM2] = cons(IM2, k, j, i) * oo_sqrt_detgamma;
      cons_pt[IM3] = cons_old_pt[IM3] = cons(IM3, k, j, i) * oo_sqrt_detgamma;
      cons_pt[IEN] = cons_old_pt[IEN] = cons(IEN, k, j, i) * oo_sqrt_detgamma;

      // Extract the scalars
      for(int n=0; n<NSCALARS; n++){
        cons_pt[IYD + n] = cons_old_pt[IYD + n] = cons_scalar(n,k,j,i) * oo_sqrt_detgamma;
      }

      // Find the primitive variables.
      Real prim_pt[NPRIM] = {0.0};
      Real b3u[NMAG] = {0.0}; // Assume no magnetic field.
      Primitive::SolverResult result = ps.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);

      // retain result of c2p
      if (pmb->phydro->c2p_status(k,j,i) == 0)
      {
        pmb->phydro->c2p_status(k,j,i) = static_cast<int>(result.error);
      }

      // If the lapse or metric determinant fall below zero, we're probably in an
      // unphysical regime for a fluid, like a black hole or something. Primitive
      // failure is expected and will just result in a floor being applied.
      if(result.error != Primitive::Error::SUCCESS && detgamma > 0) {
        std::cerr << "There was an error during the primitive solve!\n";
        std::cerr << "  Iteration: " << pmy_block_->pmy_mesh->ncycle << "\n";
        std::cerr << "  Error: " << Primitive::ErrorString[(int)result.error] << "\n";
        //printf("i=%d, j=%d, k=%d\n",i,j,k);
        std::cerr << "  i=" << i << ", j=" << j << ", k=" << k << "\n";
        const Real x1 = pmy_block_->pcoord->x1v(i);
        const Real x2 = pmy_block_->pcoord->x2v(j);
        const Real x3 = pmy_block_->pcoord->x3v(k);
        std::cerr << "  (x1,x2,x3) " << x1 << "," << x2 << "," << x3 << "\n";
        std::cerr << "  g3d = [" << g3d[S11] << ", " << g3d[S12] << ", " << g3d[S13] << ", "
                  << g3d[S22] << ", " << g3d[S23] << ", " << g3d[S33] << "\n";
        std::cerr << "  g3u = [" << g3u[S11] << ", " << g3u[S12] << ", " << g3u[S13] << ", "
                  << g3u[S22] << ", " << g3u[S23] << ", " << g3u[S33] << "\n";
        std::cerr << "  detgamma  = " << detgamma << "\n";
        std::cerr << "  sqrt_detgamma = " << sqrt_detgamma << "\n";
        std::cerr << "  D = " << cons_old_pt[IDN] << "\n";
        std::cerr << "  S_1 = " << cons_old_pt[IM1] << "\n";
        std::cerr << "  S_2 = " << cons_old_pt[IM2] << "\n";
        std::cerr << "  S_3 = " << cons_old_pt[IM3] << "\n";
        std::cerr << "  tau = " << cons_old_pt[IEN] << "\n";
      }
      // Update the primitive variables.
      prim(IDN, k, j, i) = prim_pt[IDN]*ps.GetEOS()->GetBaryonMass();
      prim(IVX, k, j, i) = prim_pt[IVX];
      prim(IVY, k, j, i) = prim_pt[IVY];
      prim(IVZ, k, j, i) = prim_pt[IVZ];
      prim(IPR, k, j, i) = prim_pt[IPR];
      pmy_block_->phydro->temperature(k,j,i) = prim_pt[ITM];
      for(int n=0; n<NSCALARS; n++){
        prim_scalar(n, k, j, i) = prim_pt[IYF + n];
      }

      // Because the conserved variables may have changed, we update those, too.
      cons(IDN, k, j, i) = cons_pt[IDN]*sqrt_detgamma;
      cons(IM1, k, j, i) = cons_pt[IM1]*sqrt_detgamma;
      cons(IM2, k, j, i) = cons_pt[IM2]*sqrt_detgamma;
      cons(IM3, k, j, i) = cons_pt[IM3]*sqrt_detgamma;
      cons(IEN, k, j, i) = cons_pt[IEN]*sqrt_detgamma;
      for(int n=0; n<NSCALARS; n++){
        cons_scalar(n, k, j, i) = cons_pt[IYD + n]*sqrt_detgamma;
      }
    }
  }

#if defined(DBG_IDEMPOTENT_C2P)
  // Debug whether: Id_x = c2p o p2c ------------------------------------------
  if (dbg_err_tol_abs == 0)
  {
    return;
  }

  AthenaArray<Real> id_u(NHYDRO,   pmb->ncells3, pmb->ncells2, pmb->ncells1);
  AthenaArray<Real> id_s(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  Real dbg_err_tol_abs_ = this->dbg_err_tol_abs;
  this->dbg_err_tol_abs = 0;

  PrimitiveToConserved(prim,
                       prim_scalar,
                       pmb->pfield->bcc,
                       id_u,
                       id_s,
                       pmb->pcoord,
                       IL, IU,
                       JL, JU,
                       KL, KU);

  this->dbg_err_tol_abs = dbg_err_tol_abs_;

  auto err_abs = [&](AA & F_a, AA & F_b,
                     const int n, const int k, const int j, const int i)
  {
    // return std::abs(F_a(n,k,j,i)-F_b(n,k,j,i));
    Real F_a_ = F_a(n,k,j,i);
    Real F_b_ = F_b(n,k,j,i);

    Real mag_ab = std::max(std::abs(F_a_), std::abs(F_b_));
    if (mag_ab > 1e-10)
      return std::abs((F_a(n,k,j,i)-F_b(n,k,j,i)) / mag_ab);

    return 0.0;
  };

  auto err_rel = [&](AA & F_a, AA & F_b,
                     const int n, const int k, const int j, const int i)
  {
    return std::abs(1 - F_b(n,k,j,i) / F_a(n,k,j,i));
  };

  auto dump_cons = [&](const Real err, AA & F_a, AA & F_b,
                       const int n, const int k, const int j, const int i,
                       char * tag)
  {
    #pragma omp critical
    {
      const bool is_physical_idx = pmb->IsPhysicalIndex_cc(k, j, i);

      std::printf("=== C2P o P2C [%s] err_tol_abs not met: \n", tag);
      std::printf("field_idx=%d is_physical_idx=%d \n", n, is_physical_idx);
      std::printf("(i=%d,j=%d,k=%d) ~ (%f,%f,%f)\n",
                  i, j, k,
                  pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k));
      std::printf("(err,id:F_a,F_b) ~ (%.2e,%.5e,%.5e)\n",
                  err,
                  F_a(n,k,j,i), F_b(n,k,j,i));

    }
  };


  for (int n=0;  n<NHYDRO;  ++n)
  for (int k=KL; k<=KU;     ++k)
  for (int j=JL; j<=JU;     ++j)
  for (int i=IL; i<=IU;     ++i)
  {
    const bool report = (dbg_report_all) ? true
                                         : pmb->IsPhysicalIndex_cc(k, j, i);
    const Real u_err = err_abs(id_u, cons, n, k, j, i);

    if ((u_err > dbg_err_tol_abs) && report)
    {
      char tag[] = "u";
      dump_cons(u_err, id_u, cons, n, k, j, i, tag);
    }
  }

  for (int n=0;  n<NSCALARS;  ++n)
  for (int k=KL; k<=KU;     ++k)
  for (int j=JL; j<=JU;     ++j)
  for (int i=IL; i<=IU;     ++i)
  {
    const bool report = (dbg_report_all) ? true
                                         : pmb->IsPhysicalIndex_cc(k, j, i);

    const Real s_err = err_abs(id_s, cons_scalar, n, k, j, i);

    if ((s_err > dbg_err_tol_abs) && report)
    {
      char tag[] = "s";
      dump_cons(s_err, id_s, cons_scalar, n, k, j, i, tag);
    }
  }
#endif // DBG_IDEMPOTENT_C2P

}

//----------------------------------------------------------------------------------------
// Function for converting all primitives to conserved variables
// Inputs:
//   prim: primitives
//   bb_cc: cell-centered magnetic field (unused)
//   pco: pointer to Coordinates
//   il,iu,jl,ju,kl,ku: index bounds of region to be updated
// Outputs:
//   cons: conserved variables
// Notes:
//   single-cell function exists for other purposes; call made to that function rather
//       than having duplicate code

void EquationOfState::PrimitiveToConserved(
  AA &prim,
  AA &prim_scalar,
  AA &bb_cc,
  AA &cons,
  AA &cons_scalar,
  Coordinates *pco,
  int il, int iu,
  int jl, int ju,
  int kl, int ku)
{

  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmy_block_->pcoord);
  MeshBlock * pmb = pmy_block_;

  AT_N_sym gamma_dd_(pmb->nverts1);
  AT_N_sym sl_adm_gamma_dd(pmb->pz4c->storage.adm, Z4c::I_ADM_gxx);

  // sanitize loop limits (coarse / fine auto-switched)
  const bool coarse_flag = false;
  int IL = il; int IU = iu;
  int JL = jl; int JU = ju;
  int KL = kl; int KU = ku;
  SanitizeLoopLimits(IL, IU, JL, JU, KL, KU, coarse_flag, pco);

  for (int k=KL; k<=KU; ++k)
  for (int j=JL; j<=JU; ++j)
  {
    pco_gr->GetGeometricFieldCC(gamma_dd_, sl_adm_gamma_dd, k, j);
    // Calculate the conserved variables at every point.
    for (int i=IL; i<=IU; ++i)
    {
      PrimitiveToConservedSingle(prim,
                                 prim_scalar,
                                 cons,
                                 cons_scalar,
                                 gamma_dd_,
                                 k, j, i,
                                 ps);
    }
  }

#if defined(DBG_IDEMPOTENT_C2P)
  // Debug whether: Id_x = p2c o c2p ------------------------------------------
  if (dbg_err_tol_abs == 0)
  {
    return;
  }

  AA id_w(NHYDRO,   pmb->ncells3, pmb->ncells2, pmb->ncells1);
  AA id_r(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1);

  static const int coarseflag = 0;
  Real dbg_err_tol_abs_ = this->dbg_err_tol_abs;
  this->dbg_err_tol_abs = 0;

  ConservedToPrimitive(cons,
                       id_w,
                       id_w,
                       cons_scalar,
                       id_r,
                       pmb->pfield->bcc,
                       pmb->pcoord,
                       IL, IU,
                       JL, JU,
                       KL, KU,
                       coarseflag);

  this->dbg_err_tol_abs = dbg_err_tol_abs_;

  auto err_abs = [&](AA & F_a, AA & F_b,
                     const int n, const int k, const int j, const int i)
  {
    // return std::abs(F_a(n,k,j,i)-F_b(n,k,j,i));

    Real F_a_ = F_a(n,k,j,i);
    Real F_b_ = F_b(n,k,j,i);

    Real mag_ab = std::max(std::abs(F_a_), std::abs(F_b_));
    if (mag_ab > 1e-10)
      return std::abs((F_a(n,k,j,i)-F_b(n,k,j,i)) / mag_ab);

    return 0.0;
  };

  auto dump_prim = [&](const Real err, AA & F_a, AA & F_b,
                       const int n, const int k, const int j, const int i,
                       char * tag)
  {
    #pragma omp critical
    {
      const bool is_physical_idx = pmb->IsPhysicalIndex_cc(k, j, i);

      std::printf("=== P2C o C2P [%s] err_tol_abs not met: \n", tag);
      std::printf("field_idx=%d is_physical_idx=%d \n", n, is_physical_idx);
      std::printf("(i=%d,j=%d,k=%d) ~ (%f,%f,%f)\n",
                  i, j, k,
                  pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k));
      std::printf("(err,id:F_a,F_b) ~ (%.2e,%.5e,%.5e)\n",
                  err,
                  F_a(n,k,j,i), F_b(n,k,j,i));

    }
  };


  for (int n=0;  n<NHYDRO;  ++n)
  for (int k=KL; k<=KU;     ++k)
  for (int j=JL; j<=JU;     ++j)
  for (int i=IL; i<=IU;     ++i)
  {
    const bool report = (dbg_report_all) ? true
                                         : pmb->IsPhysicalIndex_cc(k, j, i);
    const Real w_err = err_abs(id_w, prim, n, k, j, i);

    if ((w_err > dbg_err_tol_abs) && report)
    {
      char tag[] = "w";
      dump_prim(w_err, id_w, prim, n, k, j, i, tag);
    }
  }

  for (int n=0;  n<NSCALARS;  ++n)
  for (int k=KL; k<=KU;     ++k)
  for (int j=JL; j<=JU;     ++j)
  for (int i=IL; i<=IU;     ++i)
  {
    const bool report = (dbg_report_all) ? true
                                         : pmb->IsPhysicalIndex_cc(k, j, i);
    const Real r_err = err_abs(id_r, prim_scalar, n, k, j, i);

    if ((r_err > dbg_err_tol_abs) && report)
    {
      char tag[] = "r";
      dump_prim(r_err, id_r, prim_scalar, n, k, j, i, tag);
    }
  }

#endif // DBG_IDEMPOTENT_C2P
}

//----------------------------------------------------------------------------------------
// Function for calculating relativistic sound speeds
// Inputs:
//   rho_h: enthalpy per unit volume
//   pgas: gas pressure
//   vx: 3-velocity component v^x
//   gamma_lorentz_sq: Lorentz factor \gamma^2
// Outputs:
//   plambda_plus: value set to most positive wavespeed
//   plambda_minus: value set to most negative wavespeed
// Notes:
//   same function as in adiabatic_hydro_sr.cpp
//     uses SR formula (should be called in locally flat coordinates)
//   references Mignone & Bodo 2005, MNRAS 364 126 (MB)

void EquationOfState::SoundSpeedsSR(Real n, Real T, Real vx, Real gamma_lorentz_sq,
    Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]) {
  // FIXME: Need to update to work with particle fractions.
  Real Y[MAX_SPECIES] = {0.0};
  for (int n=0; n<NSCALARS; n++) {
    Y[n] = prim_scalar[n];
  }

  Real cs = ps.GetEOS()->GetSoundSpeed(n, T, Y);
  Real csq = cs*cs;
  Real sigma_s = csq / (gamma_lorentz_sq * (1.0 - csq));
  Real relative_speed = std::sqrt(sigma_s * (1.0 + sigma_s - vx*vx));
  *plambda_plus = 1.0/(1.0 + sigma_s) * (vx + relative_speed);
  *plambda_minus = 1.0/(1.0 + sigma_s) * (vx - relative_speed);
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating relativistic sound speeds in arbitrary coordinates
// Inputs:
//   rho_h: enthalpy per unit volume
//   pgas: gas pressure
//   u0,u1: 4-velocity components u^0, u^1
//   g00,g01,g11: metric components g^00, g^01, g^11
// Outputs:
//   plambda_plus: value set to most positive wavespeed
//   plambda_minus: value set to most negative wavespeed
// Notes:
//   follows same general procedure as vchar() in phys.c in Harm
//   variables are named as though 1 is normal direction

// BD: TODO - eigenvalues, _not_ the speed; should be refactored
void EquationOfState::SoundSpeedsGR(Real n, Real T, Real vi, Real v2, Real alpha,
    Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]) {
  // Calculate comoving sound speed
  // FIXME: Need to update to work with particle fractions.
  Real Y[MAX_SPECIES] = {0.0};
  for (int l=0; l<NSCALARS; l++) {
    Y[l] = prim_scalar[l];
  }

  Real cs = ps.GetEOS()->GetSoundSpeed(n, T, Y);

  Real cs_sq = cs*cs;

  Real root_1 = alpha*(vi*(1.0-cs_sq) + cs*std::sqrt( (1-v2)*(gammaii*(1.0-v2*cs_sq) - vi*vi*(1.0-cs_sq))))/(1.0-v2*cs_sq) - betai;
  Real root_2 = alpha*(vi*(1.0-cs_sq) - cs*std::sqrt( (1-v2)*(gammaii*(1.0-v2*cs_sq) - vi*vi*(1.0-cs_sq))))/(1.0-v2*cs_sq) - betai;

  bool collapse = true;
  if (collapse) {
    if (std::isnan(root_1) || std::isnan(root_2)) {
      root_1 = 1.0;
      root_2 = 1.0;
    }
  }

  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } else {
    *plambda_plus = root_2;
    *plambda_minus = root_1;
  }
  return;
}

//-----------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// Apply density and pressure floors to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AA &prim, AA &prim_scalar,
                                           int k, int j, int i)
{
  // Extract the primitive variables and floor them using PrimitiveSolver.
  Real Y[MAX_SPECIES] = {0.0};
  Real Wvu[3] = {};
  Real P;
  Real n;

  Real mb = ps.GetEOS()->GetBaryonMass();

  // BD: TODO - this kind of switching... just use polymorphism for 1d slices?
  if (prim.GetDim4()==1)
  {
    n = prim(IDN,i)/mb;
    P = prim(IPR,i);

    for (int a=0; a<3; ++a)
    {
      Wvu[a] = prim(IVX+a,i);
    }

    for (int l=0; l<NSCALARS; l++) {
      Y[l] = prim_scalar(l,i);
    }
  }
  else
  {
    n = prim(IDN,k,j,i)/mb;
    P = prim(IPR,k,j,i);

    for (int a=0; a<3; ++a)
    {
      Wvu[a] = prim(IVX+a,k,j,i);
    }

    for (int l=0; l<NSCALARS; l++) {
      Y[l] = prim_scalar(l,k,j,i);
    }
  }

  ps.GetEOS()->ApplyDensityLimits(n);
  ps.GetEOS()->ApplySpeciesLimits(Y);
  Real T = ps.GetEOS()->GetTemperatureFromP(n, P, Y);
  ps.GetEOS()->ApplyPrimitiveFloor(n, Wvu, P, T, Y);

  // Now push the updated quantities back to Athena.
  if (prim.GetDim4()==1)
  {
    prim(IDN,i) = n*mb;
    prim(IVX,i) = Wvu[0];
    prim(IVY,i) = Wvu[1];
    prim(IVZ,i) = Wvu[2];
    prim(IPR,i) = P;
    for (int l=0; l<NSCALARS; l++) {
      prim_scalar(l,i) = Y[l];
    }
  }
  else
  {
    prim(IDN,k,j,i) = n*mb;
    prim(IVX,k,j,i) = Wvu[0];
    prim(IVY,k,j,i) = Wvu[1];
    prim(IVZ,k,j,i) = Wvu[2];
    prim(IPR,k,j,i) = P;
    for (int l=0; l<NSCALARS; l++) {
      prim_scalar(l,k,j,i) = Y[l];
    }
  }
  return;
}


//-----------------------------------------------------------------------------
namespace {

static void PrimitiveToConservedSingle(
  AA &prim,
  AA &prim_scalar,
  AA &cons,
  AA &cons_scalar,
  const AT_N_sym & adm_gamma_dd_,
  int k, int j, int i,
  PS& ps)
{
  // Extract the primitive variables
  Real prim_pt[NPRIM] = {0.0};
  Real Y[MAX_SPECIES] = {0.0};
  Real bu[NMAG] = {0.0};
  Real mb = ps.GetEOS()->GetBaryonMass();
  prim_pt[IDN] = prim(IDN, k, j, i)/mb;
  prim_pt[IVX] = prim(IVX, k, j, i);
  prim_pt[IVY] = prim(IVY, k, j, i);
  prim_pt[IVZ] = prim(IVZ, k, j, i);
  prim_pt[IPR] = prim(IPR, k, j, i);

  for (int n=0; n<NSCALARS; n++) {
    Y[n] = prim_scalar(n,k,j,i);
  }

  // Get temperature and apply floor
  ps.GetEOS()->ApplyDensityLimits(prim_pt[IDN]);
  ps.GetEOS()->ApplySpeciesLimits(Y);
  prim_pt[ITM] =
      ps.GetEOS()->GetTemperatureFromP(prim_pt[IDN], prim_pt[IPR], Y);
  bool result = ps.GetEOS()->ApplyPrimitiveFloor(prim_pt[IDN], &prim_pt[IVX],
                                                 prim_pt[IPR], prim_pt[ITM],
                                                 Y);

  for (int n=0; n<NSCALARS; n++) {
    prim_pt[IYF + n] = Y[n];
  }

  // Extract the metric and calculate the determinant.
  Real g3d[NSPMETRIC] = {adm_gamma_dd_(0,0,i),
                         adm_gamma_dd_(0,1,i),
                         adm_gamma_dd_(0,2,i),
                         adm_gamma_dd_(1,1,i),
                         adm_gamma_dd_(1,2,i),
                         adm_gamma_dd_(2,2,i)};
  Real detg = Primitive::GetDeterminant(g3d);
  Real sdetg = std::sqrt(detg);

  // Perform the primitive solve.
  Real cons_pt[NCONS];

  ps.PrimToCon(prim_pt, cons_pt, bu, g3d);

  // DEBUG ONLY
  if (!std::isfinite(cons_pt[IEN])) {
    std::cerr << "Tau is not finite!\n";
    std::cerr << "  Error occurred at (" << i << ", " << j << ", " << k << ")\n";
    std::cerr << "  Primitive variables:\n";
    std::cerr << "    rho = " << prim(IDN, k, j, i) << "\n";
    std::cerr << "    ux  = " << prim(IVX, k, j, i) << "\n";
    std::cerr << "    uy  = " << prim(IVY, k, j, i) << "\n";
    std::cerr << "    uz  = " << prim(IVZ, k, j, i) << "\n";
    std::cerr << "    P   = " << prim(IPR, k, j, i) << "\n";
    std::cerr << "  Conserved variables:\n";
    std::cerr << "    D   = " << cons_pt[IDN] << "\n";
    std::cerr << "    Sx  = " << cons_pt[IM1] << "\n";
    std::cerr << "    Sy  = " << cons_pt[IM2] << "\n";
    std::cerr << "    Sz  = " << cons_pt[IM3] << "\n";
    std::cerr << "    tau = " << cons_pt[IEN] << "\n";
    std::cerr << "  Metric:\n";
    std::cerr << "    g3d = {" << g3d[S11] << ", " << g3d[S12] << ", " << g3d[S13] << ", "
                               << g3d[S22] << ", " << g3d[S23] << ", " << g3d[S33] << "}\n";
    std::cerr << "    detg  = " << detg << "\n";
    std::cerr << "    sdetg = " << sdetg << "\n";
  }

  // Push the densitized conserved variables to Athena.
  cons(IDN, k, j, i) = cons_pt[IDN]*sdetg;
  cons(IM1, k, j, i) = cons_pt[IM1]*sdetg;
  cons(IM2, k, j, i) = cons_pt[IM2]*sdetg;
  cons(IM3, k, j, i) = cons_pt[IM3]*sdetg;
  cons(IEN, k, j, i) = cons_pt[IEN]*sdetg;
  for (int n = 0; n < NSCALARS; n++) {
      cons_scalar(n, k, j, i)= cons_pt[IYD + n]*sdetg;
    }

  // If we floored things, we'll need to readjust the primitives.
  if (result) {
    prim(IDN, k, j, i) = prim_pt[IDN]*mb;
    prim(IVX, k, j, i) = prim_pt[IVX];
    prim(IVY, k, j, i) = prim_pt[IVY];
    prim(IVZ, k, j, i) = prim_pt[IVZ];
    prim(IPR, k, j, i) = prim_pt[IPR];
    for (int n=0; n<NSCALARS; n++) {
      prim_scalar(n,k,j,i) = prim_pt[IYF + n];
    }
  }
}

static void SetPrimAtmo(
  AA &temperature,
  AA &prim,
  AA &prim_scalar,
  const int k, const int j, const int i,
  PS & ps)
{
  Real prim_pt[NPRIM] = {0.0};
  ps.GetEOS()->DoFailureResponse(prim_pt);

  // Update the primitive variables.
  prim(IDN, k, j, i) = prim_pt[IDN]*ps.GetEOS()->GetBaryonMass();
  prim(IVX, k, j, i) = prim_pt[IVX];
  prim(IVY, k, j, i) = prim_pt[IVY];
  prim(IVZ, k, j, i) = prim_pt[IVZ];
  prim(IPR, k, j, i) = prim_pt[IPR];
  temperature(k,j,i) = prim_pt[ITM];
  for(int n=0; n<NSCALARS; n++){
    prim_scalar(n, k, j, i) = prim_pt[IYF + n];
  }
}

} // namespace
