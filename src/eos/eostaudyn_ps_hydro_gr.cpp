//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eostaudyn_ps_hydro_gr.cpp
//  \brief Implements functions for going between primitive and conserved variables in
//  general-relativistic hydrodynamics, as well as for computing wavespeeds.

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

// PrimitiveSolver headers

/*#include "../z4c/primitive/primitive_solver.hpp"
#include "../z4c/primitive/eos.hpp"*/

// Declarations
static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, AthenaArray<Real> &prim_scalar,
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, int k, int j, int i,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar,
    Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>& ps);

static Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i);

//Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> eos;
//Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> ps{&eos};

//----------------------------------------------------------------------------------------
// Constructor
// Inputs:
//   pmb: pointer to MeshBlock
//   pin: pointer to runtime inputs

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) : ps{&eos} {
  pmy_block_ = pmb;
  density_floor_ = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)));
  temperature_floor_ = pin->GetOrAddReal("hydro", "tfloor", std::sqrt(1024*(FLT_MIN)));
  scalar_floor_ = pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*FLT_MIN));

  // control PrimitiveSolver tolerances / iterates
  ps.SetRootfinderTol(pin->GetOrAddReal("hydro", "c2p_acc", 1e-15));
  ps.SetRootfinderMaxIter(pin->GetOrAddInteger("hydro", "max_iter", 30));

  int ncells1 = pmb->block_size.nx1 + 2*NGHOST;
  g_.NewAthenaArray(NMETRIC, ncells1);
  g_inv_.NewAthenaArray(NMETRIC, ncells1);
  int ncells2 = (pmb->block_size.nx2 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  int ncells3 = (pmb->block_size.nx3 > 1) ? pmb->block_size.nx3 + 2*NGHOST : 1;
  fixed_.NewAthenaArray(ncells3, ncells2, ncells1);

  // Set up the EOS
  // If we're using a tabulated EOS, load the table.
  #if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
  std::string table = pin->GetString("hydro", "table");
  eos.SetCodeUnitSystem(&Primitive::GeometricSolar);
  eos.ReadTableFromFile(table);
  #endif

  #ifdef USE_IDEAL_GAS
  // Baryon mass
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos.SetBaryonMass(mb);
  #else
  Real mb = eos.GetBaryonMass();
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

void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
    const AthenaArray<Real> &prim_old, const FaceField &bb, AthenaArray<Real> &prim, 
    AthenaArray<Real> &cons_scalar, AthenaArray<Real> &prim_scalar, 
    AthenaArray<Real> &bb_cc, Coordinates *pco, int il, int iu, int jl, int ju, int kl,
    int ku, int coarse_flag) {
  int nn1;
  GRDynamical* pco_gr;
  MeshBlock* pmb = pmy_block_;
  Z4c* pz4c = pmb->pz4c;

  if (coarse_flag) {
    pco_gr = static_cast<GRDynamical*>(pmb->pmr->pcoarsec);
    nn1 = pmb->ncv1;
  }
  else {
    pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
    nn1 = pmb->nverts1;
  }
  
  // Full 3D metric quantities
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> full_gamma_dd;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> full_alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> full_chi;

  // Sliced quantities
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rchi;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;
  gamma_dd.NewAthenaTensor(nn1);
  alpha.NewAthenaTensor(nn1);

  if (!coarse_flag) {
    full_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
    full_alpha.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_alpha);
  }
  else {
    rchi.NewAthenaTensor(nn1);

    full_gamma_dd.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_gxx);
    full_alpha.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_alpha);
    full_chi.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_chi);
  }

  // sanitize loop limits (coarse / fine auto-switched)
  int IL, IU, JL, JU, KL, KU;
  pco_gr->GetGeometricFieldCCIdxRanges(
    IL, IU,
    JL, JU,
    KL, KU);

  // Restrict further the ranges to the input argument
  IL = std::max(il, IL);
  IU = std::min(iu, IU);

  JL = std::max(jl, JL);
  JU = std::min(ju, JU);

  KL = std::max(kl, KL);
  KU = std::min(ku, KU);

  // Go through the cells
  for (int k = KL; k <= KU; ++k) {
    for (int j = JL; j <= JU; ++j) {
      pco_gr->GetGeometricFieldCC(gamma_dd, full_gamma_dd, k, j);
      pco_gr->GetGeometricFieldCC(alpha, full_alpha, k, j);
      if (coarse_flag) {
        pco_gr->GetGeometricFieldCC(rchi, full_chi, k, j);

        #pragma omp simd
        for (int i = IL; i <= IU; ++i) {
          rchi(i) = 1./rchi(i);
        }

        for (int a = 0; a < NDIM; ++a) {
          for (int b = a; b < NDIM; ++b) {
            #pragma omp simd
            for (int i = IL; i <= IU; ++i) {
              gamma_dd(a, b, i) = gamma_dd(a, b, i) * rchi(i);
            }
          }
        }
      }
      // Extract the primitive variables
      #pragma omp simd
      for (int i = IL; i <= IU; ++i) {
        // Extract the local metric and stuff it into a smaller array for PrimitiveSolver.
        Real g3d[NSPMETRIC] = {gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
                              gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i)};
        // Calculate the determinant and the inverse metric
        Real detg = Primitive::GetDeterminant(g3d);
        Real sdetg = std::sqrt(detg);
        Real g3u[NSPMETRIC];
        Primitive::InvertMatrix(g3u, g3d, detg);

        // Extract and undensitize the conserved variables.
        Real cons_pt[NCONS] = {0.0};
        Real cons_old_pt[NCONS] = {0.0}; // Redundancy in case things go bad.
        cons_pt[IDN] = cons_old_pt[IDN] = cons(IDN, k, j, i)/sdetg;
        cons_pt[IM1] = cons_old_pt[IM1] = cons(IM1, k, j, i)/sdetg;
        cons_pt[IM2] = cons_old_pt[IM2] = cons(IM2, k, j, i)/sdetg;
        cons_pt[IM3] = cons_old_pt[IM3] = cons(IM3, k, j, i)/sdetg;
        cons_pt[IEN] = cons_old_pt[IEN] = cons(IEN, k, j, i)/sdetg;

        // Extract the scalars
        for(int n=0; n<NSCALARS; n++){
          cons_pt[IYD + n] = cons_old_pt[IYD + n] = cons_scalar(n,k,j,i)/sdetg;
        }

        // Find the primitive variables.
        Real prim_pt[NPRIM] = {0.0};
        Real b3u[NMAG] = {0.0}; // Assume no magnetic field.
        Primitive::SolverResult result = ps.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);

        // If the lapse or metric determinant fall below zero, we're probably in an
        // unphysical regime for a fluid, like a black hole or something. Primitive
        // failure is expected and will just result in a floor being applied.
        if(result.error != Primitive::Error::SUCCESS && detg > 0) {
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
          std::cerr << "  detg  = " << detg << "\n";
          std::cerr << "  sdetg = " << sdetg << "\n";
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
        cons(IDN, k, j, i) = cons_pt[IDN]*sdetg;
        cons(IM1, k, j, i) = cons_pt[IM1]*sdetg;
        cons(IM2, k, j, i) = cons_pt[IM2]*sdetg;
        cons(IM3, k, j, i) = cons_pt[IM3]*sdetg;
        cons(IEN, k, j, i) = cons_pt[IEN]*sdetg;
        for(int n=0; n<NSCALARS; n++){
          cons_scalar(n, k, j, i) = cons_pt[IYD + n]*sdetg;
        }
      }
    }
  }

  // clean-up
  alpha.DeleteAthenaTensor();
  gamma_dd.DeleteAthenaTensor();

  if (coarse_flag) {
    rchi.DeleteAthenaTensor();
  }
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
  AthenaArray<Real> &prim,
  AthenaArray<Real> &prim_scalar,
  AthenaArray<Real> &bb_cc,
  AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar,
  Coordinates *pco,
  int il, int iu,
  int jl, int ju,
  int kl, int ku)
{

  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmy_block_->pcoord);

  AT_N_sym gamma_dd(pmy_block_->nverts1);
  AT_N_sym full_gamma_dd(pmy_block_->pz4c->storage.adm, Z4c::I_ADM_gxx);

  // sanitize loop limits
  int IL, IU, JL, JU, KL, KU;
  pco_gr->GetGeometricFieldCCIdxRanges(
    IL, IU,
    JL, JU,
    KL, KU);

  // Restrict further the ranges to the input argument
  IL = std::max(il, IL);
  IU = std::min(iu, IU);

  JL = std::max(jl, JL);
  JU = std::min(ju, JU);

  KL = std::max(kl, KL);
  KU = std::min(ku, KU);

  for (int k=KL; k<=KU; ++k) {
    for (int j=JL; j<=JU; ++j) {
      pco_gr->GetGeometricFieldCC(gamma_dd, full_gamma_dd, k, j);
      // Calculate the conserved variables at every point.
      for (int i=IL; i<=IU; ++i)
      {
        PrimitiveToConservedSingle(prim,
                                   prim_scalar,
                                   gamma_dd,
                                   k, j, i,
                                   cons,
                                   cons_scalar,
                                   ps);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// Function for converting primitives to conserved variables in a single cell
// Inputs:
//   prim: 3D array of primitives
//   gamma_adi: ratio of specific heats
//   g,gi: 1D arrays of metric covariant and contravariant coefficients
//   k,j,i: indices of cell
//   pco: pointer to Coordinates
// Outputs:
//   cons: conserved variables set in desired cell

static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, AthenaArray<Real> &prim_scalar,
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const &gamma_dd, int k, int j, int i,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar,
    Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>& ps) {

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
  prim_pt[ITM] = ps.GetEOS()->GetTemperatureFromP(prim_pt[IDN], prim_pt[IPR], Y);
  bool result = ps.GetEOS()->ApplyPrimitiveFloor(prim_pt[IDN], &prim_pt[IVX], prim_pt[IPR], prim_pt[ITM], Y);
  
  for (int n=0; n<NSCALARS; n++) {
    prim_pt[IYF + n] = Y[n];
  }

  // Extract the metric and calculate the determinant.
  Real g3d[NSPMETRIC] = {gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
                        gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i)};
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

//---------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// \brief Apply density and pressure floors to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, AthenaArray<Real> &prim_scalar, int k, int j, int i) {
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

//---------------------------------------------------------------------------------------
// \!fn static Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform linear interpolation to the desired cell-centered grid index.

static Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.125*(((in(k, j, i) + in(k + 1, j + 1, i + 1)) // lower-left-front to upper-right-back
               + (in(k+1, j+1, i) + in(k, j, i+1))) // upper-left-back to lower-right-front
               +((in(k, j+1, i) + in(k + 1, j, i+1)) // lower-left-back to upper-right-front
               + (in(k+1, j, i) + in(k, j+1, i+1)))); // upper-left-front to lower-right-back
}
