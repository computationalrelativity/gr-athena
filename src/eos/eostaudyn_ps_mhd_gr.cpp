//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eostaudyn_ps_mhd_gr.cpp
//  \brief Implements functions for going between primitive and conserved variables in
//  general-relativistic magnetohydrodynamics, as well as for computing wavespeeds.

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
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../utils/interp_intergrid.hpp"

static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, AthenaArray<Real>& bb_cc,
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, 
    int k, int j, int i,
    AthenaArray<Real> &cons,
    Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>& ps);

static Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i);

using RescaleFunction = void(*)(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2>& gamma_dd,
                                AthenaArray<Real>& vcchi,
                                AthenaTensor<Real, TensorSymm::NONE, NDIM, 0>& chi,
                                InterpIntergridLocal* interp,
                                int il, int iu, int j, int k);

//----------------------------------------------------------------------------------------
// Constructor
// Inputs:
//   pmb: pointer to MeshBlock
//   pin: pointer to runtime inputs

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) : ps{&eos} {
  pmy_block_ = pmb;
  density_floor_ = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)));
  pressure_floor_ = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*(FLT_MIN)));

  int ncells1 = pmb->block_size.nx1 + 2*NGHOST;
  g_.NewAthenaArray(NMETRIC, ncells1);
  g_inv_.NewAthenaArray(NMETRIC, ncells1);
  int ncells2 = (pmb->block_size.nx2 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  int ncells3 = (pmb->block_size.nx3 > 1) ? pmb->block_size.nx3 + 2*NGHOST : 1;
  fixed_.NewAthenaArray(ncells3, ncells2, ncells1);

  // Set up the EOS
  // If we're using a tabulated EOS, load the table.
  #ifdef USE_COMPOSE_EOS
  std::string table = pin->GetString("hydro", "table");
  eos.ReadTableFromFile(table);
  #endif
  #ifdef USE_IDEAL_GAS
  // Baryon mass
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos.SetBaryonMass(mb);
  #else
  Real mb = eos.GetBaryonMass();
  #endif
  eos.SetDensityFloor(density_floor_/mb);
  Real threshold = pin->GetOrAddReal("hydro", "dthreshold", 1.0);
  eos.SetThreshold(threshold);
  // Set the number density floor.
  eos.SetPressureFloor(pressure_floor_);
  for (int i = 0; i < eos.GetNSpecies(); i++) {
    std::stringstream ss;
    ss << "y" << i << "_atmosphere";
    Real atmosphere = pin->GetOrAddReal("hydro", ss.str(), 0.5);
    eos.SetSpeciesAtmosphere(atmosphere, i);
  }

  // If we're working wtih an ideal gas, we need to fix the adiabatic constant.
  #ifdef USE_IDEAL_GAS
  gamma_ = pin->GetOrAddReal("hydro", "gamma", 2.0);
  eos.SetGamma(gamma_);
  #else
  // If we're not using a gamma-law EOS, we should not ever reference gamma.
  // Make sure that's the case by setting it to NaN.
  gamma_ = std::numeric_limits<double>::quiet_NaN();
  #endif

  // FIXME: Set some parameters here for MHD.
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
     AthenaArray<Real> &bb_cc, Coordinates *pco, int il, int iu, int jl, int ju, int kl,
     int ku, int coarse_flag) {
  // Parameters
  int nn1 = iu + 1;

  // Vertex-centered containers for the metric.
  AthenaArray<Real> vcgamma_xx, vcgamma_xy, vcgamma_xz, vcgamma_yy,
                    vcgamma_yz, vcgamma_zz, vcchi;
  
  // Operations that change based on whether or not we have coarse variables;
  // this avoids an extra branch during the interpolation loop.
  InterpIntergridLocal* interp;
  RescaleFunction rescale_metric;

  // Metric at cell centers
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> chi;
  if (coarse_flag == 0) {
    vcgamma_xx.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
    vcgamma_xy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
    vcgamma_xz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
    vcgamma_yy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
    vcgamma_yz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
    vcgamma_zz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
    interp = pmy_block_->pz4c->ig;
    auto lambda = [](AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2>& gamma_dd,
                     AthenaArray<Real>& vcchi,
                     AthenaTensor<Real, TensorSymm::NONE, NDIM, 0>& chi,
                     InterpIntergridLocal* interp,
                     int il, int iu, int j, int k) -> void {return;};
    rescale_metric = lambda;
  }
  else {
    chi.NewAthenaTensor(nn1);
    vcgamma_xx.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gxx,1);
    vcgamma_xy.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gxy,1);
    vcgamma_xz.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gxz,1);
    vcgamma_yy.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gyy,1);
    vcgamma_yz.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gyz,1);
    vcgamma_zz.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gzz,1);
    vcchi.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_chi,1);
    interp = pmy_block_->pz4c->ig_coarse;
    auto lambda = [](AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2>& gamma_dd,
                     AthenaArray<Real>& vcchi,
                     AthenaTensor<Real, TensorSymm::NONE, NDIM, 0>& chi,
                     InterpIntergridLocal* interp,
                     int il, int iu, int j, int k) -> void {
      #pragma omp simd
      for (int i = il; i <= iu; i++) {
        chi(i) = interp->map3d_VC2CC(vcchi(k, j, i));
        gamma_dd(0, 0, i) = gamma_dd(0, 0, i)/chi(i);
        gamma_dd(0, 1, i) = gamma_dd(0, 1, i)/chi(i);
        gamma_dd(0, 2, i) = gamma_dd(0, 2, i)/chi(i);
        gamma_dd(1, 1, i) = gamma_dd(1, 1, i)/chi(i);
        gamma_dd(1, 2, i) = gamma_dd(1, 2, i)/chi(i);
        gamma_dd(2, 2, i) = gamma_dd(2, 2, i)/chi(i);
      }
    };
    rescale_metric = lambda;
  }
  pmy_block_->pfield->CalculateCellCenteredField(bb, bb_cc, pco, il, iu, jl, ju, kl, ku);

  // Go through the cells
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      // Extract the metric at the vertex centers and interpolate to cell centers.
      #pragma omp simd
      for (int i = il; i <= iu; ++i) {
        gamma_dd(0,0,i) = interp->map3d_VC2CC(vcgamma_xx(k,j,i));
        gamma_dd(0,1,i) = interp->map3d_VC2CC(vcgamma_xy(k,j,i));
        gamma_dd(0,2,i) = interp->map3d_VC2CC(vcgamma_xz(k,j,i));
        gamma_dd(1,1,i) = interp->map3d_VC2CC(vcgamma_yy(k,j,i));
        gamma_dd(1,2,i) = interp->map3d_VC2CC(vcgamma_yz(k,j,i));
        gamma_dd(2,2,i) = interp->map3d_VC2CC(vcgamma_zz(k,j,i));
        // DEBUG ONLY
        /*gamma_dd(0,0,i) = VCInterpolation(vcgamma_xx, k, j, i);
        gamma_dd(0,1,i) = VCInterpolation(vcgamma_xy, k, j, i);
        gamma_dd(0,2,i) = VCInterpolation(vcgamma_xz, k, j, i);
        gamma_dd(1,1,i) = VCInterpolation(vcgamma_yy, k, j, i);
        gamma_dd(1,2,i) = VCInterpolation(vcgamma_yz, k, j, i);
        gamma_dd(2,2,i) = VCInterpolation(vcgamma_zz, k, j, i);*/
      }
      rescale_metric(gamma_dd, vcchi, chi, interp, il, iu, j, k);

      // Extract the primitive variables
      #pragma omp simd
      for (int i = il; i <= iu; ++i) {
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
        // FIXME: Need to generalize to particle fractions.

        // Extract the magnetic field.
        Real b3u[NMAG] = {bb_cc(IB1, k, j, i)/sdetg,
                          bb_cc(IB2, k, j, i)/sdetg,
                          bb_cc(IB3, k, j, i)/sdetg};

        // Find the primitive variables.
        Real prim_pt[NPRIM] = {0.0};
        Primitive::SolverResult result = ps.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
        
        if (result.error != Primitive::Error::SUCCESS) {
          std::cerr << "There was an error during the primitive solve!\n";
          std::cerr << "  Iteration: " << pmy_block_->pmy_mesh->ncycle << "\n";
          std::cerr << "  Error: " << Primitive::ErrorString[(int)result.error] << "\n";
          std::cerr << "  i=" << i << ", j=" << j << ", k=" << k << "\n";
          std::cerr << "  g3d = [" << g3d[S11] << ", " << g3d[S12] << ", " << g3d[S13] << ", "
                    << g3d[S22] << ", " << g3d[S23] << ", " << g3d[S33] << "]\n";
          std::cerr << "  g3u = [" << g3u[S11] << ", " << g3u[S12] << ", " << g3u[S13] << ", "
                    << g3u[S22] << ", " << g3u[S23] << ", " << g3u[S33] << "]\n";
          std::cerr << "  detg = " << detg << "\n";
          std::cerr << "  sdetg = " << sdetg << "\n";
          std::cerr << "  D = " << cons_old_pt[IDN] << "\n";
          std::cerr << "  S_1 = " << cons_old_pt[IM1] << "\n";
          std::cerr << "  S_2 = " << cons_old_pt[IM2] << "\n";
          std::cerr << "  S_3 = " << cons_old_pt[IM3] << "\n";
          std::cerr << "  tau = " << cons_old_pt[IEN] << "\n";
          // FIXME: Add particle fractions
          std::cerr << "  b_u = [" << bb_cc(IB1, k, j, i) << ", " << bb_cc(IB2, k, j, i)
                                   << bb_cc(IB3, k, j, i) << "]\n";
        }
        // Update the primitive variables.
        prim(IDN, k, j, i) = prim_pt[IDN]*ps.GetEOS()->GetBaryonMass();
        prim(IVX, k, j, i) = prim_pt[IVX];
        prim(IVY, k, j, i) = prim_pt[IVY];
        prim(IVZ, k, j, i) = prim_pt[IVZ];
        prim(IPR, k, j, i) = prim_pt[IPR];
        
        // Because the conserved variables may have changed, we update those, too.
        cons(IDN, k, j, i) = cons_pt[IDN]*sdetg;
        cons(IM1, k, j, i) = cons_pt[IM1]*sdetg;
        cons(IM2, k, j, i) = cons_pt[IM2]*sdetg;
        cons(IM3, k, j, i) = cons_pt[IM3]*sdetg;
        cons(IEN, k, j, i) = cons_pt[IEN]*sdetg;

      }
    }
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

void EquationOfState::PrimitiveToConserved(AthenaArray<Real> &prim,
     AthenaArray<Real> &bb_cc, AthenaArray<Real> &cons, Coordinates *pco, int il,
     int iu, int jl, int ju, int kl, int ku) {
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
  int nn1 = iu+1;
  gamma_dd.NewAthenaTensor(nn1);
  AthenaArray<Real> vcgamma_xx, vcgamma_xy, vcgamma_xz, vcgamma_yy,
                    vcgamma_yz, vcgamma_zz;
  vcgamma_xx.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
  vcgamma_xy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
  vcgamma_xz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
  vcgamma_yy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
  vcgamma_yz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
  vcgamma_zz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
  
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        gamma_dd(0,0,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
        gamma_dd(0,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
        gamma_dd(0,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
        gamma_dd(1,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
        gamma_dd(1,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
        gamma_dd(2,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));
      }
      // Calculate the conserved variables at every point.
      for (int i=il; i<=iu; ++i) {
        PrimitiveToConservedSingle(prim, bb_cc, gamma_dd, k, j, i, cons, ps);
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

static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, AthenaArray<Real>& bb_cc,
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, 
    int k, int j, int i,
    AthenaArray<Real> &cons,
    Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>& ps) {
  // Extract the primitive variables
  Real prim_pt[NPRIM] = {0.0};
  Real Y[MAX_SPECIES] = {0.0}; // FIXME: Need to add support for particle fractions.
  Real mb = ps.GetEOS()->GetBaryonMass();
  prim_pt[IDN] = prim(IDN, k, j, i)/mb;
  prim_pt[IVX] = prim(IVX, k, j, i);
  prim_pt[IVY] = prim(IVY, k, j, i);
  prim_pt[IVZ] = prim(IVZ, k, j, i);
  prim_pt[IPR] = prim(IPR, k, j, i);
  prim_pt[ITM] = ps.GetEOS()->GetTemperatureFromP(prim_pt[IDN], prim_pt[IPR], Y);

  // Apply the floor to ensure that we get physical conserved variables.
  bool result = ps.GetEOS()->ApplyPrimitiveFloor(prim_pt[IDN], &prim_pt[IVX], prim_pt[IPR], prim_pt[ITM], Y);

  // Extract the metric and calculate the determinant..
  Real g3d[NSPMETRIC] = {gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
                         gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i)};
  Real detg = Primitive::GetDeterminant(g3d);
  Real sdetg = std::sqrt(detg);

  // Extract and undensitize the magnetic field.
  Real bu[NMAG] = {bb_cc(IB1, k, j, i)/sdetg, bb_cc(IB2, k, j, i)/sdetg,
                   bb_cc(IB3, k, j, i)/sdetg};

  // Perform the primitive solve.
  Real cons_pt[NCONS];

  ps.PrimToCon(prim_pt, cons_pt, bu, g3d);

  // Push the densitized conserved variables to Athena.
  cons(IDN, k, j, i) = cons_pt[IDN]*sdetg;
  cons(IM1, k, j, i) = cons_pt[IM1]*sdetg;
  cons(IM2, k, j, i) = cons_pt[IM2]*sdetg;
  cons(IM3, k, j, i) = cons_pt[IM3]*sdetg;
  cons(IEN, k, j, i) = cons_pt[IEN]*sdetg;

  // If we floored things, we'll need to readjust the primitives.
  if (result) {
    prim(IDN, k, j, i) = prim_pt[IDN]*mb;
    prim(IVX, k, j, i) = prim_pt[IVX];
    prim(IVY, k, j, i) = prim_pt[IVY];
    prim(IVZ, k, j, i) = prim_pt[IVZ];
    prim(IPR, k, j, i) = prim_pt[IPR];
  }
}

void EquationOfState::FastMagnetosonicSpeedsGR(Real n, Real T, Real bsq, Real vi, Real v2, 
    Real alpha, Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus) {
  // Constants and stuff
  Real Wlor = std::sqrt(1.0 - v2);
  Wlor = 1.0/Wlor;
  Real u0 = Wlor/alpha;
  Real g00 = -1.0/(alpha*alpha);
  Real g01 = betai/(alpha*alpha);
  Real u1 = vi*Wlor;
  Real g11 = gammaii - betai*betai/(alpha*alpha);
  // Calculate comoving fast magnetosonic speed
  // FIXME: Need to update to work with particle fractions.
  Real Y[MAX_SPECIES] = {0.0};
  Real cs = ps.GetEOS()->GetSoundSpeed(n, T, Y);
  Real cs_sq = cs*cs;
  Real mb = ps.GetEOS()->GetBaryonMass();
  Real va_sq = bsq/(bsq + n*mb*ps.GetEOS()->GetEnthalpy(n, T, Y));
  Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

  // Set fast magnetosonic speeds in appropriate coordinates
  Real a = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
  Real b = -2.0 * (u0*u1 - (g01 + u0*u1) * cms_sq);
  Real c = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
  Real d = std::max(SQR(b) - 4.0*a*c, 0.0);
  Real d_sqrt = std::sqrt(d);
  Real root_1 = (-b + d_sqrt) / (2.0*a);
  Real root_2 = (-b - d_sqrt) / (2.0*a);
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

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  // Extract the primitive variables and floor them using PrimitiveSolver.
  Real mb = ps.GetEOS()->GetBaryonMass();
  Real n = prim(IDN, i)/mb;
  Real Wvu[3] = {prim(IVX, i), prim(IVY, i), prim(IVZ, i)};
  Real P = prim(IPR, i);
  // FIXME: Update to work with particle species.
  Real Y[MAX_SPECIES] = {0.0};
  Real T = ps.GetEOS()->GetTemperatureFromP(n, P, Y);
  ps.GetEOS()->ApplyPrimitiveFloor(n, Wvu, P, T, Y);

  // Now push the updated quantities back to Athena.
  prim(IDN, i) = n*mb;
  prim(IVX, i) = Wvu[0];
  prim(IVY, i) = Wvu[1];
  prim(IVZ, i) = Wvu[2];
  prim(IPR, i) = P;
  return;
}
