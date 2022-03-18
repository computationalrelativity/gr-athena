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

// Athena++ headers
#include "eos.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"

// PrimitiveSolver headers

/*#include "../z4c/primitive/primitive_solver.hpp"
#include "../z4c/primitive/eos.hpp"*/

// Declarations
static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, 
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, int k, int j, int i,
    AthenaArray<Real> &cons,
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
  pressure_floor_ = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*(FLT_MIN)));

  int ncells1 = pmb->block_size.nx1 + 2*NGHOST;
  g_.NewAthenaArray(NMETRIC, ncells1);
  g_inv_.NewAthenaArray(NMETRIC, ncells1);
  int ncells2 = (pmb->block_size.nx2 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  int ncells3 = (pmb->block_size.nx3 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  fixed_.NewAthenaArray(ncells3, ncells2, ncells1);

  // Set up the EOS
  // Baryon mass
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  Real threshold = pin->GetOrAddReal("hydro", "dthreshold", 1.0);
  eos.SetThreshold(threshold);
  eos.SetBaryonMass(mb);
  // Set the number density floor.
  eos.SetDensityFloor(density_floor_/mb);
  // Set the pressure floor -- we first need to retrieve the temperature from the pressure.
  // That means we need to initialize an empty array of particle fractions.
  eos.SetPressureFloor(pressure_floor_);

  // If we're working with an ideal gas, we need to fix the adiabatic constant.
#if EOS_POLICY == IdealGas
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 2.0);
  eos.SetGamma(gamma);
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
    AthenaArray<Real> &bb_cc, Coordinates *pco, int il, int iu, int jl, int ju, int kl,
    int ku) {
  // Parametes
  int nn1 = iu+1;
  
  // Vertex-centered containers for the metric.
  AthenaArray<Real> vcgamma_xx, vcgamma_xy, vcgamma_xz, vcgamma_yy,
                    vcgamma_yz, vcgamma_zz;

  // Metric at cell centers.
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
  gamma_dd.NewAthenaTensor(nn1);
  vcgamma_xx.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
  vcgamma_xy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
  vcgamma_xz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
  vcgamma_yy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
  vcgamma_yz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
  vcgamma_zz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gzz,1);

  // Go through the cells
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      // Extract the metric at the vertex centers and interpolate to cell centers.
      //#pragma omp simd
      for (int i = il; i <= iu; ++i) {
        gamma_dd(0,0,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
        gamma_dd(0,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
        gamma_dd(0,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
        gamma_dd(1,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
        gamma_dd(1,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
        gamma_dd(2,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));

        /*gamma_dd(0,0,i) = VCInterpolation(vcgamma_xx, k, j, i);
        gamma_dd(0,1,i) = VCInterpolation(vcgamma_xy, k, j, i);
        gamma_dd(0,2,i) = VCInterpolation(vcgamma_xz, k, j, i);
        gamma_dd(1,1,i) = VCInterpolation(vcgamma_yy, k, j, i);
        gamma_dd(1,2,i) = VCInterpolation(vcgamma_yz, k, j, i);
        gamma_dd(2,2,i) = VCInterpolation(vcgamma_zz, k, j, i);*/
      }

      // Extract the primitive variables
      //#pragma omp simd
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
        // FIXME: Need to generalize this for particle fractions.

        // Find the primitive variables.
        Real prim_pt[NPRIM] = {0.0};
        Real b3u[NMAG] = {0.0}; // Assume no magnetic field.
        Primitive::Error result = ps.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);

        if(result != Primitive::Error::SUCCESS) {
          std::cerr << "There was an error during the primitive solve!\n";
          std::cerr << "  Error: " << Primitive::ErrorString[(int)result] << "\n";
          //printf("i=%d, j=%d, k=%d\n",i,j,k);
          std::cerr << "  i=" << i << ", j=" << j << ", k=" << k << "\n";
          std::cerr << "  g3d = [" << g3d[S11] << ", " << g3d[S12] << ", " << g3d[S13] << ", "
                    << g3d[S22] << ", " << g3d[S23] << ", " << g3d[S33] << "\n";
          std::cerr << "  g3u = [" << g3u[S11] << ", " << g3u[S12] << ", " << g3u[S13] << ", "
                    << g3u[S22] << ", " << g3u[S23] << ", " << g3u[S33] << "\n";
          std::cerr << "  sdetg = " << sdetg << "\n";
          std::cerr << "  D = " << cons_old_pt[IDN] << "\n";
          std::cerr << "  S_1 = " << cons_old_pt[IM1] << "\n";
          std::cerr << "  S_2 = " << cons_old_pt[IM2] << "\n";
          std::cerr << "  S_3 = " << cons_old_pt[IM3] << "\n";
          std::cerr << "  tau = " << cons_old_pt[IEN] << "\n";
        }
        // Update the primitive variables.
        prim(IDN, k, j, i) = prim_pt[IDN];
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
      // Extract the metric at the cell centers.
      for (int i=il; i<=iu; ++i) {
        gamma_dd(0,0,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
        gamma_dd(0,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
        gamma_dd(0,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
        gamma_dd(1,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
        gamma_dd(1,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
        gamma_dd(2,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));

        /*gamma_dd(0,0,i) = VCInterpolation(vcgamma_xx, k, j, i);
        gamma_dd(0,1,i) = VCInterpolation(vcgamma_xy, k, j, i);
        gamma_dd(0,2,i) = VCInterpolation(vcgamma_xz, k, j, i);
        gamma_dd(1,1,i) = VCInterpolation(vcgamma_yy, k, j, i);
        gamma_dd(1,2,i) = VCInterpolation(vcgamma_yz, k, j, i);
        gamma_dd(2,2,i) = VCInterpolation(vcgamma_zz, k, j, i);*/
      }

      // Calculate the conserved variables at every point.
      for (int i=il; i<=iu; ++i) {
        PrimitiveToConservedSingle(prim, gamma_dd, k, j, i, cons, ps);
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

static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, 
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const &gamma_dd, int k, int j, int i,
    AthenaArray<Real> &cons,
    Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>& ps) {

  // Extract the primitive variables
  Real prim_pt[NPRIM] = {0.0};
  Real Y[MAX_SPECIES] = {0.0}; // FIXME: Need to add support for particle fractions.
  Real bu[NMAG] = {0.0};
  Real mb = ps.GetEOS()->GetBaryonMass();
  prim_pt[IDN] = prim(IDN, k, j, i)/mb;
  prim_pt[IVX] = prim(IVX, k, j, i);
  prim_pt[IVY] = prim(IVY, k, j, i);
  prim_pt[IVZ] = prim(IVZ, k, j, i);
  prim_pt[IPR] = prim(IPR, k, j, i);
  prim_pt[ITM] = ps.GetEOS()->GetTemperatureFromP(prim_pt[IDN], prim_pt[IPR], Y);

  // Apply the floor to ensure that we get physical conserved variables.
  bool result = ps.GetEOS()->ApplyPrimitiveFloor(prim_pt[IDN], &prim_pt[IVX], prim_pt[IPR], prim_pt[ITM], Y);

  // Extract the metric and calculate the determinant.
  Real g3d[NSPMETRIC] = {gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
                        gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i)};
  Real sdetg = std::sqrt(Primitive::GetDeterminant(g3d));

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
    std::cerr << "    sdetg = " << sdetg << "\n";
  }

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
    Real *plambda_plus, Real *plambda_minus) {
  // FIXME: Need to update to work with particle fractions.
  Real Y[MAX_SPECIES] = {0};
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

void EquationOfState::SoundSpeedsGR(Real n, Real T, Real vi, Real v2, Real alpha,
    Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus) {
  // Calculate comoving sound speed
  // FIXME: Need to update to work with particle fractions.
  Real Y[MAX_SPECIES] = {0};
  Real cs = ps.GetEOS()->GetSoundSpeed(n, T, Y);
  Real cs_sq = cs*cs;

  Real root_1 = alpha*(vi*(1.0-cs_sq) + cs*std::sqrt( (1-v2)*(gammaii*(1.0-v2*cs_sq) - vi*vi*(1.0-cs_sq))))/(1.0-v2*cs_sq) - betai;
  Real root_2 = alpha*(vi*(1.0-cs_sq) - cs*std::sqrt( (1-v2)*(gammaii*(1.0-v2*cs_sq) - vi*vi*(1.0-cs_sq))))/(1.0-v2*cs_sq) - betai;

  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  }
  else {
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
  //Real n = prim(IDN, k, j, i)/mb;
  Real n = prim(IDN, i)/mb;
  //Real Wvu[3] = {prim(IVX, k, j, i), prim(IVY, k, j, i), prim(IVZ, k, j, i)};
  Real Wvu[3] = {prim(IVX, i), prim(IVY, i), prim(IVZ, i)};
  //Real P = prim(IPR, k, j, i);
  Real P = prim(IPR, i);
  // FIXME: Update to work with particle species.
  Real Y[MAX_SPECIES] = {0.0};
  Real T = ps.GetEOS()->GetTemperatureFromP(n, P, Y);
  ps.GetEOS()->ApplyPrimitiveFloor(n, Wvu, P, T, Y);

  // Now push the updated quantities back to Athena.
  //prim(IDN, k, j, i) = n*mb;
  //prim(IVX, k, j, i) = Wvu[0];
  //prim(IVY, k, j, i) = Wvu[1];
  //prim(IVZ, k, j, i) = Wvu[2];
  //prim(IPR, k, j, i) = P;
  prim(IDN, i) = n*mb;
  prim(IVX, i) = Wvu[0];
  prim(IVY, i) = Wvu[1];
  prim(IVZ, i) = Wvu[2];
  prim(IPR, i) = P;

  return;
}

//---------------------------------------------------------------------------------------
// \!fn static Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform linear interpolation to the desired cell-centered grid index.

static Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.125*( (in(k, j, i) + in(k + 1, j + 1, i + 1)) // lower-left-front to upper-right-back
               + (in(k, j+1, i) + in(k + 1, j, i+1)) // lower-left-back to upper-right-front
               + (in(k+1, j, i) + in(k, j+1, i+1)) // upper-left-front to lower-right-back
               + (in(k+1, j+1, i) + in(k, j, i+1))); // upper-left-back to lower-right-front
}
