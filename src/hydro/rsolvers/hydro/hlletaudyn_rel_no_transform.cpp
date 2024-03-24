//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file llf_rel_no_transform.cpp
//  \brief Implements HLLE Riemann solver for relativistic hydrodynamics in pure GR.

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../z4c/z4c.hpp"
#include "../../../utils/interp_intergrid.hpp"
#include "../../../athena.hpp"                   // enums, macros
#include "../../../athena_arrays.hpp"            // AthenaArray
#include "../../../coordinates/coordinates.hpp"  // Coordinates
#include "../../../eos/eos.hpp"                  // EquationOfState
#include "../../../mesh/mesh.hpp"                // MeshBlock

//----------------------------------------------------------------------------------------
// Riemann solver
// Inputs
//   kl,ku,jl,ju,il,iu: lower and upper x1-, x2-, and x3-indices
//   ivx: type of interface (IVX for x1, IVY for x2, IVZ for x3)
//   bb: 3D array of normal magnetic fields (not used)
//   prim_l,prim_r: 3D arrays of left and right primitive states
// Outputs:
//   flux: 3D array of hydrodynamical fluxes across interfaces
//   ey,ez: 3D arrays of magnetic fluxes (electric fields) across interfaces (not used)
// Notes:
//   implements relativistic HLLE algorithm.
//   We assume D, S, and tau are the conservative variables in a dynamic spacetime.
namespace {

Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33);
Real Determinant(Real a11, Real a12, Real a21, Real a22);
Real Det3Metric(Real g_dd[NSPMETRIC]);
void Inverse3Metric(Real const detginv, Real const g_dd[NSPMETRIC],
                    Real g_uu[NSPMETRIC]);

} // namespace

void Hydro::RiemannSolver(const int k, const int j, const int il, const int iu,
                          const int ivx,
                          AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r,
                          AthenaArray<Real> &flux, const AthenaArray<Real> &dxw) {
  // Calculate cyclic permutations of indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  const int nn1 = pmy_block->nverts1;  // utilize the verts

  // Ad-hoc value for adding diffusion near the weakly hyperbolic limit.
  constexpr Real speed_eps = 1e-3;

  #if USETM
  // Extract baryon mass for PrimitiveSolver framework
  const Real mb = pmy_block->peos->GetEOS().GetBaryonMass();
  #else
  // Extract ratio of specific heats
  const Real gamma_adi = pmy_block->peos->GetGamma();
  #endif

  // Metric components
  Real g_dd[NSPMETRIC],
       g_uu[NSPMETRIC],
       beta_u[NDIM],
       alpha;
  int mdir;
  switch(ivx) {
    case IVX:
      mdir = S11;
    case IVY:
      mdir = S22;
    case IVZ:
      mdir = S33;
  }

  // perform variable resampling when required
  Z4c * pz4c = pmy_block->pz4c;

  const int D = NDIM + 1;
  const int N = NDIM;

  typedef AthenaArray< Real>                         AA;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 2> AT_N_mat;

  // Slice z4c metric quantities  (NDIM=3 in z4c.hpp)
  AT_N_sym sl_g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca sl_alpha( pz4c->storage.u,   Z4c::I_Z4c_alpha);
  AT_N_vec sl_beta_u(pz4c->storage.u,   Z4c::I_Z4c_betax);

  // Scratch
  AT_N_sca rc_alpha_(   nn1);
  AT_N_vec rc_beta_u_(  nn1);
  AT_N_sym rc_g_dd_(nn1);

  // Reconstruction to FC
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmy_block->pcoord);
  pco_gr->GetGeometricFieldFC(rc_g_dd_,   sl_g_dd,   ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(rc_alpha_,  sl_alpha,  ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(rc_beta_u_, sl_beta_u, ivx-1, k, j);


  // Go through each interface
  #pragma omp simd
  for (int i = il; i <= iu; ++i) {
    // Extract left primitives
    const Real &rho_l  = prim_l(IDN, i);
    const Real &pgas_l = prim_l(IPR, i);
    const Real u_l[NDIM] = {prim_l(IVX, i),
                            prim_l(IVY, i),
                            prim_l(IVZ, i)};

    // Extract right primitives
    const Real &rho_r  = prim_r(IDN, i);
    const Real &pgas_r = prim_r(IPR, i);
    const Real u_r[NDIM] = {prim_r(IVX, i),
                            prim_r(IVY, i),
                            prim_r(IVZ, i)};

    // Extract metric components
    g_dd[S11] = rc_g_dd_(0,0,i);
    g_dd[S12] = rc_g_dd_(0,1,i);
    g_dd[S13] = rc_g_dd_(0,2,i);
    g_dd[S22] = rc_g_dd_(1,1,i);
    g_dd[S23] = rc_g_dd_(1,2,i);
    g_dd[S33] = rc_g_dd_(2,2,i);

    beta_u[0] = rc_beta_u_(0,i);
    beta_u[1] = rc_beta_u_(1,i);
    beta_u[2] = rc_beta_u_(2,i);

    alpha = rc_alpha_(i);

    // Calculate the determinant
    Real detgamma = Det3Metric(g_dd);
    Real sdetgamma = std::sqrt(detgamma);
    Inverse3Metric(1.0/detgamma, g_dd, g_uu);

    // Calculate the Lorentz factor for the left and right states.
    Real usq_l = u_l[0]*u_l[0]*g_dd[S11] + 2.0*u_l[0]*u_l[1]*g_dd[S12] + 2.0*u_l[0]*u_l[2]*g_dd[S13]
               + u_l[1]*u_l[1]*g_dd[S22] + 2.0*u_l[1]*u_l[2]*g_dd[S23]
               + u_l[2]*u_l[2]*g_dd[S33];
    Real usq_r = u_r[0]*u_r[0]*g_dd[S11] + 2.0*u_r[0]*u_r[1]*g_dd[S12] + 2.0*u_r[0]*u_r[2]*g_dd[S13]
               + u_r[1]*u_r[1]*g_dd[S22] + 2.0*u_r[1]*u_r[2]*g_dd[S23]
               + u_r[2]*u_r[2]*g_dd[S33];
    Real Wsq_l = 1.0 + usq_l;
    Real Wsq_r = 1.0 + usq_r;
    Real W_l = std::sqrt(Wsq_l);
    Real W_r = std::sqrt(Wsq_r);

    // Extract the three-velocity and its squares.
    Real v_l[NDIM] = {u_l[0]/W_l, u_l[1]/W_l, u_l[2]/W_l};
    Real v_r[NDIM] = {u_r[0]/W_r, u_r[1]/W_r, u_r[2]/W_r};
    Real vsq_l = usq_l/Wsq_l;
    Real vsq_r = usq_r/Wsq_r;

    // Calculate the lowered velocity components.
    Real vd_l[NDIM], vd_r[NDIM];
    vd_l[0] = v_l[0]*g_dd[S11] + v_l[1]*g_dd[S12] + v_l[2]*g_dd[S13];
    vd_l[1] = v_l[0]*g_dd[S12] + v_l[1]*g_dd[S22] + v_l[2]*g_dd[S23];
    vd_l[2] = v_l[0]*g_dd[S13] + v_l[1]*g_dd[S23] + v_l[2]*g_dd[S33];
    vd_r[0] = v_r[0]*g_dd[S11] + v_r[1]*g_dd[S12] + v_r[2]*g_dd[S13];
    vd_r[1] = v_r[0]*g_dd[S12] + v_r[1]*g_dd[S22] + v_r[2]*g_dd[S23];
    vd_r[2] = v_r[0]*g_dd[S13] + v_r[1]*g_dd[S23] + v_r[2]*g_dd[S33];

    // Calculate wave speeds
    Real lambda_p_l, lambda_p_r,
         lambda_m_l, lambda_m_r;
    #if USETM
    // If using the PrimitiveSolver framework, get the number density
    // and temperature to help calculate enthalpy.
    Real nl = rho_l/mb;
    Real nr = rho_r/mb;
    // FIXME: Generalize to work with EOSes accepting particle fractions.
    Real Y[MAX_SPECIES] = {0.0}; // TODO PH: Fix
    Real Tl = pmy_block->peos->GetEOS().GetTemperatureFromP(nl, pgas_l, Y);
    Real Tr = pmy_block->peos->GetEOS().GetTemperatureFromP(nr, pgas_r, Y);
    Real wgas_l = rho_l*pmy_block->peos->GetEOS().GetEnthalpy(nl, Tl, Y);
    Real wgas_r = rho_r*pmy_block->peos->GetEOS().GetEnthalpy(nr, Tr, Y);

    // Calculate the sound speeds
    pmy_block->peos->SoundSpeedsGR(nl, Tl, v_l[ivx-1], vsq_l,
                                   alpha, beta_u[ivx-1], g_uu[mdir],
                                   &lambda_p_l, &lambda_m_l);
    pmy_block->peos->SoundSpeedsGR(nr, Tr, v_r[ivx-1], vsq_r,
                                   alpha, beta_u[ivx-1], g_uu[mdir],
                                   &lambda_p_r, &lambda_m_r);
    #else
    // Calculate the enthalpy
    Real wgas_l = rho_l + gamma_adi/(gamma_adi - 1.0)*pgas_l;
    Real wgas_r = rho_r + gamma_adi/(gamma_adi - 1.0)*pgas_r;

    // Calculate the sound speeds
    pmy_block->peos->SoundSpeedsGR(wgas_l, pgas_l, v_l[ivx-1], vsq_l,
                                   alpha, beta_u[ivx-1], g_uu[mdir],
                                   &lambda_p_l, &lambda_m_l);
    pmy_block->peos->SoundSpeedsGR(wgas_r, pgas_r, v_r[ivx-1], vsq_r,
                                   alpha, beta_u[ivx-1], g_uu[mdir],
                                   &lambda_p_r, &lambda_m_r);
    #endif

    // Calculate extremal wave speeds
    Real lambda_ms = std::min(lambda_m_l, lambda_m_r);
    Real lambda_ps = std::max(lambda_p_l, lambda_p_r);
    Real lambda_p = std::max(0.0, lambda_ps);
    Real lambda_m = std::min(0.0, lambda_ms);

    // Add diffusion if the wave speeds are too close to zero.
    Real cbound = std::sqrt(alpha/g_dd[mdir]);
    if (lambda_p < speed_eps*cbound && lambda_m > -speed_eps*cbound) {
      lambda_p = cbound;
      lambda_m = -cbound;
    }

    // Calculate left conserved quantities
    Real cons_l[NHYDRO];
    cons_l[IDN] = rho_l*W_l*sdetgamma;
    cons_l[IEN] = (wgas_l*Wsq_l - pgas_l)*sdetgamma - cons_l[IDN];
    cons_l[IM1] = wgas_l*Wsq_l*vd_l[0]*sdetgamma;
    cons_l[IM2] = wgas_l*Wsq_l*vd_l[1]*sdetgamma;
    cons_l[IM3] = wgas_l*Wsq_l*vd_l[2]*sdetgamma;

    // Calculate left flux quantities
    Real flux_l[NHYDRO];
    Real ucov_l = alpha*(v_l[ivx-1] - beta_u[ivx-1]/alpha);
    flux_l[IDN] = cons_l[IDN]*ucov_l;
    flux_l[IEN] = cons_l[IEN]*ucov_l + alpha*sdetgamma*pgas_l*v_l[ivx-1];
    flux_l[IM1] = cons_l[IM1]*ucov_l;
    flux_l[IM2] = cons_l[IM2]*ucov_l;
    flux_l[IM3] = cons_l[IM3]*ucov_l;
    flux_l[ivx] += sdetgamma*alpha*pgas_l;

    // Calculate right conserved quantities
    Real cons_r[NHYDRO];
    cons_r[IDN] = rho_r*W_r*sdetgamma;
    cons_r[IEN] = (wgas_r*Wsq_r - pgas_r)*sdetgamma - cons_r[IDN];
    cons_r[IM1] = wgas_r*Wsq_r*vd_r[0]*sdetgamma;
    cons_r[IM2] = wgas_r*Wsq_r*vd_r[1]*sdetgamma;
    cons_r[IM3] = wgas_r*Wsq_r*vd_r[2]*sdetgamma;

    // Calculate right flux quantities
    Real flux_r[NHYDRO];
    Real ucov_r = alpha*(v_r[ivx-1] - beta_u[ivx-1]/alpha);
    flux_r[IDN] = cons_r[IDN]*ucov_r;
    flux_r[IEN] = cons_r[IEN]*ucov_r + alpha*sdetgamma*pgas_r*v_r[ivx-1];
    flux_r[IM1] = cons_r[IM1]*ucov_r;
    flux_r[IM2] = cons_r[IM2]*ucov_r;
    flux_r[IM3] = cons_r[IM3]*ucov_r;
    flux_r[ivx] += sdetgamma*alpha*pgas_r;

    // Calculate net flux term
    for (int n = 0; n < NHYDRO; n++) {
      flux(n, k, j, i) = ((lambda_p*flux_l[n] - lambda_m*flux_r[n])
                         + lambda_p*lambda_m*(cons_r[n] - cons_l[n]))/(lambda_p - lambda_m);
    }
  }
}

namespace {

Real Det3Metric(Real g_dd[NSPMETRIC]) {
  return SQR(g_dd[S13])*g_dd[S22] +
         2*g_dd[S12]*g_dd[S13]*g_dd[S23] -
         g_dd[S11]*SQR(g_dd[S23]) - SQR(g_dd[S12])*g_dd[S33] +
         g_dd[S11]*g_dd[S22]*g_dd[S33];
}

void Inverse3Metric(Real const detginv, Real const g_dd[NSPMETRIC],
                    Real g_uu[NSPMETRIC]) {
  g_uu[S11] = (-SQR(g_dd[S23]) + g_dd[S22]*g_dd[S33])*detginv;
  g_uu[S12] = (g_dd[S13]*g_dd[S23] - g_dd[S12]*g_dd[S33])*detginv;
  g_uu[S13] = (-g_dd[S13]*g_dd[S22] + g_dd[S12]*g_dd[S23])*detginv;
  g_uu[S22] = (-SQR(g_dd[S13]) + g_dd[S11]*g_dd[S33])*detginv;
  g_uu[S23] = (g_dd[S12]*g_dd[S13] - g_dd[S11]*g_dd[S23])*detginv;
  g_uu[S33] = (-SQR(g_dd[S12]) + g_dd[S11]*g_dd[S22])*detginv;
}

}
