//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file llf_rel_no_transform.cpp
//  \brief Implements local Lax-Friedrichs Riemann solver for relativistic hydrodynamics
//  in pure GR.

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../athena.hpp"                   // enums, macros
#include "../../../athena_arrays.hpp"            // AthenaArray
#include "../../../coordinates/coordinates.hpp"  // Coordinates
#include "../../../eos/eos.hpp"                  // EquationOfState
#include "../../../mesh/mesh.hpp"                // MeshBlock

//----------------------------------------------------------------------------------------
// Riemann solver
// Inputs:
//   kl,ku,jl,ju,il,iu: lower and upper x1-, x2-, and x3-indices
//   ivx: type of interface (IVX for x1, IVY for x2, IVZ for x3)
//   bb: 3D array of normal magnetic fields (not used)
//   prim_l,prim_r: 3D arrays of left and right primitive states
// Outputs:
//   flux: 3D array of hydrodynamical fluxes across interfaces
//   ey,ez: 3D arrays of magnetic fluxes (electric fields) across interfaces (not used)
// Notes:
//   implements LLF algorithm similar to that of fluxcalc() in step_ch.c in Harm
//   cf. LLFNonTransforming() in llf_rel.cpp
// Here we use the D, S, tau variable choice for conservatives, and assume a dynamically evolving spacetime
// so a factor of sqrt(detgamma) is included
// compare with modification to add_flux_divergence_dyn, where factors of face area, cell volume etc are missing
// since they are included here.
namespace{
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33);
Real Determinant(Real a11, Real a12, Real a21, Real a22);
}

void Hydro::RiemannSolver(const int k, const int j,
    const int il, const int iu, const int ivx,
    AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r, AthenaArray<Real> &flux, const AthenaArray<Real> &dxw) {
  // Calculate cyclic permutations of indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  // Extract ratio of specific heats
  const Real gamma_adi = pmy_block->peos->GetGamma();

  // Go through 1D arrays of interfaces
//  for (int k = kl; k <= ku; ++k) {
//    for (int j = jl; j <= ju; ++j) {

//    TODO replace FaceNMetric with local calculation of metric at FaceCenter.
//    Returning alpha(i), beta_u(a,i) gamma_dd(a,b,i)
      // Get metric components
      switch (ivx) {
        case IVX:
          pmy_block->pcoord->Face1Metric(k, j, il, iu, g_, gi_);
          break;
        case IVY:
          pmy_block->pcoord->Face2Metric(k, j, il, iu, g_, gi_);
          break;
        case IVZ:
          pmy_block->pcoord->Face3Metric(k, j, il, iu, g_, gi_);
          break;
      }

      // Go through each interface
      #pragma omp simd
      for (int i = il; i <= iu; ++i) {

        // Extract metric
        const Real
            &g_00 = g_(I00,i), &g_01 = g_(I01,i), &g_02 = g_(I02,i), &g_03 = g_(I03,i),
            &g_10 = g_(I01,i), &g_11 = g_(I11,i), &g_12 = g_(I12,i), &g_13 = g_(I13,i),
            &g_20 = g_(I02,i), &g_21 = g_(I12,i), &g_22 = g_(I22,i), &g_23 = g_(I23,i),
            &g_30 = g_(I03,i), &g_31 = g_(I13,i), &g_32 = g_(I23,i), &g_33 = g_(I33,i);
        const Real
            &g00 = gi_(I00,i), &g01 = gi_(I01,i), &g02 = gi_(I02,i), &g03 = gi_(I03,i),
            &g10 = gi_(I01,i), &g11 = gi_(I11,i), &g12 = gi_(I12,i), &g13 = gi_(I13,i),
            &g20 = gi_(I02,i), &g21 = gi_(I12,i), &g22 = gi_(I22,i), &g23 = gi_(I23,i),
            &g30 = gi_(I03,i), &g31 = gi_(I13,i), &g32 = gi_(I23,i), &g33 = gi_(I33,i);
        Real alpha = std::sqrt(-1.0/g00);
        Real gii, g0i;
        switch (ivx) {
          case IVX:
            gii = g11;
            g0i = g01;
            break;
          case IVY:
            gii = g22;
            g0i = g02;
            break;
          case IVZ:
            gii = g33;
            g0i = g03;
            break;
        }
//Define determinanit
        Real detgamma = Determinant(g_11,g_12,g_13,g_21,g_22,g_23,g_31,g_32,g_33);
        Real detg = detgamma*SQR(alpha);
        // Extract left primitives
        // NB, primitive velocities are gamma^i_mu u^mu = tilde{u}^i = v^i * gamma_lorentz BEWARE LORENTZ FACTORS
        const Real &rho_l = prim_l(IDN,i);
        const Real &pgas_l = prim_l(IPR,i);
        const Real &uu1_l = prim_l(IVX,i);
        const Real &uu2_l = prim_l(IVY,i);
        const Real &uu3_l = prim_l(IVZ,i);

        // Extract right primitives
        const Real &rho_r = prim_r(IDN,i);
        const Real &pgas_r = prim_r(IPR,i);
        const Real &uu1_r = prim_r(IVX,i);
        const Real &uu2_r = prim_r(IVY,i);
        const Real &uu3_r = prim_r(IVZ,i);

        // Calculate 4-velocity in left state
        Real ucon_l[4], ucov_l[4];
        Real tmp = g_11*SQR(uu1_l) + 2.0*g_12*uu1_l*uu2_l + 2.0*g_13*uu1_l*uu3_l
                 + g_22*SQR(uu2_l) + 2.0*g_23*uu2_l*uu3_l
                 + g_33*SQR(uu3_l);
        Real gamma_l = std::sqrt(1.0 + tmp);
        ucon_l[0] = gamma_l / alpha;
        ucon_l[1] = uu1_l - alpha * gamma_l * g01;
        ucon_l[2] = uu2_l - alpha * gamma_l * g02;
        ucon_l[3] = uu3_l - alpha * gamma_l * g03;
        ucov_l[0] = g_00*ucon_l[0] + g_01*ucon_l[1] + g_02*ucon_l[2] + g_03*ucon_l[3];
        ucov_l[1] = g_10*ucon_l[0] + g_11*ucon_l[1] + g_12*ucon_l[2] + g_13*ucon_l[3];
        ucov_l[2] = g_20*ucon_l[0] + g_21*ucon_l[1] + g_22*ucon_l[2] + g_23*ucon_l[3];
        ucov_l[3] = g_30*ucon_l[0] + g_31*ucon_l[1] + g_32*ucon_l[2] + g_33*ucon_l[3];
        
        Real utilde_d_l[3];
        Real utilde_u_l[3];
        Real v_d_l[3];
        Real v_u_l[3];

        utilde_d_l[0] = g_11 * uu1_l + g_12 * uu2_l + g_13 * uu3_l;
        utilde_d_l[1] = g_21 * uu1_l + g_22 * uu2_l + g_23 * uu3_l;
        utilde_d_l[2] = g_31 * uu1_l + g_32 * uu2_l + g_33 * uu3_l;

        utilde_u_l[0] = uu1_l;
        utilde_u_l[1] = uu2_l;
        utilde_u_l[2] = uu3_l;

        v_d_l[0] = utilde_d_l[0]/gamma_l ;
        v_d_l[1] = utilde_d_l[1]/gamma_l ;
        v_d_l[2] = utilde_d_l[2]/gamma_l ;

        v_u_l[0] = utilde_u_l[0]/gamma_l ;
        v_u_l[1] = utilde_u_l[1]/gamma_l ;
        v_u_l[2] = utilde_u_l[2]/gamma_l ;
        // Calculate 4-velocity in right state
        Real ucon_r[4], ucov_r[4];
        tmp = g_11*SQR(uu1_r) + 2.0*g_12*uu1_r*uu2_r + 2.0*g_13*uu1_r*uu3_r
            + g_22*SQR(uu2_r) + 2.0*g_23*uu2_r*uu3_r
            + g_33*SQR(uu3_r);
        Real gamma_r = std::sqrt(1.0 + tmp);
        ucon_r[0] = gamma_r / alpha;
        ucon_r[1] = uu1_r - alpha * gamma_r * g01;
        ucon_r[2] = uu2_r - alpha * gamma_r * g02;
        ucon_r[3] = uu3_r - alpha * gamma_r * g03;
        ucov_r[0] = g_00*ucon_r[0] + g_01*ucon_r[1] + g_02*ucon_r[2] + g_03*ucon_r[3];
        ucov_r[1] = g_10*ucon_r[0] + g_11*ucon_r[1] + g_12*ucon_r[2] + g_13*ucon_r[3];
        ucov_r[2] = g_20*ucon_r[0] + g_21*ucon_r[1] + g_22*ucon_r[2] + g_23*ucon_r[3];
        ucov_r[3] = g_30*ucon_r[0] + g_31*ucon_r[1] + g_32*ucon_r[2] + g_33*ucon_r[3];
      
        Real utilde_d_r[3];
        Real utilde_u_r[3];
        Real v_d_r[3];
        Real v_u_r[3];

        utilde_d_r[0] = g_11 * uu1_r + g_12 * uu2_r + g_13 * uu3_r;
        utilde_d_r[1] = g_21 * uu1_r + g_22 * uu2_r + g_23 * uu3_r;
        utilde_d_r[2] = g_31 * uu1_r + g_32 * uu2_r + g_33 * uu3_r;
       
        utilde_u_r[0] = uu1_r;
        utilde_u_r[1] = uu2_r;
        utilde_u_r[2] = uu3_r;
        
        v_d_r[0] = utilde_d_r[0]/gamma_r ;
        v_d_r[1] = utilde_d_r[1]/gamma_r ;
        v_d_r[2] = utilde_d_r[2]/gamma_r ;

        v_u_r[0] = utilde_u_r[0]/gamma_r ;
        v_u_r[1] = utilde_u_r[1]/gamma_r ;
        v_u_r[2] = utilde_u_r[2]/gamma_r ;
        // Calculate wavespeeds in left state
        Real lambda_p_l, lambda_m_l;
        Real wgas_l = rho_l + gamma_adi/(gamma_adi-1.0) * pgas_l;
        pmy_block->peos->SoundSpeedsGR(wgas_l, pgas_l, ucon_l[0], ucon_l[ivx], g00, g0i,
            gii, &lambda_p_l, &lambda_m_l);

        // Calculate wavespeeds in right state
        Real lambda_p_r, lambda_m_r;
        Real wgas_r = rho_r + gamma_adi/(gamma_adi-1.0) * pgas_r;
        pmy_block->peos->SoundSpeedsGR(wgas_r, pgas_r, ucon_r[0], ucon_r[ivx], g00, g0i,
            gii, &lambda_p_r, &lambda_m_r);

        // Calculate extremal wavespeed
        Real lambda_l = std::min(lambda_m_l, lambda_m_r);
        Real lambda_r = std::max(lambda_p_l, lambda_p_r);
        Real lambda = std::max(lambda_r, -lambda_l);

        // Calculate conserved quantities in L region including factor of sqrt(detgamma)
        Real cons_l[NWAVE];
        // D = rho * gamma_lorentz
        cons_l[IDN] = rho_l*gamma_l*std::sqrt(detgamma);
        // tau = (rho * h = ) wgas * gamma_lorentz**2 - rho * gamma_lorentz - p
        cons_l[IEN] = (wgas_l * SQR(gamma_l) - rho_l*gamma_l - pgas_l)*std::sqrt(detgamma);
        // S_i = wgas * gamma_lorentz**2 * v_i = wgas * gamma_lorentz * u_i
//        cons_l[IVX] = wgas_l * gamma_l * ucov_l[1]*std::sqrt(detgamma);
//        cons_l[IVY] = wgas_l * gamma_l * ucov_l[2]*std::sqrt(detgamma);
//        cons_l[IVZ] = wgas_l * gamma_l * ucov_l[3]*std::sqrt(detgamma);
//NB TODO double check velocity has chenged here (also in right state)
        cons_l[IVX] = wgas_l * gamma_l * utilde_d_l[0]*std::sqrt(detgamma);
        cons_l[IVY] = wgas_l * gamma_l * utilde_d_l[1]*std::sqrt(detgamma);
        cons_l[IVZ] = wgas_l * gamma_l * utilde_d_l[2]*std::sqrt(detgamma);
        // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
        Real flux_l[NWAVE];
        // D flux: D(v^i - beta^i/alpha)
         flux_l[IDN] = cons_l[IDN]*alpha*(v_u_l[ivx-1] - g0i*alpha);

        // tau flux: alpha(S^i - Dv^i) - beta^i tau
          flux_l[IEN] = cons_l[IEN] * alpha * (v_u_l[ivx-1] - g0i*alpha) + std::sqrt(detg)*pgas_l*v_u_l[ivx-1];
 
        //S_i flux alpha S^j_i - beta^j S_i
        flux_l[IVX] = cons_l[IVX] * alpha * (v_u_l[ivx-1] - g0i*alpha);      
        flux_l[IVY] = cons_l[IVY] * alpha * (v_u_l[ivx-1] - g0i*alpha);      
        flux_l[IVZ] = cons_l[IVZ] * alpha * (v_u_l[ivx-1] - g0i*alpha);      
        flux_l[ivx] += pgas_l*std::sqrt(detg);

        // Calculate conserved quantities in R region (rho u^0 and T^0_\mu)
        Real cons_r[NWAVE];

        // D = rho * gamma_lorentz
        cons_r[IDN] = rho_r*gamma_r*std::sqrt(detgamma);
        // tau = (rho * h = ) wgas * gamma_rorentz**2 - rho * gamma_rorentz - p
        cons_r[IEN] = (wgas_r * SQR(gamma_r) - rho_r*gamma_r - pgas_r)*std::sqrt(detgamma);
        // S_i = wgas * gamma_rorentz**2 * v_i = wgas * gamma_rorentz * u_i NB this is T^0_i * alpha!
        cons_r[IVX] = wgas_r * gamma_r * utilde_d_r[0]*std::sqrt(detgamma);
        cons_r[IVY] = wgas_r * gamma_r * utilde_d_r[1]*std::sqrt(detgamma);
        cons_r[IVZ] = wgas_r * gamma_r * utilde_d_r[2]*std::sqrt(detgamma);
//        cons_r[IVX] = wgas_r * gamma_r * ucov_r[1]*std::sqrt(detgamma);
//        cons_r[IVY] = wgas_r * gamma_r * ucov_r[2]*std::sqrt(detgamma);
//        cons_r[IVZ] = wgas_r * gamma_r * ucov_r[3]*std::sqrt(detgamma);
        Real flux_r[NWAVE];
        // D flux: D(v^i - beta^i/alpha)
         flux_r[IDN] = cons_r[IDN]*alpha*(v_u_r[ivx-1] - g0i*alpha);

        // tau flux: alpha(S^i - Dv^i) - beta^i tau
          flux_r[IEN] = cons_r[IEN] * alpha * (v_u_r[ivx-1] - g0i*alpha) + std::sqrt(detg)*pgas_r*v_u_r[ivx-1];
 
        //S_i flux alpha S^j_i - beta^j S_i
        flux_r[IVX] = cons_r[IVX] * alpha * (v_u_r[ivx-1] - g0i*alpha);      
        flux_r[IVY] = cons_r[IVY] * alpha * (v_u_r[ivx-1] - g0i*alpha);      
        flux_r[IVZ] = cons_r[IVZ] * alpha * (v_u_r[ivx-1] - g0i*alpha);      
        flux_r[ivx] += std::sqrt(detg)*pgas_r;
      // Set fluxes
        for (int n = 0; n < NHYDRO; ++n) {
          flux(n,k,j,i) =
              0.5 * (flux_l[n] + flux_r[n] - lambda * (cons_r[n] - cons_l[n]));
        }
      }
    
  
  return;
}
namespace{
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33) {
  Real det = a11 * Determinant(a22, a23, a32, a33)
             - a12 * Determinant(a21, a23, a31, a33)
             + a13 * Determinant(a21, a22, a31, a32);
  return det;
}

Real Determinant(Real a11, Real a12, Real a21, Real a22) {
  return a11 * a22 - a12 * a21;
}
}
