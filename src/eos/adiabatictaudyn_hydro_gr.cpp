//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatictaudyn_hydro_gr.cpp
//  \brief Implements functions for going between primitive and conserved variables in
//  general-relativistic hydrodynamics, as well as for computing wavespeeds.

// Here we use the conserved variables D, S^i, tau, assuming dynamical spacetime, so factors
// of sqrt(detgamma) are included.

// C++ headers
#include <algorithm>  // max()
#include <cfloat>     // FLT_MIN
#include <cmath>      // NAN, sqrt(), abs(), isfinite(), isnan(), pow()

// Athena++ headers
#include "eos.hpp"
#include "../athena.hpp"                   // enums, macros
#include "../athena_arrays.hpp"            // AthenaArray
#include "../parameter_input.hpp"          // ParameterInput
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../field/field.hpp"              // FaceField
#include "../mesh/mesh.hpp"                // MeshBlock

// Declarations
static void PrimitiveToConservedSingle(const AthenaArray<Real> &prim, Real gamma_adi,
    const AthenaArray<Real> &g, const AthenaArray<Real> &gi, int k, int j, int i,
    AthenaArray<Real> &cons, Coordinates *pco);
static Real QNResidual(Real w_guess, Real d, Real q_n, Real qq_sq, Real gamma_adi);
static Real QNResidualPrime(Real w_guess, Real d, Real qq_sq, Real gamma_adi);
static Real PresResidual(Real pres_guess, Real D, Real tau, Real S_sq, Real gamma_adi);
static Real PresResidualPrime(Real pres_guess, Real D, Real tau, Real S_sq, Real gamma_adi);
Real fthr, fatm, rhoc;
Real k_adi, gamma_adi;
namespace{
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33);
Real Determinant(Real a11, Real a12, Real a21, Real a22);
}
//----------------------------------------------------------------------------------------
// Constructor
// Inputs:
//   pmb: pointer to MeshBlock
//   pin: pointer to runtime inputs

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) {
  pmy_block_ = pmb;
  gamma_ = pin->GetReal("hydro", "gamma");
  density_floor_ = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)) );
  pressure_floor_ = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*(FLT_MIN)) );
  rho_min_ = pin->GetOrAddReal("hydro", "rho_min", density_floor_);
  rho_pow_ = pin->GetOrAddReal("hydro", "rho_pow", 0.0);
  pgas_min_ = pin->GetOrAddReal("hydro", "pgas_min", pressure_floor_);
  pgas_pow_ = pin->GetOrAddReal("hydro", "pgas_pow", 0.0);
  gamma_max_ = pin->GetOrAddReal("hydro", "gamma_max", 1000.0);
  int ncells1 = pmb->block_size.nx1 + 2*NGHOST;
  g_.NewAthenaArray(NMETRIC, ncells1);
  g_inv_.NewAthenaArray(NMETRIC, ncells1);
  int ncells2 = (pmb->block_size.nx2 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  int ncells3 = (pmb->block_size.nx3 > 1) ? pmb->block_size.nx3 + 2*NGHOST : 1;
  fixed_.NewAthenaArray(ncells3, ncells2, ncells1);
  fthr = pin -> GetReal("problem","fthr");
  fatm = pin -> GetReal("problem","fatm");
  rhoc = pin -> GetReal("problem","rhoc");
  k_adi = pin -> GetReal("hydro","k_adi");
  gamma_adi = pin -> GetReal("hydro","gamma");
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
// Notes - old scheme:
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
  // Parameters
  const Real max_wgas_rel = 1.0e8;
  const Real initial_guess_multiplier = 10.0;
  const int initial_guess_multiplications = 10;

  // Extract ratio of specific heats
  const Real &gamma_adi = gamma_;

  // Go through cells
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      pco->CellMetric(k, j, il, iu, g_, g_inv_);
      #pragma omp simd
      for (int i=il; i<=iu; ++i) {

        // Extract metric
        const Real
            // unused:
            &g_00 = g_(I00,i), &g_01 = g_(I01,i), &g_02 = g_(I02,i), &g_03 = g_(I03,i),
             &g_10 = g_(I01,i),
             &g_20 = g_(I02,i), &g_21 = g_(I12,i),
             &g_30 = g_(I03,i), &g_31 = g_(I13,i), &g_32 = g_(I23,i),
            &g_11 = g_(I11,i), &g_12 = g_(I12,i), &g_13 = g_(I13,i),
            &g_22 = g_(I22,i), &g_23 = g_(I23,i),
            &g_33 = g_(I33,i);
        const Real &g00 = g_inv_(I00,i), &g01 = g_inv_(I01,i), &g02 = g_inv_(I02,i),
                   &g03 = g_inv_(I03,i), &g10 = g_inv_(I01,i), &g11 = g_inv_(I11,i),
                   &g12 = g_inv_(I12,i), &g13 = g_inv_(I13,i), &g20 = g_inv_(I02,i),
                   &g21 = g_inv_(I12,i), &g22 = g_inv_(I22,i), &g23 = g_inv_(I23,i),
                   &g30 = g_inv_(I03,i), &g31 = g_inv_(I13,i), &g32 = g_inv_(I23,i),
                   &g33 = g_inv_(I33,i);
        Real alpha = 1.0/std::sqrt(-g00);
        Real detgamma = std::sqrt(Determinant(g_11,g_12,g_13,g_12,g_22,g_23,g_13,g_23,g_33));
	Real detg = detgamma*alpha;	
	// Calculate inverse spatial metric
        Real beta1 = g01*SQR(alpha);
        Real beta2 = g02*SQR(alpha);
        Real beta3 = g03*SQR(alpha);

        Real gam11 = g11 + beta1*beta1/SQR(alpha);
        Real gam12 = g12 + beta1*beta2/SQR(alpha);
        Real gam13 = g13 + beta1*beta3/SQR(alpha);
        Real gam22 = g22 + beta2*beta2/SQR(alpha);
        Real gam23 = g23 + beta2*beta3/SQR(alpha);
        Real gam33 = g33 + beta3*beta3/SQR(alpha);
// dg denotes a factor of sqrt(detgamma) is present

        // Extract conserved quantities
        const Real &Ddg = cons(IDN,k,j,i);
        const Real &taudg = cons(IEN,k,j,i);
        const Real &S_1dg = cons(IVX,k,j,i);
        const Real &S_2dg = cons(IVY,k,j,i);
        const Real &S_3dg = cons(IVZ,k,j,i);

        Real D = Ddg/detgamma;
        Real tau = taudg/detgamma;
        Real S_1 = S_1dg/detgamma;
        Real S_2 = S_2dg/detgamma;
        Real S_3 = S_3dg/detgamma;
        Real S_sq = gam11*S_1*S_1 + 2.0*gam12*S_1*S_2 + 2.0*gam13*S_1*S_3 + 
                    gam22*S_2*S_2 + 2.0*gam23*S_2*S_3 +
                    gam33*S_3*S_3;
        

        // Conserved S_i variable with index up: S^i = gam^ij S_j
        // for calculation of primitive velocity
        Real S1 = gam11*S_1 + gam12*S_2 + gam13*S_3;
        Real S2 = gam12*S_1 + gam22*S_2 + gam23*S_3;
        Real S3 = gam13*S_1 + gam23*S_2 + gam33*S_3;
        // Extract old primitives
        const Real &rho_old = prim_old(IDN,k,j,i);
//        const Real &eps_old = prim_old(IEPS,k,j,i);
        const Real &pgas_old = prim_old(IPR,k,j,i);
        const Real &uu1_old = prim_old(IVX,k,j,i);
        const Real &uu2_old = prim_old(IVY,k,j,i);
        const Real &uu3_old = prim_old(IVZ,k,j,i);



        // Apply Newton-Raphson method to find new pressure
        const int num_iterations = 5;
        Real pres_new = pgas_old;
        Real res_new = PresResidual(pres_new, D, tau, S_sq, gamma_adi);
        for (int n = 0; n < num_iterations; ++n) {
          Real pres_old = pres_new;
          Real res_old = res_new;
          Real derivative = PresResidualPrime(pres_old, D, tau, S_sq, gamma_adi);
          Real delta = -res_old / derivative;
          pres_new = pres_old + delta;
          res_new = PresResidual(pres_new, D, tau, S_sq, gamma_adi);
        }
        Real pres_true = pres_new;
        if (std::isnan(pres_true)){
         fixed_(k,j,i) = true;
         pres_true = k_adi*pow(rhoc*fatm,gamma_adi); // fix atmos value of pressure if N-R fails
        }
        // Extract primitives
        Real &rho = prim(IDN,k,j,i);
        Real &pgas = prim(IPR,k,j,i);
        Real &uu1 = prim(IVX,k,j,i);
        Real &uu2 = prim(IVY,k,j,i);
        Real &uu3 = prim(IVZ,k,j,i);



        pgas = pres_true;
        Real Wp = (tau + pres_true + D) / (std::sqrt(SQR(tau + pres_true + D) - S_sq )   );
        rho = D/Wp;
        Real v1 = S1/(tau+D+pres_true);
        Real v2 = S2/(tau+D+pres_true);
        Real v3 = S3/(tau+D+pres_true);
        uu1 = v1*Wp;
        uu2 = v2*Wp;
        uu3 = v3*Wp;
        Real gamma_rel = Wp;
        // Apply floors to density and pressure
        Real density_floor_local = density_floor_;
        if (rho_pow_ != 0.0) {
          density_floor_local = std::max(density_floor_local,
              rho_min_*std::pow(pco->x1v(i),rho_pow_));
        }
        Real pressure_floor_local = pressure_floor_;
        if (pgas_pow_ != 0.0) {
          pressure_floor_local = std::max(pressure_floor_local,
              pgas_min_ * std::pow(pco->x1v(i), pgas_pow_));
        }
//      Set as atmosphere if density is less than f_thr * f_atm * maxrho. (maxrho=rhoc)	
        if (rho < fthr*fatm*rhoc or std::isnan(rho) or rho < density_floor_local) {
          rho = fatm*rhoc;
          fixed_(k,j,i) = true;
        }
//	if point has been identified as atmosphere, set pressure to atmosphere val
        if (pgas < pressure_floor_local or std::isnan(pgas) or fixed_(k,j,i) == true) {
          pgas = k_adi*pow(rhoc*fatm,gamma_adi);
          fixed_(k,j,i) = true;
        }
	// Set atmosphere points to have 0 velocity
	if (fixed_(k,j,i) == true){
		uu1 = 0.0;
		uu2 = 0.0;
		uu3 = 0.0;
	}

        // Apply ceiling to velocity
        if (gamma_rel > gamma_max_) {
          Real factor = std::sqrt((SQR(gamma_max_)-1.0) / (SQR(gamma_rel)-1.0));
          uu1 *= factor;
          uu2 *= factor;
          uu3 *= factor;
          fixed_(k,j,i) = true;
        }
      }
    }
  }

  // Fix corresponding conserved values if any changes made
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      pco->CellMetric(k, j, il, iu, g_, g_inv_);
      for (int i=il; i<=iu; ++i) {
        if (fixed_(k,j,i)) {
          PrimitiveToConservedSingle(prim, gamma_adi, g_, g_inv_, k, j, i, cons, pco);
          fixed_(k,j,i) = false;
        }
      }
    }
  }
  return;
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

void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bb_cc, AthenaArray<Real> &cons, Coordinates *pco, int il,
     int iu, int jl, int ju, int kl, int ku) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      pco->CellMetric(k, j, il, iu, g_, g_inv_);
      //#pragma omp simd // fn is too long to inline
      for (int i=il; i<=iu; ++i) {
        PrimitiveToConservedSingle(prim, gamma_, g_, g_inv_, k, j, i, cons, pco);
      }
    }
  }
  return;
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

static void PrimitiveToConservedSingle(const AthenaArray<Real> &prim, Real gamma_adi,
    const AthenaArray<Real> &g, const AthenaArray<Real> &gi, int k, int j, int i,
    AthenaArray<Real> &cons, Coordinates *pco) {
  // Extract primitives
  const Real &rho = prim(IDN,k,j,i);
//  const Real &eps = prim(IEPS,k,j,i);
  const Real &pgas = prim(IPR,k,j,i);
  const Real &uu1 = prim(IVX,k,j,i);
  const Real &uu2 = prim(IVY,k,j,i);
  const Real &uu3 = prim(IVZ,k,j,i);

  // Calculate 4-velocity
  Real alpha = std::sqrt(-1.0/gi(I00,i));
  Real detgamma = std::sqrt(Determinant(g(I11,i),g(I12,i),g(I13,i),g(I12,i),g(I22,i),g(I23,i),g(I13,i),g(I23,i),g(I33,i)));
  Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
           + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
           + g(I33,i)*uu3*uu3;
  Real gamma = std::sqrt(1.0 + tmp);
  Real u0 = gamma/alpha;
  Real u1 = uu1 - alpha * gamma * gi(I01,i);
  Real u2 = uu2 - alpha * gamma * gi(I02,i);
  Real u3 = uu3 - alpha * gamma * gi(I03,i);
  Real u_0, u_1, u_2, u_3;
  pco->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

  Real v_1 = u_1/gamma; 
  Real v_2 = u_2/gamma;  
  Real v_3 = u_3/gamma;  
  // Extract conserved quantities
  Real &Ddg = cons(IDN,k,j,i);
  Real &taudg = cons(IEN,k,j,i);
  Real &S_1dg = cons(IM1,k,j,i);
  Real &S_2dg = cons(IM2,k,j,i);
  Real &S_3dg = cons(IM3,k,j,i);

  // Set conserved quantities
  Real wgas = rho + gamma_adi/(gamma_adi-1.0) * pgas;
  Ddg = rho*gamma*detgamma;
  taudg = wgas*SQR(gamma)*detgamma - rho*gamma*detgamma - pgas*detgamma;
  S_1dg = wgas*SQR(gamma) * v_1*detgamma  ;
  S_2dg = wgas*SQR(gamma) * v_2*detgamma  ;
  S_3dg = wgas*SQR(gamma) * v_3*detgamma  ;
  return;
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

void EquationOfState::SoundSpeedsSR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq,
    Real *plambda_plus, Real *plambda_minus) {
  const Real gamma_adi = gamma_;
  Real cs_sq = gamma_adi * pgas / rho_h;                                 // (MB 4)
  Real sigma_s = cs_sq / (gamma_lorentz_sq * (1.0-cs_sq));
  Real relative_speed = std::sqrt(sigma_s * (1.0 + sigma_s - SQR(vx)));
  *plambda_plus = 1.0/(1.0+sigma_s) * (vx + relative_speed);             // (MB 23)
  *plambda_minus = 1.0/(1.0+sigma_s) * (vx - relative_speed);            // (MB 23)
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

void EquationOfState::SoundSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1, Real g00,
    Real g01, Real g11, Real *plambda_plus, Real *plambda_minus) {
  // Parameters and constants
  const Real discriminant_tol = -1.0e-10;  // values between this and 0 are considered 0
  const Real gamma_adi = gamma_;

  // Calculate comoving sound speed
  Real cs_sq = gamma_adi * pgas / rho_h;

  // Set sound speeds in appropriate coordinates
  Real a = SQR(u0) - (g00 + SQR(u0)) * cs_sq;
  Real b = -2.0 * (u0*u1 - (g01 + u0*u1) * cs_sq);
  Real c = SQR(u1) - (g11 + SQR(u1)) * cs_sq;
  Real d = SQR(b) - 4.0*a*c;
  if (d < 0.0 and d > discriminant_tol) {
    d = 0.0;
  }
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

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct enthalpy
// Inputs:
//   w_guess: guess for enthalpy W
//   d: D = alpha * rho * u^0
//   q_n: Q_mu n^mu = -alpha^2 g^{mu 0} T^0_mu
//   qq_sq: \tilde{Q}^2 = alpha^2 g^{mu nu} T^0_mu T^0_nu + alpha^4 (g^{0 mu} T^0_mu)^2
//   gamma_adi: ratio of specific heats
// Outputs:
//   returned value: calculated minus given value of q_n
// Notes:
//   follows Noble et al. 2006, ApJ 641 626 (N)
//   implements formulas assuming no magnetic field
//remnant from old scheme - using T^0_mu conservatives,
// and inverting to find enthalpy


static Real QNResidual(Real w_guess, Real d, Real q_n, Real qq_sq, Real gamma_adi) {
  Real v_norm_sq = qq_sq / (w_guess*w_guess);        // (N 28)
  Real gamma_sq = 1.0/(1.0 - v_norm_sq);
  Real pgas = (gamma_adi-1.0)/gamma_adi
      * (w_guess/gamma_sq - d/std::sqrt(gamma_sq));  // (N 32)
  return -w_guess + pgas - q_n;                      // (N 29)
}



//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct pressure for gamma law

static Real PresResidual(Real pres_guess, Real D, Real tau, Real S_sq, Real gamma_adi) {
  Real Wp = (tau + pres_guess + D) / (std::sqrt( SQR(tau + pres_guess + D) - S_sq  )  );
  Real rhop = D / Wp;
  Real epsp = (std::sqrt( SQR(tau + pres_guess + D) - S_sq) - Wp*pres_guess -D   ) / D;
  return pres_guess - rhop*epsp*(gamma_adi-1.0);                    
}





//----------------------------------------------------------------------------------------
// Derivative of QNResidual()
// Inputs:
//   w_guess: guess for enthalpy W
//   d: D = alpha * rho * u^0
//   qq_sq: \tilde{Q}^2 = alpha^2 g^{mu nu} T^0_mu T^0_nu + alpha^4 (g^{0 mu} T^0_mu)^2
//   gamma_adi: ratio of specific heats
// Outputs:
//   returned value: derivative of calculated value of Q_mu n^mu
// Notes:
//   follows Noble et al. 2006, ApJ 641 626 (N)
//   implements formulas assuming no magnetic field
// remnant from old scheme

static Real QNResidualPrime(Real w_guess, Real d, Real qq_sq, Real gamma_adi) {
  Real v_norm_sq = qq_sq/SQR(w_guess);                             // (N 28)
  Real gamma_sq = 1.0/(1.0-v_norm_sq);
  Real gamma_4 = SQR(gamma_sq);
  Real d_v_norm_sq_dw = -2.0 * qq_sq / (w_guess*SQR(w_guess));
  Real d_gamma_sq_dw = gamma_4 * d_v_norm_sq_dw;
  Real dpgas_dw = (gamma_adi-1.0)/gamma_adi / gamma_4 * (gamma_sq
      + (0.5*d*std::sqrt(gamma_sq) - w_guess) * d_gamma_sq_dw);
  return -1.0 + dpgas_dw;
}





//----------------------------------------------------------------------------------------
// Derivative of PresResidual() for gamma law

static Real PresResidualPrime(Real pres_guess, Real D, Real tau, Real S_sq, Real gamma_adi) {
  Real Wp = (tau + pres_guess + D) / (std::sqrt( SQR(tau + pres_guess + D) - S_sq  )  );
  Real rhop = D / Wp;
  Real epsp = (std::sqrt( SQR(tau + pres_guess + D) - S_sq) - Wp*pres_guess - D  ) / D;
  Real dPdrho = epsp*(gamma_adi - 1.0);
  Real dPdeps = rhop * (gamma_adi - 1.0);
  Real drhodp = D*S_sq / ( SQR(D + pres_guess + tau) * std::sqrt( SQR(D+ pres_guess + tau) - S_sq )    ); 
  Real depsdp = pres_guess*S_sq / ( D * pow( SQR(D+ pres_guess + tau) - S_sq,1.5 )  ); 

  return 1.0 - dPdrho*drhodp - dPdeps*depsdp;
}






//---------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// \brief Apply density and pressure floors to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);
  Real& w_p  = prim(IPR,i);
  // Not applying position-dependent floors here in GR, nor using rho_min
  // apply density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

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
