#ifndef PRIMITIVE_SOLVER_HPP
#define PRIMITIVE_SOLVER_HPP

//! \file primitive_solver.hpp
//  \brief Declares PrimitiveSolver class.
//
//  PrimitiveSolver contains all the infrastructure for the inversion
//  procedure from conserved to primitive variables in GRMHD. This
//  particular implementation is based on the solver described in
//  Kastaun et al., Phys. Rev. D 103, 023018 (2021).

#include <cmath>
// FIXME: Debug only!
#include <iostream>

#include "numtoolsroot.h"

#include "eos.hpp"
#include "geom_math.hpp"
#include "ps_error.hpp"

namespace Primitive {

template<typename EOSPolicy, typename ErrorPolicy>
class PrimitiveSolver {
  private:
    /// A constant pointer to the EOS.
    /// We make this constant because the
    /// possibility of changing the EOS
    /// during implementation seems both
    /// unlikely and dangerous.
    EOS<EOSPolicy, ErrorPolicy> *const peos;
    
    //! \brief function for the upper bound of the root
    //
    //  The upper bound is the solution to the function
    //  \f$\mu\sqrt{h_0^2 + \bar{r}^2(\mu)} - 1 = 0\f$
    //
    //  \param[out] f    The value of the root function at mu
    //  \param[out] df   The derivative of the root function
    //  \param[in   mu   The guess for the root
    //  \param[in]  bsq  The square magnitude of the magnetic field
    //  \param[in]  rsq  The square magnitude of the specific momentum S/D
    //  \param[in]  rbsq The square of the product \f$r\cdot b\f$
    //  \param[in]  h_min The minimum enthalpy
    static void UpperRoot(Real &f, Real &df, Real mu, Real bsq, Real rsq, Real rbsq, Real h_min);

    //! \brief function for the upper bound of the root given
    //         a specified Lorentz factor.
    //
    //  \param[out] f    The value of the root function at mu
    //  \param[out] df   The derivative of the root function
    //  \param[in]  mu   The guess for the root
    //  \param[in]  bsq  The square magnitude of the magnetic field
    //  \param[in]  rsq  The square magnitude of the specific momentum S/D
    //  \param[in]  rbsq The square of the product /f$r\cdot b\f$
    //  \param[in]  W    The Lorentz factor
    static void MuFromW(Real &f, Real &df, Real mu, Real bsq, Real rsq, Real rbsq, Real W);

    //! \brief master function for the root solve
    //
    //  The root solve is based on the master function
    //  \f$ f(\mu) = \mu - \hat{mu}(\mu).
    //
    //  \param[in]  mu   The guess for the root
    //  \param[in]  D    The relativistic density
    //  \param[in]  q    The specific tau variable tau/D
    //  \param[in]  bsq  The square magnitude of the magnetic field
    //  \param[in]  rsq  The square magnitude of the specific momentum S/D
    //  \param[in]  rbsq The square of the product \f$r\cdot b\f$
    //  \param[in]  Y    The vector of particle fractions
    //  \param[in]  peos The pointer to the EOS
    //  \param[out] n    The resulting estimate for n
    //  \param[out] T    The resulting estimate for T
    //  \param[out] P    The resulting estimate for P
    //  
    //  \return f evaluated at mu for the given constants.
    static Real RootFunction(Real mu, Real D, Real q, Real bsq, Real rsq, Real rbsq, Real *Y,
        EOS<EOSPolicy, ErrorPolicy> *const peos, Real* n, Real* T, Real* P);

    //! \brief Check and handle the corner case for rho being too small or large.
    //
    //  Using the minimum and maximum values of rho along with some physical
    //  limitations on the velocity using S, we can predict if rho is going
    //  to violate constraints set by the EOS on how big or small it can get.
    //  We can also use these constraints to tighten the bounds on mu.
    //  
    //  \param[in,out] mul   The lower bound for mu
    //  \param[in,out] muh   The upper bound for mu
    //  \param[in]     D     The relativistic density
    //  \param[in]     bsq   The square magnitude of the magnetic field
    //  \param[in]     rsq   The square magnitude of the specific momentum S/D
    //  \param[in]     rbsq  The square of the product \f$r\cdot b\f$
    //  \param[in]     h_min The minimum enthalpy
    //
    //  \return an Error code, usually RHO_TOO_BIG, RHO_TOO_SMALL, or SUCCESS
    Error CheckDensityValid(Real& mul, Real& muh, Real D, Real bsq, Real rsq, Real rbsq, Real h_min);
  public:
    /// Constructor
    PrimitiveSolver(EOS<EOSPolicy, ErrorPolicy> *eos) : peos(eos) {
    }

    /// Destructor
    ~PrimitiveSolver() = default;

    //! \brief Get the primitive variables from the conserved variables.
    //
    //  \param[out]    prim  The array of primitive variables
    //  \param[in,out] cons  The array of conserved variables
    //  \param[in,out] bu    The magnetic field
    //  \param[in]     g3d   The 3x3 spatial metric
    //  \param[in]     g3u   The 3x3 inverse spatial metric
    //
    //  \return an error code
    Error ConToPrim(Real prim[NPRIM], Real cons[NCONS], Real b[NMAG], 
                   Real g3d[NSPMETRIC], Real g3u[NSPMETRIC]);

    //! \brief Get the conserved variables from the primitive variables.
    //
    //  \param[in]    prim  The array of primitive variables
    //  \param[out]   cons  The array of conserved variables
    //  \param[in]    bu    The magnetic field
    //  \param[in]    g3d   The 3x3 spatial metric
    //
    //  \return an error code
    Error PrimToCon(Real prim[NPRIM], Real cons[NCONS], Real b[NMAG], 
                   Real g3d[NSPMETRIC]);

    /// Get the EOS used by this PrimitiveSolver.
    inline EOS<EOSPolicy, ErrorPolicy> *const GetEOS() const {
      return peos;
    }
};

// UpperRoot {{{
template<typename EOSPolicy, typename ErrorPolicy>
void PrimitiveSolver<EOSPolicy, ErrorPolicy>::UpperRoot(Real &f, Real &df, Real mu, Real bsq, Real rsq, Real rbsq, Real min_h) {
  const Real x = 1.0/(1.0 + mu*bsq);
  const Real xsq = x*x;
  const Real rbarsq = rsq*xsq + mu*x*(1.0 + x)*rbsq;
  const Real dis = std::sqrt(min_h*min_h + rbarsq);
  const Real dx = -bsq*xsq;
  //const Real drbarsq = rbsq*x*(1.0 + x) + (mu*rbsq + 2.0*(mu*rbsq + rsq)*x)*dx;
  const Real drbarsq = rbsq*xsq + mu*rbsq*dx + x*(rbsq + 2.0*(mu*rbsq + rsq)*dx);
  f = mu*dis - 1.0;
  df = dis + mu*drbarsq/(2.0*dis);
}
// }}}

// MuFromW {{{
template<typename EOSPolicy, typename ErrorPolicy>
void PrimitiveSolver<EOSPolicy, ErrorPolicy>::MuFromW(Real &f, Real &df, Real mu, Real bsq, Real rsq, Real rbsq, Real W) {
  const Real musq = mu*mu;
  const Real x = 1.0/(1.0 + mu*bsq);
  const Real xsq = x*x;
  const Real rbarsq = rsq*xsq + mu*x*(1.0 + x)*rbsq;
  const Real vsq = musq*rbarsq;
  const Real dx = -bsq*xsq;
  //const Real drbarsq = rbsq*x*(1.0 + x) + (mu*rbsq + 2.0*(mu*rbsq + rsq)*x)*dx;
  //const Real drbarsq = rbsq*xsq + mu*rbsq*dx + x*(rbsq + 2.0*(mu*rbsq + rsq)*dx);
  const Real drbarsq = 2.0*rsq*dx + x*(1.0 + x)*rbsq + mu*dx*(1.0 + 2.0*x)*rbsq;
  const Real dvsq = 2.0*mu*rbarsq + musq*drbarsq;
  f = vsq + 1.0/(W*W) - 1.0;
  df = dvsq;
}
// }}}

// RootFunction {{{
template<typename EOSPolicy, typename ErrorPolicy>
Real PrimitiveSolver<EOSPolicy, ErrorPolicy>::RootFunction(Real mu, Real D, Real q, Real bsq, Real rsq, Real rbsq, Real *Y,
      EOS<EOSPolicy, ErrorPolicy> *const peos, Real* n, Real* T, Real* P) {
  // We need to get some utility quantities first.
  const Real x = 1.0/(1.0 + mu*bsq);
  const Real xsq = x*x;
  const Real musq = mu*mu;
  //const Real den = 1.0 + mu*bsq;
  const Real mux = mu*x;
  //const Real muxsq = mux/den;
  const Real rbarsq = rsq*xsq + mux*(1.0 + x)*rbsq;
  //const Real rbarsq = xsq*(rsq + mu*(2.0 + mu*bsq)*rbsq);
  // An alternative calculation of rbarsq that may be more accurate.
  //const Real rbarsq = rsq*xsq + (mux + muxsq)*rbsq;
  //const Real rbarsq = x*(rsq*x + mu*(x + 1.0)*rbsq);
  const Real qbar = q - 0.5*bsq - 0.5*musq*xsq*(bsq*rsq - rbsq);
  const Real mb = peos->GetBaryonMass();

  // Now we can estimate the velocity.
  //const Real v_max = peos->GetMaxVelocity();
  const Real h_min = peos->GetMinimumEnthalpy();
  const Real vsq_max = std::fmin(rsq/(h_min*h_min + rsq), peos->GetMaxVelocity());
  const Real vhatsq = std::fmin(musq*rbarsq, vsq_max);

  // Using the velocity estimate, predict the Lorentz factor.
  //const Real What = 1.0/std::sqrt(1.0 - vhatsq);
  const Real iWhat = std::sqrt(1.0 - vhatsq);

  // Now estimate the number density.
  Real rhohat = D*iWhat;
  Real nhat = rhohat/mb;
  peos->ApplyDensityLimits(nhat);

  // Estimate the energy density.
  Real eoverD = qbar - mu*rbarsq + 1.0;
  Real ehat = D*eoverD;
  //peos->ApplyEnergyLimits(ehat);
  //eoverD = ehat/D;

  // Now we can get an estimate of the temperature, and from that, the pressure and enthalpy.
  Real That = peos->GetTemperatureFromE(nhat, ehat, Y);
  peos->ApplyTemperatureLimits(That);
  ehat = peos->GetEnergy(nhat, That, Y);
  Real Phat = peos->GetPressure(nhat, That, Y);
  Real hhat = peos->GetEnthalpy(nhat, That, Y)/mb;

  // Now we can get two different estimates for nu = h/W.
  Real nu_a = hhat*iWhat;
  Real ahat = Phat/ehat;
  //Real nu_b = eoverD + Phat/D;
  Real nu_b = (1.0 + ahat)*eoverD;
  Real nuhat = std::fmax(nu_a, nu_b);

  // Finally, we can get an estimate for muhat.
  Real muhat = 1.0/(nuhat + mu*rbarsq);

  *n = nhat;
  *T = That;
  *P = Phat;

  // FIXME: Debug only!
  /*std::cout << "    D   = " << D << "\n";
  std::cout << "    q   = " << q << "\n";
  std::cout << "    bsq = " << bsq << "\n";
  std::cout << "    rsq = " << rsq << "\n";
  std::cout << "    rbsq = " << rbsq << "\n"*/

  return mu - muhat;
}
// }}}

// CheckDensityValid {{{
template<typename EOSPolicy, typename ErrorPolicy>
Error PrimitiveSolver<EOSPolicy, ErrorPolicy>::CheckDensityValid(Real& mul, Real& muh, Real D, 
      Real bsq, Real rsq, Real rbsq, Real h_min) {
  NumTools::Root& root = NumTools::Root::get_instance();

  // There are a few things considered:
  // 1. If D > rho_max, we need to make sure that W isn't too large.
  //    W_max can be estimated by considering the zero-field limit
  //    of S^2/D^2 if h = h_min.
  //    - If W is larger than W_max, then rho is just too big.
  //    - Otherwise, we can bound mu by using W to do a root solve
  //      for mu.
  // 2. If D < W_max*rho_min, then we need to make sure W isn't less
  //    than 1.
  //    - If W is less than 1, it means rho is actually smaller than
  //      rho_min.
  //    - Otherwise, we can bound mu by using W to do a root solve
  //      for mu.
  Real W_max = std::sqrt(1.0 + rsq/(h_min*h_min));
  Real rho_max = peos->GetMaximumDensity()*peos->GetBaryonMass();
  Real rho_min = peos->GetMinimumDensity()*peos->GetBaryonMass();
  if (D > rho_max) {
    Real W = D/rho_max;
    if (W > W_max) {
      // W is not physical, so rho must be larger than rho_max.
      return Error::RHO_TOO_BIG;
    }
    else {
      // We can tighten up the bounds for muh.
      root.newton_raphson(&MuFromW, muh, bsq, rsq, rbsq, W);
    }
  }
  if (D < W_max*rho_min) {
    Real W = D/rho_min;
    if (W < 1.0) {
      // W is not physical, so rho must be smaller than rho_min.
      return Error::RHO_TOO_SMALL;
    }
    else {
      // We can tighten up the bounds for mul.
      root.newton_raphson(&MuFromW, mul, bsq, rsq, rbsq, W);
    }
  }
  return Error::SUCCESS;
}
// }}}

// ConToPrim {{{
template<typename EOSPolicy, typename ErrorPolicy>
Error PrimitiveSolver<EOSPolicy, ErrorPolicy>::ConToPrim(Real prim[NPRIM], Real cons[NCONS],
      Real b[NMAG], Real g3d[NSPMETRIC], Real g3u[NSPMETRIC]) {

  // Extract the undensitized conserved variables.
  Real D = cons[IDN];
  Real S_d[3] = {cons[IM1], cons[IM2], cons[IM3]};
  Real tau = cons[IEN];
  Real B_u[3] = {b[IB1], b[IB2], b[IB3]};
  // Extract the particle fractions.
  const int n_species = peos->GetNSpecies();
  Real Y[MAX_SPECIES] = {0.0};
  for (int s = 0; s < n_species; s++) {
    Y[s] = cons[IYD + s]/cons[IDN];
  }

  // Check the conserved variables for consistency and do whatever
  // the EOSPolicy wants us to.
  bool floored = peos->ApplyConservedFloor(D, S_d, tau, Y);
  if (floored && peos->IsConservedFlooringFailure()) {
    return Error::CONS_FLOOR;
  }

  // Calculate some utility quantities.
  Real sqrtD = std::sqrt(D);
  Real b_u[3] = {B_u[0]/sqrtD, B_u[1]/sqrtD, B_u[2]/sqrtD};
  Real r_d[3] = {S_d[0]/D, S_d[1]/D, S_d[2]/D};
  Real r_u[3];
  RaiseForm(r_u, r_d, g3u);
  Real rsqr   = Contract(r_u, r_d);
  Real rb     = Contract(b_u, r_d);
  Real rbsqr  = rb*rb;
  Real bsqr   = SquareVector(b_u, g3d);
  Real q      = tau/D;

  // Make sure there are no NaNs at this point.
  if (!std::isfinite(D) || !std::isfinite(rsqr) || !std::isfinite(q) ||
      !std::isfinite(rbsqr) || !std::isfinite(bsqr)) {
    return Error::NANS_IN_CONS;
  }
  // We have to check the particle fractions separately.
  for (int s = 0; s < n_species; s++) {
    if (!std::isfinite(Y[s])) {
      return Error::NANS_IN_CONS;
    }
  }

  bool adjust_cons = false;
  // Make sure that the magnetic field is physical.
  Error error = peos->DoMagnetizationResponse(bsqr, b_u);
  if (error == Error::MAG_TOO_BIG) {
    return Error::MAG_TOO_BIG;
  }
  else if (error == Error::CONS_ADJUSTED) {
    adjust_cons = true;
    // We need to recalculate rb if b_u is rescaled.
    rb = Contract(b_u, r_d);
    rbsqr = rb*rb;
  }
  
  // Bracket the root.
  Real min_h = peos->GetMinimumEnthalpy()/peos->GetBaryonMass();
  Real mul = 0.0;
  Real muh = 1.0/min_h;
  // Check if a tighter upper bound exists.
  NumTools::Root& root = NumTools::Root::get_instance();
  if (rsqr > min_h*min_h) {
    Real mu = 0.0;
    // We don't need the bound to be that tight, so we reduce
    // the accuracy of the root solve for speed reasons.
    //NumTools::Root::tol = 1e-3;
    //NumTools::Root::iterations = 10;
    root.tol = 1e-15;
    root.iterations = 30;
    bool result = root.newton_raphson(&UpperRoot, mu, bsqr, rsqr, rbsqr, min_h);
    // DEBUG ONLY: Make sure the root is valid.
    Real n, T, P;
    /*if (RootFunction(mu, D, q, bsqr, rsqr, rbsqr, Y, peos, &n, &T, &P) < 0.0) {
      std::cout << "There was a problem bracketing the root!\n";
      return Error::BRACKETING_FAILED;
    }*/
    // Scream if the bracketing failed.
    if (!result) {
      return Error::BRACKETING_FAILED;
    }
    else {
      // To avoid the case where the root and the upper bound collide, we will perturb the
      // bound by a small factor.
      // TODO: Is there a more rigorous way of treating this?
      muh = mu*(1. + root.tol);
    }
  }

  // Check the corner case where the density is outside the permitted
  // bounds according to the ErrorPolicy.
  /*error = CheckDensityValid(mul, muh, D, bsqr, rsqr, rbsqr, min_h);
  if (error != Error::SUCCESS) {
    // TODO: This is probably something that should be handled by the ErrorPolicy.
    return error;
  }*/

  
  // Do the root solve.
  // TODO: This should be done with something like TOMS748 once it's
  // available.
  root.tol = 1e-15;
  root.iterations = 30;
  Real n, P, T, mu;
  // DEBUG ONLY
  Real mul_old = mul;
  Real muh_old = muh;
  bool result = root.false_position(&RootFunction, mul, muh, mu, D, q, bsqr, rsqr, rbsqr, Y, peos, &n, &T, &P);
  if (!result) {
    return Error::NO_SOLUTION;
  }

  // Retrieve the primitive variables.
  Real rho = n*peos->GetBaryonMass();
  Real rbmu = rb*mu;
  Real W = D/rho;
  Real Wmux = W*mu/(1.0 + mu*bsqr);
  // Before we retrieve the velocity, we need to raise S.
  Real S_u[3] = {0.0};
  RaiseForm(S_u, S_d, g3u);
  // Now we can get Wv.
  Real Wv_u[3] = {0.0};
  Wv_u[0] = Wmux*(r_u[0] + rbmu*b_u[0]);
  Wv_u[1] = Wmux*(r_u[1] + rbmu*b_u[1]);
  Wv_u[2] = Wmux*(r_u[2] + rbmu*b_u[2]);
  
  // Apply the flooring policy to the primitive variables.
  floored = peos->ApplyPrimitiveFloor(n, Wv_u, P, T, Y);
  if (floored && peos->IsPrimitiveFlooringFailure()) {
    return Error::PRIM_FLOOR;
  }
  adjust_cons = adjust_cons || floored;

  prim[IDN] = n*peos->GetBaryonMass();
  prim[IPR] = P;
  prim[ITM] = T;
  prim[IVX] = Wv_u[0];
  prim[IVY] = Wv_u[1];
  prim[IVZ] = Wv_u[2];
  for (int s = 0; s < n_species; s++) {
    prim[IYF + s] = Y[s];
  }

  // If we floored the primitive variables, we should check
  // if the EOS wants us to adjust the conserved variables back
  // in bounds. If that's the case, then we'll do it.
  if (adjust_cons && peos->KeepPrimAndConConsistent()) {
    PrimToCon(prim, cons, b, g3d);
  }

  return Error::SUCCESS;
}
// }}}

// PrimToCon {{{
template<typename EOSPolicy, typename ErrorPolicy>
Error PrimitiveSolver<EOSPolicy, ErrorPolicy>::PrimToCon(Real prim[NPRIM], Real cons[NCONS],
      Real bu[NMAG], Real g3d[NMETRIC]) {
  // Extract the primitive variables
  const Real &rho = prim[IDN]; // rest-mass density
  const Real Wv_u[3] = {prim[IVX], prim[IVY], prim[IVZ]};
  const Real &p   = prim[IPR]; // pressure
  const Real &t   = prim[ITM]; // temperature
  const Real B_u[3] = {bu[IB1], bu[IB2], bu[IB3]};
  const int n_species = peos->GetNSpecies();
  Real Y[MAX_SPECIES] = {0.0};
  for (int s = 0; s < n_species; s++) {
    Y[s] = prim[IYF + s];
  }

  // Note that Athena passes in Wv, not v.
  // Lower u.
  Real Wv_d[3];
  LowerVector(Wv_d, Wv_u, g3d);
  Real Wvsq = Contract(Wv_u, Wv_d);
  Real Wsq = 1.0 + Wvsq;
  Real W = sqrt(Wsq);
  // Get the 3-velocity.
  Real v_d[3] = {Wv_d[0]/W, Wv_d[1]/W, Wv_d[2]/W};

  // For the magnetic field contribution, we need to find
  // B_i, B^2, and B^i*v_i.
  Real B_d[3];
  LowerVector(B_d, B_u, g3d);
  Real Bsq = Contract(B_u, B_d);
  Real Bv = Contract(B_u, v_d);

  // Some utility quantities that will be helpful.
  const Real mb = peos->GetBaryonMass();

  // Extract the conserved variables
  Real &D = cons[IDN]; // Relativistic density
  Real &Sx = cons[IM1]; // Relativistic momentum density (x)
  Real &Sy = cons[IM2]; // Relativistic momentum density (y)
  Real &Sz = cons[IM3]; // Relativistic momentum density (z)
  Real &tau = cons[IEN]; // Relativistic energy - D

  // Set the conserved quantities.
  // Total enthalpy density
  Real H = rho*peos->GetEnthalpy(rho/mb, t, Y)/mb;
  Real HWsq = H*Wsq;
  D = rho*W;
  for (int s = 0; s < n_species; s++) {
    cons[IYD + s]= D*Y[s];
  }
  Real HWsqpb = HWsq + Bsq;
  Sx = (HWsqpb*v_d[0] - Bv*B_d[0]);
  Sy = (HWsqpb*v_d[1] - Bv*B_d[1]);
  Sz = (HWsqpb*v_d[2] - Bv*B_d[2]);
  tau = (HWsqpb - p - 0.5*(Bv*Bv + Bsq/Wsq)) - D;

  // DEBUG ONLY
  /*if (tau <= 0) {
    std::cout << "Tau is negative!\n";
  }*/

  return Error::SUCCESS;
}
// }}}

} // namespace
#endif
