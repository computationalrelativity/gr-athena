//! \file reset_floor_transition.cpp
//  \brief Implementation of the ResetFloorTransition policy

#include <cmath>

#include "reset_floor_transition.hpp"
#include "ps_error.hpp"

#include "../../defs.hpp"

using namespace Primitive;

/// Constructor
ResetFloorTransition::ResetFloorTransition() {
  fail_conserved_floor = false;
  fail_primitive_floor = false;
#ifdef PRIMITIVE_SOLVER_ADJUST_CONSERVED
  adjust_conserved = true;
#else
  adjust_conserved = false;
#endif
}

/// Floor for the primitive variables
bool ResetFloorTransition::PrimitiveFloor(Real& n, Real v[3], Real& T, Real *Y, int n_species) {
  if ((n < hd_n_min) and (T > ld_t_max)) {
    n = hd_n_min;
    printf("Reset at low n, high T: n=%.5e, T=%.5e\n", n, T);
    return true;
  }
  if (n < n_atm*n_threshold) {
    n = n_atm;
    v[0] = 0.0;
    v[1] = 0.0;
    v[2] = 0.0;
    T = T_atm;
    for (int i = 0; i < n_species; i++) {
      Y[i] = Y_atm[i];
    }
    // printf("Reset at atm n: n=%.5e, T=%.5e\n", n, T);
    return true;
  }
  if ((T < hd_t_min) and (n > ld_n_max)) {
    T = hd_t_min;
    printf("Reset at high n, low T: n=%.5e, T=%.5e\n", n, T);
    return true;
  }
  if (T < T_atm) {
    T = T_atm;
    // printf("Reset at atm T: n=%.5e, T=%.5e\n", n, T);
    return true;
  }
  return false;
}

/// Floor for the conserved variables
/// FIXME: Take a closer look at how the tau floor is performed.
bool ResetFloorTransition::ConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y, Real D_floor,
      Real tau_floor, Real tau_abs_floor, int n_species) {
  if (D < D_floor*n_threshold) {
    D = D_floor;
    Sd[0] = 0.0;
    Sd[1] = 0.0;
    Sd[2] = 0.0;
    tau = tau_abs_floor;
    for (int i = 0; i < n_species; i++) {
      Y[i] = Y_atm[i];
    }
    // printf("Reset conserved at atm D: D=%.5e, tau=%.5e\n", D, tau);
    return true;
  }
  else if (tau < tau_floor) {
    tau = tau_floor;
    // printf("Reset conserved at tau floor: D=%.5e, tau=%.5e tau_floor=%.5e\n", D, tau, tau_floor);
    return true;
  }
  return false;
}

/// Reset excess magnetization
Error ResetFloorTransition::MagnetizationResponse(Real& bsq, Real b_u[3]) {
  if (bsq > max_bsq) {
    Real factor = std::sqrt(max_bsq/bsq);
    bsq = max_bsq;

    b_u[0] /= factor;
    b_u[1] /= factor;
    b_u[2] /= factor;

    return Error::CONS_ADJUSTED;
  }
  return Error::SUCCESS;
}

/// Apply density limiter
void ResetFloorTransition::DensityLimits(Real& n, Real n_min, Real n_max) {
  n = std::fmax(n_min, std::fmin(n_max, n));
}

/// Apply temperature limiter
void ResetFloorTransition::TemperatureLimits(Real& T, Real T_min, Real T_max) {
  T = std::fmax(T_min, std::fmin(T_max, T));
}

/// Apply pressure limiter
void ResetFloorTransition::PressureLimits(Real& P, Real P_min, Real P_max) {
  P = std::fmax(P_min, std::fmin(P_max, P));
}

/// Apply energy density limiter
void ResetFloorTransition::EnergyLimits(Real& e, Real e_min, Real e_max) {
  e = std::fmax(e_min, std::fmin(e_max, e));
}

/// Apply species limits
bool ResetFloorTransition::SpeciesLimits(Real* Y, Real* Y_min, Real* Y_max, int n_species) {
  bool adjusted = false;
  for (int i = 0; i < n_species; i++) {
    if (Y[i] < Y_min[i]) {
      adjusted = true;
      Y[i] = Y_min[i];
    } else if (Y[i] > Y_max[i]) {
      adjusted = true;
      Y[i] = Y_max[i];
    }
  }
  return adjusted;
}

/// Perform failure response.
/// In this case, we simply floor everything.
bool ResetFloorTransition::FailureResponse(Real prim[NPRIM]) {
  prim[IDN] = n_atm;
  prim[IVX] = 0.0;
  prim[IVY] = 0.0;
  prim[IVZ] = 0.0;
  prim[ITM] = T_atm;
  for (int i = 0; i < MAX_SPECIES; i++) {
    prim[IYF + i] = Y_atm[i];
  }
  return true;
}
