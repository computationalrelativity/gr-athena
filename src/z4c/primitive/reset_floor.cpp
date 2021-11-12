//! \file reset_floor.cpp
//  \brief Implementation of the ResetFloor policy

#include <cmath>

#include "reset_floor.hpp"
#include "ps_error.hpp"

using namespace Primitive;

/// Constructor
ResetFloor::ResetFloor() {
  fail_conserved_floor = false;
  fail_primitive_floor = false;
  adjust_conserved = true;
}

/// Floor for the primitive variables
bool ResetFloor::PrimitiveFloor(Real& n, Real v[3], Real& t) {
  if (n < n_atm) {
    n = n_atm;
    v[0] = 0.0;
    v[1] = 0.0;
    v[2] = 0.0;
    t = T_atm;
    return true;
  }
  else if (t < T_atm) {
    t = T_atm;
    return true;
  }
  return false;
}

/// Floor for the conserved variables
/// FIXME: This currently sets tau to the pressure atmosphere,
/// but it needs to be replaced with something which is
/// thermodynamically consistent.
bool ResetFloor::ConservedFloor(Real& D, Real Sd[3], Real& tau, Real D_floor, Real tau_floor) {
  if (D < D_floor) {
    D = D_floor;
    Sd[0] = 0.0;
    Sd[1] = 0.0;
    Sd[2] = 0.0;
    tau = tau_floor;
    return true;
  }
  else if (tau < tau_floor) {
    tau = tau_floor;
    return true;
  }
  return false;
}

/// Reset excess magnetization
Error ResetFloor::MagnetizationResponse(Real& bsq, Real b_u[3]) {
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
void ResetFloor::DensityLimits(Real& n, Real n_min, Real n_max) {
  n = std::fmax(n_min, std::fmin(n_max, n));
}

/// Apply energy limiter
void ResetFloor::TemperatureLimits(Real& T, Real T_min, Real T_max) {
  T = std::fmax(T_min, std::fmin(T_max, T));
}
