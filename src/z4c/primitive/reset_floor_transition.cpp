//! \file reset_floor_transition.cpp
//  \brief Implementation of the ResetFloorTransition policy

#include "reset_floor_transition.hpp"

#include <algorithm>
#include <cmath>

#include "../../defs.hpp"
#include "ps_error.hpp"

using namespace Primitive;

/// Constructor
ResetFloorTransition::ResetFloorTransition()
{
  fail_conserved_floor = false;
  fail_primitive_floor = false;
#ifdef PRIMITIVE_SOLVER_ADJUST_CONSERVED
  adjust_conserved = true;
#else
  adjust_conserved = false;
#endif
}

/// Floor for the primitive variables
bool ResetFloorTransition::PrimitiveFloor(Real& n,
                                          Real v[3],
                                          Real& T,
                                          Real* Y,
                                          int n_species)
{
  bool adjusted = false;

  if (n < n_atm * n_threshold)
  {
    n    = n_atm;
    v[0] = 0.0;
    v[1] = 0.0;
    v[2] = 0.0;
    T    = T_atm;
    for (int i = 0; i < n_species; i++)
    {
      Y[i] = Y_atm[i];
    }
    return true;
  }
  if (T < T_atm)
  {
    T        = T_atm;
    adjusted = true;
  }
  if ((T > ld_t_max) and (n < hd_n_min))
  {
    n        = hd_n_min;
    adjusted = true;
  }
  if ((n > ld_n_max) and (T < hd_t_min))
  {
    T        = hd_t_min;
    adjusted = true;
  }
  return adjusted;
}

/// Floor for the conserved variables
/// FIXME: Take a closer look at how the tau floor is performed.
bool ResetFloorTransition::ConservedFloor(Real& D,
                                          Real Sd[3],
                                          Real& tau,
                                          Real* Y,
                                          Real D_floor,
                                          Real tau_floor,
                                          Real tau_abs_floor,
                                          int n_species)
{
  bool adjusted = false;

  if (D < D_floor * n_threshold)
  {
    D     = D_floor;
    Sd[0] = 0.0;
    Sd[1] = 0.0;
    Sd[2] = 0.0;
    tau   = tau_abs_floor;
    for (int i = 0; i < n_species; i++)
    {
      Y[i] = Y_atm[i];
    }
    adjusted = true;
  }
  else if (tau < tau_floor)
  {
    tau      = tau_floor;
    adjusted = true;
  }
  return adjusted;
}

/// Reset excess magnetization
Error ResetFloorTransition::MagnetizationResponse(Real& bsq, Real b_u[3])
{
  if (bsq > max_bsq)
  {
    Real factor = std::sqrt(max_bsq / bsq);
    bsq         = max_bsq;

    b_u[0] /= factor;
    b_u[1] /= factor;
    b_u[2] /= factor;

    return Error::CONS_ADJUSTED;
  }
  return Error::SUCCESS;
}

/// Apply density limiter
bool ResetFloorTransition::DensityLimits(Real& n, Real n_min, Real n_max)
{
  Real n_old = n;
  n          = std::fmax(n_min, std::fmin(n_max, n));
  return n != n_old;
}

/// Apply temperature limiter
bool ResetFloorTransition::TemperatureLimits(Real& T, Real T_min, Real T_max)
{
  Real T_old = T;
  T          = std::fmax(T_min, std::fmin(T_max, T));
  return T != T_old;
}

/// Apply pressure limiter
bool ResetFloorTransition::PressureLimits(Real& P, Real P_min, Real P_max)
{
  Real P_old = P;
  P          = std::fmax(P_min, std::fmin(P_max, P));
  return P != P_old;
}

/// Apply energy density limiter
bool ResetFloorTransition::EnergyLimits(Real& e, Real e_min, Real e_max)
{
  Real e_old = e;
  e          = std::fmax(e_min, std::fmin(e_max, e));
  return e != e_old;
}

/// Apply species limits
bool ResetFloorTransition::SpeciesLimits(Real* Y,
                                         Real* Y_min,
                                         Real* Y_max,
                                         int n_species)
{
  bool adjusted = false;
  for (int i = 0; i < n_species; i++)
  {
    if (Y[i] < Y_min[i])
    {
      adjusted = true;
      Y[i]     = Y_min[i];
    }
    else if (Y[i] > Y_max[i])
    {
      adjusted = true;
      Y[i]     = Y_max[i];
    }
  }
  return adjusted;
}

/// Perform failure response.
/// In this case, we simply floor everything.
bool ResetFloorTransition::FailureResponse(Real prim[NPRIM], int n_species)
{
  prim[IDN] = n_atm;
  prim[IVX] = 0.0;
  prim[IVY] = 0.0;
  prim[IVZ] = 0.0;
  prim[ITM] = T_atm;
  for (int i = 0; i < n_species; i++)
  {
    prim[IYF + i] = Y_atm[i];
  }
  return true;
}

/// Cap momentum to enforce S^i S_i <= Ssq_max.
/// Rescales all three S_i components uniformly by sqrt(Ssq_max / Ssq),
/// preserving the momentum direction while clamping the magnitude.
bool ResetFloorTransition::MomentumLimits(Real Sd[3], Real Ssq, Real Ssq_max)
{
  if (limit_momenta && Ssq > Ssq_max && Ssq > 0.0)
  {
    Real factor = std::sqrt(Ssq_max / Ssq);
    Sd[0] *= factor;
    Sd[1] *= factor;
    Sd[2] *= factor;
    return true;
  }
  return false;
}
