//! \file eos_transition.cpp
//  \brief Implementation of EOSTransition

#include "eos_transition.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "unit_system.hpp"

using namespace Primitive;
using namespace std;

EOSTransition::EOSTransition()
{
  compose_eos   = new EOSCompOSE();
  helmholtz_eos = new EOSHelmholtz();
  n_species     = 7;
  eos_units     = &Nuclear;
  min_Y[SCYE]   = 0.0;  // will be overwritten by update_bounds
  min_Y[SCXN]   = 0.0;
  min_Y[SCXP]   = 0.0;
  min_Y[SCXA]   = 0.0;
  min_Y[SCXH]   = 0.0;
  min_Y[SCAH]   = 1.0;
  min_Y[SCEB]   = 0.0;

  max_Y[SCYE] = 1.0;  // will be overwritten by update_bounds
  max_Y[SCXN] = 1.0;
  max_Y[SCXP] = 1.0;
  max_Y[SCXA] = 1.0;
  max_Y[SCXH] = 1.0;
  max_Y[SCAH] = 300.0;  // RHINE network domain cap (rhine_optim.hpp)
  max_Y[SCEB] = 1e-2;  // will be overwritten by SetBaryonMass

  m_min_h          = numeric_limits<Real>::max();
  m_trans_T_width  = numeric_limits<Real>::quiet_NaN();
  m_trans_ln_width = numeric_limits<Real>::quiet_NaN();
  trans_T_start    = numeric_limits<Real>::quiet_NaN();
  trans_T_end      = numeric_limits<Real>::quiet_NaN();
  trans_ln_start   = numeric_limits<Real>::quiet_NaN();
  trans_ln_end     = numeric_limits<Real>::quiet_NaN();
  m_helm_n_max     = numeric_limits<Real>::quiet_NaN();
  m_helm_T_max     = numeric_limits<Real>::quiet_NaN();
  // validity-ramp widths below the helmholtz cutoffs
  m_helm_n_ramp_dec = 1.0;
  m_helm_T_ramp_dec = 0.5;
  m_ln_n_h0         = numeric_limits<Real>::quiet_NaN();
  m_ln_T_h0         = numeric_limits<Real>::quiet_NaN();
  m_id_ln_n_ramp    = numeric_limits<Real>::quiet_NaN();
  m_id_ln_T_ramp    = numeric_limits<Real>::quiet_NaN();
  m_initialized     = false;
  root.iterations   = 500;
}

bool EOSTransition::s_printed_parameters      = false;
bool EOSTransition::s_printed_nucleon_masses  = false;

EOSTransition::~EOSTransition()
{
  delete compose_eos;
  delete helmholtz_eos;
}

Real EOSTransition::TemperatureFromEps(Real n, Real eps, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->TemperatureFromEps(n, eps, &yq);

  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  return TemperatureFromEpsSanitized(n, eps, Y_norm);
}

Real EOSTransition::TemperatureFromEpsSanitized(Real n, Real eps, Real* Y_norm)
{
  if (n < compose_eos->min_n)
    return helmholtz_eos->TemperatureFromEps(n, eps, Y_norm);
  // Bounds are evaluated lazily: floor-clamped states (the atmosphere) hit
  // the eps_min early-out, so the max bound is only computed when needed.
  Real eps_min = MinimumSpecificInternalEnergySanitized(n, Y_norm);
  if (eps <= eps_min)
    return (log(n) >= m_ln_n_h0) ? compose_eos->min_T : min_T;
  Real eps_max = MaximumSpecificInternalEnergySanitized(n, Y_norm);
  if (eps >= eps_max)
    return max_T;

  // Pure-regime candidates: invert each sub-EOS and accept the result if
  // it is self-consistent with the transition weight there. (The blended
  // eps(T) coincides with the sub-EOS wherever w is 0 or 1, and eps is
  // monotone in T.) Inside the density-validity ramp the weight is
  // nonzero for every valid temperature, and a Helmholtz inverse clamped
  // to its table minimum would spuriously pass the w == 0 check through
  // the compose-validity clause, so the Helmholtz candidate is skipped
  // there.
  if (log(n) < m_ln_n_h0)
  {
    Real T_h = helmholtz_eos->TemperatureFromEps(n, eps, Y_norm);
    if (TransitionFactor(n, T_h) == 0.0)
      return T_h;
  }
  // Search the blended region before accepting a pure-compose root: the
  // blended eps(T) can be non-monotone (binding-energy release across the
  // strip), and the lowest root is the deterministic choice.
  Real T_b = temperature_from_var_trans(
    compose_eos->ECLOGE, log(n * mb * (1 + eps)), n, Y_norm);
  if (!std::isnan(T_b))
    return T_b;
  Real T_c = compose_eos->TemperatureFromEps(n, eps, Y_norm);
  if (TransitionFactor(n, T_c) == 1.0)
    return T_c;

  // Falling back to the compose inverse is an expected outcome for some
  // surface/floor states, not a per-cell error -- warn once instead of
  // flooding stdout every cell, every substep, every rank (an I/O stall in
  // production, especially over a networked filesystem).
  static bool warned = false;
  if (!warned)
  {
    warned = true;
    printf(
      "EOSTransition::TemperatureFromEps: no consistent root (first "
      "occurrence at n = %.5e, eps = %.5e); returning the compose inverse "
      "%.5e. Silencing further reports.\n",
      n,
      eps,
      T_c);
  }
  return T_c;
}

Real EOSTransition::TemperatureFromEntropy(Real n, Real s, Real* Y)
{
  throw std::logic_error(
    "EOSTransition::TemperatureFromEntropy not currently implemented.");
}

Real EOSTransition::TemperatureFromE(Real n, Real e, Real* Y)
{
  assert(m_initialized);
  if (n > m_helm_n_max)
    return compose_eos->TemperatureFromE(n, e, Y);
  return TemperatureFromEps(n, e / (mb * n) - 1.0, Y);
}

Real EOSTransition::TemperatureFromP(Real n, Real p, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->TemperatureFromP(n, p, &yq);

  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  return TemperatureFromPSanitized(n, p, Y_norm);
}

Real EOSTransition::TemperatureFromPSanitized(Real n, Real p, Real* Y_norm)
{
  if (n < compose_eos->min_n)
    return helmholtz_eos->TemperatureFromP(n, p, Y_norm);
  // Lazy bounds, as in TemperatureFromEpsSanitized.
  Real p_min = MinimumPressureSanitized(n, Y_norm);
  if (p <= p_min)
    return (log(n) >= m_ln_n_h0) ? compose_eos->min_T : min_T;
  Real p_max = MaximumPressureSanitized(n, Y_norm);
  if (p >= p_max)
    return max_T;

  // Pure-regime candidates and blended search (see TemperatureFromEps).
  if (log(n) < m_ln_n_h0)
  {
    Real T_h = helmholtz_eos->TemperatureFromP(n, p, Y_norm);
    if (TransitionFactor(n, T_h) == 0.0)
      return T_h;
  }
  Real T_b = temperature_from_var_trans(compose_eos->ECLOGP, log(p), n, Y_norm);
  if (!std::isnan(T_b))
    return T_b;
  Real T_c = compose_eos->TemperatureFromP(n, p, Y_norm);
  if (TransitionFactor(n, T_c) == 1.0)
    return T_c;

  // See TemperatureFromEps: warn once, not per cell.
  static bool warned = false;
  if (!warned)
  {
    warned = true;
    printf(
      "EOSTransition::TemperatureFromP: no consistent root (first occurrence "
      "at n = %.5e, p = %.5e); returning the compose inverse %.5e. Silencing "
      "further reports.\n",
      n,
      p,
      T_c);
  }
  return T_c;
}

Real EOSTransition::SanitizeMassFractions(Real* Y, Real* Y_norm) const
{
  for (int i = 0; i < SCNVAR; ++i)
  {
    Y_norm[i] = max(min_Y[i], min(Y[i], max_Y[i]));
  }
  // Normalize the clamped fractions; using the raw values here could
  // produce normalized fractions outside [0, 1].
  Real Xsum = Y_norm[SCXN] + Y_norm[SCXP] + Y_norm[SCXA] + Y_norm[SCXH];
  if (Xsum <= 0.0)
  {
    // Degenerate composition (e.g. a floored-atmosphere cell whose mass
    // fractions are all zero) is an expected fallback, not an error -- warn
    // once instead of flooding stdout every cell, every substep, every rank.
    static bool warned = false;
    if (!warned)
    {
      warned = true;
      printf(
        "EOSTransition::SanitizeMassFractions: got invalid mass fractions "
        "(sum <= 0, first occurrence sum = %.5e); falling back to free "
        "nucleons at Ye = 0.5. Silencing further reports.\n",
        Xsum);
    }
    // Fall back to free nucleons at Ye = 0.5.
    Y_norm[SCYE] = 0.5;
    Y_norm[SCXN] = 0.5;
    Y_norm[SCXP] = 0.5;
    Y_norm[SCXA] = 0.0;
    Y_norm[SCXH] = 0.0;
    return 1.0;
  }
  Y_norm[SCXN] /= Xsum;
  Y_norm[SCXP] /= Xsum;
  Y_norm[SCXA] /= Xsum;
  Y_norm[SCXH] /= Xsum;
  // Charge consistency: the heavy-nucleus charge implied by (Ye, X) is
  // q_h = Ye - X_p - X_a/2 and must lie in [0, X_h/2] (0 <= Z_h <= A_h/2).
  // Advection keeps real states within a few percent; a gross violation is
  // a floor-clamp chimera (e.g. pure alpha at Ye = 0.01) that poisons the
  // RHINE rates and the Helmholtz abar/zbar. Remap those to free nucleons
  // at the cell's Ye.
  const Real q_h = Y_norm[SCYE] - Y_norm[SCXP] - 0.5 * Y_norm[SCXA];
  if (q_h < -0.05 || q_h > 0.5 * Y_norm[SCXH] + 0.05)
  {
    Y_norm[SCXN] = 1.0 - Y_norm[SCYE];
    Y_norm[SCXP] = Y_norm[SCYE];
    Y_norm[SCXA] = 0.0;
    Y_norm[SCXH] = 0.0;
  }
  return Xsum;
}

Real EOSTransition::TransitionFactor(Real n, Real T) const
{
  return TransitionFactor(n, T, log(n), log(T));
}

// Log-space variant: callers that already hold ln = log(n) and lT = log(T)
// (e.g. the strip solver, where lT is a compose grid node m_log_t[it] and
// ln is loop-invariant) pass them in to avoid recomputing the logs.
Real EOSTransition::TransitionFactor(Real n, Real T, Real ln, Real lT) const
{
  if ((n > m_helm_n_max) or (T > m_helm_T_max))
  {
    return 1.0;
  }
  if ((n < compose_eos->min_n) or (T < compose_eos->min_T))
  {
    return 0.0;
  }
  // Freeze-out strip (NSE dropout in temperature, with the density strip
  // acting as a fallback at the compose table's low-density edge).
  Real u = min(1.0, max(0.0, (T - trans_T_end) / m_trans_T_width));
  u *= min(1.0, max(0.0, (ln - trans_ln_end) / m_trans_ln_width));
  // Validity ramps toward the Helmholtz cutoffs: the weight blends
  // continuously to 1 at the cutoffs instead of jumping there ("fuzzy
  // OR" of the strip and the two ramps).
  Real v_n = min(1.0, max(0.0, (ln - m_ln_n_h0) * m_id_ln_n_ramp));
  Real v_T = min(1.0, max(0.0, (lT - m_ln_T_h0) * m_id_ln_T_ramp));
  return 1.0 - (1.0 - u) * (1.0 - v_n) * (1.0 - v_T);
}

Real EOSTransition::Energy(Real n, Real T, Real* Y)
{
  return (SpecificInternalEnergy(n, T, Y) + 1.0) * mb * n;
}

Real EOSTransition::Pressure(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->Pressure(n, T, &yq);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  return PressureSanitized(n, T, Y_norm);
}

Real EOSTransition::PressureSanitized(Real n, Real T, Real* Y_norm)
{
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->Pressure(n, T, Y_norm);
  if (w == 0.0)
    return helmholtz_eos->Pressure(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->Pressure(n, T, Y_norm);
  Real v_compose   = compose_eos->Pressure(n, T, Y_norm);
  return v_helmholtz * (1 - w) + v_compose * w;
}

Real EOSTransition::Entropy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->Entropy(n, T, &yq);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->Entropy(n, T, Y_norm);
  if (w == 0.0)
    return helmholtz_eos->Entropy(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->Entropy(n, T, Y_norm);
  Real v_compose   = compose_eos->Entropy(n, T, Y_norm);
  return v_helmholtz * (1 - w) + v_compose * w;
}

Real EOSTransition::Enthalpy(Real n, Real T, Real* Y)
{
  Real P, h;
  PressureAndEnthalpy(n, T, Y, &P, &h);
  return h;
}

Real EOSTransition::SoundSpeed(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->SoundSpeed(n, T, &yq);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->SoundSpeed(n, T, Y_norm);
  if (w == 0.0)
    return helmholtz_eos->SoundSpeed(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->SoundSpeed(n, T, Y_norm);
  Real v_compose   = compose_eos->SoundSpeed(n, T, Y_norm);
  return v_helmholtz * (1 - w) + v_compose * w;
}

Real EOSTransition::SpecificInternalEnergy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->SpecificInternalEnergy(n, T, &yq);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  return SpecificInternalEnergySanitized(n, T, Y_norm);
}

Real EOSTransition::SpecificInternalEnergySanitized(Real n, Real T,
                                                    Real* Y_norm)
{
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->SpecificInternalEnergy(n, T, Y_norm);
  if (w == 0.0)
    return helmholtz_eos->SpecificInternalEnergy(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->SpecificInternalEnergy(n, T, Y_norm);
  Real v_compose   = compose_eos->SpecificInternalEnergy(n, T, Y_norm);
  return v_helmholtz * (1 - w) + v_compose * w;
}

Real EOSTransition::BaryonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->BaryonChemicalPotential(n, T, &yq);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->BaryonChemicalPotential(n, T, Y_norm);
  if (w == 0.0)
    return helmholtz_eos->BaryonChemicalPotential(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->BaryonChemicalPotential(n, T, Y_norm);
  Real v_compose   = compose_eos->BaryonChemicalPotential(n, T, Y_norm);
  return v_helmholtz * (1 - w) + v_compose * w;
}

Real EOSTransition::ChargeChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->ChargeChemicalPotential(n, T, &yq);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->ChargeChemicalPotential(n, T, Y_norm);
  if (w == 0.0)
    return helmholtz_eos->ChargeChemicalPotential(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->ChargeChemicalPotential(n, T, Y_norm);
  Real v_compose   = compose_eos->ChargeChemicalPotential(n, T, Y_norm);
  return v_helmholtz * (1 - w) + v_compose * w;
}

Real EOSTransition::ElectronLeptonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
    return compose_eos->ElectronLeptonChemicalPotential(n, T, &yq);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  if (w == 0.0)
    return helmholtz_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  Real v_helmholtz =
    helmholtz_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  Real v_compose = compose_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  return v_helmholtz * (1 - w) + v_compose * w;
}

Real EOSTransition::InteractionPotentialDifference(Real n, Real T, Real* Y)
{
  Real w = TransitionFactor(n, T);
  if (w == 0.0)
    return 0.0;
  Real dU = compose_eos->InteractionPotentialDifference(n, T, Y);
  return w * dU;
}

Real EOSTransition::FrYn(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->FrYn(n, T, Y);
  return Y[SCXN];
}

Real EOSTransition::FrYp(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->FrYp(n, T, Y);
  return Y[SCXP];
}

Real EOSTransition::FrXa(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->FrXa(n, T, Y);
  return Y[SCXA];
}

Real EOSTransition::FrXh(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->FrXh(n, T, Y);
  return Y[SCXH];
}

Real EOSTransition::AN(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->AN(n, T, Y);
  return Y[SCAH];
}

Real EOSTransition::ZN(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
    return compose_eos->ZN(n, T, Y);
  return (Y[SCXH] > 0) ? (Y[SCYE] - Y[SCXP] - Y[SCXA] / 2) / Y[SCXH] * Y[SCAH]
                       : 0.0;
}

Real EOSTransition::MinimumEnthalpy()
{
  return m_min_h;
}

// The lowest valid temperature is the compose grid minimum once the
// density enters the validity ramp toward the Helmholtz cutoff (there the
// weight is nonzero for every T, so the compose table must be evaluable);
// below the ramp it is the Helmholtz table minimum. The highest valid
// temperature is always the compose maximum, since the weight reaches 1
// at (or above) the Helmholtz temperature cutoff.
Real EOSTransition::MinimumPressure(Real n, Real* Y)
{
  if (log(n) >= m_ln_n_h0)
  {
    return Pressure(n, compose_eos->min_T, Y);
  }
  return Pressure(n, min_T, Y);
}

Real EOSTransition::MinimumPressureSanitized(Real n, Real* Y_norm)
{
  return PressureSanitized(
    n, (log(n) >= m_ln_n_h0) ? compose_eos->min_T : min_T, Y_norm);
}

Real EOSTransition::MaximumPressure(Real n, Real* Y)
{
  return Pressure(n, max_T, Y);
}

Real EOSTransition::MaximumPressureSanitized(Real n, Real* Y_norm)
{
  return PressureSanitized(n, max_T, Y_norm);
}

Real EOSTransition::MinimumEntropy(Real n, Real* Y)
{
  if (log(n) >= m_ln_n_h0)
  {
    return Entropy(n, compose_eos->min_T, Y);
  }
  return Entropy(n, min_T, Y);
}

Real EOSTransition::MaximumEntropy(Real n, Real* Y)
{
  return Entropy(n, max_T, Y);
}

Real EOSTransition::MinimumSpecificInternalEnergy(Real n, Real* Y)
{
  if (log(n) >= m_ln_n_h0)
  {
    return SpecificInternalEnergy(n, compose_eos->min_T, Y);
  }
  return SpecificInternalEnergy(n, min_T, Y);
}

Real EOSTransition::MinimumSpecificInternalEnergySanitized(Real n,
                                                           Real* Y_norm)
{
  return SpecificInternalEnergySanitized(
    n, (log(n) >= m_ln_n_h0) ? compose_eos->min_T : min_T, Y_norm);
}

Real EOSTransition::MaximumSpecificInternalEnergy(Real n, Real* Y)
{
  return SpecificInternalEnergy(n, max_T, Y);
}

Real EOSTransition::MaximumSpecificInternalEnergySanitized(Real n,
                                                           Real* Y_norm)
{
  return SpecificInternalEnergySanitized(n, max_T, Y_norm);
}

Real EOSTransition::MinimumEnergy(Real n, Real* Y)
{
  return (MinimumSpecificInternalEnergy(n, Y) + 1.0) * mb * n;
}

Real EOSTransition::MaximumEnergy(Real n, Real* Y)
{
  return (MaximumSpecificInternalEnergy(n, Y) + 1.0) * mb * n;
}

void EOSTransition::SetTransition(Real n_start,
                                  Real n_end,
                                  Real T_start,
                                  Real T_end)
{
  if (m_initialized)
  {
    std::stringstream msg;
    msg << "### EOSTransition: Transition must be set before initialization."
        << std::endl;
    throw std::runtime_error(msg.str());
  }

  if (n_start <= n_end)
  {
    std::stringstream msg;
    msg << "### EOSTransition: density transition start: " << n_start
        << " is not larger than end: " << n_end << std::endl;
    throw std::runtime_error(msg.str());
  }

  if (T_start <= T_end)
  {
    std::stringstream msg;
    msg << "### EOSTransition: temperature transition start: " << T_start
        << " is not larger than end: " << T_end << std::endl;
    throw std::runtime_error(msg.str());
  }

  trans_ln_start   = log(n_start);
  trans_ln_end     = log(n_end);
  m_trans_ln_width = log(n_start / n_end);

  trans_T_start   = T_start;
  trans_T_end     = T_end;
  m_trans_T_width = T_start - T_end;
}

void EOSTransition::PrintParameters()
{
#pragma omp critical
  {
    if ((not s_printed_parameters) and (Globals::my_rank == 0))
    {
      printf("EOSTransition:\n");
      printf("  n min, max = %e %e\n", min_n, max_n);
      printf("  T min, max = %e %e\n", min_T, max_T);
      printf("  Ye min, max = %e %e\n", min_Y[SCYE], max_Y[SCYE]);
      printf("  Xn min, max = %e %e\n", min_Y[SCXN], max_Y[SCXN]);
      printf("  Xp min, max = %e %e\n", min_Y[SCXP], max_Y[SCXP]);
      printf("  Xa min, max = %e %e\n", min_Y[SCXA], max_Y[SCXA]);
      printf("  Xh min, max = %e %e\n", min_Y[SCXH], max_Y[SCXH]);
      printf("  Ah min, max = %e %e\n", min_Y[SCAH], max_Y[SCAH]);
      printf("  helm n_max, t_max = %e %e\n", m_helm_n_max, m_helm_T_max);
      printf("  comp n_min, t_min = %e %e\n",
             compose_eos->min_n,
             compose_eos->min_T);
      printf("  min_h = %.15e\n", m_min_h);
      printf("  mb = %.15e MeV\n", mb);
      printf("  mn, mp = %.7f %.7f MeV (Qnp = %.7f)\n",
             EOSHelmholtz::mn,
             EOSHelmholtz::mp,
             EOSHelmholtz::mn - EOSHelmholtz::mp);
      printf(
        "  T transition start, end = %e %e\n", trans_T_start, trans_T_end);
      printf("  n transition start, end = %e %e\n",
             exp(trans_ln_start),
             exp(trans_ln_end));
      printf("  i_t helm_tmax, trans_start, trans_end = %d %d %d\n",
             comp_it_helm_tmax,
             comp_it_trans_start,
             comp_it_trans_end);
      s_printed_parameters = true;
    }
  }
}

void EOSTransition::SetBaryonMass(Real new_mb)
{
  helmholtz_eos->SetBaryonMass(new_mb);
  compose_eos->SetBaryonMass(new_mb);
  min_Y[SCEB] = mFe / (56.0 * new_mb) - 1.0;  // most bound nucleus is Fe-56
  max_Y[SCEB] = EOSHelmholtz::mn / new_mb - 1.0;  // free neutron limit
  // Minimum enthalpy per baryon (in MeV, converted to per-mass by the
  // EOS wrapper): cold, pressureless Fe-56, h = e/n = m(Fe56)/56.
  // N.B. not min_Y[SCEB], which is the binding *excess* (h/mb - 1).
  m_min_h = mFe / 56.0;
  mb      = new_mb;
}

void EOSTransition::update_bounds()
{
  // Check and update bounds
  // -------------------------------------------------------------------------

  min_n       = helmholtz_eos->min_n;
  min_T       = helmholtz_eos->min_T;
  max_n       = compose_eos->max_n;
  max_T       = compose_eos->max_T;
  min_Y[SCYE] = compose_eos->min_Y[SCYE];
  max_Y[SCYE] = compose_eos->max_Y[SCYE];

  // If not set by the user, choose the Helmholtz cutoffs. The density
  // cutoff protects the high-density regime (star, disk), which must use
  // the NSE table regardless of temperature; 1e-6 fm^-3 (~1.6e9 g/cc) is
  // about an order of magnitude above typical NSE freeze-out densities
  // and well below the crust. It is a modeling choice and can be
  // overridden with SetHelmholtzNMax.
  if (isnan(m_helm_n_max))
    m_helm_n_max = min(helmholtz_eos->max_n, 1e-6);
  if (isnan(m_helm_T_max))
    m_helm_T_max = min(helmholtz_eos->max_T, compose_eos->max_T);

  // Validity ramps toward the cutoffs (see TransitionFactor).
  Real const ln10 = log(10.0);
  m_ln_n_h0       = log(m_helm_n_max) - m_helm_n_ramp_dec * ln10;
  m_id_ln_n_ramp  = 1.0 / (m_helm_n_ramp_dec * ln10);
  m_ln_T_h0       = log(m_helm_T_max) - m_helm_T_ramp_dec * ln10;
  m_id_ln_T_ramp  = 1.0 / (m_helm_T_ramp_dec * ln10);

  // The transition strips must lie inside the domain where both tables
  // are evaluable, otherwise the blended weight is discontinuous at the
  // table edges (and the compose table would be extrapolated).
  Real const edge_tol = 1.0 - 1e-10;
  if (trans_T_end < compose_eos->min_T * edge_tol)
  {
    std::stringstream msg;
    msg << "### EOSTransition: trans_T_end = " << trans_T_end
        << " lies below the compose table minimum temperature "
        << compose_eos->min_T << std::endl;
    throw std::runtime_error(msg.str());
  }
  if (trans_T_start > min(compose_eos->max_T, helmholtz_eos->max_T))
  {
    std::stringstream msg;
    msg << "### EOSTransition: trans_T_start = " << trans_T_start
        << " lies above the joint table maximum temperature "
        << min(compose_eos->max_T, helmholtz_eos->max_T) << std::endl;
    throw std::runtime_error(msg.str());
  }
  if (exp(trans_ln_end) < compose_eos->min_n * edge_tol)
  {
    std::stringstream msg;
    msg << "### EOSTransition: trans_n_end = " << exp(trans_ln_end)
        << " lies below the compose table minimum density "
        << compose_eos->min_n << std::endl;
    throw std::runtime_error(msg.str());
  }
  if (exp(trans_ln_start) > min(compose_eos->max_n, helmholtz_eos->max_n))
  {
    std::stringstream msg;
    msg << "### EOSTransition: trans_n_start = " << exp(trans_ln_start)
        << " lies above the joint table maximum density "
        << min(compose_eos->max_n, helmholtz_eos->max_n) << std::endl;
    throw std::runtime_error(msg.str());
  }
  if (m_helm_n_max > helmholtz_eos->max_n)
  {
    std::stringstream msg;
    msg << "### EOSTransition: helm_n_max = " << m_helm_n_max
        << " lies above the helmholtz table maximum density "
        << helmholtz_eos->max_n << std::endl;
    throw std::runtime_error(msg.str());
  }
  if (m_helm_n_max < exp(trans_ln_start))
    printf(
      "EOSTransition::update_bounds: warning: helmholtz density cutoff = "
      "%.5e is less than the transition density start = %.5e; the "
      "density strip is cut short\n",
      m_helm_n_max,
      exp(trans_ln_start));

  // indices for transition and helmholtz max in compose table
  comp_it_trans_start =
    (log(trans_T_start) - compose_eos->m_log_t[0]) * compose_eos->m_id_log_t +
    1;
  comp_it_trans_end =
    (log(trans_T_end) - compose_eos->m_log_t[0]) * compose_eos->m_id_log_t;
  comp_it_helm_tmax =
    (log(m_helm_T_max) - compose_eos->m_log_t[0]) * compose_eos->m_id_log_t +
    1;

  // The transition boundaries may lie outside the compose table (e.g.
  // trans_T_end below the compose T-grid, which yields a negative index);
  // clamp so that temperature_from_var_trans never indexes out of bounds
  // and always has a bracket of at least one cell.
  int const it_max    = compose_eos->m_nt - 1;
  comp_it_trans_end   = max(0, min(comp_it_trans_end, it_max - 1));
  comp_it_trans_start = max(comp_it_trans_end + 1,
                            min(comp_it_trans_start, it_max));
  comp_it_helm_tmax   = max(comp_it_trans_end + 1,
                            min(comp_it_helm_tmax, it_max));
}

/// Fused temperature + pressure + enthalpy from energy: avoiding redundant
/// lookups
void EOSTransition::TemperaturePressureAndEnthalpyFromE(Real n,
                                                        Real e,
                                                        Real* Y,
                                                        Real* T,
                                                        Real* P,
                                                        Real* h,
                                                        int* guess_it)
{
  // Pure-NSE interior (n above the Helmholtz cutoff => weight == 1 for every
  // T): the star bulk. Use the fused compose path, which inverts T and looks
  // up P/e with shared interpolation weights and a guess_it-warm-started
  // bracket -- instead of inverting T, discarding the weights, then
  // recomputing them for a separate P/e lookup.
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
  {
    compose_eos->TemperaturePressureAndEnthalpyFromE(
      n, e, &yq, T, P, h, guess_it);
    return;
  }
  // Below the cutoff (InteriorYq false <=> n <= m_helm_n_max, the same
  // branch TemperatureFromE takes): sanitize once and reuse for both the
  // inversion and the P/h evaluation.
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  *T = TemperatureFromEpsSanitized(n, e / (mb * n) - 1.0, Y_norm);
  PressureAndEnthalpySanitized(n, *T, Y_norm, P, h);
}

void EOSTransition::PressureAndEnthalpyFromE(Real n,
                                             Real e,
                                             Real* Y,
                                             Real* P,
                                             Real* h,
                                             int* guess_it)
{
  // Interior fast path (see TemperaturePressureAndEnthalpyFromE). This is the
  // hot path of the conservative-to-primitive root find.
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
  {
    compose_eos->PressureAndEnthalpyFromE(n, e, &yq, P, h, guess_it);
    return;
  }
  // Sanitize once for the whole chain (see
  // TemperaturePressureAndEnthalpyFromE).
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real T = TemperatureFromEpsSanitized(n, e / (mb * n) - 1.0, Y_norm);
  PressureAndEnthalpySanitized(n, T, Y_norm, P, h);
}

/// Fused pressure + enthalpy: sanitize the composition and compute the
/// transition weight only once, and evaluate each sub-EOS at most once.
/// This is the hot path of the conservative-to-primitive root find.
void EOSTransition::PressureAndEnthalpy(Real n,
                                        Real T,
                                        Real* Y,
                                        Real* P,
                                        Real* h)
{
  assert(m_initialized);
  Real yq;
  if (InteriorYq(n, Y, yq))
  {
    compose_eos->PressureAndEnthalpy(n, T, &yq, P, h);
    return;
  }
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  PressureAndEnthalpySanitized(n, T, Y_norm, P, h);
}

void EOSTransition::PressureAndEnthalpySanitized(Real n, Real T,
                                                 Real* Y_norm,
                                                 Real* P, Real* h)
{
  Real w = TransitionFactor(n, T);
  if (w == 1.0)
  {
    compose_eos->PressureAndEnthalpy(n, T, Y_norm, P, h);
    return;
  }
  Real P_pt, e_pt;
  if (w == 0.0)
  {
    P_pt = helmholtz_eos->Pressure(n, T, Y_norm);
    e_pt = helmholtz_eos->Energy(n, T, Y_norm);
  }
  else
  {
    Real P_h   = helmholtz_eos->Pressure(n, T, Y_norm);
    Real P_c   = compose_eos->Pressure(n, T, Y_norm);
    Real eps_h = helmholtz_eos->SpecificInternalEnergy(n, T, Y_norm);
    Real eps_c = compose_eos->SpecificInternalEnergy(n, T, Y_norm);
    P_pt       = P_h * (1 - w) + P_c * w;
    e_pt       = (eps_h * (1 - w) + eps_c * w + 1.0) * mb * n;
  }
  *P = P_pt;
  *h = (P_pt + e_pt) / n;
}

Real EOSTransition::GetNSEBindingEnergy(Real n, Real T, Real* Y)
{
  if (n > helmholtz_eos->max_n or T > helmholtz_eos->max_T)
  {
    return 0.0;
  }
  else if (n < compose_eos->min_n or T < compose_eos->min_T)
  {
    return Y[SCEB];
  }
  else
  {
    Real Y_NSE[SCNVAR];
    Y_NSE[SCYE] = Y[SCYE];
    Y_NSE[SCXN] = compose_eos->FrYn(n, T, Y);
    Y_NSE[SCXP] = compose_eos->FrYp(n, T, Y);
    Y_NSE[SCXA] = compose_eos->FrXa(n, T, Y);
    Y_NSE[SCXH] = compose_eos->FrXh(n, T, Y);
    Y_NSE[SCAH] = compose_eos->AN(n, T, Y);
    Y_NSE[SCEB] = 0.0;

    Real eps_helm = helmholtz_eos->SpecificInternalEnergy(n, T, Y_NSE);
    Real eps_comp = compose_eos->SpecificInternalEnergy(n, T, Y);

    Real Zh = (Y_NSE[SCXH] > 0)
              ? (Y_NSE[SCYE] - Y_NSE[SCXP] - Y_NSE[SCXA] / 2) / Y_NSE[SCXH] *
                  Y_NSE[SCAH]
              : 0.0;
    Real Ah = Y_NSE[SCAH];
    // Bounds on the mass excess per baryon. The compose table energies are
    // in the ATOMIC mass convention (electron rest masses included), so use
    // atomic hydrogen/helium masses for the light species; mFe is already
    // the atomic 56Fe mass. Both bounds are mass fractions times mass per
    // BARYON of each species.
    Real const mn   = EOSHelmholtz::mn;
    Real const mH   = EOSHelmholtz::mp + me;  // atomic 1H
    Real const yq_h = (Ah > 0) ? Zh / Ah : 0.5;
    Real min_EB = (Y_NSE[SCXN] * mn + Y_NSE[SCXP] * mH +
                   Y_NSE[SCXA] * (ma + 2 * me) / 4 +
                   Y_NSE[SCXH] * mFe / 56.0) /
                    mb -
                  1;
    // upper bound: everything dissociated into free nucleons (+ electrons)
    Real max_EB = (Y_NSE[SCXN] * mn + Y_NSE[SCXP] * mH +
                   Y_NSE[SCXA] * (mn + mH) / 2 +
                   Y_NSE[SCXH] * ((1 - yq_h) * mn + yq_h * mH)) /
                    mb -
                  1;
    Real eb     = max(min_EB, min(eps_comp - eps_helm, max_EB));
    return eb;
  }
}

void EOSTransition::SyncNucleonMasses()
{
  // Deviation from the CODATA defaults that is worth reporting. Real tables
  // sit ~1e-4 MeV away; anything larger suggests a different convention in
  // the table (CompOSE stores the baryon mass in the same "mn" dataset).
  constexpr Real tol = 1e-3;  // MeV

  Real const table_mn = compose_eos->GetTableNeutronMass();
  Real const table_mp = compose_eos->GetTableProtonMass();

  bool const no_table_masses = std::isnan(table_mn) or std::isnan(table_mp);
  bool const deviates =
    (std::abs(table_mn - EOSHelmholtz::mn_codata) >= tol) or
    (std::abs(table_mp - EOSHelmholtz::mp_codata) >= tol);

  // One EOSTransition is constructed per MeshBlock, report at most once.
#pragma omp critical
  {
    if ((not s_printed_nucleon_masses) and (Globals::my_rank == 0))
    {
      if (no_table_masses)
      {
        printf(
          "EOSTransition: compose table carries no nucleon masses, "
          "keeping CODATA (mn = %.7f, mp = %.7f MeV)\n",
          EOSHelmholtz::mn_codata,
          EOSHelmholtz::mp_codata);
        s_printed_nucleon_masses = true;
      }
      else if (deviates)
      {
        printf(
          "EOSTransition: compose table nucleon masses deviate from CODATA "
          "by more than %.0e MeV, using the table values.\n"
          "  mn: table %.7f vs CODATA %.7f MeV\n"
          "  mp: table %.7f vs CODATA %.7f MeV\n"
          "  Qnp = mn - mp: table %.7f vs CODATA %.7f MeV\n",
          tol,
          table_mn,
          EOSHelmholtz::mn_codata,
          table_mp,
          EOSHelmholtz::mp_codata,
          table_mn - table_mp,
          EOSHelmholtz::mn_codata - EOSHelmholtz::mp_codata);
        s_printed_nucleon_masses = true;
      }
    }
  }

  if (no_table_masses)
  {
    return;
  }

  helmholtz_eos->SetNucleonMasses(table_mn, table_mp);
}

void EOSTransition::InitializeTables(std::string fname,
                                     std::string helm_fname,
                                     Real baryon_mass)
{
  if (not m_initialized)
  {
    compose_eos->ReadTableFromFile(fname);
    helmholtz_eos->ReadTableFromFile(
      helm_fname, compose_eos->min_Y[SCYE], compose_eos->max_Y[SCYE]);
    // Default transition strips if the user did not call SetTransition:
    //  - Temperature: 0.5 -> 0.6 MeV, just below the NSE-dropout condition
    //    used by RHINE and most nucleosynthesis tools (T9 = 7, i.e.
    //    7 GK ~ 0.6 MeV). This is the primary transition.
    //  - Density: one decade above the compose table's low-density edge;
    //    a fallback that should only rarely be active.
    if (std::isnan(m_trans_T_width) or std::isnan(m_trans_ln_width))
    {
      SetTransition(10.0 * compose_eos->min_n, compose_eos->min_n, 0.6, 0.5);
    }
    // Before SetBaryonMass: max_Y[SCEB] is built from the neutron mass.
    SyncNucleonMasses();
    SetBaryonMass(baryon_mass);
    update_bounds();
    PrintParameters();
    m_initialized = true;
  }
  else
  {
    std::stringstream msg;
    msg << "### EOSTransition: InitializeTables should only be called once."
        << std::endl;
    throw std::runtime_error(msg.str());
  }
}

Real EOSTransition::temperature_from_var_trans(int iv,
                                               Real var,
                                               Real n,
                                               Real* Y) const
{
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  Real Yq         = Y[SCYE];
  Real const ln_n = log(n);
  compose_eos->weight_idx_ln(&wn0, &wn1, &in, ln_n);
  compose_eos->weight_idx_yq(&wy0, &wy1, &iy, Yq);

  auto f = [=](int it)
  {
    // lT = m_log_t[it] is log(T) exactly, so pass it (and the hoisted
    // ln_n) to the log-space TransitionFactor instead of exp-then-re-log.
    Real lT = compose_eos->m_log_t[it];
    Real T  = exp(lT);
    Real w  = TransitionFactor(n, T, ln_n, lT);
    Real var_helm;
    if (iv == compose_eos->ECLOGP)
      var_helm = helmholtz_eos->Pressure(n, T, Y);
    else if (iv == compose_eos->ECLOGE)
      var_helm = helmholtz_eos->Energy(n, T, Y);
    else
      throw std::logic_error(
        "EOSTransition::temperature_from_var_trans only implemented for "
        "log(P) and log(e).");

    Real var_comp =
      wn0 *
        (wy0 *
           compose_eos->m_table[compose_eos->index(iv, in + 0, iy + 0, it)] +
         wy1 *
           compose_eos->m_table[compose_eos->index(iv, in + 0, iy + 1, it)]) +
      wn1 *
        (wy0 *
           compose_eos->m_table[compose_eos->index(iv, in + 1, iy + 0, it)] +
         wy1 *
           compose_eos->m_table[compose_eos->index(iv, in + 1, iy + 1, it)]);

    return var - log(var_helm * (1 - w) + exp(var_comp) * w);
  };

  // Inside the density-validity ramp (n >= n_h0) the blend extends in
  // temperature all the way down to the compose grid; otherwise it starts
  // at the lower edge of the temperature strip.
  int ilo_0 = (ln_n >= m_ln_n_h0) ? 0 : comp_it_trans_end;
  // Below trans_ln_start the blend extends in temperature up to the
  // Helmholtz temperature cutoff; otherwise only the temperature strip
  // matters.
  int ihi_0 = (ln_n < trans_ln_start)
              ? min(comp_it_helm_tmax, compose_eos->m_nt - 1)
              : comp_it_trans_start;
  ihi_0 = max(ihi_0, ilo_0 + 1);

  // Scan for the *lowest* bracketing cell. Crossing the transition strip
  // releases the difference between the advected and the NSE binding
  // energy, so the blended e(T) need not be monotone and the inversion
  // can have several roots; the lowest one is continuous with the
  // low-temperature (Helmholtz) side and is the deterministic choice.
  int ilo  = ilo_0;
  int ihi  = ihi_0;
  Real flo = f(ilo);
  Real fhi = flo;
  bool bracketed = false;
  for (int i = ilo_0; i < ihi_0; ++i)
  {
    Real fnext = f(i + 1);
    if (flo * fnext <= 0)
    {
      ilo       = i;
      ihi       = i + 1;
      fhi       = fnext;
      bracketed = true;
      break;
    }
    flo = fnext;
  }
  if (!bracketed)
  {
    // No root in the blended bracket; the caller decides what to do
    // (typically fall back to the pure-compose candidate).
    return numeric_limits<Real>::quiet_NaN();
  }
  Real lthi = compose_eos->m_log_t[ihi];
  Real ltlo = compose_eos->m_log_t[ilo];

  if (flo == 0)
  {
    return exp(ltlo);
  }
  if (fhi == 0)
  {
    return exp(lthi);
  }

  // Pre-compute the four base offsets; it and it+1 are contiguous.
  ptrdiff_t const b00 = compose_eos->index(iv, in, iy, ilo);
  ptrdiff_t const b01 = compose_eos->index(iv, in, iy + 1, ilo);
  ptrdiff_t const b10 = compose_eos->index(iv, in + 1, iy, ilo);
  ptrdiff_t const b11 = compose_eos->index(iv, in + 1, iy + 1, ilo);

  Real* m_table = compose_eos->m_table;
  Real v0       = (wn0 * (wy0 * m_table[b00] + wy1 * m_table[b01]) +
                   wn1 * (wy0 * m_table[b10] + wy1 * m_table[b11]));
  Real v1       = (wn0 * (wy0 * m_table[b00 + 1] + wy1 * m_table[b01 + 1]) +
                   wn1 * (wy0 * m_table[b10 + 1] + wy1 * m_table[b11 + 1]));
  Real dv       = v1 - v0;

  Real lt0 = compose_eos->m_log_t[ilo];
  Real dlt = compose_eos->m_log_t[ihi] - compose_eos->m_log_t[ilo];

  Real wt;
  Real lb     = 0.0;
  Real ub     = 1.0;
  bool result = root.FalsePositionModified(RootFunction,
                                           lb,
                                           ub,
                                           wt,
                                           1e-13,
                                           1e-15,
                                           iv,
                                           ilo,
                                           v0,
                                           dv,
                                           lt0,
                                           dlt,
                                           n,
                                           ln_n,
                                           Y,
                                           var,
                                           this);
  // The diagnostic below recomputes the Helmholtz endpoints and weights only
  // to print them; gate the whole block on a warn-once flag so a
  // non-converged corner state does not cost two Helmholtz evaluations plus
  // an stdout write on every cell/substep/rank.
  static bool warned_conv = false;
  if (!result && !warned_conv)
  {
    warned_conv = true;
    Real vh0, vh1;
    if (iv == compose_eos->ECLOGP)
    {
      vh0 = log(helmholtz_eos->Pressure(n, exp(lt0), Y));
      vh1 = log(helmholtz_eos->Pressure(n, exp(lt0 + dlt), Y));
    }
    else if (iv == compose_eos->ECLOGE)
    {
      vh0 = log(helmholtz_eos->Energy(n, exp(lt0), Y));
      vh1 = log(helmholtz_eos->Energy(n, exp(lt0 + dlt), Y));
    }
    else
    {
      throw std::logic_error(
        "EOSTransition::temperature_from_var_trans only implemented for "
        "log(P) and log(e).");
    }

    Real w0 = TransitionFactor(n, exp(lt0));
    Real w1 = TransitionFactor(n, exp(lt0 + dlt));
    printf(
      "Root not converged in FalsePosition (first occurrence; silencing "
      "further reports):"
      "n, Ye, iv, var: %e, %e, %d, %e\n"
      "temperature: [%e, %e]\n"
      "var_helm: [%e, %e]\n"
      "var_comp: [%e, %e]\n"
      "w: [%e, %e]\n"
      "combined: [%e, %e]\n"
      "f(ilo), f(ihi): %e, %e\n",
      n,
      Y[SCYE],
      iv,
      exp(var),
      exp(lt0),
      exp(lt0 + dlt),
      exp(vh0),
      exp(vh1),
      exp(v0),
      exp(v1),
      w0,
      w1,
      exp(vh0) * (1 - w0) + exp(v0) * w0,
      exp(vh1) * (1 - w1) + exp(v1) * w1,
      exp(var - flo),
      exp(var - fhi));
  }

  Real lt = lt0 + wt * dlt;
  return exp(lt);
}
