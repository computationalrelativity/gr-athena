//! \file eos_transition.cpp
//  \brief Implementation of EOSTransition

#include "eos_transition.hpp"

#include <hdf5.h>
#include <hdf5_hl.h>

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
  max_Y[SCAH] = 500.0;
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
  m_initialized    = false;
  root.iterations  = 100;
}

bool EOSTransition::s_printed_parameters = false;

EOSTransition::~EOSTransition()
{
  delete compose_eos;
  delete helmholtz_eos;
}

Real EOSTransition::TemperatureFromEps(Real n, Real eps, Real* Y)
{
  assert(m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);

  if (n > m_helm_n_max)
    return compose_eos->TemperatureFromEps(n, eps, Y_norm);
  if (log(n) < trans_ln_end)
    return helmholtz_eos->TemperatureFromEps(n, eps, Y_norm);
  Real eps_min = MinimumSpecificInternalEnergy(n, Y_norm);
  Real eps_max = MaximumSpecificInternalEnergy(n, Y_norm);
  if (eps <= eps_min)
    return min_T;
  if (eps >= eps_max)
  {
    if (n < exp(trans_ln_start))
      return m_helm_T_max;
    else
      return max_T;
  }
  Real eps_trans_end = SpecificInternalEnergy(n, trans_T_end, Y_norm);
  if (eps <= eps_trans_end)
    return helmholtz_eos->TemperatureFromEps(n, eps, Y_norm);
  Real eps_trans_start = SpecificInternalEnergy(n, trans_T_start, Y_norm);
  if ((eps >= eps_trans_start) and (log(n) > trans_ln_start))
    return compose_eos->TemperatureFromEps(n, eps, Y_norm);
  return temperature_from_var_trans(
    compose_eos->ECLOGE, log(n * mb * (1 + eps)), n, Y_norm);
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
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);

  if (n > m_helm_n_max)
    return compose_eos->TemperatureFromP(n, p, Y_norm);
  if (log(n) < trans_ln_end)
    return helmholtz_eos->TemperatureFromP(n, p, Y_norm);
  Real p_min = MinimumPressure(n, Y_norm);
  Real p_max = MaximumPressure(n, Y_norm);
  if (p <= p_min)
    return min_T;
  if (p >= p_max)
  {
    if (n < exp(trans_ln_start))
      return m_helm_T_max;
    else
      return max_T;
  }
  Real p_trans_end = Pressure(n, trans_T_end, Y_norm);
  if (p <= p_trans_end)
    return helmholtz_eos->TemperatureFromP(n, p, Y_norm);
  Real p_trans_start = Pressure(n, trans_T_start, Y_norm);
  if ((p >= p_trans_start) and (log(n) > trans_ln_start))
    return compose_eos->TemperatureFromP(n, p, Y_norm);
  return temperature_from_var_trans(compose_eos->ECLOGP, log(p), n, Y_norm);
}

Real EOSTransition::SanitizeMassFractions(Real* Y, Real* Y_norm) const
{
  for (int i = 0; i < SCNVAR; ++i)
  {
    Y_norm[i] = max(min_Y[i], min(Y[i], max_Y[i]));
  }
  Real Xsum = Y[SCXN] + Y[SCXP] + Y[SCXA] + Y[SCXH];
  if (Xsum <= 0.0)
  {
    printf(
      "EOSTransition::SanitizeMassFractions: got invalid mass fractions, sum "
      "is %.5e\n",
      Xsum);
    Y_norm[SCYE] = 0.5;
    Y_norm[SCXN] = 0.5;
    Y_norm[SCXP] = 0.5;
    Y_norm[SCXA] = 1.0;
    Y_norm[SCXH] = 0.0;
    return 1.0;
  }
  Y_norm[SCXN] = Y[SCXN] / Xsum;
  Y_norm[SCXP] = Y[SCXP] / Xsum;
  Y_norm[SCXA] = Y[SCXA] / Xsum;
  Y_norm[SCXH] = Y[SCXH] / Xsum;
  return Xsum;
}

Real EOSTransition::TransitionFactor(Real n, Real T) const
{
  Real ln = log(n);
  if ((n > m_helm_n_max) or (T > m_helm_T_max))
  {
    return 1.0;
  }
  if ((n < compose_eos->min_n) or (T < compose_eos->min_T))
  {
    return 0.0;
  }
  Real w = min(1.0, max(0.0, (T - trans_T_end) / m_trans_T_width));
  w *= min(1.0, max(0.0, (ln - trans_ln_end) / m_trans_ln_width));
  return w;
}

Real EOSTransition::Energy(Real n, Real T, Real* Y)
{
  return (SpecificInternalEnergy(n, T, Y) + 1.0) * mb * n;
}

Real EOSTransition::Pressure(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
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
  assert(m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e) / n;
}

Real EOSTransition::SoundSpeed(Real n, Real T, Real* Y)
{
  assert(m_initialized);
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
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
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

Real EOSTransition::MinimumPressure(Real n, Real* Y)
{
  if (n > exp(trans_ln_end))
  {
    return Pressure(n, compose_eos->min_T, Y);
  }
  return Pressure(n, min_T, Y);
}

Real EOSTransition::MaximumPressure(Real n, Real* Y)
{
  if (n < exp(trans_ln_start))
  {
    return Pressure(n, m_helm_T_max, Y);
  }
  return Pressure(n, max_T, Y);
}

Real EOSTransition::MinimumEntropy(Real n, Real* Y)
{
  if (n > exp(trans_ln_end))
  {
    return Entropy(n, compose_eos->min_T, Y);
  }
  return Entropy(n, min_T, Y);
}

Real EOSTransition::MaximumEntropy(Real n, Real* Y)
{
  if (n < exp(trans_ln_start))
  {
    return Entropy(n, m_helm_T_max, Y);
  }
  return Entropy(n, max_T, Y);
}

Real EOSTransition::MinimumSpecificInternalEnergy(Real n, Real* Y)
{
  if (n > exp(trans_ln_end))
  {
    return SpecificInternalEnergy(n, compose_eos->min_T, Y);
  }
  return SpecificInternalEnergy(n, min_T, Y);
}

Real EOSTransition::MaximumSpecificInternalEnergy(Real n, Real* Y)
{
  if (n < exp(trans_ln_start))
  {
    return SpecificInternalEnergy(n, m_helm_T_max, Y);
  }
  return SpecificInternalEnergy(n, max_T, Y);
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
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }

  if (T_start <= T_end)
  {
    std::stringstream msg;
    msg << "### EOSTransition: temperature transition start: " << T_start
        << " is not larger than end: " << T_end << std::endl;
    // ATHENA_ERROR(msg);
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
  max_Y[SCEB] = mn / new_mb - 1.0;            // free neutron limit
  m_min_h =
    min_Y[SCEB];  // minimum specific enthalpy is the binding energy of Fe56
  mb = new_mb;
}

void EOSTransition::update_bounds()
{
  // Check and update bounds
  // -------------------------------------------------------------------------

  if (m_helm_n_max < exp(trans_ln_start))
    printf(
      "EOSTransition::update bounds: helmholtz max density = %.5e is less "
      "than "
      "transition density start = %.5e \n",
      m_helm_n_max,
      exp(trans_ln_start));
  if (m_helm_T_max < trans_T_start)
    printf(
      "EOSTransition::update bounds: helmholtz max temperature = %.5e is less "
      "than "
      "transition temperature start = %.5e \n",
      m_helm_T_max,
      trans_T_start);
  if (compose_eos->min_n > exp(trans_ln_end))
    printf(
      "EOSTransition::update bounds: compose min density = %.5e is greater "
      "than "
      "transition density end = %.5e \n",
      compose_eos->min_n,
      exp(trans_ln_end));
  if (compose_eos->min_T > trans_T_end)
    printf(
      "EOSTransition::update bounds: compose min temperature = %.5e is "
      "greater than "
      "transition temperature end = %.5e \n",
      compose_eos->min_T,
      trans_T_end);

  min_n       = helmholtz_eos->min_n;
  min_T       = helmholtz_eos->min_T;
  max_n       = compose_eos->max_n;
  max_T       = compose_eos->max_T;
  min_Y[SCYE] = compose_eos->min_Y[SCYE];
  max_Y[SCYE] = compose_eos->max_Y[SCYE];

  // if not udated by user set thelmholtz max
  if (isnan(m_helm_n_max))
    m_helm_n_max = min(helmholtz_eos->max_n, compose_eos->min_n);
  if (isnan(m_helm_T_max))
    m_helm_T_max = min(helmholtz_eos->max_T, compose_eos->max_T);

  // indices for transition and helmholtz max in compose table
  comp_it_trans_start =
    (log(trans_T_start) - compose_eos->m_log_t[0]) * compose_eos->m_id_log_t +
    1;
  comp_it_trans_end =
    (log(trans_T_end) - compose_eos->m_log_t[0]) * compose_eos->m_id_log_t;
  comp_it_helm_tmax =
    (log(m_helm_T_max) - compose_eos->m_log_t[0]) * compose_eos->m_id_log_t +
    1;
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
  // get T
  *T = TemperatureFromE(n, e, Y);
  PressureAndEnthalpy(n, *T, Y, P, h);
}

void EOSTransition::PressureAndEnthalpyFromE(Real n,
                                             Real e,
                                             Real* Y,
                                             Real* P,
                                             Real* h,
                                             int* guess_it)
{
  // get T
  Real T = TemperatureFromE(n, e, Y);
  PressureAndEnthalpy(n, T, Y, P, h);
}

/// Fused pressure + enthalpy: single weight computation for both P and h.
void EOSTransition::PressureAndEnthalpy(Real n,
                                        Real T,
                                        Real* Y,
                                        Real* P,
                                        Real* h)
{
  *P = Pressure(n, T, Y);
  *h = Enthalpy(n, T, Y);
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

    Real Zh     = (Y_NSE[SCXH] > 0)
                  ? (Y_NSE[SCYE] - Y_NSE[SCXP] - Y_NSE[SCXA] / 2) / Y_NSE[SCXH] *
                  Y_NSE[SCAH]
                  : 0.0;
    Real Ah     = Y_NSE[SCAH];
    Real min_EB = (Y_NSE[SCXN] * mn + Y_NSE[SCXP] * mp + Y_NSE[SCXA] * ma / 4 +
                   Y_NSE[SCXH] * mFe / 56.0) /
                    mb -
                  1;
    Real max_EB = (Y_NSE[SCXN] * mn + Y_NSE[SCXP] * mp + Y_NSE[SCXA] * ma +
                   Y_NSE[SCXH] * ((Ah - Zh) * mn + Zh * mp)) /
                    mb -
                  1;
    Real eb = max(min_EB, min(eps_comp - eps_helm, max_EB));
    return eb;
  }
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
    if (std::isnan(m_trans_T_width) or std::isnan(m_trans_ln_width))
      throw std::runtime_error(
        "EOSTransition: Transition parameters must be set before "
        "initialization.");
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
  Real Yq = Y[SCYE];
  compose_eos->weight_idx_ln(&wn0, &wn1, &in, log(n));
  compose_eos->weight_idx_yq(&wy0, &wy1, &iy, Yq);

  auto f = [=](int it)
  {
    Real T = exp(compose_eos->m_log_t[it]);
    Real w = TransitionFactor(n, T);
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

  int ilo_0 = comp_it_trans_end;
  int ihi_0 = (log(n) < trans_ln_start)
              ? ihi_0 = min(comp_it_helm_tmax, compose_eos->m_nt - 1)
              :  // we are transitioning in the density bracket
                ihi_0 = comp_it_trans_start;  // we are transitioning only in
                                              // the temperature bracket

  int ilo  = ilo_0;
  int ihi  = ihi_0;
  Real flo = f(ilo);
  Real fhi = f(ihi);
  while (flo * fhi > 0)
  {
    if (ilo == ihi - 1)
    {
      break;
    }
    else
    {
      ilo += 1;
      flo = f(ilo);
    }
  }
  if (!(flo * fhi <= 0))
  {
    Real flo_ = f(ilo_0);
    Real fhi_ = f(ihi_0);

    std::cout
      << "EOSTransition::temperature_from_var_trans failed to bracket root."
      << std::endl;
    std::cout << "iv: " << iv << std::endl;
    std::cout << "var: " << var << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "Ye, Xn, Xp, Xa, Xh, Ah: " << Y[SCYE] << " " << Y[SCXN] << " "
              << Y[SCXP] << " " << Y[SCXA] << " " << Y[SCXH] << " " << Y[SCAH]
              << std::endl;
    std::cout << "ilo_0: " << ilo_0 << std::endl;
    std::cout << "ihi_0: " << ihi_0 << std::endl;
    std::cout << "varlo_0: " << var - flo_ << std::endl;
    std::cout << "varhi_0: " << var - fhi_ << std::endl;
    std::cout << "flo_0: " << flo_ << std::endl;
    std::cout << "fhi_0: " << fhi_ << std::endl;
    std::cout << "ilo: " << ilo << std::endl;
    std::cout << "ihi: " << ihi << std::endl;
    std::cout << "varlo: " << var - flo << std::endl;
    std::cout << "varhi: " << var - fhi << std::endl;
    std::cout << "flo: " << flo << std::endl;
    std::cout << "fhi: " << fhi << std::endl;
  }
  assert(flo * fhi <= 0);
  while (ihi - ilo > 1)
  {
    int ip  = ilo + (ihi - ilo) / 2;
    Real fp = f(ip);
    if (fp * flo <= 0)
    {
      ihi = ip;
      fhi = fp;
    }
    else
    {
      ilo = ip;
      flo = fp;
    }
  }
  assert(ihi - ilo == 1);
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

  // Real operator()(Real wt,
  //                 int iv,
  //                 int it,
  //                 Real v0,
  //                 Real v1,
  //                 Real t0,
  //                 Real t1,
  //                 Real n,
  //                 Real* Y,
  //                 Real var,
  //                 EOSTransition* peos) const

  Real wt;
  Real lb     = 0.0;
  Real ub     = 1.0;
  bool result = root.FalsePosition(RootFunction,
                                   lb,
                                   ub,
                                   wt,
                                   1e-15,
                                   iv,
                                   ilo,
                                   v0,
                                   dv,
                                   lt0,
                                   dlt,
                                   n,
                                   Y,
                                   var,
                                   this);
  if (!result)
  {
    printf("Root not converged in FalsePosition: nb=%e, Y[0]=%e\n", n, Y[0]);
  }

  Real lt = lt0 + wt * dlt;
  return exp(lt);
}
