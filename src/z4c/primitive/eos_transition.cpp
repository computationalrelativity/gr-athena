//! \file eos_transition.cpp
//  \brief Implementation of EOSTransition

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "eos_transition.hpp"
#include "unit_system.hpp"

using namespace Primitive;
using namespace std;


EOSTransition::EOSTransition() {
  compose_eos = new EOSCompOSE();
  helmholtz_eos = new EOSHelmholtz();
  n_species = 7;
  eos_units = &Nuclear;
  min_Y[SCYE] = 0.0; // will be overwritten by update_bounds
  min_Y[SCXN] = 0.0;
  min_Y[SCXP] = 0.0;
  min_Y[SCXA] = 0.0;
  min_Y[SCXH] = 0.0;
  min_Y[SCAH] = 1.0;
  min_Y[SCEB] = 0.0;

  max_Y[SCYE] = 1.0; // will be overwritten by update_bounds
  max_Y[SCXN] = 1.0;
  max_Y[SCXP] = 1.0;
  max_Y[SCXA] = 1.0;
  max_Y[SCXH] = 1.0;
  max_Y[SCAH] = 500.0;
  max_Y[SCEB] = 1e-2; // will be overwritten by SetBaryonMass

  m_min_h = numeric_limits<Real>::max();
  m_trans_T_width = numeric_limits<Real>::quiet_NaN();
  m_trans_ln_width = numeric_limits<Real>::quiet_NaN();
  trans_T_start = numeric_limits<Real>::quiet_NaN();
  trans_T_end = numeric_limits<Real>::quiet_NaN();
  trans_ln_start = numeric_limits<Real>::quiet_NaN();
  trans_ln_end = numeric_limits<Real>::quiet_NaN();
  m_helm_n_max = numeric_limits<Real>::quiet_NaN();
  m_helm_T_max = numeric_limits<Real>::quiet_NaN();
  m_initialized = false;
}

bool EOSTransition::s_printed_parameters = false;

EOSTransition::~EOSTransition() {
  delete compose_eos;
  delete helmholtz_eos;
}


Real EOSTransition::TemperatureFromEps(Real n, Real eps, Real *Y) {
  assert (m_initialized);

  if (n > m_helm_n_max) return compose_eos->TemperatureFromEps(n, eps, Y);
  Real eps_min = MinimumSpecificInternalEnergy(n, Y);
  Real eps_max = MaximumSpecificInternalEnergy(n, Y);
  if (eps <= eps_min) return min_T;
  if (eps <= eps_max) {
    if (n < compose_eos->MinimumDensity())
      return m_helm_T_max;
    else
      return max_T;
  }
  if (log(n) < trans_ln_end) return helmholtz_eos->TemperatureFromEps(n, eps, Y);
  Real eps_trans_end = SpecificInternalEnergy(n, trans_T_end, Y);
  if (eps <= eps_trans_end) return helmholtz_eos->TemperatureFromEps(n, eps, Y);
  Real eps_trans_start = SpecificInternalEnergy(n, trans_T_start, Y);
  if ((eps >= eps_trans_start) and (log(n) > trans_ln_start)) return compose_eos->TemperatureFromEps(n, eps, Y);
  return temperature_from_var_trans(compose_eos->ECLOGE, log(n*mb*(1+eps)), n, Y);
}

Real EOSTransition::TemperatureFromEntropy(Real n, Real s, Real *Y) {
  throw std::logic_error("EOSTransition::TemperatureFromEntropy not currently implemented.");
}

Real EOSTransition::TemperatureFromE(Real n, Real e, Real *Y) {
  assert (m_initialized);
  return TemperatureFromEps(n, e/(mb*n) - 1.0, Y);
}

Real EOSTransition::TemperatureFromP(Real n, Real p, Real *Y) {
  assert (m_initialized);

  if (n > m_helm_n_max) return compose_eos->TemperatureFromP(n, p, Y);
  Real p_min = MinimumPressure(n, Y);
  Real p_max = MaximumPressure(n, Y);
  if (log(n) < trans_ln_start) {
    if (p <= p_min) return min_T;
    if (p >= p_max) return m_helm_T_max;
  } else {
    if (p <= p_min) return compose_eos->MinimumDensity();
    if (p >= p_max) return max_T;
  }
  Real p_trans_start = Pressure(n, trans_T_start, Y);
  Real p_trans_end = Pressure(n, trans_T_end, Y);
  if (p >= p_trans_start) return compose_eos->TemperatureFromP(n, p, Y);
  if (p <= p_trans_end) return helmholtz_eos->TemperatureFromP(n, p, Y);
  return temperature_from_var_trans(compose_eos->ECLOGP, log(p), n, Y);
}

Real EOSTransition::SanitizeMassFractions(Real *Y, Real *Y_norm) const {
    for (int i = 0; i < SCNVAR; ++i) {
      Y_norm[i] = max(min_Y[i], min(Y[i], max_Y[i]));
    }
    Real Xsum = Y[SCXN] + Y[SCXP] + Y[SCXA] + Y[SCXH];
    if (Xsum <= 0.0) {
      printf("EOSTransition::SanitizeMassFractions: got invalid mass fractions, sum is %.5e\n", Xsum);
      Y_norm[SCYE] = 0.5;
      Y_norm[SCXN] = 0.5;
      Y_norm[SCXP] = 0.5;
      Y_norm[SCXA] = 1.0;
      Y_norm[SCXH] = 0.0;
      return 1.0;
    }
    Y_norm[SCXN] = Y[SCXN]/Xsum;
    Y_norm[SCXP] = Y[SCXP]/Xsum;
    Y_norm[SCXA] = Y[SCXA]/Xsum;
    Y_norm[SCXH] = Y[SCXH]/Xsum;
    return Xsum;
}

Real EOSTransition::TransitionFactor(Real n, Real T) const {
  Real ln = log(n);
  if ((n > m_helm_n_max) or
      (T > m_helm_T_max)) {
    return 1.0;
  }
  if ((n < compose_eos->MinimumDensity()) or
      (T < compose_eos->MinimumTemperature())) {
    return 0.0;
  }
  Real w = min(1.0, max(0.0, (T - trans_T_end)/m_trans_T_width));
  w *= min(1.0, max(0.0, (ln - trans_ln_end)/m_trans_ln_width));
  return w;
}

Real EOSTransition::Energy(Real n, Real T, Real *Y) {
  return (SpecificInternalEnergy(n, T, Y) + 1.0) * mb * n;
}

Real EOSTransition::Pressure(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->Pressure(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->Pressure(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->Pressure(n, T, Y_norm);
  Real v_compose = compose_eos->Pressure(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::Entropy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->Entropy(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->Entropy(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->Entropy(n, T, Y_norm);
  Real v_compose = compose_eos->Entropy(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::Abar(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->Abar(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->Abar(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->Abar(n, T, Y_norm);
  Real v_compose = compose_eos->Abar(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::Enthalpy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e)/n;
}

Real EOSTransition::SoundSpeed(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->SoundSpeed(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->SoundSpeed(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->SoundSpeed(n, T, Y_norm);
  Real v_compose = compose_eos->SoundSpeed(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->SpecificInternalEnergy(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->SpecificInternalEnergy(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->SpecificInternalEnergy(n, T, Y_norm);
  Real v_compose = compose_eos->SpecificInternalEnergy(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::BaryonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->BaryonChemicalPotential(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->BaryonChemicalPotential(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->BaryonChemicalPotential(n, T, Y_norm);
  Real v_compose = compose_eos->BaryonChemicalPotential(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::ChargeChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->ChargeChemicalPotential(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->ChargeChemicalPotential(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->ChargeChemicalPotential(n, T, Y_norm);
  Real v_compose = compose_eos->ChargeChemicalPotential(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  if (w == 0.0) return helmholtz_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  Real v_helmholtz = helmholtz_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  Real v_compose = compose_eos->ElectronLeptonChemicalPotential(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYn(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYn(n, T, Y_norm);
  if (w == 0.0) return Y_norm[SCXN];
  Real v_helmholtz = Y_norm[SCXN];
  Real v_compose = compose_eos->FrYn(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYp(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYp(n, T, Y_norm);
  if (w == 0.0) return Y_norm[SCXP];
  Real v_helmholtz = Y_norm[SCXP];
  Real v_compose = compose_eos->FrYp(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYa(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYa(n, T, Y_norm);
  if (w == 0.0) return Y_norm[SCXA];
  Real v_helmholtz = Y_norm[SCXA];
  Real v_compose = compose_eos->FrYa(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYh(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYh(n, T, Y_norm);
  if (w == 0.0) return Y_norm[SCXH];
  Real v_helmholtz = Y_norm[SCXH];
  Real v_compose = compose_eos->FrYh(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::AN(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->AN(n, T, Y_norm);
  if (w == 0.0) return Y_norm[SCAH];
  Real v_helmholtz = Y_norm[SCAH];
  Real v_compose = compose_eos->AN(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::ZN(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->ZN(n, T, Y_norm);
  Real v_helmholtz = (Y_norm[SCXH] > 0) ? (Y_norm[SCYE] - Y_norm[SCXP] - Y_norm[SCXA]/2)/Y_norm[SCXH] * Y_norm[SCAH] : 0.0;
  if (w == 0.0) return v_helmholtz;
  Real v_compose = compose_eos->ZN(n, T, Y_norm);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::MinimumEnthalpy() {
  return m_min_h;
}

Real EOSTransition::MinimumPressure(Real n, Real *Y) {
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  if (log(n) > trans_ln_end) {
    return Pressure(n, compose_eos->MinimumDensity(), Y_norm);
  }
  return Pressure(n, min_T, Y_norm);
}

Real EOSTransition::MaximumPressure(Real n, Real *Y) {
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  if (log(n) < trans_ln_start) {
    return Pressure(n, m_helm_T_max, Y_norm);
  }
  return Pressure(n, max_T, Y_norm);
}

Real EOSTransition::MinimumSpecificInternalEnergy(Real n, Real *Y) {
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  if (log(n) > trans_ln_end) {
    return SpecificInternalEnergy(n, compose_eos->MinimumDensity(), Y_norm);
  }
  return SpecificInternalEnergy(n, min_T, Y_norm);
}

Real EOSTransition::MaximumSpecificInternalEnergy(Real n, Real *Y) {
  Real Y_norm[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  if (log(n) < trans_ln_start) {
    return SpecificInternalEnergy(n, m_helm_T_max, Y_norm);
  }
  return SpecificInternalEnergy(n, max_T, Y_norm);
}

Real EOSTransition::MinimumEnergy(Real n, Real *Y) {
  return (MinimumSpecificInternalEnergy(n, Y) + 1.0) * mb * n;
}

Real EOSTransition::MaximumEnergy(Real n, Real *Y) {
  return (MaximumSpecificInternalEnergy(n, Y) + 1.0) * mb * n;
}

void EOSTransition::SetTransition(Real n_start, Real n_end, Real T_start, Real T_end) {
  if (m_initialized) {
    std::stringstream msg;
    msg << "### EOSTransition: Transition must be set before initialization." << std::endl;
    throw std::runtime_error(msg.str());
  }

  if (n_start <= n_end) {
    std::stringstream msg;
    msg << "### EOSTransition: density transition start: " << n_start <<
        " is not larger than end: " << n_end << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }

  if (T_start <= T_end) {
    std::stringstream msg;
    msg << "### EOSTransition: temperature transition start: " << T_start <<
        " is not larger than end: " << T_end << std::endl;
    // ATHENA_ERROR(msg);
    throw std::runtime_error(msg.str());
  }

  trans_ln_start = log(n_start);
  trans_ln_end = log(n_end);
  m_trans_ln_width = log(n_start/n_end);

  trans_T_start = T_start;
  trans_T_end = T_end;
  m_trans_T_width = T_start - T_end;
}

void EOSTransition::PrintParameters() {
  #pragma omp critical
  {if ((not s_printed_parameters) and (Globals::my_rank == 0)) {
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
    printf("  comp n_min, t_min = %e %e\n", compose_eos->MinimumDensity(), compose_eos->MinimumTemperature());
    printf("  min_h = %.15e\n", m_min_h);
    printf("  mb = %.15e MeV\n", mb);
    printf("  T transition start, end = %e %e\n", trans_T_start, trans_T_end);
    printf("  n transition start, end = %e %e\n", exp(trans_ln_start), exp(trans_ln_end));
    printf("  i_t helm_tmax, trans_start, trans_end = %d %d %d\n",
           comp_it_helm_tmax, comp_it_trans_start, comp_it_trans_end);
    s_printed_parameters = true;
  }}
}

void EOSTransition::SetBaryonMass(Real new_mb) {
  helmholtz_eos->SetBaryonMass(new_mb);
  compose_eos->SetBaryonMass(new_mb);
  min_Y[SCEB] = mFe/(56.0*new_mb) - 1.0; // most bound nucleus is Fe-56
  max_Y[SCEB] = mn/new_mb - 1.0; // free neutron limit
  m_min_h = min_Y[SCEB]; // minimum specific enthalpy is the binding energy of Fe56
  mb = new_mb;
}

void EOSTransition::update_bounds() {
  // Check and update bounds
  // -------------------------------------------------------------------------

  if (m_helm_n_max < exp(trans_ln_start))
    printf("EOSTransition::update bounds: helmholtz max density = %.5e is less than "
           "transition density start = %.5e \n",
           m_helm_n_max, exp(trans_ln_start));
  if (m_helm_T_max < trans_T_start)
    printf("EOSTransition::update bounds: helmholtz max temperature = %.5e is less than "
           "transition temperature start = %.5e \n",
           m_helm_T_max, trans_T_start);
  if (compose_eos->MinimumDensity() > exp(trans_ln_end))
    printf("EOSTransition::update bounds: compose min density = %.5e is greater than "
           "transition density end = %.5e \n",
           compose_eos->MinimumDensity(), exp(trans_ln_end));
  if (compose_eos->MinimumTemperature() > trans_T_end)
    printf("EOSTransition::update bounds: compose min temperature = %.5e is greater than "
           "transition temperature end = %.5e \n",
           compose_eos->MinimumTemperature(), trans_T_end);

  min_n = helmholtz_eos->MinimumDensity();
  min_T = helmholtz_eos->MinimumTemperature();
  max_n = compose_eos->MaximumDensity();
  max_T = compose_eos->MaximumTemperature();
  min_Y[SCYE] = compose_eos->min_Y[SCYE];
  max_Y[SCYE] = compose_eos->max_Y[SCYE];

  comp_it_trans_start = (log(trans_T_start) - compose_eos->m_log_t[0])*compose_eos->m_id_log_t + 1;
  comp_it_trans_end = (log(trans_T_end) - compose_eos->m_log_t[0])*compose_eos->m_id_log_t;
  comp_it_helm_tmax = (log(m_helm_T_max) - compose_eos->m_log_t[0])*compose_eos->m_id_log_t + 1;

  if (isnan(m_helm_n_max)) m_helm_n_max = helmholtz_eos->MaximumDensity(); // has not been udated by user, so set to helmholtz max density
  if (isnan(m_helm_T_max)) m_helm_T_max = helmholtz_eos->MaximumTemperature(); // has not been udated by user, so set to helmholtz max temperature
}

Real EOSTransition::GetNSEBindingEnergy(Real n, Real T, Real *Y) {
  if (n > m_helm_n_max or T > m_helm_T_max) {
    return 0.0;
  } else if (n < compose_eos->MinimumDensity() or T < compose_eos->MinimumTemperature()) {
    return Y[SCEB];
  } else {
    Real Y_norm[SCNVAR];
    SanitizeMassFractions(Y, Y_norm);
    Y_norm[SCEB] = 0.0;
    Real eps_helm = helmholtz_eos->SpecificInternalEnergy(n, T, Y_norm);
    Real eps_comp = compose_eos->SpecificInternalEnergy(n, T, Y);


    Real Zh = (Y_norm[SCXH] > 0) ? (Y_norm[SCYE] - Y_norm[SCXP] - Y_norm[SCXA]/2)/Y_norm[SCXH] * Y_norm[SCAH] : 0.0;
    Real Ah = Y_norm[SCAH];
    Real min_EB = (Y_norm[SCXN]*mn + Y_norm[SCXP]*mp + Y_norm[SCXA]*ma/4 + Y_norm[SCXH]*mFe/56.0) / mb - 1;
    Real max_EB = (Y_norm[SCXN]*mn + Y_norm[SCXP]*mp + Y_norm[SCXA]*ma + Y_norm[SCXH]*((Ah-Zh)*mn + Zh*mp)) / mb - 1;
    Real eb = max(min_EB, min(eps_comp - eps_helm, max_EB));
    if (eb > 1e-2) {
      printf("EOSTransition::GetNSEBindingEnergy: got binding energy per baryon = %.5e MeV for n=%.5e, T=%.5e\n", eb*mb, n, T);
      printf("  Ye=%.5e, Xn=%.5e, Xp=%.5e, Xa=%.5e, Xh=%.5e, Xah=%.5e\n",
             Y[SCYE], Y[SCXN], Y[SCXP], Y[SCXA], Y[SCXH], Y[SCAH]);
      printf("  eps_helm = %.5e, eps_comp = %.5e\n", eps_helm, eps_comp);
      printf("  min_EB = %.5e, max_EB = %.5e\n", min_EB, max_EB);
    }
    return eb;
  }
}

void EOSTransition::InitializeTables(std::string fname, std::string helm_fname, std::string heating_fname, Real baryon_mass) {
  if (not m_initialized) {
    compose_eos->ReadTableFromFile(fname);
    helmholtz_eos->ReadTableFromFile(helm_fname, compose_eos->min_Y[SCYE], compose_eos->max_Y[SCYE]);
    if (std::isnan(m_trans_T_width) or std::isnan(m_trans_ln_width))
      throw std::runtime_error("EOSTransition: Transition parameters must be set before initialization.");
    SetBaryonMass(baryon_mass);
    update_bounds();
    PrintParameters();
    m_initialized = true;
  } else {
    std::stringstream msg;
    msg << "### EOSTransition: InitializeTables should only be called once." << std::endl;
    throw std::runtime_error(msg.str());
  }
}

Real EOSTransition::temperature_from_var_trans(int iv, Real var, Real n, Real *Y) const {
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  Real * Y_norm = new Real[SCNVAR];
  SanitizeMassFractions(Y, Y_norm);
  Real Yq = Y_norm[SCYE];
  compose_eos->weight_idx_ln(&wn0, &wn1, &in, log(n));
  compose_eos->weight_idx_yq(&wy0, &wy1, &iy, Yq);

  auto f = [=](int it){
    Real T = exp(compose_eos->m_log_t[it]);
    Real w = TransitionFactor(n, T);
    Real var_helm;
    if (iv == compose_eos->ECLOGP) var_helm = helmholtz_eos->Pressure(n, T, Y_norm);
    else if (iv == compose_eos->ECLOGE) var_helm = helmholtz_eos->Energy(n, T, Y_norm);
    else throw std::logic_error("EOSTransition::temperature_from_var_trans only implemented for log(P) and log(e).");

    Real var_comp =
      wn0 * (wy0 * compose_eos->m_table[compose_eos->index(iv, in+0, iy+0, it)]  +
             wy1 * compose_eos->m_table[compose_eos->index(iv, in+0, iy+1, it)]) +
      wn1 * (wy0 * compose_eos->m_table[compose_eos->index(iv, in+1, iy+0, it)]  +
             wy1 * compose_eos->m_table[compose_eos->index(iv, in+1, iy+1, it)]);

    return var - log(var_helm*(1-w) + exp(var_comp)*w);
  };

  int ilo_0 = comp_it_trans_end;
  int ihi_0 = (log(n) < trans_ln_start) ?
    ihi_0 = min(comp_it_helm_tmax, compose_eos->m_nt-1) : // we are transitioning in the density bracket
    ihi_0 = comp_it_trans_start; // we are transitioning only in the temperature bracket

  int ilo = ilo_0;
  int ihi = ihi_0;
  Real flo = f(ilo);
  Real fhi = f(ihi);
  while (flo*fhi>0){
    if (ilo == ihi - 1) {
      break;
    } else {
      ilo += 1;
      flo = f(ilo);
    }
  }
  if (!(flo*fhi <= 0)) {

    Real flo_ = f(ilo_0);
    Real fhi_ = f(ihi_0);

    std::cout<<"EOSTransition::temperature_from_var_trans failed to bracket root."<<std::endl;
    std::cout<<"iv: "<<iv<<std::endl;
    std::cout<<"var: "<<var<<std::endl;
    std::cout<<"n: "<<n<<std::endl;
    std::cout<<"Ye, Xn, Xp, Xa, Xh, Ah: "<<Y_norm[SCYE]<<" "<<Y_norm[SCXN]<<" "<<Y_norm[SCXP]<<" "<<Y_norm[SCXA]<<" "<<Y_norm[SCXH]<<" "<<Y_norm[SCAH]<<std::endl;
    std::cout<<"ilo_0: "<<ilo_0<<std::endl;
    std::cout<<"ihi_0: "<<ihi_0<<std::endl;
    std::cout<<"varlo_0: "<<var - flo_<<std::endl;
    std::cout<<"varhi_0: "<<var - fhi_<<std::endl;
    std::cout<<"flo_0: "<<flo_<<std::endl;
    std::cout<<"fhi_0: "<<fhi_<<std::endl;
    std::cout<<"ilo: "<<ilo<<std::endl;
    std::cout<<"ihi: "<<ihi<<std::endl;
    std::cout<<"varlo: "<<var - flo<<std::endl;
    std::cout<<"varhi: "<<var - fhi<<std::endl;
    std::cout<<"flo: "<<flo<<std::endl;
    std::cout<<"fhi: "<<fhi<<std::endl;
  }
  assert(flo*fhi <= 0);
  while (ihi - ilo > 1) {
    int ip = ilo + (ihi - ilo)/2;
    Real fp = f(ip);
    if (fp*flo <= 0) {
      ihi = ip;
      fhi = fp;
    }
    else {
      ilo = ip;
      flo = fp;
    }
  }
  assert(ihi - ilo == 1);
  Real lthi = compose_eos->m_log_t[ihi];
  Real ltlo = compose_eos->m_log_t[ilo];

  if (flo == 0) {
    return exp(ltlo);
  }
  if (fhi == 0) {
    return exp(lthi);
  }

  Real lt = compose_eos->m_log_t[ilo] - flo*(lthi - ltlo)/(fhi - flo);
  return exp(lt);
}
