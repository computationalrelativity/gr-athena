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
  max_iter = 50;
  T_tol = 1e-10;
  min_Y[SCXN] = 0.0;
  min_Y[SCXP] = 0.0;
  min_Y[SCXA] = 0.0;
  min_Y[SCXH] = 0.0;
  min_Y[SCAH] = 1.0;
  min_Y[SCEB] = 0.0;
  max_Y[SCXN] = 1.0;
  max_Y[SCXP] = 1.0;
  max_Y[SCXA] = 1.0;
  max_Y[SCXH] = 1.0;
  max_Y[SCAH] = 500.0;
}

EOSTransition::~EOSTransition() {
  delete compose_eos;
  delete helmholtz_eos;
}

//Definitions for static members
Real EOSTransition::max_Y[MAX_SPECIES] = {0};
Real EOSTransition::min_Y[MAX_SPECIES] = {0};

Real EOSTransition::m_min_h = numeric_limits<Real>::max();
Real EOSTransition::m_trans_T_width = numeric_limits<Real>::quiet_NaN();
Real EOSTransition::m_trans_ln_width = numeric_limits<Real>::quiet_NaN();
Real EOSTransition::trans_T_start = numeric_limits<Real>::quiet_NaN();
Real EOSTransition::trans_T_end = numeric_limits<Real>::quiet_NaN();
Real EOSTransition::trans_ln_start = numeric_limits<Real>::quiet_NaN();
Real EOSTransition::trans_ln_end = numeric_limits<Real>::quiet_NaN();
bool EOSTransition::m_initialized = false;

Real EOSTransition::TemperatureFromEps(Real n, Real eps, Real *Y) {
  assert (m_initialized);
  Real eps_min = MinimumSpecificInternalEnergy(n, Y);
  Real eps_max = MaximumSpecificInternalEnergy(n, Y);
  if (n < helmholtz_eos->MaximumDensity()) {
    if (eps <= eps_min) return min_T;
    if (eps >= eps_max) return helmholtz_eos->MaximumTemperature();
  } else {
    if (eps <= eps_min) return compose_eos->MinimumDensity();
    if (eps >= eps_max) return max_T;
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

  Real p_min = MinimumPressure(n, Y);
  Real p_max = MaximumPressure(n, Y);
  if (log(n) < trans_ln_start) {
    if (p <= p_min) return min_T;
    if (p >= p_max) return helmholtz_eos->MaximumTemperature();
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

Real EOSTransition::TransitionFactor(Real n, Real T) const {
  Real ln = log(n);
  Real lt = log(T);
  if ((n > helmholtz_eos->MaximumDensity()) or
      (T > helmholtz_eos->MaximumTemperature())) {
    return 1.0;
  }
  if ((n < compose_eos->MinimumDensity()) or
      (lt < compose_eos->MinimumTemperature())) {
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
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->Pressure(n, T, Y);
  if (w == 0.0) return helmholtz_eos->Pressure(n, T, Y);
  Real v_helmholtz = helmholtz_eos->Pressure(n, T, Y);
  Real v_compose = compose_eos->Pressure(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::Entropy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->Entropy(n, T, Y);
  if (w == 0.0) return helmholtz_eos->Entropy(n, T, Y);
  Real v_helmholtz = helmholtz_eos->Entropy(n, T, Y);
  Real v_compose = compose_eos->Entropy(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::Abar(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->Abar(n, T, Y);
  if (w == 0.0) return helmholtz_eos->Abar(n, T, Y);
  Real v_helmholtz = helmholtz_eos->Abar(n, T, Y);
  Real v_compose = compose_eos->Abar(n, T, Y);
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
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->SoundSpeed(n, T, Y);
  if (w == 0.0) return helmholtz_eos->SoundSpeed(n, T, Y);
  Real v_helmholtz = helmholtz_eos->SoundSpeed(n, T, Y);
  Real v_compose = compose_eos->SoundSpeed(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->SpecificInternalEnergy(n, T, Y);
  if (w == 0.0) return helmholtz_eos->SpecificInternalEnergy(n, T, Y);
  Real v_helmholtz = helmholtz_eos->SpecificInternalEnergy(n, T, Y);
  Real v_compose = compose_eos->SpecificInternalEnergy(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::BaryonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->BaryonChemicalPotential(n, T, Y);
  if (w == 0.0) return helmholtz_eos->BaryonChemicalPotential(n, T, Y);
  Real v_helmholtz = helmholtz_eos->BaryonChemicalPotential(n, T, Y);
  Real v_compose = compose_eos->BaryonChemicalPotential(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::ChargeChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->ChargeChemicalPotential(n, T, Y);
  if (w == 0.0) return helmholtz_eos->ChargeChemicalPotential(n, T, Y);
  Real v_helmholtz = helmholtz_eos->ChargeChemicalPotential(n, T, Y);
  Real v_compose = compose_eos->ChargeChemicalPotential(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->ElectronLeptonChemicalPotential(n, T, Y);
  if (w == 0.0) return helmholtz_eos->ElectronLeptonChemicalPotential(n, T, Y);
  Real v_helmholtz = helmholtz_eos->ElectronLeptonChemicalPotential(n, T, Y);
  Real v_compose = compose_eos->ElectronLeptonChemicalPotential(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYn(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYn(n, T, Y);
  if (w == 0.0) return Y[SCXN];
  Real v_helmholtz = Y[SCXN];
  Real v_compose = compose_eos->FrYn(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYp(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYp(n, T, Y);
  if (w == 0.0) return Y[SCXP];
  Real v_helmholtz = Y[SCXP];
  Real v_compose = compose_eos->FrYp(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYa(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYa(n, T, Y);
  if (w == 0.0) return Y[SCXA]/4;
  Real v_helmholtz = Y[SCXA]/4;
  Real v_compose = compose_eos->FrYa(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::FrYh(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->FrYh(n, T, Y);
  if (w == 0.0) return Y[SCXH];
  Real v_helmholtz = Y[SCXH];
  Real v_compose = compose_eos->FrYh(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::AN(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->AN(n, T, Y);
  if (w == 0.0) return Y[SCAH];
  Real v_helmholtz = Y[SCAH];
  Real v_compose = compose_eos->AN(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::ZN(Real n, Real T, Real *Y) {
  assert (m_initialized);
  Real w = TransitionFactor(n, T);
  if (w == 1.0) return compose_eos->ZN(n, T, Y);
  Real v_helmholtz = (Y[SCXH] > 0) ? (Y[SCYE] - Y[SCXP] - Y[SCXA]/2)/Y[SCXH] * Y[SCAH] : 0.0;
  if (w == 0.0) return v_helmholtz;
  Real v_compose = compose_eos->ZN(n, T, Y);
  return v_helmholtz*(1-w) + v_compose*w;
}

Real EOSTransition::MinimumEnthalpy() {
  return m_min_h;
}

Real EOSTransition::MinimumPressure(Real n, Real *Y) {
  if (log(n) > trans_ln_end) {
    return Pressure(n, compose_eos->MinimumDensity(), Y);
  }
  return Pressure(n, min_T, Y);
}

Real EOSTransition::MaximumPressure(Real n, Real *Y) {
  if (log(n) < trans_ln_start) {
    return Pressure(n, helmholtz_eos->MaximumTemperature(), Y);
  }
  return Pressure(n, max_T, Y);
}

Real EOSTransition::MinimumSpecificInternalEnergy(Real n, Real *Y) {
  if (log(n) > trans_ln_end) {
    return SpecificInternalEnergy(n, compose_eos->MinimumDensity(), Y);
  }
  return SpecificInternalEnergy(n, min_T, Y);
}

Real EOSTransition::MaximumSpecificInternalEnergy(Real n, Real *Y) {
  if (log(n) < trans_ln_start) {
    return SpecificInternalEnergy(n, helmholtz_eos->MaximumTemperature(), Y);
  }
  return SpecificInternalEnergy(n, max_T, Y);
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

  comp_it_trans_start = (log(trans_T_start) - compose_eos->m_log_t[0])*compose_eos->m_id_log_t + 1;
  comp_it_trans_end = (log(trans_T_end) - compose_eos->m_log_t[0])*compose_eos->m_id_log_t;
}

void EOSTransition::PrintParameters() {
  printf("EOSTransition:\n");
  printf("  min_n = %e\n", min_n);
  printf("  max_n = %e\n", max_n);
  printf("  min_T = %e\n", min_T);
  printf("  max_T = %e\n", max_T);
  printf("  min_Y = %e\n", min_Y[SCYE]);
  printf("  max_Y = %e\n", max_Y[SCYE]);
  printf("  helmholtz n_max = %e\n", helmholtz_eos->MaximumDensity());
  printf("  helm t_max = %e\n", helmholtz_eos->MaximumTemperature());
  printf("  max_Y = %e\n", max_Y[SCYE]);
  printf("  min_h = %.15e MeV\n", m_min_h);
  printf("  mb = %.15e MeV\n", mb);
  printf("  T transition start = %e\n", trans_T_start);
  printf("  T transition end = %e\n", trans_T_end);
  printf("  n transition start = %e\n", exp(trans_ln_start));
  printf("  n transition end = %e\n", exp(trans_ln_end));
  printf("  dens conversion = %.15e\n", eos_units->DensityConversion(CGS)*mb*eos_units->MassConversion(CGS));
  printf("  temp conversion = %.15e\n", eos_units->TemperatureConversion(CGS));
  printf("  pres conversion = %.15e\n", CGS.PressureConversion(*eos_units));
  printf("  inte conversion = %.15e\n", CGS.SpecificInternalEnergyConversion(*eos_units));
  printf("  entr conversion = %.15e\n", CGS.EntropyConversion(*eos_units)*mb*eos_units->MassConversion(CGS));
}

void EOSTransition::SetBaryonMass(Real new_mb) {
  helmholtz_eos->SetBaryonMass(new_mb);
  compose_eos->SetBaryonMass(new_mb);
  max_Y[SCEB] = 939.5654133/new_mb - 1.0;
}

void EOSTransition::update_bounds() {
  // Check and update bounds
  // -------------------------------------------------------------------------

  if (helmholtz_eos->MaximumDensity() < exp(trans_ln_start))
    printf("EOSTransition::update bounds: helmholtz max density = %.5e is less than "
           "transition density start = %.5e \n",
           helmholtz_eos->MaximumDensity(), exp(trans_ln_start));
  if (helmholtz_eos->MaximumTemperature() < trans_T_start)
    printf("EOSTransition::update bounds: helmholtz max temperature = %.5e is less than "
           "transition temperature start = %.5e \n",
           helmholtz_eos->MaximumTemperature(), trans_T_start);
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

  m_min_h = helmholtz_eos->MinimumEnthalpy();
}

Real EOSTransition::GetNSEBindingEnergy(Real n, Real T, Real *Y) {
  if (n > helmholtz_eos->MaximumDensity() or T > helmholtz_eos->MaximumTemperature()) {
    return 0.0;
  } else if (n < compose_eos->MinimumDensity() or T < compose_eos->MinimumTemperature()) {
    return Y[SCEB];
  } else {
    Real eps_helm = helmholtz_eos->SpecificInternalEnergy(n, T, Y);
    Real eps_comp = compose_eos->SpecificInternalEnergy(n, T, Y);
    return eps_comp - eps_helm;
  }
}

void EOSTransition::InitializeTables(std::string fname, std::string helm_fname, std::string heating_fname, Real baryon_mass) {
  #pragma omp critical
  {
    if (not m_initialized) {

      compose_eos->ReadTableFromFile(fname);
      helmholtz_eos->ReadTableFromFile(helm_fname);

      // Set the baryon mass in the Helmholtz EOS to the current mb
      Real mb_cgs = mb*eos_units->MassConversion(CGS);

      // Initialize the transitions if they have not been initialized
      // -------------------------------------------------------------------------
      if (std::isnan(m_trans_T_width)) {
        m_trans_T_width = 1e9 * CGS.TemperatureConversion(*eos_units); // 1 GK
        trans_T_end = min_T;
        trans_T_start = min_T + m_trans_T_width;
      }

      if (std::isnan(m_trans_ln_width)) {
        m_trans_ln_width = log(5); // start = 5 times end of transition
        trans_ln_end = log(min_n);
        trans_ln_start = trans_ln_end + m_trans_ln_width;
      }

      SetBaryonMass(baryon_mass);
      update_bounds();

      // PrintParameters();
      m_initialized = true;
    }
  }
}

Real EOSTransition::temperature_from_var_trans(int iv, Real var, Real n, Real *Y) const {
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  Real Yq = Y[SCYE];
  compose_eos->weight_idx_ln(&wn0, &wn1, &in, log(n));
  compose_eos->weight_idx_yq(&wy0, &wy1, &iy, Yq);

  auto f = [=](int it){
    Real T = exp(compose_eos->m_log_t[it]);
    Real w = TransitionFactor(n, T);
    Real var_helm;
    if (iv == compose_eos->ECLOGP) var_helm = helmholtz_eos->Pressure(n, T, Y);
    else if (iv == compose_eos->ECLOGE) var_helm = helmholtz_eos->Energy(n, T, Y);
    else throw std::logic_error("EOSTransition::temperature_from_var_trans only implemented for log(P) and log(e).");

    Real var_comp =
      wn0 * (wy0 * compose_eos->m_table[compose_eos->index(iv, in+0, iy+0, it)]  +
             wy1 * compose_eos->m_table[compose_eos->index(iv, in+0, iy+1, it)]) +
      wn1 * (wy0 * compose_eos->m_table[compose_eos->index(iv, in+1, iy+0, it)]  +
             wy1 * compose_eos->m_table[compose_eos->index(iv, in+1, iy+1, it)]);

    return var - log(var_helm*(1-w) + exp(var_comp)*w);
  };

  int ilo = comp_it_trans_end;
  int ihi = comp_it_trans_start;
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

    Real flo_ = f(comp_it_trans_end);
    Real fhi_ = f(comp_it_trans_start);

    std::cout<<"iv: "<<iv<<std::endl;
    std::cout<<"var: "<<var<<std::endl;
    std::cout<<"n: "<<n<<std::endl;
    std::cout<<"Yq: "<<Yq<<std::endl;
    std::cout<<"flo: "<<flo<<std::endl;
    std::cout<<"fhi: "<<fhi<<std::endl;
    std::cout<<"varlo: "<<var - flo<<std::endl;
    std::cout<<"varhi: "<<var - fhi<<std::endl;
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
