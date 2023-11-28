//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fake_rates.cpp
//  \brief this class implements fake neutrino rates for testing purposes

// C++ standard headers
#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>

// Athena++ headers
#include "../athena.hpp"
//#include "../athena_arrays.hpp"
#include "fake_rates.hpp"
#include "../parameter_input.hpp"
class ParameterInput;

#define INDEX(g, i) ((g)*nspecies + (i))

FakeRates::FakeRates(ParameterInput *pin, int _nspecies, int _ngroups)
{
  nspecies = _nspecies;
  ngroups = _ngroups;

  eta = new Real[nspecies*ngroups];
  kappa_scat = new Real[nspecies*ngroups];
  kappa_abs = new Real[nspecies*ngroups];

  avg_atomic_mass = pin->GetOrAddReal("FakeRates", "avg_atomic_mass", 1.0);
  avg_baryon_mass = pin->GetOrAddReal("FakeRates", "avg_baryon_mass", 1.0);

  // Read-in eta and kappas
  std::string parname;  // X_<group>_<species>

  parname = "eta_";
  for (int g = 0; g < ngroups; ++g) {
    std::string gstr = std::to_string(g);
    for (int i = 0; i < nspecies; ++i) {
      std::string istr = std::to_string(i);
      parname += gstr;
      parname += "_";
      parname += istr;
      eta[INDEX(g,i)] = pin->GetOrAddReal("FakeRates", parname, 0.0);
    }
  }

  parname = "kappa_scat_";
  for (int g = 0; g < ngroups; ++g) {
    std::string gstr = std::to_string(g);
    for (int i = 0; i < nspecies; ++i) {
      std::string istr = std::to_string(i);
      parname += gstr;
      parname += "_";
      parname += istr;
      kappa_scat[INDEX(g,i)] = pin->GetOrAddReal("FakeRates", parname, 0.0);
    }
  }

  parname = "kappa_abs";
  for (int g = 0; g < ngroups; ++g) {
    std::string gstr = std::to_string(g);
    for (int i = 0; i < nspecies; ++i) {
      std::string istr = std::to_string(i);
      parname += gstr;
      parname += "_";
      parname += istr;
      kappa_abs[INDEX(g,i)] = pin->GetOrAddReal("FakeRates", parname, 0.0);
    }
  }

}

FakeRates::~FakeRates()
{
  delete [] eta;
  delete [] kappa_scat;
  delete [] kappa_abs;
}

int FakeRates::NeutrinoEmission(Real const rho, Real const temp, Real const Y_e, Real *R)
{
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    R[ig] = rho * eta[ig];
  }
  return 0;
}

int FakeRates::NeutrinoOpacity(Real const rho, Real const temp, Real const Y_e, Real * kappa)
{
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    kappa[ig] = rho * (kappa_scat[ig] + kappa_abs[ig]);
  }
  return 0;
}

int FakeRates::NeutrinoAbsorptionRate(Real const rho, Real const temp, Real const Y_e, Real * abs)
{
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    abs[ig] = rho * kappa_abs[ig];
  }
  return 0;
}

int FakeRates::NeutrinoDensity(Real const rho, Real const temp, Real const Y_e,
			       Real * num, Real * ene)
{
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    num[ig] = 1.0;
    ene[ig] = 1.0;
    if(rho*kappa_abs[ig] > FLT_EPSILON*eta[ig]) {
      num[ig] = eta[ig]/(rho*kappa_abs[ig]);
      ene[ig] = eta[ig]/(rho*kappa_abs[ig]);
    } 
  }
  return 0;
}

int FakeRates::WeakEquilibrium(Real const rho, Real const temp, Real const Y_e,
			       Real const & num,  Real const & ene,
			       Real * temp_eq, Real * ye_eq,
			       Real * num_eq, Real * ene_eq)
{
  *temp_eq = temp;
  *ye_eq = Y_e;
  return NeutrinoDensity(rho, *temp_eq, *ye_eq, num_eq, ene_eq);
}

int FakeRates::AverageAtomicMass(Real const rho, Real const temp, Real const Y_e, Real * Abar)
{
  *Abar = avg_atomic_mass;
  return 0;
}

Real FakeRates::AverageBaryonMass()
{
  return avg_baryon_mass;
}

