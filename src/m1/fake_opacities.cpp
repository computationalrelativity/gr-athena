//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fake_opacities.cpp
//  \brief this class implements fake opacities opacities for testing purposes

// C++ standard headers
#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>

// Athena++ headers
#include "../athena.hpp"
//#include "../athena_arrays.hpp"
#include "fake_opacities.hpp"
#include "../parameter_input.hpp"
class ParameterInput;

#define INDEX(g, i) ((g)*nspecies + (i))

FakeOpacities::FakeOpacities(ParameterInput *pin, int _nspecies, int _ngroups)
{
  nspecies = _nspecies;
  ngroups = _ngroups;

  eta = new Real[nspecies*ngroups];
  kappa_sca = new Real[nspecies*ngroups];
  kappa_abs = new Real[nspecies*ngroups];

  avg_atomic_mass = pin->GetOrAddReal("FakeOpacities", "avg_atomic_mass", 1.0);
  avg_baryon_mass = pin->GetOrAddReal("FakeOpacities", "avg_baryon_mass", 1.0);

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
      eta[INDEX(g,i)] = pin->GetOrAddReal("FakeOpacities", parname, 0.0);
    }
  }

  parname = "kappa_sca_";
  for (int g = 0; g < ngroups; ++g) {
    std::string gstr = std::to_string(g);
    for (int i = 0; i < nspecies; ++i) {
      std::string istr = std::to_string(i);
      parname += gstr;
      parname += "_";
      parname += istr;
      kappa_sca[INDEX(g,i)] = pin->GetOrAddReal("FakeOpacities", parname, 0.0);
    }
  }

  parname = "kappa_abs_";
  for (int g = 0; g < ngroups; ++g) {
    std::string gstr = std::to_string(g);
    for (int i = 0; i < nspecies; ++i) {
      std::string istr = std::to_string(i);
      parname += gstr;
      parname += "_";
      parname += istr;
      kappa_abs[INDEX(g,i)] = pin->GetOrAddReal("FakeOpacities", parname, 0.0);
    }
  }

}

FakeOpacities::~FakeOpacities()
{
  delete [] eta;
  delete [] kappa_sca;
  delete [] kappa_abs;
}

int FakeOpacities::Emission(Real const rho, Real const temp, Real const Y_e,
			    Real *_eta)
{
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    //_eta[ig] = eta[ig];
    _eta[ig] = rho * eta[ig];
  }
  return 0;
}

int FakeOpacities::Absorption_abs(Real const rho, Real const temp, Real const Y_e,
				  Real * _kappa_abs)
{
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    //_kappa_abs[ig] = kappa_abs[ig];
    _kappa_abs[ig] = rho * kappa_abs[ig];
  }
  return 0;
}

int FakeOpacities::Absorption_sca(Real const rho, Real const temp, Real const Y_e,
				  Real * _kappa_sca)
{
  for (int ig = 0; ig < ngroups*nspecies; ++ig) {
    //_kappa_sca[ig] = kappa_sca[ig];
    _kappa_sca[ig] = rho * kappa_sca[ig];
  }
  return 0;
}

int FakeOpacities::AverageAtomicMass(Real const rho, Real const temp, Real const Y_e,
				 Real * Abar)
{
  *Abar = avg_atomic_mass;
  return 0;
}

Real FakeOpacities::AverageBaryonMass()
{
  return avg_baryon_mass;
}

