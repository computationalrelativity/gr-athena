//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file general_gr_hydro.cpp
//  \brief Implements functions for going between primitive and conserved variables in
//  general-relativistic hydrodynamics, as well as for computing wavespeeds.

// C++ headers
#include <algorithm>
#include <cfloat>
#include <cmath>

// Athena++ headers
#include "../eos.hpp"

// PrimitiveSolver headers

/*#include "../../z4c/primitive/primitive_solver.hpp"
#include "../../z4c/primitive/eos.hpp"
#include "../../z4c/primitive/idealgas.hpp"
#include "../../z4c/primitive/reset_floor.hpp"*/

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//  \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  // FIXME: Adjust to work properly with particle fractions.
  Real n = rho/eos.GetBaryonMass();
  Real Y[MAX_SPECIES] = {0.0};
  Real T = eos.GetTemperatureFromE(n, egas, Y);

  return eos.GetPressure(n, T, Y);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//  \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  // FIXME: Adjust to work properly with particle fractions.
  Real n = rho/eos.GetBaryonMass();
  Real Y[MAX_SPECIES] = {0.0};
  Real T = eos.GetTemperatureFromP(n, pres, Y);

  return eos.GetEnergy(n, T, Y);
}

//----------------------------------------------------------------------------------------
//! void EquationOfState::InitEosConstants(ParameterInput* pin)
//  \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  return;
}
