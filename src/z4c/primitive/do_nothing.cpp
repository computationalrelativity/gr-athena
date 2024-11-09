//! \file do_nothing.cpp
//  \brief Implementation of the DoNothing policy

#include <cmath>

#include "do_nothing.hpp"
#include "ps_error.hpp"

using namespace Primitive;

/// Constructor
DoNothing::DoNothing() {
  fail_conserved_floor = false;
  fail_primitive_floor = false;
  adjust_conserved = false;
}

bool DoNothing::PrimitiveFloor(Real& n, Real v[3], Real& T, Real *Y, int n_species) {
  return false;
}

bool DoNothing::ConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y, Real D_floor, 
      Real tau_floor, Real tau_abs_floor, int n_species) {
  return false;
}

Error DoNothing::MagnetizationResponse(Real& bsq, Real b_u[3]) {
  return Error::SUCCESS;
}

void DoNothing::DensityLimits(Real& n, Real n_min, Real n_max) {
  return;
}

void DoNothing::TemperatureLimits(Real& T, Real T_min, Real T_max) {
  return;
}

void DoNothing::PressureLimits(Real& P, Real P_min, Real P_max) {
  return;
}

void DoNothing::EnergyLimits(Real& e, Real e_min, Real e_max) {
  return;
}

bool DoNothing::SpeciesLimits(Real* Y, Real* Y_min, Real* Y_max, int n_species) {
  return;
}

bool DoNothing::FailureResponse(Real prim[NPRIM])
{
  return false;
}
