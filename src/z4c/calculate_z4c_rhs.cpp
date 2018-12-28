//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM decomposition

// C++ standard headers
#include <cmath> // exp, pow

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../mesh/mesh.hpp"

void Z4c::Z4cRHS(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
{
  Z4c_vars z4c, rhs;
  SetZ4cAliases(u, z4c);
  SetZ4cAliases(u_rhs, rhs);
#warning "Z4cRHS not implemented yet!"
  u_rhs.Zero();
}
