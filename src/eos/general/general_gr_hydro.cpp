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
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../parameter_input.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../field/field.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"

// PrimitiveSolver headers

#include "../../z4c/primitive/primitive_solver.hpp"
#include "../../z4c/primitive/eos.hpp"
#include "../../z4c/primitive/idealgas.hpp"
#include "../../z4c/primitive/reset_floor.hpp"

// Declarations
static void PrimitiveToConservedSingle(const AthenaArray<Real> &prim, 
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, int k, int j, int i,
    AthenaArray<Real> &cons, Coordinates *pco);

Primitive::EOS<Primitive::IdealGas, Primitive::ResetFloor> eos;
Primitive::PrimitiveSolver<Primitive::IdealGas, Primitive::ResetFloor> ps{&eos};
