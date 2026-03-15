#ifndef SCALARS_SCALARS_HPP_
#define SCALARS_SCALARS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalars.hpp
//  \brief definitions for PassiveScalars class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_aliases.hpp"
#include "../bvals/cc/bvals_cc.hpp"
// #include "../coordinates/coordinates.hpp"
// #include "../eos/eos.hpp"
// #include "../mesh/mesh.hpp"
// #include "../hydro/hydro.hpp"
// #include "../reconstruct/reconstruction.hpp"

using namespace gra::aliases;

// class MeshBlock;
class ParameterInput;

//! \class PassiveScalars
//  \brief

// TODO(felker): consider renaming to Scalars
class PassiveScalars {
 public:
  // TODO(felker): pin is currently only used for checking ssprk5_4, otherwise unused.
  // Leaving as ctor parameter in case of run-time "nscalars" option
  PassiveScalars(MeshBlock *pmb, ParameterInput *pin);

  // public data:
  // "conserved vars" = passive scalar mass
  AthenaArray<Real> s, s1, s2;  // (no more than MAX_NREGISTER allowed)
  // "primitive vars" = (density-normalized) mass fraction/concentration of each species
  AthenaArray<Real> r;  // , r1;
  AthenaArray<Real> s_flux[3];  // face-averaged flux vector

  // storage for SMR/AMR
  // TODO(KGF): remove trailing underscore or revert to private:
  AthenaArray<Real> coarse_s_, coarse_r_;
  int refinement_idx{-1};

  CellCenteredBoundaryVariable sbvar;

  // public functions:
  // KGF: use inheritance for these functions / overall class?
  void AddFluxDivergence(const Real wght, AthenaArray<Real> &s_out);

 public:
  MeshBlock* pmy_block;
  // scratch space used to compute fluxes
  // 1D scratch arrays
  AthenaArray<Real> dflx_;
};
#endif // SCALARS_SCALARS_HPP_
