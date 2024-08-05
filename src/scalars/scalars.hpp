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
#include "../bvals/cc/bvals_cc.hpp"
// #include "../coordinates/coordinates.hpp"
// #include "../eos/eos.hpp"
// #include "../mesh/mesh.hpp"
// #include "../hydro/hydro.hpp"
// #include "../reconstruct/reconstruction.hpp"


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
  void CalculateFluxes(AthenaArray<Real> &s, const int order);
  void CalculateFluxesRef(AthenaArray<Real> &s, const int order);
  void CalculateFluxes_STS();

  // NOTE: for now, not creating subfolder "scalars_diffusion/", nor class ScalarDiffusion
  // that is would have an instance contained within PassiveScalars like HydroDiffusion
  // approach. Consider creating an encapsulated class as these features are generalized.
  Real nu_scalar_iso; //, nu_scalar_aniso;          // diffusion coeff
  bool scalar_diffusion_defined;
  AthenaArray<Real> diffusion_flx[3];
  // AthenaArray<Real> nu_scalar;               // diffusion array

  // No need for nu_scalar array, nor counterpart to HydroDiffusion::CalcDiffusionFlux
  // wrapper function since, currently: 1) nu_scalar_iso must be constant across the mesh
  // (does not depend on local fluid or field variables), 2) there is only one type of
  // passive scalar diffusion process (nu_scalar_aniso disabled, no "eta"l, etc.)
  // 3) nu_scalar_iso is identical for all NSCALARS
  void DiffusiveFluxIso(const AthenaArray<Real> &prim_r, const AthenaArray<Real> &w,
                        AthenaArray<Real> *flx_out);
  Real NewDiffusionDt();

  bool SpeciesWithinLimits(AthenaArray<Real> & z_, const int i);
  void ApplySpeciesLimits(AthenaArray<Real> & z_,
                          const int il, const int iu);

  void FallbackInadmissibleScalarX1_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu);

  void FallbackInadmissibleScalarX2_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu);

  void FallbackInadmissibleScalarX3_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu);

public:  // debug for combined hydro-scalar recon
  MeshBlock* pmy_block;
  // scratch space used to compute fluxes
  // 2D scratch arrays
  AthenaArray<Real> rl_, rr_, rlb_;
  AthenaArray<Real> r_rl_, r_rr_, r_rlb_;

  // 1D scratch arrays
  AthenaArray<Real> dflx_;

  void ComputeUpwindFlux(const int k, const int j, const int il,
                         const int iu, // CoordinateDirection dir,
                         AthenaArray<Real> &rl, AthenaArray<Real> &rr,
                         AthenaArray<Real> &mass_flx,
                         AthenaArray<Real> &flx_out);
  void AddDiffusionFluxes();
  // TODO(felker): dedpulicate these arrays and the same named ones in HydroDiffusion
  AthenaArray<Real> dx1_, dx2_, dx3_;
};
#endif // SCALARS_SCALARS_HPP_
