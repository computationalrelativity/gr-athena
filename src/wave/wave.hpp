#ifndef WAVE_HPP
#define WAVE_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave.hpp
//  \brief definitions for the Wave class

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"

// #include "../utils/finite_differencing.hpp"
#include "../bvals/vc/bvals_vc.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "../bvals/cx/bvals_cx.hpp"

#include "../utils/finite_differencing.hpp"

class MeshBlock;
class ParameterInput;

//! \class Wave
//  \brief Wave data and functions

class Wave {
public:
  Wave(MeshBlock *pmb, ParameterInput *pin);
  ~Wave();

  // data
  Mesh *pmy_mesh;               // pointer to Mesh containing MeshBlock
  MeshBlock *pmy_block;         // ptr to MeshBlock containing this Wave

  MB_info mbi;

  AthenaArray<Real> u;          // solution of the wave equation
  AthenaArray<Real> u1, u2;     // auxiliary arrays at intermediate steps
  AthenaArray<Real> rhs;        // wave equation rhs

  AthenaArray<Real> exact;      // exact solution of of the wave equation
  AthenaArray<Real> error;      // error with respect to the exact solution

  AthenaArray<Real> ref_tra;    // reference field for tr.extrema dbg

  Real c;                       // characteristic speed

  bool debug_inspect_error;     // allow for error inspection in work loop
  Real debug_abort_threshold;   // std.out & terminate on analytical compare >

  bool use_Dirichlet = false;
  bool use_Sommerfeld = false;

  // boundary and grid data
  CellCenteredBoundaryVariable ubvar_cc;
  VertexCenteredBoundaryVariable ubvar_vc;
  CellCenteredXBoundaryVariable ubvar_cx;

  AthenaArray<Real> empty_flux[3];

  // storage for SMR/AMR
  // TODO(KGF): remove trailing underscore or revert to private:
  AthenaArray<Real> coarse_u_;
  int refinement_idx{-1};

  // BT style integrators -----------------------------------------------------
  std::vector<AthenaArray<Real>> bt_k;

  // functions ----------------------------------------------------------------
  Real NewBlockTimeStep(void);  // compute new timestep on a MeshBlock
  void WaveRHS(AthenaArray<Real> &u);
  void WaveBoundaryRHS(AthenaArray<Real> &u);
  void AddWaveRHS(const Real wght, AthenaArray<Real> &u_out);

  static const int NWAVE_CPT = 2;      // num. of wave equation field components

  // For Dirichlet problem [need to be exposed for pgen]
  int M_, N_, O_;                      // max eigenfunction indices
  AthenaArray<Real> A_;                // field @ initial time
  AthenaArray<Real> B_;                // time-derivative @ initial time

  Real Lx1_;
  Real Lx2_;
  Real Lx3_;

private:
  AthenaArray<Real> dt1_,dt2_,dt3_;    // scratch arrays used in NewTimeStep

private:
  void WaveSommerfeld_1d_L_(AthenaArray<Real> & u,
                            int il, int iu, int jl, int ju, int kl, int ku);
  void WaveSommerfeld_1d_R_(AthenaArray<Real> & u,
                            int il, int iu, int jl, int ju, int kl, int ku);
  void WaveSommerfeld_2d_(AthenaArray<Real> & u,
                          int il, int iu, int jl, int ju, int kl, int ku);
  void WaveSommerfeld_3d_(AthenaArray<Real> & u,
                          int il, int iu, int jl, int ju, int kl, int ku);

  void WaveBoundaryDirichlet_(AthenaArray<Real> & u,
                              int il, int iu, int jl, int ju, int kl, int ku);


  FiniteDifference::Uniform *fd;

};
#endif // WAVE_HPP
