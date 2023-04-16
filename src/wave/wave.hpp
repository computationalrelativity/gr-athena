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
#include "../finite_differencing.hpp"

#include "../bvals/vc/bvals_vc.hpp"
#include "../bvals/cc/bvals_cc.hpp"

class MeshBlock;
class ParameterInput;

//! \class Wave
//  \brief Wave data and functions

class Wave {
public:
  Wave(MeshBlock *pmb, ParameterInput *pin);
  ~Wave();

  // data
  MeshBlock *pmy_block;         // ptr to MeshBlock containing this Wave

  AthenaArray<Real> u;          // solution of the wave equation
  AthenaArray<Real> u1, u2;     // auxiliary arrays at intermediate steps
  AthenaArray<Real> rhs;        // wave equation rhs

  AthenaArray<Real> exact;      // exact solution of of the wave equation
  AthenaArray<Real> error;      // error with respect to the exact solution

  Real c;                       // characteristic speed

  bool debug_inspect_error;     // allow for error inspection in work loop
  Real debug_abort_threshold;   // std.out & terminate on analytical compare >

  bool use_Dirichlet = false;
  bool use_Sommerfeld = false;

  // boundary and grid data
#if PREFER_VC
  VertexCenteredBoundaryVariable ubvar;
#else
  CellCenteredBoundaryVariable ubvar;
#endif

  AthenaArray<Real> empty_flux[3];

  // storage for SMR/AMR
  // TODO(KGF): remove trailing underscore or revert to private:
  AthenaArray<Real> coarse_u_;
  int refinement_idx{-1};

  // functions
  Real NewBlockTimeStep(void);  // compute new timestep on a MeshBlock
  void WaveRHS(AthenaArray<Real> &u);
  void WaveBoundaryRHS(AthenaArray<Real> &u);
  void AddWaveRHS(const Real wght, AthenaArray<Real> &u_out);

  static const int NWAVE_CPT = 2;      // num. of wave equation field components

  struct MB_info {
    int il, iu, jl, ju, kl, ku;        // local block iter.
    int nn1, nn2, nn3;                 // number of nodes (simplify switching)

    AthenaArray<Real> x1, x2, x3;      // for CC / VC grid switch
    AthenaArray<Real> cx1, cx2, cx3;   // for CC / VC grid switch (coarse)

    // provide coarse analogues of the above
    // int cnn1, cnn2, cnn3;
    // int cil, ciu, cjl, cju, ckl, cku;
  };

  MB_info mbi;

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

private:
  FiniteDifference::Uniform * pfd;

};
#endif // WAVE_HPP
