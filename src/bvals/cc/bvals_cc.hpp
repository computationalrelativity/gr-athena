#ifndef BVALS_CC_BVALS_CC_HPP_
#define BVALS_CC_BVALS_CC_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.hpp
//  \brief handle boundaries for any AthenaArray type variable that represents a physical
//         quantity indexed along / located around cell-centers

// C headers

// C++ headers

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../bvals.hpp"
#include "../bvals_interfaces.hpp"

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//  \brief

class CellCenteredBoundaryVariable : public BoundaryVariable {
 public:
  CellCenteredBoundaryVariable(MeshBlock *pmb,
                               AthenaArray<Real> *var, AthenaArray<Real> *coarse_var,
                               AthenaArray<Real> *var_flux);
  ~CellCenteredBoundaryVariable();

  // may want to rebind var_cc to u,u1,u2,w,w1, etc. registers for time integrator logic.
  // Also, derived class HydroBoundaryVariable needs to keep switching var and coarse_var
  // arrays between primitive and conserved variables ---> ptr members, not references
  AthenaArray<Real> *var_cc;
  AthenaArray<Real> *coarse_buf;  // may pass nullptr if mesh refinement is unsupported
  AthenaArray<Real> &x1flux, &x2flux, &x3flux;

  inline void InterchangeFundamentalCoarse() final
  {
    std::swap(var_cc, coarse_buf);
  };

  void ProlongateBoundaries(const Real time, const Real dt) final;
  void RestrictInterior(const Real time, const Real dt) final;

  // Pre-restrict the entire physical coarse interior in one pass.
  // Called from SendBoundaryBuffers before the per-neighbor loop.
  void RestrictNonGhost();

  // maximum number of reserved unique "physics ID" component of MPI tag
  // bitfield (CellCenteredXBoundaryVariable only actually uses 1x if
  // multilevel==false) must correspond to the # of "int *phys_id_"
  // private members, below. Convert to array?
  static constexpr int max_phys_id = 3;

  // BoundaryVariable:
  int ComputeVariableBufferSize(const NeighborIndexes& ni, int cng) override;
  int ComputeFluxCorrectionBufferSize(const NeighborIndexes& ni, int cng) override;

  // BoundaryCommunication:
  void SendBoundaryBuffers() override;
  void SetupPersistentMPI() override;
  void StartReceiving(BoundaryCommSubset phase) override;
  void ClearBoundary(BoundaryCommSubset phase) override;

  // BoundaryBuffer:
  void SendFluxCorrection() override;
  bool ReceiveFluxCorrection() override;

  // BoundaryPhysics:
  void ReflectInnerX1(Real time, Real dt,
                      int il, int jl, int ju, int kl, int ku, int ngh) override;
  void ReflectOuterX1(Real time, Real dt,
                      int iu, int jl, int ju, int kl, int ku, int ngh) override;
  void ReflectInnerX2(Real time, Real dt,
                      int il, int iu, int jl, int kl, int ku, int ngh) override;
  void ReflectOuterX2(Real time, Real dt,
                      int il, int iu, int ju, int kl, int ku, int ngh) override;
  void ReflectInnerX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int kl, int ngh) override;
  void ReflectOuterX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int ku, int ngh) override;

  void OutflowInnerX1(Real time, Real dt,
                      int il, int jl, int ju, int kl, int ku, int ngh) override;
  void OutflowOuterX1(Real time, Real dt,
                      int iu, int jl, int ju, int kl, int ku, int ngh) override;
  void OutflowInnerX2(Real time, Real dt,
                      int il, int iu, int jl, int kl, int ku, int ngh) override;
  void OutflowOuterX2(Real time, Real dt,
                      int il, int iu, int ju, int kl, int ku, int ngh) override;
  void OutflowInnerX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int kl, int ngh) override;
  void OutflowOuterX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int ku, int ngh) override;

  void ExtrapolateOutflowInnerX1(Real time, Real dt,
                                 int il, int jl, int ju, int kl, int ku,
                                 int ngh) override;
  void ExtrapolateOutflowOuterX1(Real time, Real dt,
                                 int iu, int jl, int ju, int kl, int ku,
                                 int ngh) override;
  void ExtrapolateOutflowInnerX2(Real time, Real dt,
                                 int il, int iu, int jl, int kl, int ku,
                                 int ngh) override;
  void ExtrapolateOutflowOuterX2(Real time, Real dt,
                                 int il, int iu, int ju, int kl, int ku,
                                 int ngh) override;
  void ExtrapolateOutflowInnerX3(Real time, Real dt,
                                 int il, int iu, int jl, int ju, int kl,
                                 int ngh) override;
  void ExtrapolateOutflowOuterX3(Real time, Real dt,
                                 int il, int iu, int jl, int ju, int ku,
                                 int ngh) override;

  void GRSommerfeldInnerX1(Real time, Real dt,
                           int il, int jl, int ju, int kl, int ku,
                           int ngh) override;
  void GRSommerfeldOuterX1(Real time, Real dt,
                           int iu, int jl, int ju, int kl, int ku,
                           int ngh) override;
  void GRSommerfeldInnerX2(Real time, Real dt,
                           int il, int iu, int jl, int kl, int ku,
                           int ngh) override;
  void GRSommerfeldOuterX2(Real time, Real dt,
                           int il, int iu, int ju, int kl, int ku,
                           int ngh) override;
  void GRSommerfeldInnerX3(Real time, Real dt,
                           int il, int iu, int jl, int ju, int kl,
                           int ngh) override;
  void GRSommerfeldOuterX3(Real time, Real dt,
                           int il, int iu, int jl, int ju, int ku,
                           int ngh) override;

  void PolarWedgeInnerX2(Real time, Real dt,
                         int il, int iu, int jl, int kl, int ku, int ngh) override;
  void PolarWedgeOuterX2(Real time, Real dt,
                         int il, int iu, int ju, int kl, int ku, int ngh) override;

 protected:
  int nl_, nu_;
  const bool *flip_across_pole_;

  // --------------------------------------------------------------------------
  // buffer / index calculators
private:

  inline void CalculateProlongationIndices(
    std::int64_t &lx, int ox, int pcng, int cix_vs, int cix_ve,
    int &set_ix_vs, int &set_ix_ve,
    bool is_dim_nontrivial)
  {
    if (ox > 0) {
      set_ix_vs = cix_ve+1;
      set_ix_ve = cix_ve+pcng;
    } else if (ox < 0) {
      set_ix_vs = cix_vs-pcng;
      set_ix_ve = cix_vs-1;
    } else {  // ox == 0
      set_ix_vs = cix_vs;
      set_ix_ve = cix_ve;
      if (is_dim_nontrivial) {
        std::int64_t &lx_ = lx;
        if ((lx_ & 1LL) == 0LL) {
          set_ix_ve += pcng;
        } else {
          set_ix_vs -= pcng;
        }
      }
    }
  }

public:

  void CalculateProlongationIndices(NeighborBlock &nb,
                                    int &si, int &ei,
                                    int &sj, int &ej,
                                    int &sk, int &ek);

  void CalculateProlongationIndicesFine(NeighborBlock &nb,
                                        int &fsi, int &fei,
                                        int &fsj, int &fej,
                                        int &fsk, int &fek);
  // --------------------------------------------------------------------------

private:
  // --------------------------------------------------------------------------
  // idx_utilities helpers (defined in idx_utilities_cc.cpp)
  void idxLoadSameLevelRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    bool is_coarse);

  void idxLoadToCoarserRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek);

  void idxLoadToFinerRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek);

  void idxSetSameLevelRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int type);

  void idxSetFromCoarserRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek);

  void idxSetFromFinerRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek);

#ifdef MPI_PARALLEL
  int MPI_BufferSizeSameLevel(const NeighborIndexes& ni,
    bool skip_coarse=false);
  int MPI_BufferSizeToCoarser(const NeighborIndexes& ni);
  int MPI_BufferSizeFromCoarser(const NeighborIndexes& ni);
  int MPI_BufferSizeToFiner(const NeighborIndexes& ni);
  int MPI_BufferSizeFromFiner(const NeighborIndexes& ni);
#endif
  // --------------------------------------------------------------------------

  // BoundaryBuffer:
  int LoadBoundaryBufferSameLevel(Real *buf, const NeighborBlock& nb) override;
  void SetBoundarySameLevel(Real *buf, const NeighborBlock& nb) override;

  int LoadBoundaryBufferToCoarser(Real *buf, const NeighborBlock& nb) override;
  int LoadBoundaryBufferToFiner(Real *buf, const NeighborBlock& nb) override;

  void SetBoundaryFromCoarser(Real *buf, const NeighborBlock& nb) override;
  void SetBoundaryFromFiner(Real *buf, const NeighborBlock& nb) override;

  void PolarBoundarySingleAzimuthalBlock() override;

#ifdef MPI_PARALLEL
  int cc_phys_id_, cc_flx_phys_id_;
#endif
};

#endif // BVALS_CC_BVALS_CC_HPP_
