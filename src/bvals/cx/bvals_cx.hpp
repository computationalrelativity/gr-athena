#ifndef BVALS_CX_BVALS_CX_HPP_
#define BVALS_CX_BVALS_CX_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_vc.hpp
//  \brief handle boundaries for any AthenaArray type variable that represents a physical
//         quantity indexed along / located around vertices

// C headers

// C++ headers

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../bvals.hpp"
#include "../bvals_interfaces.hpp"
#include <iostream>

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \class CellCenteredXBoundaryVariable
//  \brief

class CellCenteredXBoundaryVariable : public BoundaryVariable
{
 public:
  CellCenteredXBoundaryVariable(MeshBlock *pmb,
                                 AthenaArray<Real> *var, AthenaArray<Real> *coarse_var,
                                 AthenaArray<Real> *var_flux);
  ~CellCenteredXBoundaryVariable();

  AthenaArray<Real> *var_cx;
  AthenaArray<Real> *coarse_buf;
  AthenaArray<Real> &x1flux, &x2flux, &x3flux;

  // BoundaryInterface:
  inline void InterchangeFundamentalCoarse() final
  {
    std::swap(var_cx, coarse_buf);
  };

  void ProlongateBoundaries(const Real time, const Real dt) final;

  // maximum number of reserved unique "physics ID" component of MPI tag
  // bitfield (CellCenteredXBoundaryVariable only actually uses 1x if
  // multilevel==false, no shear) must correspond to the # of "int *phys_id_"
  // private members, below. Convert to array?
  static constexpr int max_phys_id = 3;

  // BoundaryVariable:
  int ComputeVariableBufferSize(const NeighborIndexes &ni, int cng) override;

  int ComputeFluxCorrectionBufferSize(const NeighborIndexes &ni,
                                      int cng) override
  {
    return 0;
  };

  // BoundaryCommunication:
  void SetupPersistentMPI() override;
  void StartReceiving(BoundaryCommSubset phase) override;
  void ClearBoundary(BoundaryCommSubset phase) override;

  void StartReceivingShear(BoundaryCommSubset phase) override {return;};
  void ComputeShear(const Real time) override {return;};

  // BoundaryBuffer:
  void SendBoundaryBuffers() override;
  void SendBoundaryBuffersFullRestriction();
  void ReceiveAndSetBoundariesWithWait() override;
  void SetBoundaries() override;
  void SendFluxCorrection() override {return;};
  bool ReceiveFluxCorrection() override {return false;};

  void RestrictNonGhost();
  void ZeroVertexGhosts();
  void FinalizeVertexConsistency();

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

private:
  // --------------------------------------------------------------------------
  // buffer / index calculators
  void AccumulateBufferSize(int sn, int en,
                            int si, int ei, int sj, int ej, int sk, int ek,
                            int &offset, int ijk_step);

  void idxLoadSameLevelRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    bool is_coarse);

  void idxLoadToCoarserRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    bool is_coarse);

  void idxLoadToFinerRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek);

  void idxSetSameLevelRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int type);

  void idxSetFromCoarserRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    bool is_node_mult);

  void idxSetFromFinerRanges(const NeighborIndexes& ni,
    int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
    int type);


  int NeighborVariableBufferSize(const NeighborIndexes& ni);

#ifdef MPI_PARALLEL
  int MPI_BufferSizeSameLevel(const NeighborIndexes& ni,
    bool is_send);

  int MPI_BufferSizeToCoarser(const NeighborIndexes& ni);
  int MPI_BufferSizeFromCoarser(const NeighborIndexes& ni);

  int MPI_BufferSizeToFiner(const NeighborIndexes& ni);
  int MPI_BufferSizeFromFiner(const NeighborIndexes& ni);
#endif

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
  // --------------------------------------------------------------------------

  // BoundaryBuffer:
  int LoadBoundaryBufferSameLevel(Real *buf, const NeighborBlock& nb) override;
  void SetBoundarySameLevel(Real *buf, const NeighborBlock& nb) override;

  int LoadBoundaryBufferToCoarser(Real *buf, const NeighborBlock& nb) override;
  int LoadBoundaryBufferToFiner(Real *buf, const NeighborBlock& nb) override;

  void SetBoundaryFromCoarser(Real *buf, const NeighborBlock& nb) override;
  void SetBoundaryFromFiner(Real *buf, const NeighborBlock& nb) override;

  void PolarBoundarySingleAzimuthalBlock() override;

  void ErrorIfPolarNotImplemented(const NeighborBlock& nb);
  void ErrorIfShearingBoxNotImplemented();

  // helper functions for assigning indices (inlined on definition)
  void SetIndexRangesSBSL(int ox, int &ix_s, int &ix_e,
                          int ix_vs, int ix_ve, int ix_ms, int ix_pe);

  void SetIndexRangesSBFC(int ox, int &ix_s, int &ix_e,
                          int ix_cvs, int ix_cve, int ix_cms, int ix_cme,
                          int ix_cps, int ix_cpe, bool level_flag);

  void SetIndexRangesSBFF(int ox, int &ix_s, int &ix_e,
                          int ix_vs, int ix_ve, int ix_ms, int ix_pe,
                          int fi1, int fi2, int axis_half_size,
                          bool size_flag, bool offset_flag);

#ifdef MPI_PARALLEL
  int cx_phys_id_; //, cc_flx_phys_id_;
#endif
};

#endif // BVALS_CX_BVALS_CX_HPP_
