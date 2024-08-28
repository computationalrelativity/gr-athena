#ifndef BVALS_VC_BVALS_VC_HPP_
#define BVALS_VC_BVALS_VC_HPP_
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

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \class VertexCenteredBoundaryVariable
//  \brief

class VertexCenteredBoundaryVariable : public BoundaryVariable {
 public:
  VertexCenteredBoundaryVariable(MeshBlock *pmb,
                                 AthenaArray<Real> *var, AthenaArray<Real> *coarse_var,
                                 AthenaArray<Real> *var_flux);
  ~VertexCenteredBoundaryVariable();

  AthenaArray<Real> *var_vc;
  AthenaArray<Real> *coarse_buf;
  AthenaArray<Real> &x1flux, &x2flux, &x3flux;

  inline void InterchangeFundamentalCoarse() final
  {
    std::swap(var_vc, coarse_buf);
  };

  inline void ProlongateBoundaries(
    const Real time, const Real dt
  ) final;

  // maximum number of reserved unique "physics ID" component of MPI tag
  // bitfield (CellCenteredXBoundaryVariable only actually uses 1x if
  // multilevel==false, no shear) must correspond to the # of "int *phys_id_"
  // private members, below. Convert to array?
  static constexpr int max_phys_id = 3;

  // BoundaryVariable:
  int ComputeVariableBufferSize(const NeighborIndexes& ni, int cng) override;
  int ComputeFluxCorrectionBufferSize(const NeighborIndexes& ni, int cng) override {return 0;};

  // BoundaryCommunication:
  void SetupPersistentMPI() override;
  void StartReceiving(BoundaryCommSubset phase) override;
  void ClearBoundary(BoundaryCommSubset phase) override;

  void StartReceivingShear(BoundaryCommSubset phase) override {return;};
  void ComputeShear(const Real time) override {return;};

  // BoundaryBuffer:
  void SendBoundaryBuffers() override;
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
                            int si, int ei, int sj, int ej,
                            int sk, int ek,
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

  // functions pertaining to vertex consistency
  void AllocateNodeMult();
  void PrepareNodeMult();

  void ApplyNodeMultiplicitesDim3(
    AthenaArray<Real> &var,
    int ims, int ivs, int ive, int ipe, int axis_half_size_x1,
    int jms, int jvs, int jve, int jpe, int axis_half_size_x2,
    int kms, int kvs, int kve, int kpe, int axis_half_size_x3);

  void ApplyNodeMultiplicitesDim2(
    AthenaArray<Real> &var,
    int ims, int ivs, int ive, int ipe, int axis_half_size_x1,
    int jms, int jvs, int jve, int jpe, int axis_half_size_x2);

  void ApplyNodeMultiplicitesDim1(
    AthenaArray<Real> &var,
    int ims, int ivs, int ive, int ipe, int axis_half_size_x1);

  // node multiplicities ------------------------------------------------------
  AthenaArray<unsigned short int> node_mult;
  // BD: TODO - flip/flop based on neighbour changes during AMR?
  bool node_mult_assembled = false;

  int c_ims = 0, c_ivs = 1, c_ive = 5, c_ipe = 6;
  int c_jms = 0, c_jvs = 1, c_jve = 5, c_jpe = 6;
  int c_kms = 0, c_kvs = 1, c_kve = 5, c_kpe = 6;
  //---------------------------------------------------------------------------

#ifdef MPI_PARALLEL
  int vc_phys_id_; //, cc_flx_phys_id_;
#endif
};

#endif // BVALS_VC_BVALS_VC_HPP_
