//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//  \brief functions that apply BCs for CELL_CENTERED variables

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>    // memcpy()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../globals.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../../utils/buffer_utils.hpp"
#include "../bvals.hpp"
#include "bvals_cc.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// constructor

CellCenteredBoundaryVariable::CellCenteredBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var, AthenaArray<Real> *coarse_var,
    AthenaArray<Real> *var_flux)
    : BoundaryVariable(pmb), var_cc(var), coarse_buf(coarse_var), x1flux(var_flux[X1DIR]),
      x2flux(var_flux[X2DIR]), x3flux(var_flux[X3DIR]), nl_(0), nu_(var->GetDim4() -1),
      flip_across_pole_(nullptr) {
  // CellCenteredBoundaryVariable should only be used w/ 4D or 3D (nx4=1) AthenaArray
  // For now, assume that full span of 4th dim of input AthenaArray should be used:
  // ---> get the index limits directly from the input AthenaArray
  // <=nu_ (inclusive), <nx4 (exclusive)
  if (nu_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CellCenteredBoundaryVariable constructor" << std::endl
        << "An 'AthenaArray<Real> *var' of nx4_ = " << var->GetDim4() << " was passed\n"
        << "Should be nx4 >= 1 (likely uninitialized)." << std::endl;
    ATHENA_ERROR(msg);
  }

  InitBoundaryData(bd_var_, BoundaryQuantity::cc);
#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // cc_phys_id_ = pbval_->ReserveTagVariableIDs(1);
  cc_phys_id_ = pbval_->bvars_next_phys_id_;
#endif
  if (pmy_mesh_->multilevel) { // SMR or AMR
    InitBoundaryData(bd_var_flcor_, BoundaryQuantity::cc_flcor);
#ifdef MPI_PARALLEL
    cc_flx_phys_id_ = cc_phys_id_ + 1;
#endif
  }

}

// destructor

CellCenteredBoundaryVariable::~CellCenteredBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
  if (pmy_mesh_->multilevel)
    DestroyBoundaryData(bd_var_flcor_);

}

int CellCenteredBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes& ni,
                                                            int cng) {
  MeshBlock *pmb = pmy_block_;
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng*(pmb->block_size.nx2 > 1 ? 1 : 0);
  cng3 = cng*(pmb->block_size.nx3 > 1 ? 1 : 0);

  int size = ((ni.ox1 == 0) ? pmb->block_size.nx1 : NGHOST)
    *((ni.ox2 == 0) ? pmb->block_size.nx2 : NGHOST)
    *((ni.ox3 == 0) ? pmb->block_size.nx3 : NGHOST);
  if (pmy_mesh_->multilevel) {
    // Same-level coarse payload (pre-restricted coarse data for receiver's
    // coarse_buf ghost zones)
    int same_coarse = ((ni.ox1 == 0) ? ((pmb->block_size.nx1+1)/2) : cng1)
      *((ni.ox2 == 0) ? ((pmb->block_size.nx2+1)/2) : cng2)
      *((ni.ox3 == 0) ? ((pmb->block_size.nx3+1)/2) : cng3);
    size += same_coarse;
    int f2c = ((ni.ox1 == 0) ? ((pmb->block_size.nx1+1)/2) : NGHOST)
      *((ni.ox2 == 0) ? ((pmb->block_size.nx2+1)/2) : NGHOST)
      *((ni.ox3 == 0) ? ((pmb->block_size.nx3+1)/2) : NGHOST);
    int c2f = ((ni.ox1 == 0) ?((pmb->block_size.nx1+1)/2 + cng1) : cng)
      *((ni.ox2 == 0) ? ((pmb->block_size.nx2+1)/2 + cng2) : cng)
      *((ni.ox3 == 0) ? ((pmb->block_size.nx3+1)/2 + cng3) : cng);
    size = std::max(size, c2f);
    size = std::max(size, f2c);
  }
  size *= nu_ + 1;

  return size;
}

int CellCenteredBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes& ni, int cng) {
  MeshBlock *pmb = pmy_block_;
  int size = 0;
  if (ni.ox1 != 0)
    size = (pmb->block_size.nx2 + 1)/2*(pmb->block_size.nx3 + 1)/2*(nu_ + 1);
  if (ni.ox2 != 0)
    size = (pmb->block_size.nx1 + 1)/2*(pmb->block_size.nx3 + 1)/2*(nu_ + 1);
  if (ni.ox3 != 0)
    size = (pmb->block_size.nx1 + 1)/2*(pmb->block_size.nx2 + 1)/2*(nu_ + 1);
  return size;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the same level

int CellCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
                                                               const NeighborBlock& nb) {
  int si, sj, sk, ei, ej, ek;

  idxLoadSameLevelRanges(nb.ni, si, ei, sj, ej, sk, ek, false);
  int p = 0;
  AthenaArray<Real> &var = *var_cc;

  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  // If multilevel, also pack pre-restricted coarse data so that the receiver
  // can fill its coarse_buf ghost zones directly (future-proofing for
  // higher-order CC restriction).
  if (pmy_mesh_->multilevel) {
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    // Skip coarse payload when the receiver has all same-level neighbors
    // and therefore does not need our coarse data for prolongation.
    if (!nb.neighbor_all_same_level) {
#endif
    AthenaArray<Real> &coarse_var = *coarse_buf;
    int csi, cei, csj, cej, csk, cek;
    idxLoadSameLevelRanges(nb.ni, csi, cei, csj, cej, csk, cek, true);
    BufferUtility::PackData(coarse_var, buf, nl_, nu_,
                            csi, cei, csj, cej, csk, cek, p);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    }
#endif
  }

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the coarser level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
                                                               const NeighborBlock& nb) {
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  idxLoadToCoarserRanges(nb.ni, si, ei, sj, ej, sk, ek);

  int p = 0;

  // coarse_buf interior is pre-populated by RestrictNonGhost() in SendBoundaryBuffers
  BufferUtility::PackData(coarse_var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered boundary buffers for sending to a block on the finer level

int CellCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
                                                             const NeighborBlock& nb) {
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &var = *var_cc;

  idxLoadToFinerRanges(nb.ni, si, ei, sj, ej, sk, ek);

  int p = 0;

  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  return p;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on the same level

void CellCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                         const NeighborBlock& nb) {
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &var = *var_cc;

  idxSetSameLevelRanges(nb.ni, si, ei, sj, ej, sk, ek, 1);

  int p = 0;

  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign = 1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i) {
            var(n,k,j,i) = sign * buf[p++];
          }
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }

  // If multilevel, unpack the coarse payload into coarse_buf ghost zones.
  // The sender packed its pre-restricted coarse interior cells that map to
  // our coarse ghost zone in each direction.
  if (pmy_mesh_->multilevel) {
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    // Skip coarse unpack when this block has all same-level neighbors:
    // no prolongation is needed, so coarse ghost data is unused.
    // The sender also skipped packing the coarse payload in this case.
    if (!pmy_block_->NeighborBlocksSameLevel()) {
#endif
    AthenaArray<Real> &coarse_var = *coarse_buf;
    int csi, cei, csj, cej, csk, cek;
    idxSetSameLevelRanges(nb.ni, csi, cei, csj, cej, csk, cek, 2);
    BufferUtility::UnpackData(buf, coarse_var, nl_, nu_,
                              csi, cei, csj, cej, csk, cek, p);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    }
#endif
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set cell-centered prolongation buffer received from a block on a coarser level

void CellCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
                                                          const NeighborBlock& nb) {
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  idxSetFromCoarserRanges(nb.ni, si, ei, sj, ej, sk, ek);

  int p = 0;
  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign = 1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i)
            coarse_var(n,k,j,i) = sign * buf[p++];
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, coarse_var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set cell-centered boundary received from a block on a finer level

void CellCenteredBoundaryVariable::SetBoundaryFromFiner(Real *buf,
                                                        const NeighborBlock& nb) {
  AthenaArray<Real> &var = *var_cc;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;

  idxSetFromFinerRanges(nb.ni, si, ei, sj, ej, sk, ek);

  int p = 0;
  if (nb.polar) {
    for (int n=nl_; n<=nu_; ++n) {
      Real sign=1.0;
      if (flip_across_pole_ != nullptr) sign = flip_across_pole_[n] ? -1.0 : 1.0;
      for (int k=sk; k<=ek; ++k) {
        for (int j=ej; j>=sj; --j) {
#pragma omp simd linear(p)
          for (int i=si; i<=ei; ++i)
            var(n,k,j,i) = sign * buf[p++];
        }
      }
    }
  } else {
    BufferUtility::UnpackData(buf, var, nl_, nu_, si, ei, sj, ej, sk, ek, p);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::PolarBoundarySingleAzimuthalBlock()
// \brief polar boundary edge-case: single MeshBlock spans the entire azimuthal (x3) range

void CellCenteredBoundaryVariable::PolarBoundarySingleAzimuthalBlock() {
  MeshBlock *pmb = pmy_block_;

  if (pmb->loc.level  ==  pmy_mesh_->root_level && pmy_mesh_->nrbx3 == 1
      && pmb->block_size.nx3 > 1) {
    AthenaArray<Real> &var = *var_cc;
    if (pbval_->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) {
      int nx3_half = (pmb->ke - pmb->ks + 1) / 2;
      for (int n=nl_; n<=nu_; ++n) {
        for (int j=pmb->js-NGHOST; j<=pmb->js-1; ++j) {
          for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k)
              pbval_->azimuthal_shift_(k) = var(n,k,j,i);
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
              int k_shift = k;
              k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
              var(n,k,j,i) = pbval_->azimuthal_shift_(k_shift);
            }
          }
        }
      }
    }

    if (pbval_->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
      int nx3_half = (pmb->ke - pmb->ks + 1) / 2;
      for (int n=nl_; n<=nu_; ++n) {
        for (int j=pmb->je+1; j<=pmb->je+NGHOST; ++j) {
          for (int i=pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k)
              pbval_->azimuthal_shift_(k) = var(n,k,j,i);
            for (int k=pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
              int k_shift = k;
              k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
              var(n,k,j,i) = pbval_->azimuthal_shift_(k_shift);
            }
          }
        }
      }
    }
  }
  return;
}

void CellCenteredBoundaryVariable::ProlongateBoundaries(
  const Real time, const Real dt)
{
  MeshBlock * pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;

  const int mylevel = pbval_->loc.level;
  const int nneighbor = pbval_->nneighbor;

  // dimensionality of variable common
  const int nu = var_cc->GetDim4() - 1;

  for (int n=0; n<nneighbor; ++n)
  {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.level >= mylevel) continue;

    int si, ei, sj, ej, sk, ek;

    CalculateProlongationIndices(nb, si, ei, sj, ej, sk, ek);
    pmr->ProlongateCellCenteredValues(*coarse_buf, *var_cc, 0, nu,
                                      si, ei, sj, ej, sk, ek);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SendBoundaryBuffers()
//  \brief Pre-restrict the coarse interior, then send boundary buffers.
//         Overrides BoundaryVariable::SendBoundaryBuffers to insert the
//         upfront restriction pass before the per-neighbor loop.

void CellCenteredBoundaryVariable::SendBoundaryBuffers() {
  if (pmy_mesh_->multilevel) {
    RestrictNonGhost();
  }
  BoundaryVariable::SendBoundaryBuffers();
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::RestrictNonGhost()
//  \brief Pre-restrict the entire physical coarse interior in one pass.
//         Called from SendBoundaryBuffers before the per-neighbor loop.
//         Analogous to CellCenteredXBoundaryVariable::RestrictNonGhost.

void CellCenteredBoundaryVariable::RestrictNonGhost() {
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;

  // Skip restriction when no neighbor (including ourselves) needs coarse data
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  if (pmb->NeighborBlocksSameLevel()) {
    // All our neighbors are same-level, so we have no coarser neighbor that
    // needs our restricted data via LoadBoundaryBufferToCoarser.  But a
    // same-level neighbor may still need our coarse payload for its own
    // prolongation if *it* has a finer neighbor.  Skip only when every
    // same-level neighbor also has all same-level neighbors.
    bool any_neighbor_needs_coarse = false;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      if (!pmb->pbval->neighbor[n].neighbor_all_same_level) {
        any_neighbor_needs_coarse = true;
        break;
      }
    }
    if (!any_neighbor_needs_coarse)
      return;
  }
#endif // DBG_NO_REF_NN_SAME_LEVEL

  const int nu = var_cc->GetDim4() - 1;
  pmr->RestrictCellCenteredValues(*var_cc, *coarse_buf, 0, nu,
                                  pmb->cis, pmb->cie,
                                  pmb->cjs, pmb->cje,
                                  pmb->cks, pmb->cke);
}

void CellCenteredBoundaryVariable::RestrictInterior(
  const Real time, const Real dt)
{
  // No-op: CC pre-restricts the entire coarse_buf interior in
  // SendBoundaryBuffers (via RestrictNonGhost).  Same-level coarse ghost
  // zones are filled by the coarse payload in SetBoundarySameLevel.
  // Coarser-neighbor ghost zones are filled by SetBoundaryFromCoarser.
}

void CellCenteredBoundaryVariable::SetupPersistentMPI() {
#ifdef MPI_PARALLEL
  MeshBlock* pmb = pmy_block_;
  int &mylevel = pmb->loc.level;

  int ssize, rsize;
  int tag;
  // Initialize non-polar neighbor communications to other ranks
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      if (nb.snb.level == mylevel) { // same
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
        // Skip coarse portion when the destination block has all same-level
        // neighbors and therefore does not need the coarse payload.
        ssize = MPI_BufferSizeSameLevel(nb.ni, nb.neighbor_all_same_level);
        rsize = MPI_BufferSizeSameLevel(nb.ni, pmb->NeighborBlocksSameLevel());
#else
        ssize = MPI_BufferSizeSameLevel(nb.ni);
        rsize = ssize;
#endif
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = MPI_BufferSizeToCoarser(nb.ni);
        rsize = MPI_BufferSizeFromCoarser(nb.ni);
      } else { // finer
        ssize = MPI_BufferSizeToFiner(nb.ni);
        rsize = MPI_BufferSizeFromFiner(nb.ni);
      }
      // specify the offsets in the view point of the target block: flip ox? signs

      // Initialize persistent communication requests attached to specific BoundaryData
      // cell-centered hydro: bd_hydro_
      tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      MPI_Send_init(bd_var_.send[nb.bufid], ssize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid]));
      tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      MPI_Recv_init(bd_var_.recv[nb.bufid], rsize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));

      if (FLUID_ENABLED || M1_ENABLED) {
        // hydro flux correction: bd_var_flcor_
        if (pmy_mesh_->multilevel && nb.ni.type == NeighborConnect::face) {
          int size;
          if (nb.fid == 0 || nb.fid == 1)
            size = ((pmb->block_size.nx2 + 1)/2)*((pmb->block_size.nx3 + 1)/2);
          else if (nb.fid == 2 || nb.fid == 3)
            size = ((pmb->block_size.nx1 + 1)/2)*((pmb->block_size.nx3 + 1)/2);
          else // (nb.fid == 4 || nb.fid == 5)
            size = ((pmb->block_size.nx1 + 1)/2)*((pmb->block_size.nx2 + 1)/2);
          size *= (nu_ + 1);
          if (nb.snb.level < mylevel) { // send to coarser
            tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cc_flx_phys_id_);
            if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
              MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
            MPI_Send_init(bd_var_flcor_.send[nb.bufid], size, MPI_ATHENA_REAL,
                          nb.snb.rank, tag, MPI_COMM_WORLD,
                          &(bd_var_flcor_.req_send[nb.bufid]));
          } else if (nb.snb.level > mylevel) { // receive from finer
            tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cc_flx_phys_id_);
            if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
              MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
            MPI_Recv_init(bd_var_flcor_.recv[nb.bufid], size, MPI_ATHENA_REAL,
                          nb.snb.rank, tag, MPI_COMM_WORLD,
                          &(bd_var_flcor_.req_recv[nb.bufid]));
          }
        }
      }

    }
  }
#endif
  return;
}

void CellCenteredBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
#ifdef MPI_PARALLEL
  MeshBlock *pmb = pmy_block_;
  int mylevel = pmb->loc.level;
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      if (phase != BoundaryCommSubset::matter_flux_corrected)
        MPI_Start(&(bd_var_.req_recv[nb.bufid]));

      if(FLUID_ENABLED || M1_ENABLED)
      {
        const bool phase_comm_fc = (
          (phase == BoundaryCommSubset::all) ||
          (phase == BoundaryCommSubset::m1) ||
          (phase == BoundaryCommSubset::matter_flux_corrected)
        );

        if (phase_comm_fc &&
            nb.ni.type == NeighborConnect::face &&
            nb.snb.level > mylevel) // opposite condition in ClearBoundary()
        {
          MPI_Start(&(bd_var_flcor_.req_recv[nb.bufid]));
        }
      }
    }

  }
#endif
  return;
}


void CellCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;

    if (FLUID_ENABLED || M1_ENABLED)
    {
      if (nb.ni.type == NeighborConnect::face) {
        bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::waiting;
        bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::waiting;
      }
    }

#ifdef MPI_PARALLEL
    MeshBlock *pmb = pmy_block_;
    int mylevel = pmb->loc.level;
    if (nb.snb.rank != Globals::my_rank) {
      // Wait for Isend
      if (phase != BoundaryCommSubset::matter_flux_corrected)
        MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE);

      if (FLUID_ENABLED || M1_ENABLED)
      {
        const bool phase_comm_fc = (
          (phase == BoundaryCommSubset::all) ||
          (phase == BoundaryCommSubset::m1) ||
          (phase == BoundaryCommSubset::matter_flux_corrected)
        );
        if (phase_comm_fc &&
            nb.ni.type == NeighborConnect::face &&
            nb.snb.level < mylevel)
        {
          MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
        }
      }

    }

#endif
  }

  return;
}
