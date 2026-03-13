//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file bvals_fc.cpp
//  \brief functions that apply BCs for FACE_CENTERED variables

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring> // memcpy()
#include <iomanip>
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

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
#include "bvals_fc.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// constructor

FaceCenteredBoundaryVariable::FaceCenteredBoundaryVariable(
    MeshBlock *pmb, FaceField *var, FaceField *coarse_buf, EdgeField &var_flux)
    : BoundaryVariable(pmb), var_fc(var), coarse_buf(coarse_buf),
      e1(var_flux.x1e), e2(var_flux.x2e), e3(var_flux.x3e),
      flip_across_pole_(flip_across_pole_field) {
  // assuming Field, not generic FaceCenteredBoundaryVariable:
  // flip_across_pole_ = flip_across_pole_field;

  InitBoundaryData(bd_var_, BoundaryQuantity::fc);
  InitBoundaryData(bd_var_flcor_, BoundaryQuantity::fc_flcor);

#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // fc_phys_id_ = pbval_->ReserveTagVariableIDs(2);
  fc_phys_id_ = pbval_->bvars_next_phys_id_;
  fc_flx_phys_id_ = fc_phys_id_ + 1;
  if (pbval_->num_north_polar_blocks_ > 0 ||
      pbval_->num_south_polar_blocks_ > 0) {
    fc_flx_pole_phys_id_ = fc_flx_phys_id_ + 1;
  }
#endif

  if (pbval_->num_north_polar_blocks_ > 0) {
    flux_north_send_ = new Real *[pbval_->num_north_polar_blocks_];
    flux_north_recv_ = new Real *[pbval_->num_north_polar_blocks_];
    flux_north_flag_ =
        new std::atomic<BoundaryStatus>[pbval_->num_north_polar_blocks_];
#ifdef MPI_PARALLEL
    req_flux_north_send_ = new MPI_Request[pbval_->num_north_polar_blocks_];
    req_flux_north_recv_ = new MPI_Request[pbval_->num_north_polar_blocks_];
#endif
    for (int n = 0; n < pbval_->num_north_polar_blocks_; ++n) {
      flux_north_send_[n] = nullptr;
      flux_north_recv_[n] = nullptr;
      flux_north_flag_[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
      req_flux_north_send_[n] = MPI_REQUEST_NULL;
      req_flux_north_recv_[n] = MPI_REQUEST_NULL;
#endif
    }
  }
  if (pbval_->num_south_polar_blocks_ > 0) {
    flux_south_send_ = new Real *[pbval_->num_south_polar_blocks_];
    flux_south_recv_ = new Real *[pbval_->num_south_polar_blocks_];
    flux_south_flag_ =
        new std::atomic<BoundaryStatus>[pbval_->num_south_polar_blocks_];
#ifdef MPI_PARALLEL
    req_flux_south_send_ = new MPI_Request[pbval_->num_south_polar_blocks_];
    req_flux_south_recv_ = new MPI_Request[pbval_->num_south_polar_blocks_];
#endif
    for (int n = 0; n < pbval_->num_south_polar_blocks_; ++n) {
      flux_south_send_[n] = nullptr;
      flux_south_recv_[n] = nullptr;
      flux_south_flag_[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
      req_flux_south_send_[n] = MPI_REQUEST_NULL;
      req_flux_south_recv_[n] = MPI_REQUEST_NULL;
#endif
    }
  }

  // Allocate buffers for polar neighbor communication
  if (pbval_->num_north_polar_blocks_ > 0) {
    for (int n = 0; n < pbval_->num_north_polar_blocks_; ++n) {
      flux_north_send_[n] = new Real[pmb->block_size.nx1];
      flux_north_recv_[n] = new Real[pmb->block_size.nx1];
    }
  }
  if (pbval_->num_south_polar_blocks_ > 0) {
    for (int n = 0; n < pbval_->num_south_polar_blocks_; ++n) {
      flux_south_send_[n] = new Real[pmb->block_size.nx1];
      flux_south_recv_[n] = new Real[pmb->block_size.nx1];
    }
  }
}

// destructor

FaceCenteredBoundaryVariable::~FaceCenteredBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
  DestroyBoundaryData(bd_var_flcor_);

  if (pbval_->num_north_polar_blocks_ > 0) {
    for (int n = 0; n < pbval_->num_north_polar_blocks_; ++n) {
      delete[] flux_north_send_[n];
      delete[] flux_north_recv_[n];
#ifdef MPI_PARALLEL
      if (req_flux_north_send_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_north_send_[n]);
      if (req_flux_north_recv_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_north_recv_[n]);
#endif
    }
    delete[] flux_north_send_;
    delete[] flux_north_recv_;
    delete[] flux_north_flag_;
#ifdef MPI_PARALLEL
    delete[] req_flux_north_send_;
    delete[] req_flux_north_recv_;
#endif
  }
  if (pbval_->num_south_polar_blocks_ > 0) {
    for (int n = 0; n < pbval_->num_south_polar_blocks_; ++n) {
      delete[] flux_south_send_[n];
      delete[] flux_south_recv_[n];
#ifdef MPI_PARALLEL
      if (req_flux_south_send_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_south_send_[n]);
      if (req_flux_south_recv_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_south_recv_[n]);
#endif
    }
    delete[] flux_south_send_;
    delete[] flux_south_recv_;
    delete[] flux_south_flag_;
#ifdef MPI_PARALLEL
    delete[] req_flux_south_send_;
    delete[] req_flux_south_recv_;
#endif
  }
}

int FaceCenteredBoundaryVariable::ComputeVariableBufferSize(
    const NeighborIndexes &ni, int cng) {
  MeshBlock *pmb = pmy_block_;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int f2 = pmy_mesh_->f2, f3 = pmy_mesh_->f3;
  int cng1, cng2, cng3;
  cng1 = cng;
  cng2 = cng * f2;
  cng3 = cng * f3;

  int size1 = ((ni.ox1 == 0) ? (nx1 + 1) : NGHOST) *
              ((ni.ox2 == 0) ? (nx2) : NGHOST) *
              ((ni.ox3 == 0) ? (nx3) : NGHOST);
  int size2 = ((ni.ox1 == 0) ? (nx1) : NGHOST) *
              ((ni.ox2 == 0) ? (nx2 + f2) : NGHOST) *
              ((ni.ox3 == 0) ? (nx3) : NGHOST);
  int size3 = ((ni.ox1 == 0) ? (nx1) : NGHOST) *
              ((ni.ox2 == 0) ? (nx2) : NGHOST) *
              ((ni.ox3 == 0) ? (nx3 + f3) : NGHOST);
  int size = size1 + size2 + size3;
  if (pmy_mesh_->multilevel) {
    if (ni.type != NeighborConnect::face) {
      if (ni.ox1 != 0)
        size1 = size1 / NGHOST * (NGHOST + 1);
      if (ni.ox2 != 0)
        size2 = size2 / NGHOST * (NGHOST + 1);
      if (ni.ox3 != 0)
        size3 = size3 / NGHOST * (NGHOST + 1);
    }
    size = size1 + size2 + size3;
    // Same-level coarse payload (pre-restricted coarse data for receiver's
    // coarse_buf ghost zones, one pack per face component)
    int sc1 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + 1) : cng) *
              ((ni.ox2 == 0) ? ((nx2 + 1) / 2) : cng2) *
              ((ni.ox3 == 0) ? ((nx3 + 1) / 2) : cng3);
    int sc2 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2) : cng) *
              ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + f2) : cng2) *
              ((ni.ox3 == 0) ? ((nx3 + 1) / 2) : cng3);
    int sc3 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2) : cng) *
              ((ni.ox2 == 0) ? ((nx2 + 1) / 2) : cng2) *
              ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + f3) : cng3);
    size += sc1 + sc2 + sc3;
    int f2c1 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + 1) : NGHOST) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2) : NGHOST) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2) : NGHOST);
    int f2c2 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2) : NGHOST) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + f2) : NGHOST) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2) : NGHOST);
    int f2c3 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2) : NGHOST) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2) : NGHOST) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + f3) : NGHOST);
    if (ni.type != NeighborConnect::face) {
      if (ni.ox1 != 0)
        f2c1 = f2c1 / NGHOST * (NGHOST + 1);
      if (ni.ox2 != 0)
        f2c2 = f2c2 / NGHOST * (NGHOST + 1);
      if (ni.ox3 != 0)
        f2c3 = f2c3 / NGHOST * (NGHOST + 1);
    }
    int fsize = f2c1 + f2c2 + f2c3;
    int c2f1 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1 + 1) : cng + 1) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2) : cng) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3) : cng);
    int c2f2 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1) : cng) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2 + f2) : cng + 1) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3) : cng);
    int c2f3 = ((ni.ox1 == 0) ? ((nx1 + 1) / 2 + cng1) : cng) *
               ((ni.ox2 == 0) ? ((nx2 + 1) / 2 + cng2) : cng) *
               ((ni.ox3 == 0) ? ((nx3 + 1) / 2 + cng3 + f3) : cng + 1);
    int csize = c2f1 + c2f2 + c2f3;
    size = std::max(size, std::max(csize, fsize));
  }
  return size;
}

int FaceCenteredBoundaryVariable::ComputeFluxCorrectionBufferSize(
    const NeighborIndexes &ni, int cng) {
  MeshBlock *pmb = pmy_block_;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int size = 0;

  if (ni.type == NeighborConnect::face) {
    if (nx3 > 1) { // 3D
      if (ni.ox1 != 0)
        size = (nx2 + 1) * (nx3) + (nx2) * (nx3 + 1);
      else if (ni.ox2 != 0)
        size = (nx1 + 1) * (nx3) + (nx1) * (nx3 + 1);
      else
        size = (nx1 + 1) * (nx2) + (nx1) * (nx2 + 1);
    } else if (nx2 > 1) { // 2D
      if (ni.ox1 != 0)
        size = (nx2 + 1) + nx2;
      else
        size = (nx1 + 1) + nx1;
    } else { // 1D
      size = 2;
    }
  } else if (ni.type == NeighborConnect::edge) {
    if (nx3 > 1) { // 3D
      if (ni.ox3 == 0)
        size = nx3;
      if (ni.ox2 == 0)
        size = nx2;
      if (ni.ox1 == 0)
        size = nx1;
    } else if (nx2 > 1) {
      size = 1;
    }
  }
  return size;
}

//----------------------------------------------------------------------------------------
//! \fn int FaceCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
//                                                               const
//                                                               NeighborBlock&
//                                                               nb)
//  \brief Set face-centered boundary buffers for sending to a block on the same
//  level

int FaceCenteredBoundaryVariable::LoadBoundaryBufferSameLevel(
    Real *buf, const NeighborBlock &nb) {
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  // bx1 (stagger_axis=0)
  idxLoadSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 0, false);
  BufferUtility::PackData((*var_fc).x1f, buf, si, ei, sj, ej, sk, ek, p);

  // bx2 (stagger_axis=1)
  idxLoadSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 1, false);
  BufferUtility::PackData((*var_fc).x2f, buf, si, ei, sj, ej, sk, ek, p);

  // bx3 (stagger_axis=2)
  idxLoadSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 2, false);
  BufferUtility::PackData((*var_fc).x3f, buf, si, ei, sj, ej, sk, ek, p);

  // If multilevel, also pack pre-restricted coarse data so that the receiver
  // can fill its coarse_buf ghost zones directly.
  if (pmy_mesh_->multilevel) {
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    // Skip coarse payload when the receiver has all same-level neighbors
    // and therefore does not need our coarse data for prolongation.
    if (!nb.neighbor_all_same_level) {
#endif
      // coarse bx1 (stagger_axis=0)
      idxLoadSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 0, true);
      BufferUtility::PackData(coarse_buf->x1f, buf, si, ei, sj, ej, sk, ek, p);

      // coarse bx2 (stagger_axis=1)
      idxLoadSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 1, true);
      BufferUtility::PackData(coarse_buf->x2f, buf, si, ei, sj, ej, sk, ek, p);

      // coarse bx3 (stagger_axis=2)
      idxLoadSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 2, true);
      BufferUtility::PackData(coarse_buf->x3f, buf, si, ei, sj, ej, sk, ek, p);
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    }
#endif
  }

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int FaceCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
//                                                                const
//                                                                NeighborBlock&
//                                                                nb)
//  \brief Set face-centered boundary buffers for sending to a block on the
//  coarser level

int FaceCenteredBoundaryVariable::LoadBoundaryBufferToCoarser(
    Real *buf, const NeighborBlock &nb) {
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  // coarse_buf is already populated by RestrictNonGhost() in
  // SendBoundaryBuffers. Just compute indices and pack from the pre-restricted
  // data.

  // bx1 (stagger_axis=0)
  idxLoadToCoarserRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 0);
  BufferUtility::PackData(coarse_buf->x1f, buf, si, ei, sj, ej, sk, ek, p);

  // bx2 (stagger_axis=1)
  idxLoadToCoarserRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 1);
  BufferUtility::PackData(coarse_buf->x2f, buf, si, ei, sj, ej, sk, ek, p);

  // bx3 (stagger_axis=2)
  idxLoadToCoarserRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 2);
  BufferUtility::PackData(coarse_buf->x3f, buf, si, ei, sj, ej, sk, ek, p);

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int FaceCenteredBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
//                                                                const
//                                                                NeighborBlock&
//                                                                nb)
//  \brief Set face-centered boundary buffers for sending to a block on the
//  finer level

int FaceCenteredBoundaryVariable::LoadBoundaryBufferToFiner(
    Real *buf, const NeighborBlock &nb) {
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  // send the data first and later prolongate on the target block
  // need to add edges for faces, add corners for edges
  // bx1 (stagger_axis=0)
  idxLoadToFinerRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 0);
  BufferUtility::PackData((*var_fc).x1f, buf, si, ei, sj, ej, sk, ek, p);

  // bx2 (stagger_axis=1)
  idxLoadToFinerRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 1);
  BufferUtility::PackData((*var_fc).x2f, buf, si, ei, sj, ej, sk, ek, p);

  // bx3 (stagger_axis=2)
  idxLoadToFinerRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 2);
  BufferUtility::PackData((*var_fc).x3f, buf, si, ei, sj, ej, sk, ek, p);

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::SetBoundarySameLevel(Real *buf,
//                                                              const
//                                                              NeighborBlock&
//                                                              nb)
//  \brief Set face-centered boundary received from a block on the same level

void FaceCenteredBoundaryVariable::SetBoundarySameLevel(
    Real *buf, const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  int p = 0;
  // bx1 (stagger_axis=0)
  idxSetSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 0, 1);
  if (nb.polar) {
    Real sign = flip_across_pole_[IB1] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          (*var_fc).x1f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, (*var_fc).x1f, si, ei, sj, ej, sk, ek, p);
  }

  // bx2 (stagger_axis=1)
  idxSetSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 1, 1);
  if (nb.polar) {
    Real sign = flip_across_pole_[IB2] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          (*var_fc).x2f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, (*var_fc).x2f, si, ei, sj, ej, sk, ek, p);
  }
  if (pmb->block_size.nx2 == 1) { // 1D
#pragma omp simd
    for (int i = si; i <= ei; ++i)
      (*var_fc).x2f(sk, sj + 1, i) = (*var_fc).x2f(sk, sj, i);
  }

  // bx3 (stagger_axis=2)
  idxSetSameLevelRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 2, 1);
  if (nb.polar) {
    Real sign = flip_across_pole_[IB3] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          (*var_fc).x3f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, (*var_fc).x3f, si, ei, sj, ej, sk, ek, p);
  }
  if (pmb->block_size.nx3 == 1) { // 1D or 2D
    for (int j = sj; j <= ej; ++j) {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        (*var_fc).x3f(sk + 1, j, i) = (*var_fc).x3f(sk, j, i);
    }
  }

  // If multilevel, unpack the coarse payload into coarse_buf ghost zones.
  // The sender packed its pre-restricted coarse interior for each face
  // component.
  if (pmy_mesh_->multilevel) {
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    // Skip coarse unpack when this block has all same-level neighbors:
    // no prolongation is needed, so coarse ghost data is unused.
    // The sender also skipped packing the coarse payload in this case.
    if (!pmy_block_->NeighborBlocksSameLevel()) {
#endif
      int csi, cei, csj, cej, csk, cek;

      // coarse bx1 (stagger_axis=0)
      idxSetSameLevelRanges_FC(nb.ni, csi, cei, csj, cej, csk, cek, 0, 2);
      BufferUtility::UnpackData(buf, coarse_buf->x1f, csi, cei, csj, cej, csk,
                                cek, p);

      // coarse bx2 (stagger_axis=1)
      idxSetSameLevelRanges_FC(nb.ni, csi, cei, csj, cej, csk, cek, 1, 2);
      BufferUtility::UnpackData(buf, coarse_buf->x2f, csi, cei, csj, cej, csk,
                                cek, p);
      if (pmb->block_size.nx2 == 1) { // 1D
#pragma omp simd
        for (int i = csi; i <= cei; ++i)
          coarse_buf->x2f(csk, csj + 1, i) = coarse_buf->x2f(csk, csj, i);
      }

      // coarse bx3 (stagger_axis=2)
      idxSetSameLevelRanges_FC(nb.ni, csi, cei, csj, cej, csk, cek, 2, 2);
      BufferUtility::UnpackData(buf, coarse_buf->x3f, csi, cei, csj, cej, csk,
                                cek, p);
      if (pmb->block_size.nx3 == 1) { // 1D or 2D
        for (int j = csj; j <= cej; ++j) {
#pragma omp simd
          for (int i = csi; i <= cei; ++i)
            coarse_buf->x3f(csk + 1, j, i) = coarse_buf->x3f(csk, j, i);
        }
      }
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
    }
#endif
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
//                                                                const
//                                                                NeighborBlock&
//                                                                nb)
//  \brief Set face-centered prolongation buffer received from a block on the
//  same level

void FaceCenteredBoundaryVariable::SetBoundaryFromCoarser(
    Real *buf, const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  // bx1 (stagger_axis=0)
  idxSetFromCoarserRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 0);

  if (nb.polar) {
    Real sign = flip_across_pole_[IB1] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          coarse_buf->x1f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, coarse_buf->x1f, si, ei, sj, ej, sk, ek, p);
  }

  // bx2 (stagger_axis=1)
  idxSetFromCoarserRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 1);

  if (nb.polar) {
    Real sign = flip_across_pole_[IB2] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          coarse_buf->x2f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, coarse_buf->x2f, si, ei, sj, ej, sk, ek, p);
    if (pmb->block_size.nx2 == 1) { // 1D
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        coarse_buf->x2f(sk, sj + 1, i) = coarse_buf->x2f(sk, sj, i);
    }
  }

  // bx3 (stagger_axis=2)
  idxSetFromCoarserRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 2);

  if (nb.polar) {
    Real sign = flip_across_pole_[IB3] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          coarse_buf->x3f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, coarse_buf->x3f, si, ei, sj, ej, sk, ek, p);
    if (pmb->block_size.nx3 == 1) { // 2D
      for (int j = sj; j <= ej; ++j) {
        for (int i = si; i <= ei; ++i)
          coarse_buf->x3f(sk + 1, j, i) = coarse_buf->x3f(sk, j, i);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::SetFielBoundaryFromFiner(Real *buf,
//                                                                const
//                                                                NeighborBlock&
//                                                                nb)
//  \brief Set face-centered boundary received from a block on the same level

void FaceCenteredBoundaryVariable::SetBoundaryFromFiner(
    Real *buf, const NeighborBlock &nb) {
  MeshBlock *pmb = pmy_block_;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  // bx1 (stagger_axis=0)
  idxSetFromFinerRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 0);

  if (nb.polar) {
    Real sign = flip_across_pole_[IB1] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          (*var_fc).x1f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, (*var_fc).x1f, si, ei, sj, ej, sk, ek, p);
  }

  // bx2 (stagger_axis=1)
  idxSetFromFinerRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 1);

  if (nb.polar) {
    Real sign = flip_across_pole_[IB2] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          (*var_fc).x2f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, (*var_fc).x2f, si, ei, sj, ej, sk, ek, p);
  }
  if (pmb->block_size.nx2 == 1) { // 1D
#pragma omp simd
    for (int i = si; i <= ei; ++i)
      (*var_fc).x2f(sk, sj + 1, i) = (*var_fc).x2f(sk, sj, i);
  }

  // bx3 (stagger_axis=2)
  idxSetFromFinerRanges_FC(nb.ni, si, ei, sj, ej, sk, ek, 2);

  if (nb.polar) {
    Real sign = flip_across_pole_[IB3] ? -1.0 : 1.0;
    for (int k = sk; k <= ek; ++k) {
      for (int j = ej; j >= sj; --j) {
#pragma omp simd linear(p)
        for (int i = si; i <= ei; ++i)
          (*var_fc).x3f(k, j, i) = sign * buf[p++];
      }
    }
  } else {
    BufferUtility::UnpackData(buf, (*var_fc).x3f, si, ei, sj, ej, sk, ek, p);
  }
  if (pmb->block_size.nx3 == 1) { // 1D or 2D
    for (int j = sj; j <= ej; ++j) {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        (*var_fc).x3f(sk + 1, j, i) = (*var_fc).x3f(sk, j, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::SetBoundaries()
//  \brief set the face-centered boundary data

void FaceCenteredBoundaryVariable::SetBoundaries() {
  BoundaryVariable::SetBoundaries();
  if (pbval_->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar ||
      pbval_->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
    PolarFieldBoundaryAverage();
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::ReceiveAndSetBoundariesWithWait()
//  \brief receive and set the face-centered boundary data for initialization

// TODO(KGF): nearly identical to CellCenteredBoundaryVariable counterpart
// (extra call to PolarFieldBoundaryAverage())
void FaceCenteredBoundaryVariable::ReceiveAndSetBoundariesWithWait() {
  BoundaryVariable::ReceiveAndSetBoundariesWithWait();
  if (pbval_->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar ||
      pbval_->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
    PolarFieldBoundaryAverage();
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::PolarBoundarySingleAzimuthalBlock()
// \brief polar boundary edge-case: single MeshBlock spans the entire azimuthal
// (x3) range

void FaceCenteredBoundaryVariable::PolarBoundarySingleAzimuthalBlock() {
  MeshBlock *pmb = pmy_block_;
  if (pmb->loc.level == pmy_mesh_->root_level && pmy_mesh_->nrbx3 == 1 &&
      pmb->block_size.nx3 > 1) {
    if (pbval_->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) {
      int nx3_half = (pmb->ke - pmb->ks + 1) / 2;
      for (int j = pmb->js - NGHOST; j <= pmb->js - 1; ++j) {
        for (int i = pmb->is - NGHOST; i <= pmb->ie + NGHOST + 1; ++i) {
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k)
            pbval_->azimuthal_shift_(k) = (*var_fc).x1f(k, j, i);
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k) {
            int k_shift = k;
            k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
            (*var_fc).x1f(k, j, i) = pbval_->azimuthal_shift_(k_shift);
          }
        }
      }
      for (int j = pmb->js - NGHOST; j <= pmb->js - 1; ++j) {
        for (int i = pmb->is - NGHOST; i <= pmb->ie + NGHOST; ++i) {
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k)
            pbval_->azimuthal_shift_(k) = (*var_fc).x2f(k, j, i);
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k) {
            int k_shift = k;
            k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
            (*var_fc).x2f(k, j, i) = pbval_->azimuthal_shift_(k_shift);
          }
        }
      }
      for (int j = pmb->js - NGHOST; j <= pmb->js - 1; ++j) {
        for (int i = pmb->is - NGHOST; i <= pmb->ie + NGHOST; ++i) {
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST + 1; ++k)
            pbval_->azimuthal_shift_(k) = (*var_fc).x3f(k, j, i);
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST + 1; ++k) {
            int k_shift = k;
            k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
            (*var_fc).x3f(k, j, i) = pbval_->azimuthal_shift_(k_shift);
          }
        }
      }
    }

    if (pbval_->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
      int nx3_half = (pmb->ke - pmb->ks + 1) / 2;
      for (int j = pmb->je + 1; j <= pmb->je + NGHOST; ++j) {
        for (int i = pmb->is - NGHOST; i <= pmb->ie + NGHOST + 1; ++i) {
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k)
            pbval_->azimuthal_shift_(k) = (*var_fc).x1f(k, j, i);
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k) {
            int k_shift = k;
            k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
            (*var_fc).x1f(k, j, i) = pbval_->azimuthal_shift_(k_shift);
          }
        }
      }
      for (int j = pmb->je + 2; j <= pmb->je + NGHOST + 1; ++j) {
        for (int i = pmb->is - NGHOST; i <= pmb->ie + NGHOST; ++i) {
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k)
            pbval_->azimuthal_shift_(k) = (*var_fc).x2f(k, j, i);
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST; ++k) {
            int k_shift = k;
            k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
            (*var_fc).x2f(k, j, i) = pbval_->azimuthal_shift_(k_shift);
          }
        }
      }
      for (int j = pmb->je + 1; j <= pmb->je + NGHOST; ++j) {
        for (int i = pmb->is - NGHOST; i <= pmb->ie + NGHOST; ++i) {
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST + 1; ++k)
            pbval_->azimuthal_shift_(k) = (*var_fc).x3f(k, j, i);
          for (int k = pmb->ks - NGHOST; k <= pmb->ke + NGHOST + 1; ++k) {
            int k_shift = k;
            k_shift += (k < (nx3_half + NGHOST) ? 1 : -1) * nx3_half;
            (*var_fc).x3f(k, j, i) = pbval_->azimuthal_shift_(k_shift);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::PolarFieldBoundaryAverage()
//  \brief set theta-component of field along axis

void FaceCenteredBoundaryVariable::PolarFieldBoundaryAverage() {
  MeshBlock *pmb = pmy_block_;
  int il = pmb->is - NGHOST;
  int iu = pmb->ie + NGHOST;
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (pmb->block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }
  if (pbval_->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) {
    int j = pmb->js;
    for (int k = kl; k <= ku; ++k) {
      for (int i = il; i <= iu; ++i) {
        (*var_fc).x2f(k, j, i) =
            0.5 * ((*var_fc).x2f(k, j - 1, i) + (*var_fc).x2f(k, j + 1, i));
      }
    }
  }
  if (pbval_->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
    int j = pmb->je + 1;
    for (int k = kl; k <= ku; ++k) {
      for (int i = il; i <= iu; ++i) {
        (*var_fc).x2f(k, j, i) =
            0.5 * ((*var_fc).x2f(k, j - 1, i) + (*var_fc).x2f(k, j + 1, i));
      }
    }
  }
  return;
}

void FaceCenteredBoundaryVariable::CountFineEdges() {
  MeshBlock *pmb = pmy_block_;
  int &mylevel = pmb->loc.level;

  // count the number of the fine meshblocks contacting on each edge
  int eid = 0;
  if (pmb->block_size.nx2 > 1) {
    for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
      for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
        int nis, nie, njs, nje;
        nis = std::max(ox1 - 1, -1), nie = std::min(ox1 + 1, 1);
        njs = std::max(ox2 - 1, -1), nje = std::min(ox2 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nj = njs; nj <= nje; nj++) {
          for (int ni = nis; ni <= nie; ni++) {
            if (pbval_->nblevel[1][nj + 1][ni + 1] > fl)
              fl++, nf = 0;
            if (pbval_->nblevel[1][nj + 1][ni + 1] == fl)
              nf++;
          }
        }
        edge_flag_[eid] = (fl == mylevel);
        nedge_fine_[eid++] = nf;
      }
    }
  }
  if (pmb->block_size.nx3 > 1) {
    for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
      for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
        int nis, nie, nks, nke;
        nis = std::max(ox1 - 1, -1), nie = std::min(ox1 + 1, 1);
        nks = std::max(ox3 - 1, -1), nke = std::min(ox3 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nk = nks; nk <= nke; nk++) {
          for (int ni = nis; ni <= nie; ni++) {
            if (pbval_->nblevel[nk + 1][1][ni + 1] > fl)
              fl++, nf = 0;
            if (pbval_->nblevel[nk + 1][1][ni + 1] == fl)
              nf++;
          }
        }
        edge_flag_[eid] = (fl == mylevel);
        nedge_fine_[eid++] = nf;
      }
    }
    for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
      for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
        int njs, nje, nks, nke;
        njs = std::max(ox2 - 1, -1), nje = std::min(ox2 + 1, 1);
        nks = std::max(ox3 - 1, -1), nke = std::min(ox3 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nk = nks; nk <= nke; nk++) {
          for (int nj = njs; nj <= nje; nj++) {
            if (pbval_->nblevel[nk + 1][nj + 1][1] > fl)
              fl++, nf = 0;
            if (pbval_->nblevel[nk + 1][nj + 1][1] == fl)
              nf++;
          }
        }
        edge_flag_[eid] = (fl == mylevel);
        nedge_fine_[eid++] = nf;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::SendBoundaryBuffers()
//  \brief Pre-restrict the coarse interior, then send boundary buffers.
//         Overrides BoundaryVariable::SendBoundaryBuffers to insert the
//         upfront restriction pass before the per-neighbor loop.

void FaceCenteredBoundaryVariable::SendBoundaryBuffers() {
  if (pmy_mesh_->multilevel) {
    RestrictNonGhost();
  }
  BoundaryVariable::SendBoundaryBuffers();
}

//----------------------------------------------------------------------------------------
//! \fn void FaceCenteredBoundaryVariable::RestrictNonGhost()
//  \brief Pre-restrict the entire physical coarse interior in one pass.
//         Called from SendBoundaryBuffers before the per-neighbor loop.
//         Analogous to CellCenteredBoundaryVariable::RestrictNonGhost.

void FaceCenteredBoundaryVariable::RestrictNonGhost() {
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

  int f2 = pmy_mesh_->f2;
  int f3 = pmy_mesh_->f3;

  // Restrict all three face-field components over the full coarse interior.
  // Each face direction has staggering: one extra index in its own direction.

  // x1-faces: staggered in x1, so i-range is [cis, cie+1]
  pmr->RestrictFieldX1((*var_fc).x1f, coarse_buf->x1f, pmb->cis, pmb->cie + 1,
                       pmb->cjs, pmb->cje, pmb->cks, pmb->cke);

  // x2-faces: staggered in x2, so j-range is [cjs, cje+f2]
  pmr->RestrictFieldX2((*var_fc).x2f, coarse_buf->x2f, pmb->cis, pmb->cie,
                       pmb->cjs, pmb->cje + f2, pmb->cks, pmb->cke);
  if (pmb->block_size.nx2 == 1) { // 1D
    for (int i = pmb->cis; i <= pmb->cie; i++)
      coarse_buf->x2f(pmb->cks, pmb->cjs + 1, i) =
          coarse_buf->x2f(pmb->cks, pmb->cjs, i);
  }

  // x3-faces: staggered in x3, so k-range is [cks, cke+f3]
  pmr->RestrictFieldX3((*var_fc).x3f, coarse_buf->x3f, pmb->cis, pmb->cie,
                       pmb->cjs, pmb->cje, pmb->cks, pmb->cke + f3);
  if (pmb->block_size.nx3 == 1) { // 1D or 2D
    for (int j = pmb->cjs; j <= pmb->cje; j++) {
      for (int i = pmb->cis; i <= pmb->cie; i++)
        coarse_buf->x3f(pmb->cks + 1, j, i) = coarse_buf->x3f(pmb->cks, j, i);
    }
  }
}

void FaceCenteredBoundaryVariable::ProlongateBoundaries(const Real time,
                                                        const Real dt) {
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;

  const int mylevel = pbval_->loc.level;
  const int nneighbor = pbval_->nneighbor;

  for (int n = 0; n < nneighbor; ++n) {
    NeighborBlock &nb = pbval_->neighbor[n];
    if (nb.snb.level >= mylevel)
      continue;

    int si, ei, sj, ej, sk, ek;
    int il, iu, jl, ju, kl, ku;

    CalculateProlongationIndices(nb, si, ei, sj, ej, sk, ek);
    CalculateProlongationSharedIndices(nb, si, ei, sj, ej, sk, ek, il, iu, jl,
                                       ju, kl, ku);

    // step 1. calculate x1 outer surface fields and slopes
    pmr->ProlongateSharedFieldX1((*coarse_buf).x1f, (*var_fc).x1f, il, iu, sj,
                                 ej, sk, ek);
    // step 2. calculate x2 outer surface fields and slopes
    pmr->ProlongateSharedFieldX2((*coarse_buf).x2f, (*var_fc).x2f, si, ei, jl,
                                 ju, sk, ek);
    // step 3. calculate x3 outer surface fields and slopes
    pmr->ProlongateSharedFieldX3((*coarse_buf).x3f, (*var_fc).x3f, si, ei, sj,
                                 ej, kl, ku);

    // step 4. calculate the internal finer fields using the Toth & Roe method
    pmr->ProlongateInternalField((*var_fc), si, ei, sj, ej, sk, ek);
  }
}

void FaceCenteredBoundaryVariable::RestrictInterior(const Real time,
                                                    const Real dt) {
  // No-op: ghost-ghost zone restriction is now handled by RestrictNonGhost()
  // (called from the overridden SendBoundaryBuffers) which restricts the
  // entire coarse interior in one pass.  Same-level neighbor coarse data is
  // supplied via the coarse payload in LoadBoundaryBufferSameLevel /
  // SetBoundarySameLevel.
}

void FaceCenteredBoundaryVariable::SetupPersistentMPI() {
  CountFineEdges();

#ifdef MPI_PARALLEL
  MeshBlock *pmb = pmy_block_;
  int nx1 = pmb->block_size.nx1;
  int nx2 = pmb->block_size.nx2;
  int nx3 = pmb->block_size.nx3;
  int &mylevel = pmb->loc.level;

  int ssize, rsize;
  int tag;
  // Initialize non-polar neighbor communications to other ranks
  for (int n = 0; n < pbval_->nneighbor; n++) {
    NeighborBlock &nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      if (nb.snb.level == mylevel) { // same refinement level
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
        // Skip coarse portion when the destination block has all same-level
        // neighbors and therefore does not need the coarse payload.
        ssize = MPI_BufferSizeSameLevel_FC(nb.ni, nb.neighbor_all_same_level);
        rsize =
            MPI_BufferSizeSameLevel_FC(nb.ni, pmb->NeighborBlocksSameLevel());
#else
        ssize = MPI_BufferSizeSameLevel_FC(nb.ni);
        rsize = ssize;
#endif
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = MPI_BufferSizeToCoarser_FC(nb.ni);
        rsize = MPI_BufferSizeFromCoarser_FC(nb.ni);
      } else { // finer
        ssize = MPI_BufferSizeToFiner_FC(nb.ni);
        rsize = MPI_BufferSizeFromFiner_FC(nb.ni);
      }

      // face-centered field: bd_var_
      tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, fc_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      MPI_Send_init(bd_var_.send[nb.bufid], ssize, MPI_ATHENA_REAL, nb.snb.rank,
                    tag, MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid]));
      tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, fc_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      MPI_Recv_init(bd_var_.recv[nb.bufid], rsize, MPI_ATHENA_REAL, nb.snb.rank,
                    tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));

      // emf correction
      int size, f2csize = 0;
      if (nb.ni.type == NeighborConnect::face) { // face
        if (nx3 > 1) {                           // 3D
          if (nb.fid == BoundaryFace::inner_x1 ||
              nb.fid == BoundaryFace::outer_x1) {
            size = (nx2 + 1) * (nx3) + (nx2) * (nx3 + 1);
            f2csize = (nx2 / 2 + 1) * (nx3 / 2) + (nx2 / 2) * (nx3 / 2 + 1);
          } else if (nb.fid == BoundaryFace::inner_x2 ||
                     nb.fid == BoundaryFace::outer_x2) {
            size = (nx1 + 1) * (nx3) + (nx1) * (nx3 + 1);
            f2csize = (nx1 / 2 + 1) * (nx3 / 2) + (nx1 / 2) * (nx3 / 2 + 1);
          } else if (nb.fid == BoundaryFace::inner_x3 ||
                     nb.fid == BoundaryFace::outer_x3) {
            size = (nx1 + 1) * (nx2) + (nx1) * (nx2 + 1);
            f2csize = (nx1 / 2 + 1) * (nx2 / 2) + (nx1 / 2) * (nx2 / 2 + 1);
          }
        } else if (nx2 > 1) { // 2D
          if (nb.fid == BoundaryFace::inner_x1 ||
              nb.fid == BoundaryFace::outer_x1) {
            size = (nx2 + 1) + nx2;
            f2csize = (nx2 / 2 + 1) + nx2 / 2;
          } else if (nb.fid == BoundaryFace::inner_x2 ||
                     nb.fid == BoundaryFace::outer_x2) {
            size = (nx1 + 1) + nx1;
            f2csize = (nx1 / 2 + 1) + nx1 / 2;
          }
        } else { // 1D
          size = f2csize = 2;
        }
      } else if (nb.ni.type == NeighborConnect::edge) { // edge
        if (nx3 > 1) {                                  // 3D
          if (nb.eid >= 0 && nb.eid < 4) {
            size = nx3;
            f2csize = nx3 / 2;
          } else if (nb.eid >= 4 && nb.eid < 8) {
            size = nx2;
            f2csize = nx2 / 2;
          } else if (nb.eid >= 8 && nb.eid < 12) {
            size = nx1;
            f2csize = nx1 / 2;
          }
        } else if (nx2 > 1) { // 2D
          size = f2csize = 1;
        }
      } else { // corner
        continue;
      }
      // field flux (emf) correction: bd_var_flcor_
      if (nb.snb.level == mylevel) { // the same level
        if ((nb.ni.type == NeighborConnect::face) ||
            ((nb.ni.type == NeighborConnect::edge) && (edge_flag_[nb.eid]))) {
          tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid,
                                          fc_flx_phys_id_);
          if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
          MPI_Send_init(bd_var_flcor_.send[nb.bufid], size, MPI_ATHENA_REAL,
                        nb.snb.rank, tag, MPI_COMM_WORLD,
                        &(bd_var_flcor_.req_send[nb.bufid]));
          tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, fc_flx_phys_id_);
          if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
          MPI_Recv_init(bd_var_flcor_.recv[nb.bufid], size, MPI_ATHENA_REAL,
                        nb.snb.rank, tag, MPI_COMM_WORLD,
                        &(bd_var_flcor_.req_recv[nb.bufid]));
        }
      }
      if (nb.snb.level > mylevel) { // finer neighbor
        tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, fc_flx_phys_id_);
        if (bd_var_flcor_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&bd_var_flcor_.req_recv[nb.bufid]);
        MPI_Recv_init(bd_var_flcor_.recv[nb.bufid], f2csize, MPI_ATHENA_REAL,
                      nb.snb.rank, tag, MPI_COMM_WORLD,
                      &(bd_var_flcor_.req_recv[nb.bufid]));
      }
      if (nb.snb.level < mylevel) { // coarser neighbor
        tag =
            pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, fc_flx_phys_id_);
        if (bd_var_flcor_.req_send[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&bd_var_flcor_.req_send[nb.bufid]);
        MPI_Send_init(bd_var_flcor_.send[nb.bufid], f2csize, MPI_ATHENA_REAL,
                      nb.snb.rank, tag, MPI_COMM_WORLD,
                      &(bd_var_flcor_.req_send[nb.bufid]));
      }
    } // neighbor block is on separate MPI process
  } // end loop over neighbors

  // Initialize polar neighbor communications to other ranks
  for (int n = 0; n < pbval_->num_north_polar_blocks_; ++n) {
    const SimpleNeighborBlock &snb = pbval_->polar_neighbor_north_[n];
    if (snb.rank != Globals::my_rank) {
      tag = pbval_->CreateBvalsMPITag(snb.lid, pmb->loc.lx3,
                                      fc_flx_pole_phys_id_);
      if (req_flux_north_send_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_north_send_[n]);
      MPI_Send_init(flux_north_send_[n], nx1, MPI_ATHENA_REAL, snb.rank, tag,
                    MPI_COMM_WORLD, &req_flux_north_send_[n]);
      tag = pbval_->CreateBvalsMPITag(pmb->lid, n, fc_flx_pole_phys_id_);
      if (req_flux_north_recv_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_north_recv_[n]);
      MPI_Recv_init(flux_north_recv_[n], nx1, MPI_ATHENA_REAL, snb.rank, tag,
                    MPI_COMM_WORLD, &req_flux_north_recv_[n]);
    }
  }
  for (int n = 0; n < pbval_->num_south_polar_blocks_; ++n) {
    const SimpleNeighborBlock &snb = pbval_->polar_neighbor_south_[n];
    if (snb.rank != Globals::my_rank) {
      tag = pbval_->CreateBvalsMPITag(snb.lid, pmb->loc.lx3,
                                      fc_flx_pole_phys_id_);
      if (req_flux_south_send_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_south_send_[n]);
      MPI_Send_init(flux_south_send_[n], nx1, MPI_ATHENA_REAL, snb.rank, tag,
                    MPI_COMM_WORLD, &req_flux_south_send_[n]);
      tag = pbval_->CreateBvalsMPITag(pmb->lid, n, fc_flx_pole_phys_id_);
      if (req_flux_south_recv_[n] != MPI_REQUEST_NULL)
        MPI_Request_free(&req_flux_south_recv_[n]);
      MPI_Recv_init(flux_south_recv_[n], nx1, MPI_ATHENA_REAL, snb.rank, tag,
                    MPI_COMM_WORLD, &req_flux_south_recv_[n]);
    }
  }
#endif
  return;
}

void FaceCenteredBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
  if ((phase == BoundaryCommSubset::all) ||
      (phase == BoundaryCommSubset::matter_flux_corrected)) {
    recv_flx_same_lvl_ = true;
  } else {
    recv_flx_same_lvl_ = false;
  }

#ifdef MPI_PARALLEL
  MeshBlock *pmb = pmy_block_;
  int mylevel = pmb->loc.level;
  for (int n = 0; n < pbval_->nneighbor; n++) {
    NeighborBlock &nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank &&
        phase != BoundaryCommSubset::matter_primitives) {

      if (phase != BoundaryCommSubset::matter_flux_corrected)
        MPI_Start(&(bd_var_.req_recv[nb.bufid]));

      // Deal with flux correction
      if (((phase == BoundaryCommSubset::all) ||
           (phase == BoundaryCommSubset::matter_flux_corrected)) &&
          (nb.ni.type == NeighborConnect::face ||
           nb.ni.type == NeighborConnect::edge)) {
        if ((nb.snb.level > mylevel) ||
            ((nb.snb.level == mylevel) &&
             ((nb.ni.type == NeighborConnect::face) ||
              ((nb.ni.type == NeighborConnect::edge) && (edge_flag_[nb.eid])))))
          MPI_Start(&(bd_var_flcor_.req_recv[nb.bufid]));
      }
    }
  }

  if (phase == BoundaryCommSubset::all) {
    for (int n = 0; n < pbval_->num_north_polar_blocks_; ++n) {
      const SimpleNeighborBlock &snb = pbval_->polar_neighbor_north_[n];
      if (snb.rank != Globals::my_rank) {
        MPI_Start(&req_flux_north_recv_[n]);
      }
    }
    for (int n = 0; n < pbval_->num_south_polar_blocks_; ++n) {
      const SimpleNeighborBlock &snb = pbval_->polar_neighbor_south_[n];
      if (snb.rank != Globals::my_rank) {
        MPI_Start(&req_flux_south_recv_[n]);
      }
    }
  }
#endif
  return;
}

void FaceCenteredBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  // Clear non-polar boundary communications
  for (int n = 0; n < pbval_->nneighbor; n++) {
    NeighborBlock &nb = pbval_->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;

    if (((nb.ni.type == NeighborConnect::face) ||
         (nb.ni.type == NeighborConnect::edge)) &&
        ((phase == BoundaryCommSubset::all) ||
         (phase == BoundaryCommSubset::matter_flux_corrected))) {
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::waiting;
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::waiting;
    }

#ifdef MPI_PARALLEL
    MeshBlock *pmb = pmy_block_;
    int mylevel = pmb->loc.level;
    if (nb.snb.rank != Globals::my_rank &&
        phase != BoundaryCommSubset::matter_primitives) {
      // Wait for Isend
      if (phase != BoundaryCommSubset::matter_flux_corrected)
        MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE);

      // Deal with flux correction
      if ((phase == BoundaryCommSubset::all) ||
          (phase == BoundaryCommSubset::matter_flux_corrected)) {
        if (nb.ni.type == NeighborConnect::face ||
            nb.ni.type == NeighborConnect::edge) {
          if (nb.snb.level < mylevel)
            MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
          else if ((nb.snb.level == mylevel) &&
                   ((nb.ni.type == NeighborConnect::face) ||
                    ((nb.ni.type == NeighborConnect::edge) &&
                     (edge_flag_[nb.eid]))))
            MPI_Wait(&(bd_var_flcor_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
        }
      }
    }
#endif
  }

  // Clear polar boundary communications (only during main integration loop)
  if (phase == BoundaryCommSubset::all) {
    for (int n = 0; n < pbval_->num_north_polar_blocks_; ++n) {
      flux_north_flag_[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
      SimpleNeighborBlock &snb = pbval_->polar_neighbor_north_[n];
      if (snb.rank != Globals::my_rank)
        MPI_Wait(&req_flux_north_send_[n], MPI_STATUS_IGNORE);
#endif
    }
    for (int n = 0; n < pbval_->num_south_polar_blocks_; ++n) {
      flux_south_flag_[n] = BoundaryStatus::waiting;
#ifdef MPI_PARALLEL
      SimpleNeighborBlock &snb = pbval_->polar_neighbor_south_[n];
      if (snb.rank != Globals::my_rank)
        MPI_Wait(&req_flux_south_send_[n], MPI_STATUS_IGNORE);
#endif
    }
  }
  return;
}
