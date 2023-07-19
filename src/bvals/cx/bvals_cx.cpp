//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//  \brief functions that apply BCs for VERTEX_CENTERED variables

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
#include "bvals_cx.hpp"

#include "../../utils/lagrange_interp.hpp"
#include "../../utils/interp_barycentric.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// constructor

CellCenteredXBoundaryVariable::CellCenteredXBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var, AthenaArray<Real> *coarse_var,
    AthenaArray<Real> *var_flux)
    : BoundaryVariable(pmb), var_cx(var), coarse_buf(coarse_var), x1flux(var_flux[X1DIR]),
      x2flux(var_flux[X2DIR]), x3flux(var_flux[X3DIR]), nl_(0), nu_(var->GetDim4() -1),
      flip_across_pole_(nullptr) {
  // CellCenteredXBoundaryVariable should only be used w/ 4D or 3D (nx4=1) AthenaArray
  // For now, assume that full span of 4th dim of input AthenaArray should be used:
  // ---> get the index limits directly from the input AthenaArray
  // <=nu_ (inclusive), <nx4 (exclusive)
  if (nu_ < 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CellCenteredXBoundaryVariable constructor" << std::endl
        << "An 'AthenaArray<Real> *var' of nx4_ = " << var->GetDim4() << " was passed\n"
        << "Should be nx4 >= 1 (likely uninitialized)." << std::endl;
    ATHENA_ERROR(msg);
  }

  InitBoundaryData(bd_var_, BoundaryQuantity::cx);
#ifdef MPI_PARALLEL
  // KGF: dead code, leaving for now:
  // cc_phys_id_ = pbval_->ReserveTagVariableIDs(1);
  cx_phys_id_ = pbval_->bvars_next_phys_id_;
#endif
  if (pmy_mesh_->multilevel) { // SMR or AMR
    // InitBoundaryData(bd_var_flcor_, BoundaryQuantity::cc_flcor);
#ifdef MPI_PARALLEL
    // cc_flx_phys_id_ = cc_phys_id_ + 1;
#endif
  }

}

// destructor
CellCenteredXBoundaryVariable::~CellCenteredXBoundaryVariable() {
  DestroyBoundaryData(bd_var_);
 // if (pmy_mesh_->multilevel)
 //   DestroyBoundaryData(bd_var_flcor_);
}

void CellCenteredXBoundaryVariable::ErrorIfPolarNotImplemented(
  const NeighborBlock& nb) {

  // BD: TODO implement polar coordinates
  if (nb.polar) {
  std::stringstream msg;
  msg << "### FATAL ERROR" << std::endl
      << "Polar coordinates not implemented for vertex-centered." << std::endl;
  ATHENA_ERROR(msg);
  }
  return;
}

void CellCenteredXBoundaryVariable::ErrorIfShearingBoxNotImplemented() {
  // BD: TODO implement shearing box
  if (SHEARING_BOX){
    std::stringstream msg;
    msg << "### FATAL ERROR" << std::endl
        << "Shearing box not implemented for vertex-centered." << std::endl;
    ATHENA_ERROR(msg);
  }
}

int CellCenteredXBoundaryVariable::ComputeVariableBufferSize(const NeighborIndexes& ni,
                                                              int cng) {
  // 'cng' to preserve function signature but is a dummy slot
  return NeighborVariableBufferSize(ni);
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredXBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set vertex-centered boundary buffers for sending to a block on the same level

int CellCenteredXBoundaryVariable::LoadBoundaryBufferSameLevel(Real *buf,
                                                                const NeighborBlock& nb) {
  //MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  int p = 0;
  AthenaArray<Real> &var = *var_cx;

  // BD: ok
  idxLoadSameLevelRanges(nb.ni, si, ei, sj, ej, sk, ek, false);
  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  // BD: opt- if nn all same level not required
  // if multilevel make use of pre-restricted internal data
  if (pmy_mesh_->multilevel) {
    // convert to coarse indices
    AthenaArray<Real> &coarse_var = *coarse_buf;

    // BD: ok
    idxLoadSameLevelRanges(nb.ni, si, ei, sj, ej, sk, ek, true);
    BufferUtility::PackData(coarse_var, buf, nl_, nu_,
                            si, ei, sj, ej, sk, ek, p);
  }

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredXBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set vertex-centered boundary buffers for sending to a block on the coarser level

int CellCenteredXBoundaryVariable::LoadBoundaryBufferToCoarser(Real *buf,
                                                                const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  AthenaArray<Real> &var = *var_cx;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  /*
  // vertices that are shared with adjacent MeshBlocks are to be copied to coarser level
  idxLoadToCoarserRanges(nb.ni, si, ei, sj, ej, sk, ek, false);
  pmr->RestrictVertexCenteredValues(var, coarse_var, nl_, nu_,
                                    si, ei, sj, ej, sk, ek);

  BufferUtility::PackData(coarse_var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  if (pmy_mesh_->multilevel) {
    // double restrict required to populate coarse buffer of coarser level
    idxLoadToCoarserRanges(nb.ni, si, ei, sj, ej, sk, ek, true);
    pmr->RestrictTwiceToBufferVertexCenteredValues(var, buf, nl_, nu_,
                                                   si, ei, sj, ej, sk, ek, p);
  }
  */

  // BD: ok
  // Take coarse rep on l and populate fundamental on l-1
  idxLoadToCoarserRanges(nb.ni, si, ei, sj, ej, sk, ek, true);

  // pmr->RestrictCellCenteredXValues(var, coarse_var,
  //                                  nl_, nu_,
  //                                  si, ei, sj, ej, sk, ek);

  BufferUtility::PackData(coarse_var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CellCenteredXBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set vertex-centered boundary buffers for sending to a block on the finer level

int CellCenteredXBoundaryVariable::LoadBoundaryBufferToFiner(Real *buf,
                                                              const NeighborBlock& nb) {
  AthenaArray<Real> &var = *var_cx;
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  // coarse->fine
  // only pack fundamental rep. on l-1;
  // this populates coarse rep. on l

  // BD: ...
  idxLoadToFinerRanges(nb.ni, si, ei, sj, ej, sk, ek);
  BufferUtility::PackData(var, buf, nl_, nu_, si, ei, sj, ej, sk, ek, p);

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::SetBoundarySameLevel(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set vertex-centered boundary received from a block on the same level

void CellCenteredXBoundaryVariable::SetBoundarySameLevel(Real *buf,
                                                          const NeighborBlock& nb) {
  //MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;
  AthenaArray<Real> &var = *var_cx;
  int p = 0;

  // BD: TODO implement
  ErrorIfPolarNotImplemented(nb);
  ErrorIfShearingBoxNotImplemented();

  // BD: ok
  idxSetSameLevelRanges(nb.ni, si, ei, sj, ej, sk, ek, 1);
  BufferUtility::UnpackData(buf, var, nl_, nu_,
                            si, ei, sj, ej, sk, ek, p);

  // BD: opt- if nn all same level not required
  if (pmy_mesh_->multilevel) {
    // note: unpacked shared nodes additively unpacked-
    // consistency conditions will need to be applied to the coarse variable

    //MeshRefinement *pmr = pmb->pmr;
    AthenaArray<Real> &coarse_var = *coarse_buf;

    // BD: ok
    idxSetSameLevelRanges(nb.ni, si, ei, sj, ej, sk, ek, 2);
    BufferUtility::UnpackData(buf, coarse_var, nl_, nu_,
                              si, ei, sj, ej, sk, ek, p);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
//                                                                const NeighborBlock& nb)
//  \brief Set vertex-centered prolongation buffer received from a block on a coarser level

void CellCenteredXBoundaryVariable::SetBoundaryFromCoarser(Real *buf,
                                                            const NeighborBlock& nb) {
  MeshBlock *pmb = pmy_block_;
  int si, sj, sk, ei, ej, ek;

  AthenaArray<Real> &coarse_var = *coarse_buf;


  int p = 0;
  // BD: TODO implement
  ErrorIfPolarNotImplemented(nb);

  // BD: ...
  idxSetFromCoarserRanges(nb.ni, si, ei, sj, ej, sk, ek, false);
  BufferUtility::UnpackData(buf, coarse_var, nl_, nu_,
                            si, ei, sj, ej, sk, ek, p);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::SetBoundaryFromFiner(Real *buf,
//                                                              const NeighborBlock& nb)
//  \brief Set vertex-centered boundary received from a block on a finer level
void CellCenteredXBoundaryVariable::SetBoundaryFromFiner(Real *buf,
                                                          const NeighborBlock& nb) {
  // populating from finer level

  //MeshBlock *pmb = pmy_block_;
  AthenaArray<Real> &var = *var_cx;
  // receive already restricted data
  int si, sj, sk, ei, ej, ek;
  int p = 0;

  // BD: TODO implement
  ErrorIfPolarNotImplemented(nb);

  // BD: ...
  idxSetFromFinerRanges(nb.ni, si, ei, sj, ej, sk, ek, 1);
  BufferUtility::UnpackData(buf, var, nl_, nu_,
                            si, ei, sj, ej, sk, ek, p);

  // if (pmy_mesh_->multilevel) {
  //   AthenaArray<Real> &coarse_var = *coarse_buf;
  //   idxSetFromFinerRanges(nb.ni, si, ei, sj, ej, sk, ek, 2);
  //   BufferUtility::UnpackData(buf, coarse_var, nl_, nu_,
  //                             si, ei, sj, ej, sk, ek, p);
  // }
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::RestrictNonGhost()
//  \brief populate coarser buffer with restricted data

void CellCenteredXBoundaryVariable::RestrictNonGhost() {
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;

  AthenaArray<Real> &var = *var_cx;
  AthenaArray<Real> &coarse_var = *coarse_buf;

  const int nu = var_cx->GetDim4() - 1;
  pmr->RestrictCellCenteredXWithInteriorValues(
    var,
    coarse_var,
    0, nu
  );


  // Ghost data is required for (symmetric) restriction to coarse.
  // Extrapolate fundamental grid (it is overwritten later).

  // if (pmb->block_size.nx3>1)
  // { // 3D

  // }
  // else if (pmb->block_size.nx2>1)
  // { // 2d

  // }
  // else
  // { // 1d
  //   ExtrapolateOutflowInnerX1(0, 0, pmb->cx_is,
  //                             0, 0,
  //                             0, 0,
  //                             NGHOST);

  //   ExtrapolateOutflowOuterX1(0, 0, pmb->cx_ie,
  //                             0, 0,
  //                             0, 0,
  //                             NGHOST);
  // }

  // pmr->RestrictCellCenteredXValues(var, coarse_var,
  //                                  nl_, nu_,
  //                                  si, ei, sj, ej, sk, ek);

  // coarse_var.print_all("%.3e");
  // std::cout << "QQ" << std::endl;

 // Mimic logic of BoundaryValues::ApplyPhysicalCellCenteredXBoundaries
 // To extrapolate to fundamental ghosts.
 // This allows for physical node restriction to coarse to take place

  // const int nu = var_cx->GetDim4() - 1;

  // coarse_var(0, 0, 0, 2) = 2.2;

  /*
  ExtrapolateOutflowInnerX1(0, nu, pmb->cx_is,
                            pmb->cx_js, pmb->cx_je,
                            pmb->cx_ks, pmb->cx_ke,
                            NGHOST);

  ExtrapolateOutflowOuterX1(0, nu, pmb->cx_ie,
                            pmb->cx_js, pmb->cx_je,
                            pmb->cx_ks, pmb->cx_ke,
                            NGHOST);

  if (pmb->block_size.nx2>1)
  {  // 2D problem
    ExtrapolateOutflowInnerX2(0, nu,
                              pmb->cx_is, pmb->cx_ie,
                              pmb->cx_js,
                              pmb->cx_ks, pmb->cx_ke,
                              NGHOST);

    ExtrapolateOutflowOuterX2(0, nu,
                              pmb->cx_is, pmb->cx_ie,
                              pmb->cx_je,
                              pmb->cx_ks, pmb->cx_ke,
                              NGHOST);

    // now deal with corner conditions
    ExtrapolateOutflowInnerX1(0, nu, pmb->cx_is,
                              pmb->cx_jms, pmb->cx_jme,
                              pmb->cx_ks, pmb->cx_ke,
                              NGHOST);

    ExtrapolateOutflowInnerX1(0, nu, pmb->cx_is,
                              pmb->cx_jps, pmb->cx_jpe,
                              pmb->cx_ks, pmb->cx_ke,
                              NGHOST);

    ExtrapolateOutflowOuterX1(0, nu, pmb->cx_ie,
                              pmb->cx_jms, pmb->cx_jme,
                              pmb->cx_ks, pmb->cx_ke,
                              NGHOST);

    ExtrapolateOutflowOuterX1(0, nu, pmb->cx_ie,
                              pmb->cx_jps, pmb->cx_jpe,
                              pmb->cx_ks, pmb->cx_ke,
                              NGHOST);

  }

  if (pmb->block_size.nx3>1)
  { // 3D
    ExtrapolateOutflowInnerX3(0, nu,
                              pmb->cx_is, pmb->cx_ie,
                              pmb->cx_js, pmb->cx_je,
                              pmb->cx_ks,
                              NGHOST);

    ExtrapolateOutflowOuterX3(0, nu,
                              pmb->cx_is, pmb->cx_ie,
                              pmb->cx_js, pmb->cx_je,
                              pmb->cx_ke,
                              NGHOST);
  }
  */

  // std::swap(var_cx, coarse_buf);

  // // extrapolate to (unknown) fine ghosts
  // int nu = var_cx->GetDim4() - 1;
  // for (int n=0; n<=nu; n++)
  // {
  //   for (int i=pmb->cx_is-1; i>=pmb->cx_ims; i--)
  //   {
  //     var(n,0,0,i) = (
  //       4.0  * var(n,0,0,i+1) +
  //       -6.0 * var(n,0,0,i+2) +
  //       4.0  * var(n,0,0,i+3) +
  //       -1.0 * var(n,0,0,i+4)
  //     );
  //   }

  //   for (int i=pmb->cx_ie+1; i<=pmb->cx_ipe; i++)
  //   {
  //     var(n,0,0,i) = (
  //       -1.0 * var(n,0,0,i-4) +
  //       4.0  * var(n,0,0,i-3) +
  //       -6.0 * var(n,0,0,i-2) +
  //       4.0  * var(n,0,0,i-1)
  //     );
  //   }
  // }


  // BD: opt- we don't actually need _all_ the internal points
  /*

  int si, sj, sk, ei, ej, ek;
  si = pmb->cx_cis; ei = pmb->cx_cie;
  sj = pmb->cx_cjs; ej = pmb->cx_cje;
  sk = pmb->cx_cks; ek = pmb->cx_cke;

  Real const origin[3] = {
    pmb->pcoord->x1v(pmb->cx_is),
    pmb->pcoord->x2v(pmb->cx_js),
    pmb->pcoord->x3v(pmb->cx_ks),
  };
  Real const delta[3] = {
    pmb->pcoord->dx1v(0),
    pmb->pcoord->dx2v(0),
    pmb->pcoord->dx3v(0),
  };
  int const size[3] = {
    pmb->block_size.nx1,
    pmb->block_size.nx2,
    pmb->block_size.nx3,
  };

  AthenaArray<Real> svar;
  svar.NewAthenaArray(nu + 1, size[2], size[1], size[0]);
  for (int n=0; n<=nu; ++n)
  for (int cx_fk=0; cx_fk<size[2]; ++cx_fk)
  for (int cx_fj=0; cx_fj<size[1]; ++cx_fj)
  for (int cx_fi=0; cx_fi<size[0]; ++cx_fi)
  {
    svar(n, cx_fk, cx_fj, cx_fi) = var(
      n, pmb->cx_ks+cx_fk, pmb->cx_js+cx_fj, pmb->cx_is+cx_fi);

    // BD: debug (inject poly, see if it is recovered at order)
    if (false)
    {
      const Real x = pmb->pcoord->x1v(pmb->cx_is+cx_fi);
      const Real y = pmb->pcoord->x2v(pmb->cx_js+cx_fj);
      // const Real z = pmb->pcoord->x3v(pmb->cx_ks+cx_fk);

      svar(n, cx_fk, cx_fj, cx_fi) = SQR(x * x) * SQR(y * y); // * z ** 2
    }
  }


  for (int n=0; n<=nu; ++n)
  for (int cx_cj=sj; cx_cj<=ej; ++cx_cj)
  for (int cx_ci=si; cx_ci<=ei; ++cx_ci)
  {
    // left child idx on fine grid
    const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;
    const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

    const Real dx1 = delta[0] / 2.0;
    const Real dx2 = delta[1] / 2.0;

    // interp to this position
    Real const pos[3] = {
      pmb->pcoord->x1v(cx_fi) + dx1,
      pmb->pcoord->x2v(cx_fj) + dx2,
      pmb->pcoord->x3v(pmb->cx_ks),
    };


    LagrangeInterpND<2*NGHOST, 2> linterp(origin, delta, size, pos);
    // LagrangeInterpND<3, 2> linterp(origin, delta, size, pos);
    // Real const ival = linterp.eval(&var(n, pmb->cx_ks, pmb->cx_js, pmb->cx_is));
    Real const ival = linterp.eval(&svar(n, 0, 0, 0));
    coarse_var(n, 0, cx_cj, cx_ci) = ival;

    if (false)
    {
      Real const dbg_poly = SQR(pos[0] * pos[0]) * SQR(pos[1] * pos[1]);
      coarse_var(n, 0, cx_cj, cx_ci) -= dbg_poly;
    }
  }
  */


  // pmr->RestrictCellCenteredXValues(var, coarse_var,
  //                                  nl_, nu_,
  //                                  si+1, ei-1, sj+1, ej-1, sk, ek);

}


void CellCenteredXBoundaryVariable::SendBoundaryBuffers() {

  // restrict all data (except ghosts) to coarse buffer
  if (pmy_mesh_->multilevel)
  {
    // BD: opt- if nn all same level not required

    AthenaArray<Real> &coarse_var = *coarse_buf;
    // coarse_var.ZeroClear();
    RestrictNonGhost();
  }

  BoundaryVariable::SendBoundaryBuffers();
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::SetBoundaries()
//  \brief set the vertex-centered boundary data

void CellCenteredXBoundaryVariable::SetBoundaries() {
  BoundaryVariable::SetBoundaries();
}


//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::ReceiveAndSetBoundariesWithWait()
//  \brief receive and set the vertex-centered boundary data for initialization

void CellCenteredXBoundaryVariable::ReceiveAndSetBoundariesWithWait() {
  BoundaryVariable::ReceiveAndSetBoundariesWithWait();
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::PolarBoundarySingleAzimuthalBlock()
// \brief polar boundary edge-case: single MeshBlock spans the entire azimuthal (x3) range

void CellCenteredXBoundaryVariable::PolarBoundarySingleAzimuthalBlock() {
  return;
}

void CellCenteredXBoundaryVariable::SetupPersistentMPI() {
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
        ssize = MPI_BufferSizeSameLevel(nb.ni, true);
        rsize = MPI_BufferSizeSameLevel(nb.ni, false);
      } else if (nb.snb.level < mylevel) { // coarser
        ssize = MPI_BufferSizeToCoarser(nb.ni);
        rsize = MPI_BufferSizeFromCoarser(nb.ni);
      } else { // finer
        ssize = MPI_BufferSizeToFiner(nb.ni);
        rsize = MPI_BufferSizeFromFiner(nb.ni);
      }
      // specify the offsets in the view point of the target block: flip ox? signs

      // Initialize persistent communication requests attached to specific BoundaryData
      // vertex-centered
      tag = pbval_->CreateBvalsMPITag(nb.snb.lid, nb.targetid, cx_phys_id_);
      if (bd_var_.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_send[nb.bufid]);
      MPI_Send_init(bd_var_.send[nb.bufid], ssize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_send[nb.bufid]));
      tag = pbval_->CreateBvalsMPITag(pmb->lid, nb.bufid, cx_phys_id_);
      if (bd_var_.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&bd_var_.req_recv[nb.bufid]);
      MPI_Recv_init(bd_var_.recv[nb.bufid], rsize, MPI_ATHENA_REAL,
                    nb.snb.rank, tag, MPI_COMM_WORLD, &(bd_var_.req_recv[nb.bufid]));

    }
  }
#endif
  return;
}

void CellCenteredXBoundaryVariable::StartReceiving(BoundaryCommSubset phase) {
#ifdef MPI_PARALLEL
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.snb.rank != Globals::my_rank) {
      MPI_Start(&(bd_var_.req_recv[nb.bufid]));
    }
  }
#endif
  return;
}


void CellCenteredXBoundaryVariable::ClearBoundary(BoundaryCommSubset phase) {
  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    bd_var_.flag[nb.bufid] = BoundaryStatus::waiting;
    bd_var_.sflag[nb.bufid] = BoundaryStatus::waiting;

#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank) {
      // Wait for Isend
      MPI_Wait(&(bd_var_.req_send[nb.bufid]), MPI_STATUS_IGNORE);
    }
#endif
  }

  return;
}
