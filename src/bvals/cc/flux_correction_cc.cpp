//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file flux_correction_cc.cpp
//  \brief functions that perform flux correction for CELL_CENTERED variables

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>
#include <cstdlib>
#include <cstring>    // std::memcpy
#include <iomanip>
#include <iostream>   // endl
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
#include "bvals_cc.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredBoundaryVariable::SendFluxCorrection()
//  \brief Restrict, pack and send the surface flux to the coarse neighbor(s)

void CellCenteredBoundaryVariable::SendFluxCorrection() {
  if (DBGPR_FLXCOR_CC){
  coutBoldGreen("CellCenteredBoundaryVariable::SendFluxCorrection\n");
  }
  MeshBlock *pmb = pmy_block_;
  Coordinates *pco = pmb->pcoord;

  // cache pointers to surface area arrays (BoundaryBase protected variable)
  AthenaArray<Real> &sarea0 = pbval_->sarea_[0];
  AthenaArray<Real> &sarea1 = pbval_->sarea_[1];

  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.ni.type != NeighborConnect::face) break;
    if (bd_var_flcor_.sflag[nb.bufid] == BoundaryStatus::completed) continue;
    if (nb.snb.level == pmb->loc.level - 1) {
      int p = 0;
      Real *sbuf = bd_var_flcor_.send[nb.bufid];
      // x1 direction
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
        int i = pmb->is + (pmb->ie-pmb->is + 1)*nb.fid;
        if (pmb->block_size.nx3>1) { // 3D
          for (int nn=nl_; nn<=nu_; nn++) {
            for (int k=pmb->ks; k<=pmb->ke; k+=2) {
              for (int j=pmb->js; j<=pmb->je; j+=2) {
                Real amm = pco->dx2v(j)*pco->dx3v(k);
                Real amp = pco->dx2v(j+1)*pco->dx3v(k);
                Real apm = pco->dx2v(j)*pco->dx3v(k+1);
                Real app = pco->dx2v(j+1)*pco->dx3v(k+1);
                Real tarea = amm + amp + apm + app;
                sbuf[p++] = (x1flux(nn, k  , j  , i)*amm
                            + x1flux(nn, k  , j+1, i)*amp
                            + x1flux(nn, k+1, j  , i)*apm
                            + x1flux(nn, k+1, j+1, i)*app)/tarea;
              }
            }
          }
        } else if (pmb->block_size.nx2>1) { // 2D
          int k = pmb->ks;
          for (int nn=nl_; nn<=nu_; nn++) {
            for (int j=pmb->js; j<=pmb->je; j+=2) {
              Real am = pco->dx2v(j)*pco->dx3v(k);
              Real ap = pco->dx2v(j+1)*pco->dx3v(k);
              Real tarea = am + ap;
              sbuf[p++] = (x1flux(nn, k, j  , i)*am + x1flux(nn, k, j+1, i)*ap)/tarea;
            }
          }
        } else { // 1D
          int k = pmb->ks, j = pmb->js;
          for (int nn=nl_; nn<=nu_; nn++)
            sbuf[p++] = x1flux(nn, k, j, i);
        }
        // x2 direction
      } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
        int j = pmb->js + (pmb->je-pmb->js + 1)*(nb.fid & 1);
        if (pmb->block_size.nx3>1) { // 3D
          for (int nn=nl_; nn<=nu_; nn++) {
            for (int k=pmb->ks; k<=pmb->ke; k+=2) {
              pco->Face2Area(k  , j, pmb->is, pmb->ie, sarea0);
              pco->Face2Area(k+1, j, pmb->is, pmb->ie, sarea1);
              for (int i=pmb->is; i<=pmb->ie; i+=2) {
                Real amm = pco->dx1v(i)*pco->dx3v(k);
                Real amp = pco->dx1v(i)*pco->dx3v(k+1);
                Real apm = pco->dx1v(i+1)*pco->dx3v(k);
                Real app = pco->dx1v(i+1)*pco->dx3v(k+1);
//                Real tarea = sarea0(i) + sarea0(i+1) + sarea1(i) + sarea1(i+1);
                Real tarea = amm + amp + apm + app;
//                sbuf[p++] = (x2flux(nn, k  , j, i  )*sarea0(i  )
//                            + x2flux(nn, k  , j, i+1)*sarea0(i+1)
//                            + x2flux(nn, k+1, j, i  )*sarea1(i  )
//                            + x2flux(nn, k+1, j, i+1)*sarea1(i+1))/tarea;
                sbuf[p++] = (x2flux(nn, k  , j, i  )*amm
                            + x2flux(nn, k  , j, i+1)*apm
                            + x2flux(nn, k+1, j, i  )*amp
                            + x2flux(nn, k+1, j, i+1)*app)/tarea;
              }
            }
          }
        } else if (pmb->block_size.nx2>1) { // 2D
          int k = pmb->ks;
          for (int nn=nl_; nn<=nu_; nn++) {
//            pco->Face2Area(0, j, pmb->is ,pmb->ie, sarea0);
            for (int i=pmb->is; i<=pmb->ie; i+=2) {
              Real am = pco->dx1v(i)*pco->dx3v(k);
              Real ap = pco->dx1v(i+1)*pco->dx3v(k);
//              Real tarea = sarea0(i) + sarea0(i+1);
              Real tarea = am + ap;
//              sbuf[p++] = (x2flux(nn, k, j, i  )*sarea0(i  )
//                          + x2flux(nn, k, j, i+1)*sarea0(i+1))/tarea;
              sbuf[p++] = (x2flux(nn, k, j, i  )*am
                          + x2flux(nn, k, j, i+1)*ap)/tarea;
            }
          }
        }
        // x3 direction - 3D onl_y
      } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
        int k = pmb->ks + (pmb->ke-pmb->ks + 1)*(nb.fid & 1);
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int j=pmb->js; j<=pmb->je; j+=2) {
//            pco->Face3Area(k, j,   pmb->is, pmb->ie, sarea0);
//            pco->Face3Area(k, j+1, pmb->is, pmb->ie, sarea1);
            for (int i=pmb->is; i<=pmb->ie; i+=2) {
              Real amm = pco->dx1v(i)*pco->dx2v(j);
              Real amp = pco->dx1v(i)*pco->dx2v(j+1);
              Real apm = pco->dx1v(i+1)*pco->dx2v(j);
              Real app = pco->dx1v(i+1)*pco->dx2v(j+1);
//              Real tarea = sarea0(i) + sarea0(i+1) + sarea1(i) + sarea1(i+1);
              Real tarea = amm + apm + amp + app;
//              sbuf[p++] = (x3flux(nn, k, j  , i  )*sarea0(i  )
//                           + x3flux(nn, k, j  , i+1)*sarea0(i+1)
//                           + x3flux(nn, k, j+1, i  )*sarea1(i  )
//                           + x3flux(nn, k, j+1, i+1)*sarea1(i+1))/tarea;
              sbuf[p++] = (x3flux(nn, k, j  , i  )*amm
                           + x3flux(nn, k, j  , i+1)*apm
                           + x3flux(nn, k, j+1, i  )*amp
                           + x3flux(nn, k, j+1, i+1)*app)/tarea;
            }
          }
        }
      }
      if (nb.snb.rank == Globals::my_rank) { // on the same node
        CopyFluxCorrectionBufferSameProcess(nb, p);
      }
#ifdef MPI_PARALLEL
      else
        MPI_Start(&(bd_var_flcor_.req_send[nb.bufid]));
#endif
      bd_var_flcor_.sflag[nb.bufid] = BoundaryStatus::completed;
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn bool CellCenteredBoundaryVariable::ReceiveFluxCorrection()
//  \brief Receive and apply the surface flux from the finer neighbor(s)

bool CellCenteredBoundaryVariable::ReceiveFluxCorrection() {
  if (DBGPR_FLXCOR_CC){
  coutBoldGreen("CellCenteredBoundaryVariable::ReceiveFluxCorrection\n");
  }
  MeshBlock *pmb = pmy_block_;
  bool bflag=true;

  for (int n=0; n<pbval_->nneighbor; n++) {
    NeighborBlock& nb = pbval_->neighbor[n];
    if (nb.ni.type != NeighborConnect::face) break;
    if (nb.snb.level == pmb->loc.level+1) {
      if (bd_var_flcor_.flag[nb.bufid] == BoundaryStatus::completed) continue;
      if (bd_var_flcor_.flag[nb.bufid] == BoundaryStatus::waiting) {
        if (nb.snb.rank == Globals::my_rank) {// on the same process
          bflag = false;
          continue;
        }
#ifdef MPI_PARALLEL
        else { // NOLINT
          int test;
          MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &test,
                     MPI_STATUS_IGNORE);
          MPI_Test(&(bd_var_flcor_.req_recv[nb.bufid]), &test, MPI_STATUS_IGNORE);
          if (!static_cast<bool>(test)) {
            bflag = false;
            continue;
          }
          bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::arrived;
        }
#endif
      }
      // boundary arrived; apply flux correction
      int p = 0;
      Real *rbuf=bd_var_flcor_.recv[nb.bufid];
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1) {
        int il = pmb->is + (pmb->ie - pmb->is)*nb.fid+nb.fid;
        int jl = pmb->js, ju = pmb->je, kl = pmb->ks, ku = pmb->ke;
        if (nb.ni.fi1 == 0) ju -= pmb->block_size.nx2/2;
        else          jl += pmb->block_size.nx2/2;
        if (nb.ni.fi2 == 0) ku -= pmb->block_size.nx3/2;
        else          kl += pmb->block_size.nx3/2;
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int k=kl; k<=ku; k++) {
            for (int j=jl; j<=ju; j++)
              x1flux(nn,k,j,il) = rbuf[p++];
          }
        }
      } else if (nb.fid == BoundaryFace::inner_x2 || nb.fid == BoundaryFace::outer_x2) {
        int jl = pmb->js + (pmb->je - pmb->js)*(nb.fid & 1) + (nb.fid & 1);
        int il = pmb->is, iu = pmb->ie, kl = pmb->ks, ku = pmb->ke;
        if (nb.ni.fi1 == 0) iu -= pmb->block_size.nx1/2;
        else          il += pmb->block_size.nx1/2;
        if (nb.ni.fi2 == 0) ku -= pmb->block_size.nx3/2;
        else          kl += pmb->block_size.nx3/2;
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int k=kl; k<=ku; k++) {
            for (int i=il; i<=iu; i++)
              x2flux(nn,k,jl,i) = rbuf[p++];
          }
        }
      } else if (nb.fid == BoundaryFace::inner_x3 || nb.fid == BoundaryFace::outer_x3) {
        int kl = pmb->ks + (pmb->ke - pmb->ks)*(nb.fid & 1) + (nb.fid & 1);
        int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je;
        if (nb.ni.fi1 == 0) iu -= pmb->block_size.nx1/2;
        else          il += pmb->block_size.nx1/2;
        if (nb.ni.fi2 == 0) ju -= pmb->block_size.nx2/2;
        else          jl += pmb->block_size.nx2/2;
        for (int nn=nl_; nn<=nu_; nn++) {
          for (int j=jl; j<=ju; j++) {
            for (int i=il; i<=iu; i++)
              x3flux(nn,kl,j,i) = rbuf[p++];
          }
        }
      }
      bd_var_flcor_.flag[nb.bufid] = BoundaryStatus::completed;
    }
  }
  return bflag;
}
