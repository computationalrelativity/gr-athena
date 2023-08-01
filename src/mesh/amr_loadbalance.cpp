//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh_amr.cpp
//  \brief implementation of Mesh::AdaptiveMeshRefinement() and related utilities

// C headers

// C++ headers
#include <algorithm>  // std::sort()
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/buffer_utils.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "meshblock_tree.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif


//----------------------------------------------------------------------------------------
// \!fn void Mesh::LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin)
// \brief Main function for adaptive mesh refinement

void Mesh::LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin) {
  int nnew = 0, ndel = 0;

  if (adaptive) {
    UpdateMeshBlockTree(nnew, ndel);
    nbnew += nnew; nbdel += ndel;
  }

  lb_flag_ |= lb_automatic_;

  UpdateCostList();

  if (nnew != 0 || ndel != 0) { // at least one (de)refinement happened
    GatherCostListAndCheckBalance();
    RedistributeAndRefineMeshBlocks(pin, nbtotal + nnew - ndel);
  } else if (lb_flag_ && step_since_lb >= lb_interval_) {
    if (!GatherCostListAndCheckBalance()) // load imbalance detected
      RedistributeAndRefineMeshBlocks(pin, nbtotal);
    lb_flag_ = false;
  }
  return;
}


//----------------------------------------------------------------------------------------
// \!fn void Mesh::CalculateLoadBalance(double *clist, int *rlist, int *slist,
//                                      int *nlist, int nb)
// \brief Calculate distribution of MeshBlocks based on the cost list

void Mesh::CalculateLoadBalance(double *clist, int *rlist, int *slist, int *nlist,
                                int nb) {
  std::stringstream msg;
  double real_max  =  std::numeric_limits<double>::max();
  double totalcost = 0, maxcost = 0.0, mincost = (real_max);

  for (int i=0; i<nb; i++) {
    totalcost += clist[i];
    mincost = std::min(mincost,clist[i]);
    maxcost = std::max(maxcost,clist[i]);
  }

  int j = (Globals::nranks) - 1;
  double targetcost = totalcost/Globals::nranks;
  double mycost = 0.0;
  // create rank list from the end: the master MPI rank should have less load
  for (int i=nb-1; i>=0; i--) {
    if (targetcost == 0.0) {
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There is at least one process which has no MeshBlock" << std::endl
          << "Decrease the number of processes or use smaller MeshBlocks." << std::endl;
      ATHENA_ERROR(msg);
    }
    mycost += clist[i];
    rlist[i] = j;
    if (mycost >= targetcost && j>0) {
      j--;
      totalcost -= mycost;
      mycost = 0.0;
      targetcost = totalcost/(j+1);
    }
  }
  slist[0] = 0;
  j = 0;
  for (int i=1; i<nb; i++) { // make the list of nbstart and nblocks
    if (rlist[i] != rlist[i-1]) {
      nlist[j] = i-slist[j];
      slist[++j] = i;
    }
  }
  nlist[j] = nb-slist[j];

  if (Globals::my_rank == 0) {
    for (int i=0; i<Globals::nranks; i++) {
      double rcost = 0.0;
      for(int n=slist[i]; n<slist[i]+nlist[i]; n++)
        rcost += clist[n];
    }
  }

#ifdef MPI_PARALLEL
  if (nb % (Globals::nranks * num_mesh_threads_) != 0
      && !adaptive && !lb_flag_ && maxcost == mincost && Globals::my_rank == 0) {
    std::cout << "### Warning in CalculateLoadBalance" << std::endl
              << "The number of MeshBlocks cannot be divided evenly. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  if ((Globals::nranks)*(num_mesh_threads_) > nb) {
    msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
        << "There are fewer MeshBlocks than OpenMP threads on each MPI rank" << std::endl
        << "Decrease the number of threads or use more MeshBlocks." << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ResetLoadBalanceVariables()
// \brief reset counters and flags for load balancing

void Mesh::ResetLoadBalanceVariables() {
  if (lb_automatic_) {
    MeshBlock *pmb = pblock;
    while (pmb != nullptr) {
      costlist[pmb->gid] = TINY_NUMBER;
      pmb->ResetTimeMeasurement();
      pmb = pmb->next;
    }
  }
  lb_flag_ = false;
  step_since_lb = 0;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateCostList()
// \brief update the cost list

void Mesh::UpdateCostList() {
  MeshBlock *pmb = pblock;
  if (lb_automatic_) {
    double w = static_cast<double>(lb_interval_-1)/static_cast<double>(lb_interval_);
    while (pmb != nullptr) {
      costlist[pmb->gid] = costlist[pmb->gid]*w+pmb->cost_;
      pmb = pmb->next;
    }
  } else if (lb_flag_) {
    while (pmb != nullptr) {
      costlist[pmb->gid] = pmb->cost_;
      pmb = pmb->next;
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel)
// \brief collect refinement flags and manipulate the MeshBlockTree

void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel) {
  // compute nleaf= number of leaf MeshBlocks per refined block
  MeshBlock *pmb;
  int nleaf = 2, dim = 1;
  if (mesh_size.nx2 > 1) nleaf = 4, dim = 2;
  if (mesh_size.nx3 > 1) nleaf = 8, dim = 3;
  (void)dim;

  // collect refinement flags from all the meshblocks
  // count the number of the blocks to be (de)refined
  nref[Globals::my_rank] = 0;
  nderef[Globals::my_rank] = 0;
  pmb = pblock;
  while (pmb != nullptr) {
    if (pmb->pmr->refine_flag_ ==  1) nref[Globals::my_rank]++;
    if (pmb->pmr->refine_flag_ == -1) nderef[Globals::my_rank]++;
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nref,   1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nderef, 1, MPI_INT, MPI_COMM_WORLD);
#endif

  // count the number of the blocks to be (de)refined and displacement
  int tnref = 0, tnderef = 0;
  for (int n=0; n<Globals::nranks; n++) {
    tnref  += nref[n];
    tnderef += nderef[n];
  }
  if (tnref == 0 && tnderef < nleaf) // nothing to do
    return;

  int rd = 0, dd = 0;
  for (int n=0; n<Globals::nranks; n++) {
    rdisp[n] = rd;
    ddisp[n] = dd;
    // technically could overflow, since sizeof() operator returns
    // std::size_t = long unsigned int > int
    // on many platforms (LP64). However, these are used below in MPI calls for
    // integer arguments (recvcounts, displs). MPI does not support > 64-bit count ranges
    bnref[n] = static_cast<int>(nref[n]*sizeof(LogicalLocation));
    bnderef[n] = static_cast<int>(nderef[n]*sizeof(LogicalLocation));
    brdisp[n] = static_cast<int>(rd*sizeof(LogicalLocation));
    bddisp[n] = static_cast<int>(dd*sizeof(LogicalLocation));
    rd += nref[n];
    dd += nderef[n];
  }

  // allocate memory for the location arrays
  LogicalLocation *lref{}, *lderef{}, *clderef{};
  if (tnref > 0)
    lref = new LogicalLocation[tnref];
  if (tnderef >= nleaf) {
    lderef = new LogicalLocation[tnderef];
    clderef = new LogicalLocation[tnderef/nleaf];
  }

  // collect the locations and costs
  int iref = rdisp[Globals::my_rank], ideref = ddisp[Globals::my_rank];
  pmb = pblock;
  while (pmb != nullptr) {
    if (pmb->pmr->refine_flag_ ==  1)
      lref[iref++] = pmb->loc;
    if (pmb->pmr->refine_flag_ == -1 && tnderef >= nleaf)
      lderef[ideref++] = pmb->loc;
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  if (tnref > 0) {
    MPI_Allgatherv(MPI_IN_PLACE, bnref[Globals::my_rank],   MPI_BYTE,
                   lref,   bnref,   brdisp, MPI_BYTE, MPI_COMM_WORLD);
  }
  if (tnderef >= nleaf) {
    MPI_Allgatherv(MPI_IN_PLACE, bnderef[Globals::my_rank], MPI_BYTE,
                   lderef, bnderef, bddisp, MPI_BYTE, MPI_COMM_WORLD);
  }
#endif

  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf) {
    int lk = 0, lj = 0;
    if (mesh_size.nx2 > 1) lj = 1;
    if (mesh_size.nx3 > 1) lk = 1;
    for (int n=0; n<tnderef; n++) {
      if ((lderef[n].lx1 & 1LL) == 0LL &&
          (lderef[n].lx2 & 1LL) == 0LL &&
          (lderef[n].lx3 & 1LL) == 0LL) {
        int r = n, rr = 0;
        for (std::int64_t k=0; k<=lk; k++) {
          for (std::int64_t j=0; j<=lj; j++) {
            for (std::int64_t i=0; i<=1; i++) {
              if (r < tnderef) {
                if ((lderef[n].lx1+i) == lderef[r].lx1
                    && (lderef[n].lx2+j) == lderef[r].lx2
                    && (lderef[n].lx3+k) == lderef[r].lx3
                    &&  lderef[n].level  == lderef[r].level)
                  rr++;
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          clderef[ctnd].lx1   = lderef[n].lx1>>1;
          clderef[ctnd].lx2   = lderef[n].lx2>>1;
          clderef[ctnd].lx3   = lderef[n].lx3>>1;
          clderef[ctnd].level = lderef[n].level-1;
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1)
    std::sort(clderef, &(clderef[ctnd-1]), LogicalLocation::Greater);

  if (tnderef >= nleaf)
    delete [] lderef;

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation
  // Step 1. perform refinement
  for (int n=0; n<tnref; n++) {
    MeshBlockTree *bt=tree.FindMeshBlock(lref[n]);
    bt->Refine(nnew);
  }
  if (tnref != 0)
    delete [] lref;

  // Step 2. perform derefinement
  for (int n=0; n<ctnd; n++) {
    MeshBlockTree *bt = tree.FindMeshBlock(clderef[n]);
    bt->Derefine(ndel);
  }
  if (tnderef >= nleaf)
    delete [] clderef;

  return;
}

//----------------------------------------------------------------------------------------
// \!fn bool Mesh::GatherCostListAndCheckBalance()
// \brief collect the cost from MeshBlocks and check the load balance

bool Mesh::GatherCostListAndCheckBalance() {
  if (lb_manual_ || lb_automatic_) {
#ifdef MPI_PARALLEL
    MPI_Allgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_DOUBLE, costlist, nblist,
                   nslist, MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    double maxcost = 0.0, avecost = 0.0;
    for (int rank=0; rank<Globals::nranks; rank++) {
      double rcost = 0.0;
      int ns = nslist[rank];
      int ne = ns + nblist[rank];
      for (int n=ns; n<ne; ++n)
        rcost += costlist[n];
      maxcost = std::max(maxcost,rcost);
      avecost += rcost;
    }
    avecost /= Globals::nranks;

    if (adaptive) lb_tolerance_ = 2.0*static_cast<double>(Globals::nranks)
                                     /static_cast<double>(nbtotal);

    if (maxcost > (1.0 + lb_tolerance_)*avecost)
      return false;
  }
  return true;
}


//----------------------------------------------------------------------------------------
// \!fn void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, int ntot)
// \brief redistribute MeshBlocks according to the new load balance

void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, int ntot) {
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (mesh_size.nx2 > 1) nleaf = 4;
  if (mesh_size.nx3 > 1) nleaf = 8;

  // Step 1. construct new lists
  LogicalLocation *newloc = new LogicalLocation[ntot];
  int *newrank = new int[ntot];
  double *newcost = new double[ntot];
  int *newtoold = new int[ntot];
  int *oldtonew = new int[nbtotal];
  int nbtold = nbtotal;
  tree.GetMeshBlockList(newloc, newtoold, nbtotal);

  // create a list mapping the previous gid to the current one
  oldtonew[0] = 0;
  int mb_idx = 1;
  for (int n=1; n<ntot; n++) {
    if (newtoold[n] == newtoold[n-1] + 1) { // normal
      oldtonew[mb_idx++] = n;
    } else if (newtoold[n] == newtoold[n-1] + nleaf) { // derefined
      for (int j=0; j<nleaf-1; j++)
        oldtonew[mb_idx++] = n-1;
      oldtonew[mb_idx++] = n;
    }
  }
  // fill the last block
  for ( ; mb_idx<nbtold; mb_idx++)
    oldtonew[mb_idx] = ntot-1;

  current_level = 0;
  for (int n=0; n<ntot; n++) {
    // "on" = "old n" = "old gid" = "old global MeshBlock ID"
    int on = newtoold[n];
    if (newloc[n].level>current_level) // set the current max level
      current_level = newloc[n].level;
    if (newloc[n].level >= loclist[on].level) { // same or refined
      newcost[n] = costlist[on];
    } else {
      double acost = 0.0;
      for (int l=0; l<nleaf; l++)
        acost += costlist[on+l];
      newcost[n] = acost/nleaf;
    }
  }
#ifdef MPI_PARALLEL
  // store old nbstart and nbend before load balancing in Step 2.
  int onbs = nslist[Globals::my_rank];
  int onbe = onbs + nblist[Globals::my_rank] - 1;
#endif
  // Step 2. Calculate new load balance
  CalculateLoadBalance(newcost, newrank, nslist, nblist, ntot);

  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;

#ifdef MPI_PARALLEL
  int bnx1 = pblock->block_size.nx1;
  int bnx2 = pblock->block_size.nx2;
  int bnx3 = pblock->block_size.nx3;
#endif

#ifdef MPI_PARALLEL
  // Step 3. count the number of the blocks to be sent / received
  int nsend = 0, nrecv = 0;
  for (int n=nbs; n<=nbe; n++) {
    int on = newtoold[n];
    if (loclist[on].level > newloc[n].level) { // f2c
      for (int k=0; k<nleaf; k++) {
        if (ranklist[on+k] != Globals::my_rank)
          nrecv++;
      }
    } else {
      if (ranklist[on] != Globals::my_rank)
        nrecv++;
    }
  }
  for (int n=onbs; n<=onbe; n++) {
    int nn = oldtonew[n];
    if (loclist[n].level < newloc[nn].level) { // c2f
      for (int k=0; k<nleaf; k++) {
        if (newrank[nn+k] != Globals::my_rank)
          nsend++;
      }
    } else {
      if (newrank[nn] != Globals::my_rank)
        nsend++;
    }
  }

  // Step 4. calculate buffer sizes
  Real **sendbuf(nullptr), **recvbuf(nullptr);
  // use the first MeshBlock in the linked list of blocks belonging to this MPI rank as a
  // representative of all MeshBlocks for counting the "load-balancing registered" and
  // "SMR/AMR-enrolled" quantities (loop over MeshBlock::vars_cc_, not MeshRefinement)

  // TODO(felker): add explicit check to ensure that elements of pb->vars_cc/fc_ and
  // pb->pmr->pvars_cc/fc_ v point to the same objects, if adaptive

  // int num_cc = pblock->pmr->pvars_cc_.size();
  int num_fc = pblock->vars_fc_.size();
  int nx4_tot = 0;
  for (AthenaArray<Real> &var_cc : pblock->vars_cc_) {
    nx4_tot += var_cc.GetDim4();
  }

  // cell-centered extended
  int nx4_cx_tot = 0;
  for (AthenaArray<Real> &var_cx : pblock->vars_cx_) {
    nx4_cx_tot += var_cx.GetDim4();
  }

  // vertex-centered
  int nx4_vc_tot = 0;
  for (AthenaArray<Real> &var_vc : pblock->vars_vc_) {
    nx4_vc_tot += var_vc.GetDim4();
  }

  // cell-centered quantities enrolled in SMR/AMR -----------------------------
  int bssame = bnx1*bnx2*bnx3*nx4_tot;
  int bsf2c = (bnx1/2)*((bnx2 + 1)/2)*((bnx3 + 1)/2)*nx4_tot;
  int bsc2f = (bnx1/2 + 2)*((bnx2 + 1)/2 + 2*f2)*((bnx3 + 1)/2 + 2*f3)*nx4_tot;

  // cell-centered extended quantities enrolled in SMR/AMR --------------------
  const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);

  const int cx_hbnx1 = bnx1 / 2;
  const int cx_hbnx2 = (bnx2 > 1) ? bnx2 / 2: 1;
  const int cx_hbnx3 = (bnx3 > 1) ? bnx3 / 2: 1;

  const int cx_ndg1 = min_cx_ng;
  const int cx_ndg2 = (f2 > 0) ? min_cx_ng : 0;
  const int cx_ndg3 = (f3 > 0) ? min_cx_ng : 0;

  bssame += bnx1 * bnx2 * bnx3 * nx4_cx_tot;
  bsc2f += nx4_cx_tot * (
    (cx_hbnx1 + 2 * cx_ndg1) *
    (cx_hbnx2 + 2 * cx_ndg2) *
    (cx_hbnx3 + 2 * cx_ndg3)
  );

  bsf2c += cx_hbnx1 * cx_hbnx2 * cx_hbnx3 * nx4_cx_tot;

  // vertex-centered quantities enrolled in SMR/AMR ---------------------------
  const int vbnx1 = bnx1 + 1;
  const int vbnx2 = (bnx2 > 1) ? bnx2 + 1 : 1;
  const int vbnx3 = (bnx3 > 1) ? bnx3 + 1 : 1;

  const int hbnx1 = bnx1 / 2 + 1;
  const int hbnx2 = bnx2 / 2 + 1;
  const int hbnx3 = bnx3 / 2 + 1;

  const int ndg1 = NGHOST;
  const int ndg2 = (f2 > 0) ? NGHOST : 0;
  const int ndg3 = (f3 > 0) ? NGHOST : 0;

  bssame += vbnx1 * vbnx2 * vbnx3 * nx4_vc_tot;
  bsc2f += nx4_vc_tot * (
    (hbnx1 + 2 * ndg1) *
    (hbnx2 + 2 * ndg2) *
    (hbnx3 + 2 * ndg3)
  );
  bsf2c += hbnx1 * hbnx2 * hbnx3 * nx4_vc_tot;

  // face-centered quantities enrolled in SMR/AMR -----------------------------
  bssame += num_fc*((bnx1 + 1)*bnx2*bnx3 + bnx1*(bnx2 + f2)*bnx3
                    + bnx1*bnx2*(bnx3 + f3));
  bsf2c += num_fc*(((bnx1/2) + 1)*((bnx2 + 1)/2)*((bnx3 + 1)/2)
                   + (bnx1/2)*(((bnx2 + 1)/2) + f2)*((bnx3 + 1)/2)
                   + (bnx1/2)*((bnx2 + 1)/2)*(((bnx3 + 1)/2) + f3));
  bsc2f += num_fc*(((bnx1/2) + 1 + 2)*((bnx2 + 1)/2 + 2*f2)*((bnx3 + 1)/2 + 2*f3)
                   + (bnx1/2 + 2)*(((bnx2 + 1)/2) + f2 + 2*f2)*((bnx3 + 1)/2 + 2*f3)
                   + (bnx1/2 + 2)*((bnx2 + 1)/2 + 2*f2)*(((bnx3 + 1)/2) + f3 + 2*f3));

  // add one more element to buffer size for storing the derefinement counter
  bssame++;

  MPI_Request *req_send(nullptr), *req_recv(nullptr);
  // Step 5. allocate and start receiving buffers
  if (nrecv != 0) {
    recvbuf = new Real*[nrecv];
    req_recv = new MPI_Request[nrecv];
    int rb_idx = 0;     // recv buffer index
    for (int n=nbs; n<=nbe; n++) {
      int on = newtoold[n];
      LogicalLocation &oloc = loclist[on];
      LogicalLocation &nloc = newloc[n];
      if (oloc.level > nloc.level) { // f2c
        for (int l=0; l<nleaf; l++) {
          if (ranklist[on+l] == Globals::my_rank) continue;
          LogicalLocation &lloc = loclist[on+l];
          int ox1 = ((lloc.lx1 & 1LL) == 1LL), ox2 = ((lloc.lx2 & 1LL) == 1LL),
              ox3 = ((lloc.lx3 & 1LL) == 1LL);
          recvbuf[rb_idx] = new Real[bsf2c];
          int tag = CreateAMRMPITag(n-nbs, ox1, ox2, ox3);
          MPI_Irecv(recvbuf[rb_idx], bsf2c, MPI_ATHENA_REAL, ranklist[on+l],
                    tag, MPI_COMM_WORLD, &(req_recv[rb_idx]));
          rb_idx++;
        }
      } else { // same level or c2f
        if (ranklist[on] == Globals::my_rank) continue;
        int size;
        if (oloc.level == nloc.level) {
          size = bssame;
        } else {
          size = bsc2f;
        }
        recvbuf[rb_idx] = new Real[size];
        int tag = CreateAMRMPITag(n-nbs, 0, 0, 0);
        MPI_Irecv(recvbuf[rb_idx], size, MPI_ATHENA_REAL, ranklist[on],
                  tag, MPI_COMM_WORLD, &(req_recv[rb_idx]));
        rb_idx++;
      }
    }
  }
  // Step 6. allocate, pack and start sending buffers
  if (nsend != 0) {
    sendbuf = new Real*[nsend];
    req_send = new MPI_Request[nsend];
    int sb_idx = 0;      // send buffer index
    for (int n=onbs; n<=onbe; n++) {
      int nn = oldtonew[n];
      LogicalLocation &oloc = loclist[n];
      LogicalLocation &nloc = newloc[nn];
      MeshBlock* pb = FindMeshBlock(n);
      if (nloc.level == oloc.level) { // same level
        if (newrank[nn] == Globals::my_rank) continue;
        sendbuf[sb_idx] = new Real[bssame];
        PrepareSendSameLevel(pb, sendbuf[sb_idx]);
        int tag = CreateAMRMPITag(nn-nslist[newrank[nn]], 0, 0, 0);
        MPI_Isend(sendbuf[sb_idx], bssame, MPI_ATHENA_REAL, newrank[nn],
                  tag, MPI_COMM_WORLD, &(req_send[sb_idx]));
        sb_idx++;
      } else if (nloc.level > oloc.level) { // c2f
        // c2f must communicate to multiple leaf blocks (unlike f2c, same2same)
        for (int l=0; l<nleaf; l++) {
          if (newrank[nn+l] == Globals::my_rank) continue;
          sendbuf[sb_idx] = new Real[bsc2f];
          PrepareSendCoarseToFineAMR(pb, sendbuf[sb_idx], newloc[nn+l]);
          int tag = CreateAMRMPITag(nn+l-nslist[newrank[nn+l]], 0, 0, 0);
          MPI_Isend(sendbuf[sb_idx], bsc2f, MPI_ATHENA_REAL, newrank[nn+l],
                    tag, MPI_COMM_WORLD, &(req_send[sb_idx]));
          sb_idx++;
        } // end loop over nleaf (unique to c2f branch in this step 6)
      } else { // f2c: restrict + pack + send
        if (newrank[nn] == Globals::my_rank) continue;
        sendbuf[sb_idx] = new Real[bsf2c];
        PrepareSendFineToCoarseAMR(pb, sendbuf[sb_idx]);
        int ox1 = ((oloc.lx1 & 1LL) == 1LL), ox2 = ((oloc.lx2 & 1LL) == 1LL),
            ox3 = ((oloc.lx3 & 1LL) == 1LL);
        int tag = CreateAMRMPITag(nn-nslist[newrank[nn]], ox1, ox2, ox3);
        MPI_Isend(sendbuf[sb_idx], bsf2c, MPI_ATHENA_REAL, newrank[nn],
                  tag, MPI_COMM_WORLD, &(req_send[sb_idx]));
        sb_idx++;
      }
    }
  } // if (nsend !=0)
#endif // MPI_PARALLEL

  // Step 7. construct a new MeshBlock list (moving the data within the MPI rank)
  MeshBlock *newlist = nullptr;
  MeshBlock *pmb = nullptr;
  RegionSize block_size = pblock->block_size;

  for (int n=nbs; n<=nbe; n++) {
    int on = newtoold[n];
    if ((ranklist[on] == Globals::my_rank) && (loclist[on].level == newloc[n].level)) {
      // on the same MPI rank and same level -> just move it
      MeshBlock* pob = FindMeshBlock(on);
      if (pob->prev == nullptr) {
        pblock = pob->next;
      } else {
        pob->prev->next = pob->next;
      }
      if (pob->next != nullptr) pob->next->prev = pob->prev;
      pob->next = nullptr;
      if (n == nbs) { // first
        pob->prev = nullptr;
        newlist = pob;
        pmb = newlist;
      } else {
        pmb->next = pob;
        pob->prev = pmb;
        pmb = pmb->next;
      }
      pmb->gid = n;
      pmb->lid = n - nbs;
    } else {
      // on a different refinement level or MPI rank - create a new block
      BoundaryFlag block_bcs[6];
      SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
      // insert new block in singly-linked list of MeshBlocks
      if (n == nbs) { // first node
        newlist = new MeshBlock(n, n-nbs, newloc[n], block_size, block_bcs, this,
                                pin, gflag, true);
        pmb = newlist;
      } else {
        pmb->next = new MeshBlock(n, n-nbs, newloc[n], block_size, block_bcs, this,
                                  pin, gflag, true);
        pmb->next->prev = pmb;
        pmb = pmb->next;
      }
      // fill the conservative variables
      if ((loclist[on].level > newloc[n].level)) { // fine to coarse (f2c)
        for (int ll=0; ll<nleaf; ll++) {
          if (ranklist[on+ll] != Globals::my_rank) continue;
          // fine to coarse on the same MPI rank (different AMR level) - restriction
          MeshBlock* pob = FindMeshBlock(on+ll);
          FillSameRankFineToCoarseAMR(pob, pmb, loclist[on+ll]);
        }
      } else if ((loclist[on].level < newloc[n].level) && // coarse to fine (c2f)
                 (ranklist[on] == Globals::my_rank)) {
        // coarse to fine on the same MPI rank (different AMR level) - prolongation
        MeshBlock* pob = FindMeshBlock(on);
        FillSameRankCoarseToFineAMR(pob, pmb, newloc[n]);
      }
    }
  }

  // discard remaining MeshBlocks
  // they could be reused, but for the moment, just throw them away for simplicity
  if (pblock != nullptr) {
    while (pblock->next  !=  nullptr)
      delete pblock->next;
    delete pblock;
  }

  // Replace the MeshBlock list
  pblock = newlist;

  // Step 8. Receive the data and load into MeshBlocks
  // This is a test: try MPI_Waitall later.
#ifdef MPI_PARALLEL
  if (nrecv != 0) {
    int rb_idx = 0;     // recv buffer index
    for (int n=nbs; n<=nbe; n++) {
      int on = newtoold[n];
      LogicalLocation &oloc = loclist[on];
      LogicalLocation &nloc = newloc[n];
      MeshBlock *pb = FindMeshBlock(n);
      if (oloc.level == nloc.level) { // same
        if (ranklist[on] == Globals::my_rank) continue;
        MPI_Wait(&(req_recv[rb_idx]), MPI_STATUS_IGNORE);
        FinishRecvSameLevel(pb, recvbuf[rb_idx]);
        rb_idx++;
      } else if (oloc.level > nloc.level) { // f2c
        for (int l=0; l<nleaf; l++) {
          if (ranklist[on+l] == Globals::my_rank) continue;
          MPI_Wait(&(req_recv[rb_idx]), MPI_STATUS_IGNORE);
          FinishRecvFineToCoarseAMR(pb, recvbuf[rb_idx], loclist[on+l]);
          rb_idx++;
        }
      } else { // c2f
        if (ranklist[on] == Globals::my_rank) continue;
        MPI_Wait(&(req_recv[rb_idx]), MPI_STATUS_IGNORE);
        FinishRecvCoarseToFineAMR(pb, recvbuf[rb_idx]);
        rb_idx++;
      }
    }
  }
#endif

  // deallocate arrays
  delete [] loclist;
  delete [] ranklist;
  delete [] costlist;
  delete [] newtoold;
  delete [] oldtonew;
#ifdef MPI_PARALLEL
  if (nsend != 0) {
    MPI_Waitall(nsend, req_send, MPI_STATUSES_IGNORE);
    for (int n=0; n<nsend; n++)
      delete [] sendbuf[n];
    delete [] sendbuf;
    delete [] req_send;
  }
  if (nrecv != 0) {
    for (int n=0; n<nrecv; n++)
      delete [] recvbuf[n];
    delete [] recvbuf;
    delete [] req_recv;
  }
#endif

  // update the lists
  loclist = newloc;
  ranklist = newrank;
  costlist = newcost;

  // re-initialize the MeshBlocks
  pmb = pblock;
  while (pmb != nullptr) {
    pmb->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
    pmb = pmb->next;
  }
  Initialize(2, pin);

  ResetLoadBalanceVariables();

  return;
}

// AMR: step 6, branch 1 (same2same: just pack+send)

void Mesh::PrepareSendSameLevel(MeshBlock* pb, Real *sendbuf) {
  // pack
  int p = 0;

  // this helper fn is used for AMR and non-refinement load balancing of
  // MeshBlocks. Therefore, unlike PrepareSendCoarseToFineAMR(), etc., it loops over
  // MeshBlock::vars_cc/fc_ containers, not MeshRefinement::pvars_cc/fc_ containers

  // TODO(felker): add explicit check to ensure that elements of pb->vars_cc/fc_ and
  // pb->pmr->pvars_cc/fc_ v point to the same objects, if adaptive

  // (C++11) range-based for loop: (automatic type deduction fails when iterating over
  // container with std::reference_wrapper; could use auto var_cc_r = var_cc.get())

  // cell-centered ------------------------------------------------------------
  for (AthenaArray<Real> &var_cc : pb->vars_cc_) {
    int nu = var_cc.GetDim4() - 1;
    BufferUtility::PackData(var_cc, sendbuf, 0, nu,
                            pb->is, pb->ie,
                            pb->js, pb->je,
                            pb->ks, pb->ke, p);
  }

  // cell-centered extended ---------------------------------------------------
  for (AthenaArray<Real> &var_cx : pb->vars_cx_) {
    int nu = var_cx.GetDim4() - 1;
    BufferUtility::PackData(var_cx, sendbuf, 0, nu,
                            pb->cx_is, pb->cx_ie,
                            pb->cx_js, pb->cx_je,
                            pb->cx_ks, pb->cx_ke, p);
  }

  // vertex-centered ----------------------------------------------------------
  for (AthenaArray<Real> &var_vc : pb->vars_vc_) {
    int nu = var_vc.GetDim4() - 1;
    BufferUtility::PackData(var_vc, sendbuf, 0, nu,
                            pb->ivs, pb->ive,
                            pb->jvs, pb->jve,
                            pb->kvs, pb->kve, p);
  }

  // face-centered ------------------------------------------------------------
  for (FaceField &var_fc : pb->vars_fc_) {
    BufferUtility::PackData(var_fc.x1f, sendbuf,
                            pb->is, pb->ie+1,
                            pb->js, pb->je,
                            pb->ks, pb->ke, p);
    BufferUtility::PackData(var_fc.x2f, sendbuf,
                            pb->is, pb->ie,
                            pb->js, pb->je+f2,
                            pb->ks, pb->ke, p);
    BufferUtility::PackData(var_fc.x3f, sendbuf,
                            pb->is, pb->ie,
                            pb->js, pb->je,
                            pb->ks, pb->ke+f3, p);
  }


  // WARNING(felker): casting from "Real *" to "int *" in order to append single integer
  // to send buffer is slightly unsafe (especially if sizeof(int) > sizeof(Real))
  if (adaptive) {
    int *dcp = reinterpret_cast<int *>(&(sendbuf[p]));
    *dcp = pb->pmr->deref_count_;
  }
  return;
}


// step 6, branch 2 (c2f: just pack+send)

void Mesh::PrepareSendCoarseToFineAMR(MeshBlock* pb, Real *sendbuf,
                                      LogicalLocation &lloc) {

  int p = 0;

  const int b_hsz1 = pb->block_size.nx1/2;
  const int b_hsz2 = pb->block_size.nx2/2;
  const int b_hsz3 = pb->block_size.nx3/2;

  const int ox1 = ((lloc.lx1 & 1LL) == 1LL);
  const int ox2 = ((lloc.lx2 & 1LL) == 1LL);
  const int ox3 = ((lloc.lx3 & 1LL) == 1LL);


  // cell-centered ------------------------------------------------------------

  // pack (c.f. FillSameRankCoarseToFineAMR indicial logic)
  int il, iu, jl, ju, kl, ku;
  if (ox1 == 0)
  {
    il = pb->is - 1;
    iu = pb->is + b_hsz1;
  }
  else
  {
    il = pb->is + b_hsz1-1;
    iu = pb->ie + 1;
  }

  if (ox2 == 0)
  {
    jl = pb->js - f2;
    ju = pb->js + b_hsz2;
  }
  else
  {
    jl = pb->js + b_hsz2 - f2;
    ju = pb->je + f2;
  }

  if (ox3 == 0)
  {
    kl = pb->ks - f3;
    ku = pb->ks + b_hsz3;
  }
  else
  {
    kl = pb->ks + b_hsz3 - f3;
    ku = pb->ke + f3;
  }

  for (auto cc_pair : pb->pmr->pvars_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    int nu = var_cc->GetDim4() - 1;
    BufferUtility::PackData(*var_cc, sendbuf, 0, nu,
                            il, iu, jl, ju, kl, ku, p);
  }

  // cell-centered extended ---------------------------------------------------

  // fill (below) with the maximum number of ghosts possible
  const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);

  int cx_il, cx_iu, cx_jl, cx_ju, cx_kl, cx_ku;
  if (ox1 == 0)
  {
    cx_il = pb->cx_is - min_cx_ng;
    cx_iu = pb->cx_is + b_hsz1 + (min_cx_ng - 1);
  }
  else
  {
    cx_il = pb->cx_is + b_hsz1 - 1 - (min_cx_ng - 1);
    cx_iu = pb->cx_ie + min_cx_ng;
  }

  if (ox2 == 0)
  {
    cx_jl = pb->cx_js - f2 * min_cx_ng;
    cx_ju = pb->cx_js + b_hsz2 + f2 * (min_cx_ng - 1);
  }
  else
  {
    cx_jl = pb->cx_js + b_hsz2 - f2 * min_cx_ng;
    cx_ju = pb->cx_je + f2 * min_cx_ng;
  }

  if (ox3 == 0)
  {
    cx_kl = pb->cx_ks - f3 * min_cx_ng;
    cx_ku = pb->cx_ks + b_hsz3 + f3 * (min_cx_ng - 1);
  }
  else
  {
    cx_kl = pb->cx_ks + b_hsz3 - f3 * min_cx_ng;
    cx_ku = pb->cx_ke + f3 * min_cx_ng;
  }

  for (auto cx_pair : pb->pmr->pvars_cx_) {
    AthenaArray<Real> *var_cx = std::get<0>(cx_pair);
    int nu = var_cx->GetDim4() - 1;
    BufferUtility::PackData(*var_cx, sendbuf, 0, nu,
                            cx_il, cx_iu,
                            cx_jl, cx_ju,
                            cx_kl, cx_ku, p);
  }

  // vertex-centered ----------------------------------------------------------
  int ndg1 = NGHOST;
  int ndg2 = (f2 > 0) ? NGHOST : 0;
  int ndg3 = (f3 > 0) ? NGHOST : 0;

  int vc_il, vc_iu;
  int vc_jl, vc_ju;
  int vc_kl, vc_ku;

  if (ox1 == 0) {
    vc_il = pb->ivs - ndg1;
    vc_iu = vc_il + (pb->cive + ndg1 - (pb->civs - ndg1));
  } else {
    vc_il = b_hsz1 + pb->ivs - ndg1;
    vc_iu = vc_il + (pb->cive + ndg1 - (pb->civs - ndg1));
  }
  if (ox2 == 0) {
    vc_jl = pb->jvs - ndg2;
    vc_ju = vc_jl + (pb->cjve + ndg2 - (pb->cjvs - ndg2));
  } else {
    vc_jl = b_hsz2 + pb->jvs - ndg2;
    vc_ju = vc_jl + (pb->cjve + ndg2 - (pb->cjvs - ndg2));
  }
  if (ox3 == 0) {
    vc_kl = pb->kvs - ndg3;
    vc_ku = vc_kl + (pb->ckve + ndg3 - (pb->ckvs - ndg3));
  } else {
    vc_kl = b_hsz3 + pb->kvs - ndg3;
    vc_ku = vc_kl + (pb->ckve + ndg3 - (pb->ckvs - ndg3));
  }

  for (auto vc_pair : pb->pmr->pvars_vc_) {
    AthenaArray<Real> *var_vc = std::get<0>(vc_pair);
    int nu = var_vc->GetDim4() - 1;

    BufferUtility::PackData(*var_vc, sendbuf, 0, nu,
                            vc_il, vc_iu,
                            vc_jl, vc_ju,
                            vc_kl, vc_ku, p);
  }

  // face-centered ------------------------------------------------------------
  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    BufferUtility::PackData((*var_fc).x1f, sendbuf,
                            il, iu+1, jl, ju, kl, ku, p);
    BufferUtility::PackData((*var_fc).x2f, sendbuf,
                            il, iu, jl, ju+f2, kl, ku, p);
    BufferUtility::PackData((*var_fc).x3f, sendbuf,
                            il, iu, jl, ju, kl, ku+f3, p);
  }

  return;
}

// step 6, branch 3 (f2c: restrict, pack, send)

void Mesh::PrepareSendFineToCoarseAMR(MeshBlock* pb, Real *sendbuf) {
  // restrict and pack
  MeshRefinement *pmr = pb->pmr;
  int p = 0;

  // cell-centered ------------------------------------------------------------
  for (auto cc_pair : pmr->pvars_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    AthenaArray<Real> *coarse_cc = std::get<1>(cc_pair);
    int nu = var_cc->GetDim4() - 1;
    pmr->RestrictCellCenteredValues(*var_cc, *coarse_cc,
                                    0, nu,
                                    pb->cis, pb->cie,
                                    pb->cjs, pb->cje,
                                    pb->cks, pb->cke);
    BufferUtility::PackData(*coarse_cc, sendbuf, 0, nu,
                            pb->cis, pb->cie,
                            pb->cjs, pb->cje,
                            pb->cks, pb->cke, p);
  }

  // cell-centered extended ---------------------------------------------------
  for (auto cx_pair : pmr->pvars_cx_) {
    AthenaArray<Real> *var_cx = std::get<0>(cx_pair);
    AthenaArray<Real> *coarse_cx = std::get<1>(cx_pair);
    int nu = var_cx->GetDim4() - 1;
    pmr->RestrictCellCenteredXValues(*var_cx, *coarse_cx,
                                     0, nu,
                                     pb->cx_cis, pb->cx_cie,
                                     pb->cx_cjs, pb->cx_cje,
                                     pb->cx_cks, pb->cx_cke);
    BufferUtility::PackData(*coarse_cx, sendbuf, 0, nu,
                            pb->cx_cis, pb->cx_cie,
                            pb->cx_cjs, pb->cx_cje,
                            pb->cx_cks, pb->cx_cke, p);
  }

  // vertex-centered ----------------------------------------------------------
  for (auto vc_pair : pmr->pvars_vc_) {
    AthenaArray<Real> *var_vc = std::get<0>(vc_pair);
    AthenaArray<Real> *coarse_vc = std::get<1>(vc_pair);
    int nu = var_vc->GetDim4() - 1;
    pmr->RestrictVertexCenteredValues(*var_vc, *coarse_vc,
                                      0, nu,
                                      pb->civs, pb->cive,
                                      pb->cjvs, pb->cjve,
                                      pb->ckvs, pb->ckve);
    BufferUtility::PackData(*coarse_vc, sendbuf, 0, nu,
                            pb->civs, pb->cive,
                            pb->cjvs, pb->cjve,
                            pb->ckvs, pb->ckve, p);

  }

  // face-centered ------------------------------------------------------------
  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);
    pmr->RestrictFieldX1((*var_fc).x1f, (*coarse_fc).x1f,
                         pb->cis, pb->cie+1,
                         pb->cjs, pb->cje,
                         pb->cks, pb->cke);
    BufferUtility::PackData((*coarse_fc).x1f, sendbuf,
                            pb->cis, pb->cie+1,
                            pb->cjs, pb->cje,
                            pb->cks, pb->cke, p);
    pmr->RestrictFieldX2((*var_fc).x2f, (*coarse_fc).x2f,
                         pb->cis, pb->cie,
                         pb->cjs, pb->cje+f2,
                         pb->cks, pb->cke);
    BufferUtility::PackData((*coarse_fc).x2f, sendbuf,
                            pb->cis, pb->cie,
                            pb->cjs, pb->cje+f2,
                            pb->cks, pb->cke, p);
    pmr->RestrictFieldX3((*var_fc).x3f, (*coarse_fc).x3f,
                         pb->cis, pb->cie,
                         pb->cjs, pb->cje,
                         pb->cks, pb->cke+f3);
    BufferUtility::PackData((*coarse_fc).x3f, sendbuf,
                            pb->cis, pb->cie,
                            pb->cjs, pb->cje,
                            pb->cks, pb->cke+f3, p);
  }


  return;
}

// step 7: f2c, same MPI rank, different level (just restrict+copy, no pack/send)

void Mesh::FillSameRankFineToCoarseAMR(MeshBlock* pob, MeshBlock* pmb,
                                       LogicalLocation &loc) {
  MeshRefinement *pmr = pob->pmr;

  const int b_hsz1 = pob->block_size.nx1/2;
  const int b_hsz2 = pob->block_size.nx2/2;
  const int b_hsz3 = pob->block_size.nx3/2;

  const bool llx1 = ((loc.lx1 & 1LL) == 1LL);
  const bool llx2 = ((loc.lx2 & 1LL) == 1LL);
  const bool llx3 = ((loc.lx3 & 1LL) == 1LL);

  // cell-centered ------------------------------------------------------------
  int il = pmb->is + llx1 * b_hsz1;
  int jl = pmb->js + llx2 * b_hsz2;
  int kl = pmb->ks + llx3 * b_hsz3;

  // absent a zip() feature for range-based for loops, manually advance the
  // iterator over "SMR/AMR-enrolled" cell-centered quantities on the new
  // MeshBlock in lock-step with pob
  auto pmb_cc_it = pmb->pmr->pvars_cc_.begin();
  // iterate MeshRefinement std::vectors on pob
  for (auto cc_pair : pmr->pvars_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    AthenaArray<Real> *coarse_cc = std::get<1>(cc_pair);
    int nu = var_cc->GetDim4() - 1;
    pmr->RestrictCellCenteredValues(*var_cc, *coarse_cc,
                                    0, nu,
                                    pob->cis, pob->cie,
                                    pob->cjs, pob->cje,
                                    pob->cks, pob->cke);
    // copy from old/original/other MeshBlock (pob) to newly created block (pmb)
    AthenaArray<Real> &src = *coarse_cc;
    AthenaArray<Real> &dst = *std::get<0>(*pmb_cc_it); // pmb->phydro->u;
    for (int nv=0; nv<=nu; nv++) {
      for (int k=kl, fk=pob->cks; fk<=pob->cke; k++, fk++) {
        for (int j=jl, fj=pob->cjs; fj<=pob->cje; j++, fj++) {
          for (int i=il, fi=pob->cis; fi<=pob->cie; i++, fi++)
            dst(nv, k, j, i) = src(nv, fk, fj, fi);
        }
      }
    }
    pmb_cc_it++;
  }

  // cell-centered extended ---------------------------------------------------
  int cx_il = pmb->cx_is + llx1 * b_hsz1;
  int cx_jl = pmb->cx_js + llx2 * b_hsz2;
  int cx_kl = pmb->cx_ks + llx3 * b_hsz3;

  auto pmb_cx_it = pmb->pmr->pvars_cx_.begin();
  // iterate MeshRefinement std::vectors on pob
  for (auto cx_pair : pmr->pvars_cx_) {
    AthenaArray<Real> *var_cx = std::get<0>(cx_pair);
    AthenaArray<Real> *coarse_cx = std::get<1>(cx_pair);
    int nu = var_cx->GetDim4() - 1;
    pmr->RestrictCellCenteredXValues(*var_cx, *coarse_cx,
                                    0, nu,
                                    pob->cx_cis, pob->cx_cie,
                                    pob->cx_cjs, pob->cx_cje,
                                    pob->cx_cks, pob->cx_cke);
    // copy from old/original/other MeshBlock (pob) to newly created block (pmb)
    AthenaArray<Real> &src = *coarse_cx;
    AthenaArray<Real> &dst = *std::get<0>(*pmb_cx_it); // pmb->phydro->u;
    for (int nv=0; nv<=nu; nv++)
    for (int k=kl, fk=pob->cx_cks; fk<=pob->cx_cke; k++, fk++)
    for (int j=jl, fj=pob->cx_cjs; fj<=pob->cx_cje; j++, fj++)
    for (int i=il, fi=pob->cx_cis; fi<=pob->cx_cie; i++, fi++)
    {
      dst(nv, k, j, i) = src(nv, fk, fj, fi);
    }
    pmb_cx_it++;
  }
  // --------------------------------------------------------------------------

  // vertex-centered ----------------------------------------------------------
  const int vc_il = pmb->ivs + llx1 * b_hsz1;
  const int vc_jl = pmb->jvs + llx2 * b_hsz2;
  const int vc_kl = pmb->kvs + llx3 * b_hsz3;

  auto pmb_vc_it = pmb->pmr->pvars_vc_.begin();
  // iterate MeshRefinement std::vectors on pob
  for (auto vc_pair : pmr->pvars_vc_) {
    AthenaArray<Real> *var_vc = std::get<0>(vc_pair);
    AthenaArray<Real> *coarse_vc = std::get<1>(vc_pair);
    int nu = var_vc->GetDim4() - 1;

    pmr->RestrictVertexCenteredValues(*var_vc, *coarse_vc,
                                      0, nu,
                                      pob->civs, pob->cive,
                                      pob->cjvs, pob->cjve,
                                      pob->ckvs, pob->ckve);
    // copy from old/original/other MeshBlock (pob)
    // to newly created block (pmb)
    AthenaArray<Real> &src = *coarse_vc;
    AthenaArray<Real> &dst = *std::get<0>(*pmb_vc_it);
    for (int nv=0; nv<=nu; nv++)
    for (int k=vc_kl, fk=pob->ckvs; fk<=pob->ckve; k++, fk++)
    for (int j=vc_jl, fj=pob->cjvs; fj<=pob->cjve; j++, fj++)
    for (int i=vc_il, fi=pob->civs; fi<=pob->cive; i++, fi++)
    {
      dst(nv, k, j, i) = src(nv, fk, fj, fi);
    }

    pmb_vc_it++;
  }

  // face-centered ------------------------------------------------------------
  auto pmb_fc_it = pmb->pmr->pvars_fc_.begin();
  for (auto fc_pair : pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);
    pmr->RestrictFieldX1((*var_fc).x1f, (*coarse_fc).x1f,
                         pob->cis, pob->cie+1,
                         pob->cjs, pob->cje,
                         pob->cks, pob->cke);
    pmr->RestrictFieldX2((*var_fc).x2f, (*coarse_fc).x2f,
                         pob->cis, pob->cie,
                         pob->cjs, pob->cje+f2,
                         pob->cks, pob->cke);
    pmr->RestrictFieldX3((*var_fc).x3f, (*coarse_fc).x3f,
                         pob->cis, pob->cie,
                         pob->cjs, pob->cje,
                         pob->cks, pob->cke+f3);
    FaceField &src_b = *coarse_fc;
    FaceField &dst_b = *std::get<0>(*pmb_fc_it); // pmb->pfield->b;
    for (int k=kl, fk=pob->cks; fk<=pob->cke; k++, fk++) {
      for (int j=jl, fj=pob->cjs; fj<=pob->cje; j++, fj++) {
        for (int i=il, fi=pob->cis; fi<=pob->cie+1; i++, fi++)
          dst_b.x1f(k, j, i) = src_b.x1f(fk, fj, fi);
      }
    }
    for (int k=kl, fk=pob->cks; fk<=pob->cke; k++, fk++) {
      for (int j=jl, fj=pob->cjs; fj<=pob->cje+f2; j++, fj++) {
        for (int i=il, fi=pob->cis; fi<=pob->cie; i++, fi++)
          dst_b.x2f(k, j, i) = src_b.x2f(fk, fj, fi);
      }
    }
    if (pmb->block_size.nx2 == 1) {
      int iu = il + b_hsz1 - 1;
      for (int i=il; i<=iu; i++)
        dst_b.x2f(pmb->ks, pmb->js+1, i) = dst_b.x2f(pmb->ks, pmb->js, i);
    }
    for (int k=kl, fk=pob->cks; fk<=pob->cke+f3; k++, fk++) {
      for (int j=jl, fj=pob->cjs; fj<=pob->cje; j++, fj++) {
        for (int i=il, fi=pob->cis; fi<=pob->cie; i++, fi++)
          dst_b.x3f(k, j, i) = src_b.x3f(fk, fj, fi);
      }
    }
    if (pmb->block_size.nx3 == 1) {
      int iu = il + b_hsz1 - 1, ju = jl + b_hsz2 - 1;
      if (pmb->block_size.nx2 == 1) ju = jl;
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++)
          dst_b.x3f(pmb->ks+1, j, i) = dst_b.x3f(pmb->ks, j, i);
      }
    }
    pmb_fc_it++;
  }

  return;
}

// step 7: c2f, same MPI rank, different level (just copy+prolongate, no pack/send)

void Mesh::FillSameRankCoarseToFineAMR(MeshBlock* pob, MeshBlock* pmb,
                                       LogicalLocation &newloc) {
  MeshRefinement *pmr = pmb->pmr;

  const int b_hsz1 = pob->block_size.nx1/2;
  const int b_hsz2 = pob->block_size.nx2/2;
  const int b_hsz3 = pob->block_size.nx3/2;

  const bool nlx1 =((newloc.lx1 & 1LL) == 1LL);
  const bool nlx2 =((newloc.lx2 & 1LL) == 1LL);
  const bool nlx3 =((newloc.lx3 & 1LL) == 1LL);

  // cell-centered ------------------------------------------------------------
  int il = pob->cis - 1;
  int iu = pob->cie + 1;
  int jl = pob->cjs - f2;
  int ju = pob->cje + f2;
  int kl = pob->cks - f3;
  int ku = pob->cke + f3;

  int cis = nlx1 * b_hsz1 + pob->is - 1;
  int cjs = nlx2 * b_hsz2 + pob->js - f2;
  int cks = nlx3 * b_hsz3 + pob->ks - f3;

  auto pob_cc_it = pob->pmr->pvars_cc_.begin();
  // iterate MeshRefinement std::vectors on new pmb
  for (auto cc_pair : pmr->pvars_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    AthenaArray<Real> *coarse_cc = std::get<1>(cc_pair);
    int nu = var_cc->GetDim4() - 1;

    AthenaArray<Real> &src = *std::get<0>(*pob_cc_it);
    AthenaArray<Real> &dst = *coarse_cc;
    // fill the coarse buffer
    for (int nv=0; nv<=nu; nv++) {
      for (int k=kl, ck=cks; k<=ku; k++, ck++) {
        for (int j=jl, cj=cjs; j<=ju; j++, cj++) {
          for (int i=il, ci=cis; i<=iu; i++, ci++)
            dst(nv, k, j, i) = src(nv, ck, cj, ci);
        }
      }
    }
    pmr->ProlongateCellCenteredValues(
        dst, *var_cc, 0, nu,
        pob->cis, pob->cie, pob->cjs, pob->cje, pob->cks, pob->cke);
    pob_cc_it++;
  }

  // cell-centered extended ---------------------------------------------------

  // fill (below) with the maximum number of ghosts possible
  const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);

  int cx_il = pob->cx_cis - min_cx_ng;
  int cx_iu = pob->cx_cie + min_cx_ng;

  int cx_jl = pob->cx_cjs - f2 * min_cx_ng;
  int cx_ju = pob->cx_cje + f2 * min_cx_ng;

  int cx_kl = pob->cx_cks - f3 * min_cx_ng;
  int cx_ku = pob->cx_cke + f3 * min_cx_ng;

  int cx_cis = nlx1 * b_hsz1 + pob->cx_is - 1  * min_cx_ng;
  int cx_cjs = nlx2 * b_hsz2 + pob->cx_js - f2 * min_cx_ng;
  int cx_cks = nlx3 * b_hsz3 + pob->cx_ks - f3 * min_cx_ng;

  auto pob_cx_it = pob->pmr->pvars_cx_.begin();  // fund. on coarse level

  // due to the structure of prolongation impl. need to pre-fill as below

  // iterate MeshRefinement std::vectors on new pmb
  for (auto cx_pair : pmr->pvars_cx_) {
    AthenaArray<Real> *var_cx = std::get<0>(cx_pair);
    AthenaArray<Real> *coarse_cx = std::get<1>(cx_pair);
    int nu = var_cx->GetDim4() - 1;

    AthenaArray<Real> &src = *std::get<0>(*pob_cx_it);
    AthenaArray<Real> &dst = *coarse_cx;
    // populate coarse on new fine level
    dst.Fill(NAN);
    for (int nv=0; nv<=nu; nv++)
    for (int k=cx_kl, ck=cx_cks; k<=cx_ku; k++, ck++)
    for (int j=cx_jl, cj=cx_cjs; j<=cx_ju; j++, cj++)
    for (int i=cx_il, ci=cx_cis; i<=cx_iu; i++, ci++)
    {
      dst(nv, k, j, i) = src(nv, ck, cj, ci);
    }

    // finally do the prolongation
    pmr->ProlongateCellCenteredXValues(
        dst, *var_cx, 0, nu,
        pob->cx_cis, pob->cx_cie,
        pob->cx_cjs, pob->cx_cje,
        pob->cx_cks, pob->cx_cke);
    pob_cx_it++;
  }
  // --------------------------------------------------------------------------

  // vertex-centered ----------------------------------------------------------

  // fill (below) with the maximum number of ghosts possible
  const int min_vc_ng = std::min(NCGHOST, NGHOST);

  int ndg1 = min_vc_ng;
  int ndg2 = (f2 > 0) ? min_vc_ng : 0;
  int ndg3 = (f3 > 0) ? min_vc_ng : 0;

  const int vc_il = pob->civs - ndg1;
  const int vc_iu = pob->cive + ndg1;

  const int vc_jl = pob->cjvs - ndg2;
  const int vc_ju = pob->cjve + ndg2;

  const int vc_kl = pob->ckvs - ndg3;
  const int vc_ku = pob->ckve + ndg3;

  const int vc_cis = nlx1 * b_hsz1 + pob->ivs - ndg1;
  const int vc_cjs = nlx2 * b_hsz2 + pob->jvs - ndg2;
  const int vc_cks = nlx3 * b_hsz3 + pob->kvs - ndg3;

  auto pob_vc_it = pob->pmr->pvars_vc_.begin();
  // iterate MeshRefinement std::vectors on new pmb
  for (auto vc_pair : pmr->pvars_vc_) {
    AthenaArray<Real> *var_vc = std::get<0>(vc_pair);
    AthenaArray<Real> *coarse_vc = std::get<1>(vc_pair);
    int nu = var_vc->GetDim4() - 1;

    AthenaArray<Real> &src = *std::get<0>(*pob_vc_it);
    AthenaArray<Real> &dst = *coarse_vc;

    // fill the coarse buffer
    for (int nv=0; nv<=nu; nv++)
    for (int k=vc_kl, ck=vc_cks; k<=vc_ku; k++, ck++)
    for (int j=vc_jl, cj=vc_cjs; j<=vc_ju; j++, cj++)
    for (int i=vc_il, ci=vc_cis; i<=vc_iu; i++, ci++)
    {
      dst(nv, k, j, i) = src(nv, ck, cj, ci);
    }

    pmr->ProlongateVertexCenteredValues(
        dst, *var_vc, 0, nu,
        pob->civs, pob->cive,
        pob->cjvs, pob->cjve,
        pob->ckvs, pob->ckve);
    pob_vc_it++;

  }

  // face-centered ------------------------------------------------------------
  auto pob_fc_it = pob->pmr->pvars_fc_.begin();
  // iterate MeshRefinement std::vectors on new pmb
  for (auto fc_pair : pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);

    FaceField &src_b = *std::get<0>(*pob_fc_it);
    FaceField &dst_b = *coarse_fc;
    for (int k=kl, ck=cks; k<=ku; k++, ck++) {
      for (int j=jl, cj=cjs; j<=ju; j++, cj++) {
        for (int i=il, ci=cis; i<=iu+1; i++, ci++)
          dst_b.x1f(k, j, i) = src_b.x1f(ck, cj, ci);
      }
    }
    for (int k=kl, ck=cks; k<=ku; k++, ck++) {
      for (int j=jl, cj=cjs; j<=ju+f2; j++, cj++) {
        for (int i=il, ci=cis; i<=iu; i++, ci++)
          dst_b.x2f(k, j, i) = src_b.x2f(ck, cj, ci);
      }
    }
    for (int k=kl, ck=cks; k<=ku+f3; k++, ck++) {
      for (int j=jl, cj=cjs; j<=ju; j++, cj++) {
        for (int i=il, ci=cis; i<=iu; i++, ci++)
          dst_b.x3f(k, j, i) = src_b.x3f(ck, cj, ci);
      }
    }
    pmr->ProlongateSharedFieldX1(
        dst_b.x1f, (*var_fc).x1f,
        pob->cis, pob->cie+1, pob->cjs, pob->cje, pob->cks, pob->cke);
    pmr->ProlongateSharedFieldX2(
        dst_b.x2f, (*var_fc).x2f,
        pob->cis, pob->cie, pob->cjs, pob->cje+f2, pob->cks, pob->cke);
    pmr->ProlongateSharedFieldX3(
        dst_b.x3f, (*var_fc).x3f,
        pob->cis, pob->cie, pob->cjs, pob->cje, pob->cks, pob->cke+f3);
    pmr->ProlongateInternalField(
        *var_fc, pob->cis, pob->cie,
        pob->cjs, pob->cje, pob->cks, pob->cke);
    pob_fc_it++;
  }

  return;
}

// step 8 (receive and load), branch 1 (same2same: unpack)
void Mesh::FinishRecvSameLevel(MeshBlock *pb, Real *recvbuf) {
  int p = 0;

  // cell-centered ------------------------------------------------------------
  for (AthenaArray<Real> &var_cc : pb->vars_cc_) {
    int nu = var_cc.GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, var_cc, 0, nu,
                              pb->is, pb->ie,
                              pb->js, pb->je,
                              pb->ks, pb->ke, p);
  }

  // cell-centered extended ---------------------------------------------------
  for (AthenaArray<Real> &var_cx : pb->vars_cx_) {
    int nu = var_cx.GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, var_cx, 0, nu,
                              pb->cx_is, pb->cx_ie,
                              pb->cx_js, pb->cx_je,
                              pb->cx_ks, pb->cx_ke, p);
  }

  // vertex-centered ----------------------------------------------------------
  for (AthenaArray<Real> &var_vc : pb->vars_vc_) {
    int nu = var_vc.GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, var_vc, 0, nu,
                              pb->ivs, pb->ive,
                              pb->jvs, pb->jve,
                              pb->kvs, pb->kve, p);
  }

  // face-centered ------------------------------------------------------------
  for (FaceField &var_fc : pb->vars_fc_) {
    BufferUtility::UnpackData(recvbuf, var_fc.x1f,
                              pb->is, pb->ie+1,
                              pb->js, pb->je,
                              pb->ks, pb->ke, p);
    BufferUtility::UnpackData(recvbuf, var_fc.x2f,
                              pb->is, pb->ie,
                              pb->js, pb->je+f2,
                              pb->ks, pb->ke, p);
    BufferUtility::UnpackData(recvbuf, var_fc.x3f,
                              pb->is, pb->ie,
                              pb->js, pb->je,
                              pb->ks, pb->ke+f3, p);
    if (pb->block_size.nx2 == 1) {
      for (int i=pb->is; i<=pb->ie; i++)
        var_fc.x2f(pb->ks, pb->js+1, i) = var_fc.x2f(pb->ks, pb->js, i);
    }
    if (pb->block_size.nx3 == 1) {
      for (int j=pb->js; j<=pb->je; j++) {
        for (int i=pb->is; i<=pb->ie; i++)
          var_fc.x3f(pb->ks+1, j, i) = var_fc.x3f(pb->ks, j, i);
      }
    }
  }


  // WARNING(felker): casting from "Real *" to "int *" in order to read single
  // appended integer from received buffer is slightly unsafe
  if (adaptive) {
    int *dcp = reinterpret_cast<int *>(&(recvbuf[p]));
    pb->pmr->deref_count_ = *dcp;
  }
  return;
}

// step 8 (receive and load), branch 2 (f2c: unpack)
void Mesh::FinishRecvFineToCoarseAMR(MeshBlock *pb, Real *recvbuf,
                                     LogicalLocation &lloc) {

  const int b_hsz1 = pb->block_size.nx1/2;
  const int b_hsz2 = pb->block_size.nx2/2;
  const int b_hsz3 = pb->block_size.nx3/2;

  const int ox1 = ((lloc.lx1 & 1LL) == 1LL);
  const int ox2 = ((lloc.lx2 & 1LL) == 1LL);
  const int ox3 = ((lloc.lx3 & 1LL) == 1LL);

  int p = 0;

  // cell-centered ------------------------------------------------------------
  int il, iu, jl, ju, kl, ku;

  if (ox1 == 0)
  {
    il = pb->is;
    iu = pb->is + b_hsz1 - 1;
  }
  else
  {
    il = pb->is + b_hsz1;
    iu = pb->ie;
  }

  if (ox2 == 0)
  {
    jl = pb->js;
    ju = pb->js + b_hsz2 - f2;
  }
  else
  {
    jl = pb->js + b_hsz2;
    ju = pb->je;
  }

  if (ox3 == 0)
  {
    kl = pb->ks;
    ku = pb->ks + b_hsz3 - f3;
  }
  else
  {
    kl = pb->ks + b_hsz3;
    ku = pb->ke;
  }

  for (auto cc_pair : pb->pmr->pvars_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    int nu = var_cc->GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, *var_cc, 0, nu,
                              il, iu, jl, ju, kl, ku, p);
  }

  // cell-centered extended ---------------------------------------------------
  // fill (below) with the maximum number of ghosts possible
  const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);

  int cx_il, cx_iu, cx_jl, cx_ju, cx_kl, cx_ku;

  if (ox1 == 0)
  {
    cx_il = pb->cx_is;
    cx_iu = pb->cx_is + b_hsz1 - 1;
  }
  else
  {
    cx_il = pb->cx_is + b_hsz1;
    cx_iu = pb->cx_ie;
  }

  if (ox2 == 0)
  {
    cx_jl = pb->cx_js;
    cx_ju = pb->cx_js + b_hsz2 - f2;
  }
  else
  {
    cx_jl = pb->cx_js + b_hsz2;
    cx_ju = pb->cx_je;
  }

  if (ox3 == 0)
  {
    cx_kl = pb->cx_ks;
    cx_ku = pb->cx_ks + b_hsz3 - f3;
  }
  else
  {
    cx_kl = pb->cx_ks + b_hsz3;
    cx_ku = pb->cx_ke;
  }

  for (auto cx_pair : pb->pmr->pvars_cx_) {
    AthenaArray<Real> *var_cx = std::get<0>(cx_pair);
    int nu = var_cx->GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, *var_cx, 0, nu,
                              cx_il, cx_iu,
                              cx_jl, cx_ju,
                              cx_kl, cx_ku, p);
  }
  // vertex-centered ----------------------------------------------------------
  int vc_il, vc_iu;
  int vc_jl, vc_ju;
  int vc_kl, vc_ku;

  if (ox1 == 0) {
    vc_il = pb->ivs;
    vc_iu = pb->ivs + b_hsz1;
  } else {
    vc_il = pb->ivs + b_hsz1;
    vc_iu = pb->ive;}
  if (ox2 == 0) {
    vc_jl = pb->jvs;
    vc_ju = pb->jvs + b_hsz2;
  } else {
    vc_jl = pb->jvs + b_hsz2;
    vc_ju = pb->jve;
  }
  if (ox3 == 0) {
    vc_kl = pb->kvs;
    vc_ku = pb->kvs + b_hsz3;
  } else {
    vc_kl = pb->kvs + b_hsz3;
    vc_ku = pb->kve;
  }

  for (auto vc_pair : pb->pmr->pvars_vc_) {
    AthenaArray<Real> *var_vc = std::get<0>(vc_pair);
    int nu = var_vc->GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, *var_vc, 0, nu,
                              vc_il, vc_iu,
                              vc_jl, vc_ju,
                              vc_kl, vc_ku, p);
  }

  // face-centered ------------------------------------------------------------
  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField &dst_b = *var_fc;
    BufferUtility::UnpackData(recvbuf, dst_b.x1f,
                              il, iu+1, jl, ju, kl, ku, p);
    BufferUtility::UnpackData(recvbuf, dst_b.x2f,
                              il, iu, jl, ju+f2, kl, ku, p);
    BufferUtility::UnpackData(recvbuf, dst_b.x3f,
                              il, iu, jl, ju, kl, ku+f3, p);
    if (pb->block_size.nx2 == 1) {
      for (int i=il; i<=iu; i++)
        dst_b.x2f(pb->ks, pb->js+1, i) = dst_b.x2f(pb->ks, pb->js, i);
    }
    if (pb->block_size.nx3 == 1) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++)
          dst_b.x3f(pb->ks+1, j, i) = dst_b.x3f(pb->ks, j, i);
      }
    }
  }

  return;
}

// step 8 (receive and load), branch 2 (c2f: unpack+prolongate)
void Mesh::FinishRecvCoarseToFineAMR(MeshBlock *pb, Real *recvbuf) {
  MeshRefinement *pmr = pb->pmr;
  int p = 0;

  // cell-centered ------------------------------------------------------------
  int il = pb->cis - 1;
  int iu = pb->cie + 1;

  int jl = pb->cjs - f2;
  int ju = pb->cje + f2;

  int kl = pb->cks - f3;
  int ku = pb->cke + f3;

  for (auto cc_pair : pb->pmr->pvars_cc_) {
    AthenaArray<Real> *var_cc = std::get<0>(cc_pair);
    AthenaArray<Real> *coarse_cc = std::get<1>(cc_pair);
    int nu = var_cc->GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, *coarse_cc,
                              0, nu, il, iu, jl, ju, kl, ku, p);
    pmr->ProlongateCellCenteredValues(
        *coarse_cc, *var_cc, 0, nu,
        pb->cis, pb->cie, pb->cjs, pb->cje, pb->cks, pb->cke);
  }

  // cell-centered extended ---------------------------------------------------
  // fill (below) with the maximum number of ghosts possible
  const int min_cx_ng = std::min(NCGHOST_CX, NGHOST);

  const int cx_il = pb->cx_cis - min_cx_ng;
  const int cx_iu = pb->cx_cie + min_cx_ng;

  const int cx_jl = pb->cx_cjs - f2 * min_cx_ng;
  const int cx_ju = pb->cx_cje + f2 * min_cx_ng;

  const int cx_kl = pb->cx_cks - f3 * min_cx_ng;
  const int cx_ku = pb->cx_cke + f3 * min_cx_ng;

  for (auto cx_pair : pb->pmr->pvars_cx_) {
    AthenaArray<Real> *var_cx = std::get<0>(cx_pair);
    AthenaArray<Real> *coarse_cx = std::get<1>(cx_pair);
    int nu = var_cx->GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, *coarse_cx,
                              0, nu,
                              cx_il, cx_iu,
                              cx_jl, cx_ju,
                              cx_kl, cx_ku, p);
    pmr->ProlongateCellCenteredXValues(
        *coarse_cx, *var_cx, 0, nu,
        pb->cx_cis, pb->cx_cie,
        pb->cx_cjs, pb->cx_cje,
        pb->cx_cks, pb->cx_cke);
  }

  // vertex-centered ----------------------------------------------------------
  int ndg1 = NGHOST;
  int ndg2 = (f2 > 0) ? NGHOST : 0;
  int ndg3 = (f3 > 0) ? NGHOST : 0;

  const int vc_il = pb->civs - ndg1;
  const int vc_iu = pb->cive + ndg1;

  const int vc_jl = pb->cjvs - ndg2;
  const int vc_ju = pb->cjve + ndg2;

  const int vc_kl = pb->ckvs - ndg3;
  const int vc_ku = pb->ckve + ndg3;

  for (auto vc_pair : pmr->pvars_vc_) {
    AthenaArray<Real> *var_vc = std::get<0>(vc_pair);
    AthenaArray<Real> *coarse_vc = std::get<1>(vc_pair);
    int nu = var_vc->GetDim4() - 1;
    BufferUtility::UnpackData(recvbuf, *coarse_vc,
                              0, nu,
                              vc_il, vc_iu,
                              vc_jl, vc_ju,
                              vc_kl, vc_ku, p);

    pmr->ProlongateVertexCenteredValues(
        *coarse_vc, *var_vc, 0, nu,
        pb->civs, pb->cive, pb->cjvs, pb->cjve, pb->ckvs, pb->ckve);

  }

  // face-centered ------------------------------------------------------------
  for (auto fc_pair : pb->pmr->pvars_fc_) {
    FaceField *var_fc = std::get<0>(fc_pair);
    FaceField *coarse_fc = std::get<1>(fc_pair);

    BufferUtility::UnpackData(recvbuf, (*coarse_fc).x1f,
                              il, iu+1, jl, ju, kl, ku, p);
    BufferUtility::UnpackData(recvbuf, (*coarse_fc).x2f,
                              il, iu, jl, ju+f2, kl, ku, p);
    BufferUtility::UnpackData(recvbuf, (*coarse_fc).x3f,
                              il, iu, jl, ju, kl, ku+f3, p);
    pmr->ProlongateSharedFieldX1(
        (*coarse_fc).x1f, (*var_fc).x1f,
        pb->cis, pb->cie+1, pb->cjs, pb->cje, pb->cks, pb->cke);
    pmr->ProlongateSharedFieldX2(
        (*coarse_fc).x2f, (*var_fc).x2f,
        pb->cis, pb->cie, pb->cjs, pb->cje+f2, pb->cks, pb->cke);
    pmr->ProlongateSharedFieldX3(
        (*coarse_fc).x3f, (*var_fc).x3f,
        pb->cis, pb->cie, pb->cjs, pb->cje, pb->cks, pb->cke+f3);
    pmr->ProlongateInternalField(
        *var_fc, pb->cis, pb->cie,
        pb->cjs, pb->cje, pb->cks, pb->cke);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3)
//  \brief calculate an MPI tag for AMR block transfer
// tag = local id of destination (remaining bits) + ox1(1 bit) + ox2(1 bit) + ox3(1 bit)
//       + physics(5 bits)

// See comments on BoundaryBase::CreateBvalsMPITag()

int Mesh::CreateAMRMPITag(int lid, int ox1, int ox2, int ox3) {
  // former "AthenaTagMPI" AthenaTagMPI::amr=8 redefined to 0
  int tag = (lid<<8) | (ox1<<7)| (ox2<<6) | (ox3<<5) | 0;
  assert (tag <= Globals::mpi_tag_ub);
  return tag;
}
