//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh_amr.cpp
//  \brief implementation of Mesh::AdaptiveMeshRefinement() and related
//  utilities

// C headers

// C++ headers
#include <algorithm>  // std::sort()
#include <cassert>
#include <cstdint>
#include <cstring>  // std::memcpy
#include <iostream>
#include <limits>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../comm/amr_registry.hpp"
#include "../comm/amr_transfer.hpp"
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

Mesh::AMRStatus Mesh::LoadBalancingAndAdaptiveMeshRefinement(
  ParameterInput* pin)
{
  int nnew = 0, ndel = 0;

  if (adaptive)
  {
    UpdateMeshBlockTree(nnew, ndel);
    nbnew += nnew;
    nbdel += ndel;
  }

  // at least one (de)refinement happened
  AMRStatus status =
    (nnew != 0 || ndel != 0) ? AMRStatus::refined : AMRStatus::unchanged;

  lb_flag_ |= lb_automatic_;

  UpdateCostList();

  if (status == AMRStatus::refined)
  {
    // Return value intentionally discarded: redistribution is unconditional
    // when the mesh topology changed.  The call is needed for its MPI gather
    // side-effect.
    GatherCostListAndCheckBalance();
    RedistributeAndRefineMeshBlocks(pin, nbtotal + nnew - ndel);
  }
  else if (lb_flag_ && step_since_lb >= lb_interval_)
  {
    if (!GatherCostListAndCheckBalance())  // load imbalance detected
    {
      RedistributeAndRefineMeshBlocks(pin, nbtotal);
      status = AMRStatus::redistributed;
    }
    lb_flag_ = false;
  }

  return status;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::CalculateLoadBalance(double *clist, int *rlist, int *slist,
//                                      int *nlist, int nb)
// \brief Calculate distribution of MeshBlocks based on the cost list

void Mesh::CalculateLoadBalance(double* clist,
                                int* rlist,
                                int* slist,
                                int* nlist,
                                int nb)
{
  std::stringstream msg;
  double real_max  = std::numeric_limits<double>::max();
  double totalcost = 0, maxcost = 0.0, mincost = (real_max);

  for (int i = 0; i < nb; i++)
  {
    totalcost += clist[i];
    mincost = std::min(mincost, clist[i]);
    maxcost = std::max(maxcost, clist[i]);
  }

  int j             = (Globals::nranks)-1;
  double targetcost = totalcost / Globals::nranks;
  double mycost     = 0.0;
  // create rank list from the end: the master MPI rank should have less load
  for (int i = nb - 1; i >= 0; i--)
  {
    if (targetcost == 0.0)
    {
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There is at least one process which has no MeshBlock"
          << std::endl
          << "Decrease the number of processes or use smaller MeshBlocks."
          << std::endl;
      ATHENA_ERROR(msg);
    }
    mycost += clist[i];
    rlist[i] = j;
    if (mycost >= targetcost && j > 0)
    {
      j--;
      totalcost -= mycost;
      mycost     = 0.0;
      targetcost = totalcost / (j + 1);
    }
  }
  slist[0] = 0;
  j        = 0;
  for (int i = 1; i < nb; i++)
  {  // make the list of nbstart and nblocks
    if (rlist[i] != rlist[i - 1])
    {
      nlist[j]   = i - slist[j];
      slist[++j] = i;
    }
  }
  nlist[j] = nb - slist[j];

#ifdef MPI_PARALLEL
  if (nb % (Globals::nranks * num_mesh_threads_) != 0 && !adaptive &&
      !lb_flag_ && maxcost == mincost && Globals::my_rank == 0)
  {
    std::cout << "### Warning in CalculateLoadBalance" << std::endl
              << "The number of MeshBlocks cannot be divided evenly. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  if ((Globals::nranks) * (num_mesh_threads_) > nb)
  {
    msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
        << "There are fewer MeshBlocks than OpenMP threads on each MPI rank"
        << std::endl
        << "Decrease the number of threads or use more MeshBlocks."
        << std::endl
        << "nb=" << nb
        << ", n_threads*n_procs=" << ((Globals::nranks) * (num_mesh_threads_))
        << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ResetLoadBalanceVariables()
// \brief reset counters and flags for load balancing

void Mesh::ResetLoadBalanceVariables()
{
  if (lb_automatic_)
  {
    const auto& pmb_array = GetMeshBlocksCached();
    const int nmb         = pmb_array.size();
    for (int i = 0; i < nmb; ++i)
    {
      MeshBlock* pmb     = pmb_array[i];
      costlist[pmb->gid] = TINY_NUMBER;
      pmb->ResetTimeMeasurement();
    }
  }
  lb_flag_      = false;
  step_since_lb = 0;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateCostList()
// \brief update the cost list

void Mesh::UpdateCostList()
{
  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();
  if (lb_automatic_)
  {
    double w = static_cast<double>(lb_interval_ - 1) /
               static_cast<double>(lb_interval_);
    for (int i = 0; i < nmb; ++i)
    {
      MeshBlock* pmb     = pmb_array[i];
      costlist[pmb->gid] = costlist[pmb->gid] * w + pmb->cost_;
    }
  }
  else if (lb_flag_)
  {
    for (int i = 0; i < nmb; ++i)
    {
      MeshBlock* pmb     = pmb_array[i];
      costlist[pmb->gid] = pmb->cost_;
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel)
// \brief collect refinement flags and manipulate the MeshBlockTree

void Mesh::UpdateMeshBlockTree(int& nnew, int& ndel)
{
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2, dim = 1;
  if (mesh_size.nx2 > 1)
    nleaf = 4, dim = 2;
  if (mesh_size.nx3 > 1)
    nleaf = 8, dim = 3;
  (void)dim;

  const auto& pmb_array = GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

  // collect refinement flags from all the meshblocks
  // count the number of the blocks to be (de)refined
  nref[Globals::my_rank]   = 0;
  nderef[Globals::my_rank] = 0;
  for (int i = 0; i < nmb; ++i)
  {
    MeshBlock* pmb = pmb_array[i];
    if (pmb->pmr->refine_flag_ == 1)
      nref[Globals::my_rank]++;
    if (pmb->pmr->refine_flag_ == -1)
      nderef[Globals::my_rank]++;
  }
#ifdef MPI_PARALLEL
  // Merge two MPI_Allgather calls into one by packing nref+nderef per rank
  {
    std::vector<int> rd_buf(2 * Globals::nranks);
    rd_buf[2 * Globals::my_rank]     = nref[Globals::my_rank];
    rd_buf[2 * Globals::my_rank + 1] = nderef[Globals::my_rank];
    MPI_Allgather(
      MPI_IN_PLACE, 2, MPI_INT, rd_buf.data(), 2, MPI_INT, MPI_COMM_WORLD);
    for (int n = 0; n < Globals::nranks; n++)
    {
      nref[n]   = rd_buf[2 * n];
      nderef[n] = rd_buf[2 * n + 1];
    }
  }
#endif

  // count the number of the blocks to be (de)refined and displacement
  int tnref = 0, tnderef = 0;
  for (int n = 0; n < Globals::nranks; n++)
  {
    tnref += nref[n];
    tnderef += nderef[n];
  }
  if (tnref == 0 && tnderef < nleaf)  // nothing to do
    return;

  int rd = 0, dd = 0;
  for (int n = 0; n < Globals::nranks; n++)
  {
    rdisp[n] = rd;
    ddisp[n] = dd;
    // technically could overflow, since sizeof() operator returns
    // std::size_t = long unsigned int > int
    // on many platforms (LP64). However, these are used below in MPI calls for
    // integer arguments (recvcounts, displs). MPI does not support > 64-bit
    // count ranges
    bnref[n]   = static_cast<int>(nref[n] * sizeof(LogicalLocation));
    bnderef[n] = static_cast<int>(nderef[n] * sizeof(LogicalLocation));
    brdisp[n]  = static_cast<int>(rd * sizeof(LogicalLocation));
    bddisp[n]  = static_cast<int>(dd * sizeof(LogicalLocation));
    rd += nref[n];
    dd += nderef[n];
  }

  // allocate memory for the location arrays
  LogicalLocation *lref{}, *lderef{}, *clderef{};
  if (tnref > 0)
    lref = new LogicalLocation[tnref];
  if (tnderef >= nleaf)
  {
    lderef  = new LogicalLocation[tnderef];
    clderef = new LogicalLocation[tnderef / nleaf];
  }

  // collect the locations and costs
  int iref = rdisp[Globals::my_rank], ideref = ddisp[Globals::my_rank];
  for (int i = 0; i < nmb; ++i)
  {
    MeshBlock* pmb = pmb_array[i];
    if (pmb->pmr->refine_flag_ == 1)
      lref[iref++] = pmb->loc;
    if (pmb->pmr->refine_flag_ == -1 && tnderef >= nleaf)
      lderef[ideref++] = pmb->loc;
  }
#ifdef MPI_PARALLEL
  if (tnref > 0)
  {
    MPI_Allgatherv(MPI_IN_PLACE,
                   bnref[Globals::my_rank],
                   MPI_BYTE,
                   lref,
                   bnref,
                   brdisp,
                   MPI_BYTE,
                   MPI_COMM_WORLD);
  }
  if (tnderef >= nleaf)
  {
    MPI_Allgatherv(MPI_IN_PLACE,
                   bnderef[Globals::my_rank],
                   MPI_BYTE,
                   lderef,
                   bnderef,
                   bddisp,
                   MPI_BYTE,
                   MPI_COMM_WORLD);
  }
#endif

  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf)
  {
    int lk = 0, lj = 0;
    if (mesh_size.nx2 > 1)
      lj = 1;
    if (mesh_size.nx3 > 1)
      lk = 1;
    for (int n = 0; n < tnderef; n++)
    {
      if ((lderef[n].lx1 & 1LL) == 0LL && (lderef[n].lx2 & 1LL) == 0LL &&
          (lderef[n].lx3 & 1LL) == 0LL)
      {
        int r = n, rr = 0;
        for (std::int64_t k = 0; k <= lk; k++)
        {
          for (std::int64_t j = 0; j <= lj; j++)
          {
            for (std::int64_t i = 0; i <= 1; i++)
            {
              if (r < tnderef)
              {
                if ((lderef[n].lx1 + i) == lderef[r].lx1 &&
                    (lderef[n].lx2 + j) == lderef[r].lx2 &&
                    (lderef[n].lx3 + k) == lderef[r].lx3 &&
                    lderef[n].level == lderef[r].level)
                  rr++;
                r++;
              }
            }
          }
        }
        if (rr == nleaf)
        {
          clderef[ctnd].lx1   = lderef[n].lx1 >> 1;
          clderef[ctnd].lx2   = lderef[n].lx2 >> 1;
          clderef[ctnd].lx3   = lderef[n].lx3 >> 1;
          clderef[ctnd].level = lderef[n].level - 1;
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1)
    std::sort(clderef, clderef + ctnd, LogicalLocation::Greater);

  if (tnderef >= nleaf)
    delete[] lderef;

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation
  // Step 1. perform refinement
  for (int n = 0; n < tnref; n++)
  {
    MeshBlockTree* bt = tree.FindMeshBlock(lref[n]);
    bt->Refine(nnew);
  }
  if (tnref != 0)
    delete[] lref;

  // Step 2. perform derefinement
  for (int n = 0; n < ctnd; n++)
  {
    MeshBlockTree* bt = tree.FindMeshBlock(clderef[n]);
    bt->Derefine(ndel);
  }
  if (tnderef >= nleaf)
    delete[] clderef;

  return;
}

//----------------------------------------------------------------------------------------
// \!fn bool Mesh::GatherCostListAndCheckBalance()
// \brief collect the cost from MeshBlocks and check the load balance

bool Mesh::GatherCostListAndCheckBalance()
{
  if (lb_manual_ || lb_automatic_)
  {
#ifdef MPI_PARALLEL
    MPI_Allgatherv(MPI_IN_PLACE,
                   nblist[Globals::my_rank],
                   MPI_DOUBLE,
                   costlist,
                   nblist,
                   nslist,
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);
#endif
    double maxcost = 0.0, avecost = 0.0;
    for (int rank = 0; rank < Globals::nranks; rank++)
    {
      double rcost = 0.0;
      int ns       = nslist[rank];
      int ne       = ns + nblist[rank];
      for (int n = ns; n < ne; ++n)
        rcost += costlist[n];
      maxcost = std::max(maxcost, rcost);
      avecost += rcost;
    }
    avecost /= Globals::nranks;

    if (maxcost > (1.0 + lb_tolerance_) * avecost)
      return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, int
// ntot)
// \brief redistribute MeshBlocks according to the new load balance

void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput* pin, int ntot)
{
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (mesh_size.nx2 > 1)
    nleaf = 4;
  if (mesh_size.nx3 > 1)
    nleaf = 8;

  // Step 1. construct new lists
  LogicalLocation* newloc = new LogicalLocation[ntot];
  int* newrank            = new int[ntot];
  double* newcost         = new double[ntot];
  int* newtoold           = new int[ntot];
  int* oldtonew           = new int[nbtotal];
  int nbtold              = nbtotal;
  tree.GetMeshBlockList(newloc, newtoold, nbtotal);

  // create a list mapping the previous gid to the current one
  oldtonew[0] = 0;
  int mb_idx  = 1;
  for (int n = 1; n < ntot; n++)
  {
    if (newtoold[n] == newtoold[n - 1] + 1)
    {  // normal
      oldtonew[mb_idx++] = n;
    }
    else if (newtoold[n] == newtoold[n - 1] + nleaf)
    {  // derefined
      for (int j = 0; j < nleaf - 1; j++)
        oldtonew[mb_idx++] = n - 1;
      oldtonew[mb_idx++] = n;
    }
  }
  // fill the last block
  for (; mb_idx < nbtold; mb_idx++)
    oldtonew[mb_idx] = ntot - 1;

  current_level = 0;
  for (int n = 0; n < ntot; n++)
  {
    // "on" = "old n" = "old gid" = "old global MeshBlock ID"
    int on = newtoold[n];
    if (newloc[n].level > current_level)  // set the current max level
      current_level = newloc[n].level;
    if (newloc[n].level >= loclist[on].level)
    {  // same or refined
      newcost[n] = costlist[on];
    }
    else
    {
      double acost = 0.0;
      for (int l = 0; l < nleaf; l++)
        acost += costlist[on + l];
      newcost[n] = acost / nleaf;
    }
  }
  // Store old nbstart and nbend before load balancing in Step 2.
  int onbs = nslist[Globals::my_rank];
  int onbe = onbs + nblist[Globals::my_rank] - 1;

  // Step 2. Calculate new load balance
  CalculateLoadBalance(newcost, newrank, nslist, nblist, ntot);

  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;

  // Steps 3-6: Compute buffer sizes from AMRRegistry and construct AMRTransfer
  // to handle all MPI communication + same-rank cross-level data movement.
  // AMRTransfer is created unconditionally (it is a no-op for nsend=nrecv=0
  // and is used for same-rank cross-level fills even in non-MPI builds).
  comm::AMRRegistry* pamr_rep = pblock->pamr;

  int bssame_amr = pamr_rep->TotalSameLevelSize();
  int bsf2c_amr  = 0;
  int bsc2f_amr  = 0;
  for (int g = 0; g < static_cast<int>(comm::AMRGroup::NumGroups); ++g)
  {
    const comm::AMRBufferSizes& gs =
      pamr_rep->GetBufferSizes(static_cast<comm::AMRGroup>(g));
    bsf2c_amr += gs.fine_to_coarse;
    bsc2f_amr += gs.coarse_to_fine;
  }

  // +1 for deref_count_ when adaptive
  if (adaptive)
    bssame_amr++;

  comm::AMRTransferMap xfer_map;
  xfer_map.newloc   = newloc;
  xfer_map.loclist  = loclist;
  xfer_map.newtoold = newtoold;
  xfer_map.oldtonew = oldtonew;
  xfer_map.ranklist = ranklist;
  xfer_map.newrank  = newrank;
  xfer_map.nslist   = nslist;
  xfer_map.nbs      = nbs;
  xfer_map.nbe      = nbe;
  xfer_map.onbs     = onbs;
  xfer_map.onbe     = onbe;
  xfer_map.nleaf    = nleaf;
  xfer_map.adaptive = adaptive;
  xfer_map.bssame   = bssame_amr;
  xfer_map.bsf2c    = bsf2c_amr;
  xfer_map.bsc2f    = bsc2f_amr;

  comm::AMRTransfer amr_xfer(this, xfer_map);

#ifdef MPI_PARALLEL
  // Steps 5+6: post receives, then pack + send (old blocks still alive).
  amr_xfer.PostReceives();

  // deref_count_ pack callback: appends the derefinement counter after
  // AMRRegistry's same-level pack.
  comm::AMRTransfer::DerefPackFn deref_pack_fn = nullptr;
  if (adaptive)
  {
    deref_pack_fn = [](MeshBlock* pb, Real* buf, int offset)
    {
      static_assert(
        sizeof(int) <= sizeof(Real),
        "int must fit in a single Real element of the send buffer");
      int deref_tmp = pb->pmr->deref_count_;
      std::memcpy(&buf[offset], &deref_tmp, sizeof(int));
    };
  }
  amr_xfer.PackAndSend(deref_pack_fn);
#endif  // MPI_PARALLEL

  // Step 7. construct a new MeshBlock list (moving the data within the MPI
  // rank)
  MeshBlock* newlist    = nullptr;
  MeshBlock* pmb        = nullptr;
  RegionSize block_size = pblock->block_size;

  for (int n = nbs; n <= nbe; n++)
  {
    int on = newtoold[n];
    if ((ranklist[on] == Globals::my_rank) &&
        (loclist[on].level == newloc[n].level))
    {
      // on the same MPI rank and same level -> just move it
      MeshBlock* pob = FindMeshBlock(on);
      if (pob->prev == nullptr)
      {
        pblock = pob->next;
      }
      else
      {
        pob->prev->next = pob->next;
      }
      if (pob->next != nullptr)
        pob->next->prev = pob->prev;
      pob->next = nullptr;
      if (n == nbs)
      {  // first
        pob->prev = nullptr;
        newlist   = pob;
        pmb       = newlist;
      }
      else
      {
        pmb->next = pob;
        pob->prev = pmb;
        pmb       = pmb->next;
      }
      pmb->gid = n;
      pmb->lid = n - nbs;
    }
    else
    {
      // on a different refinement level or MPI rank - create a new block
      BoundaryFlag block_bcs[6];
      SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
      // insert new block in singly-linked list of MeshBlocks
      if (n == nbs)
      {  // first node
        newlist = new MeshBlock(
          n, n - nbs, newloc[n], block_size, block_bcs, this, pin, true);
        pmb = newlist;
      }
      else
      {
        pmb->next = new MeshBlock(
          n, n - nbs, newloc[n], block_size, block_bcs, this, pin, true);
        pmb->next->prev = pmb;
        pmb             = pmb->next;
      }
      // Finalize AMR registration so that buffer sizes are available
      // for WaitAndUnpack (Step 8) on cross-rank f2c/c2f transfers.
      if (pmb->pamr != nullptr)
        pmb->pamr->Finalize();
      // fill the conservative variables
      if ((loclist[on].level > newloc[n].level))
      {  // fine to coarse (f2c)
        for (int ll = 0; ll < nleaf; ll++)
        {
          if (ranklist[on + ll] != Globals::my_rank)
            continue;
          // fine to coarse on the same MPI rank (different AMR level) -
          // restriction
          MeshBlock* pob = FindMeshBlock(on + ll);
          amr_xfer.FillFineToCoarseSameRank(pob, pmb, loclist[on + ll]);
        }
      }
      else if ((loclist[on].level <
                newloc[n].level) &&  // coarse to fine (c2f)
               (ranklist[on] == Globals::my_rank))
      {
        // coarse to fine on the same MPI rank (different AMR level) -
        // prolongation
        MeshBlock* pob = FindMeshBlock(on);
        amr_xfer.FillCoarseToFineSameRank(pob, pmb, newloc[n]);
      }
    }
  }

  // discard remaining MeshBlocks
  // they could be reused, but for the moment, just throw them away for
  // simplicity
  if (pblock != nullptr)
  {
    while (pblock->next != nullptr)
      delete pblock->next;
    delete pblock;
  }

  // Replace the MeshBlock list
  pblock = newlist;
  RebuildBlockByGid();

  // Step 8. Receive the data and load into MeshBlocks
#ifdef MPI_PARALLEL
  {
    // deref_count_ unpack callback
    comm::AMRTransfer::DerefUnpackFn deref_unpack_fn = nullptr;
    if (adaptive)
    {
      deref_unpack_fn = [](MeshBlock* pb, Real* buf, int offset)
      {
        int deref_tmp;
        std::memcpy(&deref_tmp, &buf[offset], sizeof(int));
        pb->pmr->deref_count_ = deref_tmp;
      };
    }
    amr_xfer.WaitAndUnpack(deref_unpack_fn);
    amr_xfer.WaitSendsAndCleanup();
  }
#endif

  // deallocate arrays
  delete[] loclist;
  delete[] ranklist;
  delete[] costlist;
  delete[] newtoold;
  delete[] oldtonew;

  // update the lists
  loclist  = newloc;
  ranklist = newrank;
  costlist = newcost;

  // re-initialize the MeshBlocks
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  tree.ComputeNeighborLevelFlags();
#endif
  // Cache is fresh - RebuildBlockByGid() was called above.
  const auto& pmb_array = GetMeshBlocksCached();
  // Each block writes only to its own NeighborConnectivity; tree/ranklist/
  // nslist are read-only at this point, so the loop is embarrassingly
  // parallel.
  const int n_pmb = static_cast<int>(pmb_array.size());
#pragma omp parallel for schedule(static)
  for (int idx = 0; idx < n_pmb; ++idx)
  {
    pmb_array[idx]->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  Initialize(initialize_style::regrid, pin);

  ResetLoadBalanceVariables();

  return;
}

// Legacy AMR helper functions removed - now handled by comm::AMRTransfer and
// comm::AMRRegistry.  The following functions have been superseded:
//   PrepareSendSameLevel        -> AMRRegistry::PackSameLevel
//   PrepareSendCoarseToFineAMR  -> AMRRegistry::PackCoarseToFine
//   PrepareSendFineToCoarseAMR  -> AMRRegistry::PackFineToCoarse
//   FillSameRankFineToCoarseAMR -> AMRTransfer::FillFineToCoarseSameRank
//   FillSameRankCoarseToFineAMR -> AMRTransfer::FillCoarseToFineSameRank
//   FinishRecvSameLevel         -> AMRRegistry::UnpackSameLevel
//   FinishRecvFineToCoarseAMR   -> AMRRegistry::UnpackFineToCoarse
//   FinishRecvCoarseToFineAMR   -> AMRRegistry::UnpackCoarseToFine
//   CreateAMRMPITag             -> AMRTransfer::CreateAMRMPITag
