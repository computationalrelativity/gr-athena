//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file amr_transfer.cpp
//  \brief AMRTransfer: MPI lifecycle for AMR block redistribution.
//
//  See amr_transfer.hpp for the full design overview.  This file implements
//  the lifecycle methods (PostReceives, PackAndSend, FillFineToCoarseSameRank,
//  FillCoarseToFineSameRank, WaitAndUnpack, WaitSendsAndCleanup) and the
//  static CreateAMRMPITag helper.

// C headers

// C++ headers
#include <cstring>  // std::memcpy
#include <iostream>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "amr_registry.hpp"
#include "amr_spec.hpp"
#include "amr_transfer.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace comm
{

//========================================================================================
// Constructor / Destructor
//========================================================================================

AMRTransfer::AMRTransfer(Mesh* pm, const AMRTransferMap& map)
    : pmy_mesh_(pm),
      map_(map),
      nsend_(0),
      nrecv_(0),
#ifdef MPI_PARALLEL
      req_send_(nullptr),
      req_recv_(nullptr),
#endif
      sendbuf_(nullptr),
      recvbuf_(nullptr)
{
  CountSendRecv();
}

//----------------------------------------------------------------------------------------

AMRTransfer::~AMRTransfer()
{
  // Free any remaining send buffers (should be freed by WaitSendsAndCleanup,
  // but guard against early destruction).
#ifdef MPI_PARALLEL
  if (sendbuf_ != nullptr)
  {
    for (int n = 0; n < nsend_; ++n)
    {
      delete[] sendbuf_[n];
    }
    delete[] sendbuf_;
    delete[] req_send_;
  }
  if (recvbuf_ != nullptr)
  {
    for (int n = 0; n < nrecv_; ++n)
    {
      delete[] recvbuf_[n];
    }
    delete[] recvbuf_;
    delete[] req_recv_;
  }
#endif
}

//========================================================================================
// CountSendRecv - count cross-rank messages
//========================================================================================
// Mirrors the logic from amr_loadbalance.cpp Step 3 (L426-452).
// nrecv: for each new block this rank will own, count cross-rank sources.
// nsend: for each old block this rank currently owns, count cross-rank
// destinations.

void AMRTransfer::CountSendRecv()
{
  nsend_ = 0;
  nrecv_ = 0;

  const int nbs    = map_.nbs;
  const int nbe    = map_.nbe;
  const int onbs   = map_.onbs;
  const int onbe   = map_.onbe;
  const int nleaf  = map_.nleaf;
  const int myrank = Globals::my_rank;

  // Count receives: iterate over new blocks this rank will own.
  for (int n = nbs; n <= nbe; ++n)
  {
    int on                = map_.newtoold[n];
    LogicalLocation& oloc = map_.loclist[on];
    LogicalLocation& nloc = map_.newloc[n];

    if (oloc.level > nloc.level)
    {
      // f2c: each child that lives on a different rank is one recv
      for (int k = 0; k < nleaf; ++k)
      {
        if (map_.ranklist[on + k] != myrank)
          ++nrecv_;
      }
    }
    else
    {
      // same or c2f: one recv if cross-rank
      if (map_.ranklist[on] != myrank)
        ++nrecv_;
    }
  }

  // Count sends: iterate over old blocks this rank currently owns.
  for (int n = onbs; n <= onbe; ++n)
  {
    int nn                = map_.oldtonew[n];
    LogicalLocation& oloc = map_.loclist[n];
    LogicalLocation& nloc = map_.newloc[nn];

    if (nloc.level > oloc.level)
    {
      // c2f: each child that goes to a different rank is one send
      for (int k = 0; k < nleaf; ++k)
      {
        if (map_.newrank[nn + k] != myrank)
          ++nsend_;
      }
    }
    else
    {
      // same or f2c: one send if cross-rank
      if (map_.newrank[nn] != myrank)
        ++nsend_;
    }
  }
}

//========================================================================================
// CreateAMRMPITag - static helper
//========================================================================================
// Bit layout: [lid (upper bits)] [ox1 (1 bit)] [ox2 (1 bit)] [ox3 (1 bit)]
// This matches the legacy Mesh::CreateAMRMPITag exactly.

int AMRTransfer::CreateAMRMPITag(int lid, int ox1, int ox2, int ox3)
{
  int tag = (lid << 3) | (ox1 << 2) | (ox2 << 1) | ox3;
  if (tag > Globals::mpi_tag_ub)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in AMRTransfer::CreateAMRMPITag" << std::endl
        << "MPI tag " << tag << " exceeds MPI_TAG_UB=" << Globals::mpi_tag_ub
        << std::endl
        << "Too many MeshBlocks per rank (lid=" << lid << ")." << std::endl;
    ATHENA_ERROR(msg);
  }
  return tag;
}

//========================================================================================
// PostReceives - Step 5 equivalent
//========================================================================================

void AMRTransfer::PostReceives()
{
#ifdef MPI_PARALLEL
  if (nrecv_ == 0)
    return;

  recvbuf_   = new Real*[nrecv_];
  req_recv_  = new MPI_Request[nrecv_];
  int rb_idx = 0;

  const int nbs    = map_.nbs;
  const int nbe    = map_.nbe;
  const int nleaf  = map_.nleaf;
  const int myrank = Globals::my_rank;

  for (int n = nbs; n <= nbe; ++n)
  {
    int on                = map_.newtoold[n];
    LogicalLocation& oloc = map_.loclist[on];
    LogicalLocation& nloc = map_.newloc[n];

    if (oloc.level > nloc.level)
    {
      // f2c: post one recv per cross-rank child
      for (int l = 0; l < nleaf; ++l)
      {
        if (map_.ranklist[on + l] == myrank)
          continue;

        LogicalLocation& lloc = map_.loclist[on + l];
        int ox1               = ((lloc.lx1 & 1LL) == 1LL);
        int ox2               = ((lloc.lx2 & 1LL) == 1LL);
        int ox3               = ((lloc.lx3 & 1LL) == 1LL);

        recvbuf_[rb_idx] = new Real[map_.bsf2c];
        int tag          = CreateAMRMPITag(n - nbs, ox1, ox2, ox3);
        MPI_Irecv(recvbuf_[rb_idx],
                  map_.bsf2c,
                  MPI_ATHENA_REAL,
                  map_.ranklist[on + l],
                  tag,
                  MPI_COMM_WORLD,
                  &(req_recv_[rb_idx]));
        ++rb_idx;
      }
    }
    else
    {
      // same level or c2f: one recv
      if (map_.ranklist[on] == myrank)
        continue;

      int size         = (oloc.level == nloc.level) ? map_.bssame : map_.bsc2f;
      recvbuf_[rb_idx] = new Real[size];
      int tag          = CreateAMRMPITag(n - nbs, 0, 0, 0);
      MPI_Irecv(recvbuf_[rb_idx],
                size,
                MPI_ATHENA_REAL,
                map_.ranklist[on],
                tag,
                MPI_COMM_WORLD,
                &(req_recv_[rb_idx]));
      ++rb_idx;
    }
  }
#endif  // MPI_PARALLEL
}

//========================================================================================
// PackAndSend - Step 6 equivalent
//========================================================================================
// Must be called while old MeshBlocks are still alive (before Step 7 deletes
// them).

void AMRTransfer::PackAndSend(DerefPackFn deref_pack_fn)
{
#ifdef MPI_PARALLEL
  if (nsend_ == 0)
    return;

  sendbuf_   = new Real*[nsend_];
  req_send_  = new MPI_Request[nsend_];
  int sb_idx = 0;

  const int onbs   = map_.onbs;
  const int onbe   = map_.onbe;
  const int nleaf  = map_.nleaf;
  const int myrank = Globals::my_rank;

  for (int n = onbs; n <= onbe; ++n)
  {
    int nn                = map_.oldtonew[n];
    LogicalLocation& oloc = map_.loclist[n];
    LogicalLocation& nloc = map_.newloc[nn];
    MeshBlock* pb         = pmy_mesh_->FindMeshBlock(n);

    if (nloc.level == oloc.level)
    {
      // --- same level ---
      if (map_.newrank[nn] == myrank)
        continue;

      sendbuf_[sb_idx] = new Real[map_.bssame];

      // Delegate packing to AMRRegistry
      int offset = pb->pamr->PackSameLevel(pb, sendbuf_[sb_idx]);

      // Orchestrator appends deref_count_ via callback
      if (deref_pack_fn != nullptr)
        deref_pack_fn(pb, sendbuf_[sb_idx], offset);

      int tag = CreateAMRMPITag(nn - map_.nslist[map_.newrank[nn]], 0, 0, 0);
      MPI_Isend(sendbuf_[sb_idx],
                map_.bssame,
                MPI_ATHENA_REAL,
                map_.newrank[nn],
                tag,
                MPI_COMM_WORLD,
                &(req_send_[sb_idx]));
      ++sb_idx;
    }
    else if (nloc.level > oloc.level)
    {
      // --- coarse to fine ---
      // Must send to each child that lives on a different rank.
      for (int l = 0; l < nleaf; ++l)
      {
        if (map_.newrank[nn + l] == myrank)
          continue;

        sendbuf_[sb_idx] = new Real[map_.bsc2f];

        // Pack all groups into one contiguous buffer
        int offset = 0;
        for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
        {
          offset += pb->pamr->PackCoarseToFine(pb,
                                               sendbuf_[sb_idx] + offset,
                                               map_.newloc[nn + l],
                                               static_cast<AMRGroup>(g));
        }

        int tag =
          CreateAMRMPITag(nn + l - map_.nslist[map_.newrank[nn + l]], 0, 0, 0);
        MPI_Isend(sendbuf_[sb_idx],
                  map_.bsc2f,
                  MPI_ATHENA_REAL,
                  map_.newrank[nn + l],
                  tag,
                  MPI_COMM_WORLD,
                  &(req_send_[sb_idx]));
        ++sb_idx;
      }
    }
    else
    {
      // --- fine to coarse ---
      if (map_.newrank[nn] == myrank)
        continue;

      sendbuf_[sb_idx] = new Real[map_.bsf2c];

      // Pack all groups into one contiguous buffer
      int offset = 0;
      for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
      {
        offset += pb->pamr->PackFineToCoarse(
          pb, sendbuf_[sb_idx] + offset, static_cast<AMRGroup>(g));
      }

      int ox1 = ((oloc.lx1 & 1LL) == 1LL);
      int ox2 = ((oloc.lx2 & 1LL) == 1LL);
      int ox3 = ((oloc.lx3 & 1LL) == 1LL);
      int tag =
        CreateAMRMPITag(nn - map_.nslist[map_.newrank[nn]], ox1, ox2, ox3);
      MPI_Isend(sendbuf_[sb_idx],
                map_.bsf2c,
                MPI_ATHENA_REAL,
                map_.newrank[nn],
                tag,
                MPI_COMM_WORLD,
                &(req_send_[sb_idx]));
      ++sb_idx;
    }
  }
#endif  // MPI_PARALLEL
}

//========================================================================================
// FillFineToCoarseSameRank - same-rank f2c data movement
//========================================================================================
// Called inline by the orchestrator during Step 7 for each same-rank child
// that is being merged into a coarser parent.  The orchestrator passes
// explicit old/new block pointers because FindMeshBlock cannot locate both
// simultaneously.

void AMRTransfer::FillFineToCoarseSameRank(MeshBlock* old_fine,
                                           MeshBlock* new_coarse,
                                           LogicalLocation& old_loc)
{
  for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
  {
    old_fine->pamr->FillSameRankFineToCoarse(old_fine,
                                             new_coarse,
                                             old_loc,
                                             static_cast<AMRGroup>(g),
                                             new_coarse->pamr);
  }
}

//========================================================================================
// FillCoarseToFineSameRank - same-rank c2f data movement
//========================================================================================
// Called inline by the orchestrator during Step 7 for each same-rank parent
// that is being refined into finer children.

void AMRTransfer::FillCoarseToFineSameRank(MeshBlock* old_coarse,
                                           MeshBlock* new_fine,
                                           LogicalLocation& new_loc)
{
  for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
  {
    old_coarse->pamr->FillSameRankCoarseToFine(old_coarse,
                                               new_fine,
                                               new_loc,
                                               static_cast<AMRGroup>(g),
                                               old_coarse->pamr,
                                               new_fine->pamr);
  }
}

//========================================================================================
// WaitAndUnpack - Step 8 equivalent
//========================================================================================
// Must be called AFTER the new MeshBlock list is built (Step 7) so that
// FindMeshBlock can locate destination blocks by new GID.

void AMRTransfer::WaitAndUnpack(DerefUnpackFn deref_unpack_fn)
{
#ifdef MPI_PARALLEL
  if (nrecv_ == 0)
    return;

  MPI_Waitall(nrecv_, req_recv_, MPI_STATUSES_IGNORE);

  int rb_idx = 0;

  const int nbs    = map_.nbs;
  const int nbe    = map_.nbe;
  const int nleaf  = map_.nleaf;
  const int myrank = Globals::my_rank;

  for (int n = nbs; n <= nbe; ++n)
  {
    int on                = map_.newtoold[n];
    LogicalLocation& oloc = map_.loclist[on];
    LogicalLocation& nloc = map_.newloc[n];
    MeshBlock* pb         = pmy_mesh_->FindMeshBlock(n);

    if (oloc.level == nloc.level)
    {
      // --- same level ---
      if (map_.ranklist[on] == myrank)
        continue;

      pb->pamr->UnpackSameLevel(pb, recvbuf_[rb_idx]);

      // Orchestrator reads deref_count_ via callback
      if (deref_unpack_fn != nullptr)
      {
        int offset = pb->pamr->TotalSameLevelSize();
        deref_unpack_fn(pb, recvbuf_[rb_idx], offset);
      }

      ++rb_idx;
    }
    else if (oloc.level > nloc.level)
    {
      // --- f2c ---
      for (int l = 0; l < nleaf; ++l)
      {
        if (map_.ranklist[on + l] == myrank)
          continue;

        // Unpack all groups from contiguous buffer
        int offset = 0;
        for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
        {
          // UnpackFineToCoarse reads from recvbuf + offset and internally
          // knows how many Reals to consume for this group.
          pb->pamr->UnpackFineToCoarse(pb,
                                       recvbuf_[rb_idx] + offset,
                                       map_.loclist[on + l],
                                       static_cast<AMRGroup>(g));
          offset +=
            pb->pamr->GetBufferSizes(static_cast<AMRGroup>(g)).fine_to_coarse;
        }

        ++rb_idx;
      }
    }
    else
    {
      // --- c2f ---
      if (map_.ranklist[on] == myrank)
        continue;

      // Unpack all groups from contiguous buffer
      int offset = 0;
      for (int g = 0; g < static_cast<int>(AMRGroup::NumGroups); ++g)
      {
        pb->pamr->UnpackCoarseToFine(
          pb, recvbuf_[rb_idx] + offset, static_cast<AMRGroup>(g));
        offset +=
          pb->pamr->GetBufferSizes(static_cast<AMRGroup>(g)).coarse_to_fine;
      }

      ++rb_idx;
    }
  }
#endif  // MPI_PARALLEL
}

//========================================================================================
// WaitSendsAndCleanup
//========================================================================================
// Wait for all sends to complete and free send/recv buffers + MPI requests.
// The recv buffers are also freed here (they've already been consumed by
// WaitAndUnpack).

void AMRTransfer::WaitSendsAndCleanup()
{
#ifdef MPI_PARALLEL
  if (nsend_ != 0 && sendbuf_ != nullptr)
  {
    MPI_Waitall(nsend_, req_send_, MPI_STATUSES_IGNORE);
    for (int n = 0; n < nsend_; ++n)
      delete[] sendbuf_[n];
    delete[] sendbuf_;
    delete[] req_send_;
    sendbuf_  = nullptr;
    req_send_ = nullptr;
  }
  if (nrecv_ != 0 && recvbuf_ != nullptr)
  {
    for (int n = 0; n < nrecv_; ++n)
      delete[] recvbuf_[n];
    delete[] recvbuf_;
    delete[] req_recv_;
    recvbuf_  = nullptr;
    req_recv_ = nullptr;
  }
#endif  // MPI_PARALLEL
}

}  // namespace comm
