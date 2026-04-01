//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file comm_channel.cpp
//  \brief CommChannel implementation: buffer lifecycle, MPI setup,
//  pack/unpack.

#include "comm_channel.hpp"

#include <algorithm>  // std::max
#include <cstdio>     // std::printf
#include <cstring>    // std::memcpy
#include <sstream>
#include <stdexcept>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/buffer_utils.hpp"
#include "comm_enums.hpp"
#include "comm_registry.hpp"
#include "comm_spec.hpp"
#include "index_utilities.hpp"
#include "neighbor_connectivity.hpp"
#include "parity.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>

namespace
{
void CheckMPIResult(int rc,
                    const char* call_site,
                    int my_rank,
                    int my_gid,
                    int nb_rank,
                    int nb_gid,
                    int bufid,
                    int channel_id)
{
  if (rc == MPI_SUCCESS)
    return;
  int err_class = 0;
  MPI_Error_class(rc, &err_class);
  char err_str[MPI_MAX_ERROR_STRING];
  int err_len = 0;
  MPI_Error_string(rc, err_str, &err_len);
  std::printf(
    "[MPI ERROR] %s failed on rank %d (gid %d)\n"
    "  error_class=%d error_string=\"%.*s\"\n"
    "  nb_rank=%d nb_gid=%d bufid=%d channel=%d\n",
    call_site,
    my_rank,
    my_gid,
    err_class,
    err_len,
    err_str,
    nb_rank,
    nb_gid,
    bufid,
    channel_id);
  MPI_Abort(MPI_COMM_WORLD, rc);
}
}  // anonymous namespace
#endif

namespace comm
{

//----------------------------------------------------------------------------------------
// Tag construction.
// Layout: (lid << (bufid_bits + channel_bits)) | (bufid << channel_bits) |
// channel_id Bit widths are computed from actual counts; validated against
// MPI_TAG_UB.

static int MakeTag(int lid,
                   int bufid,
                   int channel_id,
                   int bufid_bits,
                   int channel_bits)
{
  return (lid << (bufid_bits + channel_bits)) | (bufid << channel_bits) |
         channel_id;
}

// Determine how many bits are needed to represent [0, max_val].
static int BitsNeeded(int max_val)
{
  if (max_val <= 0)
    return 0;
  int bits = 0;
  int v    = max_val;
  while (v > 0)
  {
    v >>= 1;
    ++bits;
  }
  return bits;
}

// Offset added to channel_id for flux correction tags, so ghost-exchange and
// flux-correction tags occupy disjoint channel_id ranges within the same bit
// layout.
static constexpr int kFluxCorrTagChannelOffset = kMaxNeighbor;

//----------------------------------------------------------------------------------------
// Constructor: capture spec, zero-init buffers.

CommChannel::CommChannel(const CommSpec& spec, MeshBlock* pmb, int channel_id)
    : spec_(spec),
      pmy_block_(pmb),
      channel_id_(channel_id),
      finalized_(false),
      nbmax_flcor_(0),
      recv_flx_same_lvl_(true)
{
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    send_buf_[n]       = nullptr;
    recv_buf_[n]       = nullptr;
    target_channel_[n] = nullptr;
    recv_flag_[n].store(BoundaryStatus::waiting, std::memory_order_relaxed);
    send_flag_[n].store(BoundaryStatus::waiting, std::memory_order_relaxed);
    // Flux correction buffers
    flcor_send_buf_[n] = nullptr;
    flcor_recv_buf_[n] = nullptr;
    flcor_recv_flag_[n].store(BoundaryStatus::waiting,
                              std::memory_order_relaxed);
    flcor_send_flag_[n].store(BoundaryStatus::waiting,
                              std::memory_order_relaxed);
#ifdef MPI_PARALLEL
    req_send_[n]       = MPI_REQUEST_NULL;
    req_recv_[n]       = MPI_REQUEST_NULL;
    req_flcor_send_[n] = MPI_REQUEST_NULL;
    req_flcor_recv_[n] = MPI_REQUEST_NULL;
#endif
  }
  for (int e = 0; e < 12; ++e)
  {
    edge_flag_[e]  = true;
    nedge_fine_[e] = 1;
  }
}

//----------------------------------------------------------------------------------------
// Destructor: free buffers and MPI requests.

CommChannel::~CommChannel()
{
  FreeFluxCorrMPIRequests();
  FreeFluxCorrBuffers();
  FreeMPIRequests();
  FreeBuffers();
}

//----------------------------------------------------------------------------------------
// Move constructor.

CommChannel::CommChannel(CommChannel&& other) noexcept
    : spec_(std::move(other.spec_)),
      pmy_block_(other.pmy_block_),
      channel_id_(other.channel_id_),
      finalized_(other.finalized_),
      nbmax_flcor_(other.nbmax_flcor_),
      recv_flx_same_lvl_(other.recv_flx_same_lvl_)
{
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    send_buf_[n]             = other.send_buf_[n];
    recv_buf_[n]             = other.recv_buf_[n];
    target_channel_[n]       = other.target_channel_[n];
    other.send_buf_[n]       = nullptr;
    other.recv_buf_[n]       = nullptr;
    other.target_channel_[n] = nullptr;
    recv_flag_[n].store(other.recv_flag_[n].load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    send_flag_[n].store(other.send_flag_[n].load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    // Flux correction buffers
    flcor_send_buf_[n]       = other.flcor_send_buf_[n];
    flcor_recv_buf_[n]       = other.flcor_recv_buf_[n];
    other.flcor_send_buf_[n] = nullptr;
    other.flcor_recv_buf_[n] = nullptr;
    flcor_recv_flag_[n].store(
      other.flcor_recv_flag_[n].load(std::memory_order_relaxed),
      std::memory_order_relaxed);
    flcor_send_flag_[n].store(
      other.flcor_send_flag_[n].load(std::memory_order_relaxed),
      std::memory_order_relaxed);
#ifdef MPI_PARALLEL
    req_send_[n]             = other.req_send_[n];
    req_recv_[n]             = other.req_recv_[n];
    other.req_send_[n]       = MPI_REQUEST_NULL;
    other.req_recv_[n]       = MPI_REQUEST_NULL;
    req_flcor_send_[n]       = other.req_flcor_send_[n];
    req_flcor_recv_[n]       = other.req_flcor_recv_[n];
    other.req_flcor_send_[n] = MPI_REQUEST_NULL;
    other.req_flcor_recv_[n] = MPI_REQUEST_NULL;
#endif
  }
  for (int e = 0; e < 12; ++e)
  {
    edge_flag_[e]  = other.edge_flag_[e];
    nedge_fine_[e] = other.nedge_fine_[e];
  }
  // Move scratch arrays
  sarea_[0]          = std::move(other.sarea_[0]);
  sarea_[1]          = std::move(other.sarea_[1]);
  other.finalized_   = false;
  other.nbmax_flcor_ = 0;
}

//----------------------------------------------------------------------------------------
// Move assignment.

CommChannel& CommChannel::operator=(CommChannel&& other) noexcept
{
  if (this != &other)
  {
    FreeFluxCorrMPIRequests();
    FreeFluxCorrBuffers();
    FreeMPIRequests();
    FreeBuffers();
    spec_              = std::move(other.spec_);
    pmy_block_         = other.pmy_block_;
    channel_id_        = other.channel_id_;
    finalized_         = other.finalized_;
    nbmax_flcor_       = other.nbmax_flcor_;
    recv_flx_same_lvl_ = other.recv_flx_same_lvl_;
    for (int n = 0; n < kMaxNeighbor; ++n)
    {
      send_buf_[n]             = other.send_buf_[n];
      recv_buf_[n]             = other.recv_buf_[n];
      target_channel_[n]       = other.target_channel_[n];
      other.send_buf_[n]       = nullptr;
      other.recv_buf_[n]       = nullptr;
      other.target_channel_[n] = nullptr;
      recv_flag_[n].store(other.recv_flag_[n].load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
      send_flag_[n].store(other.send_flag_[n].load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
      // Flux correction buffers
      flcor_send_buf_[n]       = other.flcor_send_buf_[n];
      flcor_recv_buf_[n]       = other.flcor_recv_buf_[n];
      other.flcor_send_buf_[n] = nullptr;
      other.flcor_recv_buf_[n] = nullptr;
      flcor_recv_flag_[n].store(
        other.flcor_recv_flag_[n].load(std::memory_order_relaxed),
        std::memory_order_relaxed);
      flcor_send_flag_[n].store(
        other.flcor_send_flag_[n].load(std::memory_order_relaxed),
        std::memory_order_relaxed);
#ifdef MPI_PARALLEL
      req_send_[n]             = other.req_send_[n];
      req_recv_[n]             = other.req_recv_[n];
      other.req_send_[n]       = MPI_REQUEST_NULL;
      other.req_recv_[n]       = MPI_REQUEST_NULL;
      req_flcor_send_[n]       = other.req_flcor_send_[n];
      req_flcor_recv_[n]       = other.req_flcor_recv_[n];
      other.req_flcor_send_[n] = MPI_REQUEST_NULL;
      other.req_flcor_recv_[n] = MPI_REQUEST_NULL;
#endif
    }
    for (int e = 0; e < 12; ++e)
    {
      edge_flag_[e]  = other.edge_flag_[e];
      nedge_fine_[e] = other.nedge_fine_[e];
    }
    sarea_[0]          = std::move(other.sarea_[0]);
    sarea_[1]          = std::move(other.sarea_[1]);
    other.finalized_   = false;
    other.nbmax_flcor_ = 0;
  }
  return *this;
}

//----------------------------------------------------------------------------------------
// Finalize: allocate buffers and create persistent MPI requests.

void CommChannel::Finalize(const NeighborConnectivity& nc, int max_channel_id)
{
  AllocateBuffers(nc);
  SetupPersistentMPI(nc, max_channel_id);

  // Cache target CommChannel pointers for same-rank neighbors so that
  // zero-copy pack (ResolveTargetRecvBuffer) and flag-setting
  // (SetTargetRecvFlag) avoid repeated FindMeshBlock + FindForBlock lookups
  // every communication cycle.
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.snb.rank == Globals::my_rank)
    {
      MeshBlock* ptarget = pmy_block_->pmy_mesh->FindMeshBlock(nb.snb.gid);
      CommRegistry* target_reg  = CommRegistry::FindForBlock(ptarget);
      target_channel_[nb.bufid] = &target_reg->channel(channel_id_);
    }
    // Off-rank slots remain nullptr (from constructor / Reinitialize reset).
  }

  // Pre-compute polar sign array from component_groups.
  // Used during Unpack() for neighbors with nb.polar == true.
  polar_signs_ = ComputeSignArray(spec_, FlipContext::Polar);

  // Pre-compute vertex sharing multiplicity for VC channels.
  // Purely topological - depends only on neighbor list and block position.
  if (spec_.sampling == Sampling::VC)
    node_mult_.Precompute(pmy_block_, nc, spec_.nghost);

  // Set up flux correction infrastructure if this channel participates.
  if (spec_.flcor_mode != FluxCorrMode::None)
  {
    // FC EMF needs per-edge counts for the two-phase averaging protocol.
    if (spec_.flcor_mode == FluxCorrMode::AccumulateAverage)
      CountFineEdges(nc);
    AllocateFluxCorrBuffers(nc);
    SetupFluxCorrMPI(nc, max_channel_id);
    // Flux correction needs scratch arrays for area/edge-length weighting (CC
    // and FC).
    if (spec_.flcor_mode != FluxCorrMode::None)
    {
      int ncells1 = pmy_block_->block_size.nx1 + 2 * NGHOST;
      sarea_[0].NewAthenaArray(ncells1);
      sarea_[1].NewAthenaArray(ncells1);
    }
  }

  finalized_ = true;
}

//----------------------------------------------------------------------------------------
// Reinitialize: tear down and rebuild after regrid.

void CommChannel::Reinitialize(const NeighborConnectivity& nc,
                               int max_channel_id)
{
  FreeFluxCorrMPIRequests();
  FreeFluxCorrBuffers();
  FreeMPIRequests();
  FreeBuffers();
  // Free scratch area arrays before Finalize re-allocates them.
  // NewAthenaArray does not free existing data, so this prevents a leak.
  sarea_[0].DeleteAthenaArray();
  sarea_[1].DeleteAthenaArray();
  // Reset cached target channel pointers (Finalize will repopulate them).
  for (int n = 0; n < kMaxNeighbor; ++n)
    target_channel_[n] = nullptr;
  finalized_ = false;
  Finalize(nc, max_channel_id);
}

//----------------------------------------------------------------------------------------
// Compute buffer size (in Reals) for one neighbor.
// Delegates to idx::ComputeBufferSizeFromRanges(), which computes actual index
// ranges for all communication patterns (same-level, to-coarser, to-finer) and
// takes the max. This works correctly for all samplings (CC, VC, CX) because
// the per-dimension helpers in index_utilities use sampling-specific base
// indices from MeshBlock.

int CommChannel::ComputeBufferSize(const NeighborConnectivity& nc,
                                   int nb_idx) const
{
  const NeighborBlock& nb = nc.neighbor(nb_idx);
  return idx::ComputeBufferSizeFromRanges(
    pmy_block_, nb.ni, spec_.nvar, spec_.nghost, spec_.sampling);
}

//----------------------------------------------------------------------------------------
// Allocate flat send/recv buffers for each neighbor.

void CommChannel::AllocateBuffers(const NeighborConnectivity& nc)
{
  // Use sparse nb.bufid indexing so the receiver's recv_buf_[nb.targetid]
  // and MPI persistent requests (bound to req_send_[nb.bufid]) always
  // reference the correct buffer slot.
  //
  // Zero-copy optimization: same-rank neighbors don't need a send_buf_
  // because PackAndSend packs directly into the target's recv_buf_ via
  // ResolveTargetRecvBuffer().  We skip the allocation to save memory.
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    int size                = ComputeBufferSize(nc, n);
    if (nb.snb.rank != Globals::my_rank)
    {
      send_buf_[nb.bufid] = new Real[size];
    }
    // send_buf_[nb.bufid] remains nullptr for same-rank (initialized in ctor).
    recv_buf_[nb.bufid] = new Real[size];
    recv_flag_[nb.bufid].store(BoundaryStatus::waiting,
                               std::memory_order_relaxed);
    send_flag_[nb.bufid].store(BoundaryStatus::waiting,
                               std::memory_order_relaxed);
  }
}

//----------------------------------------------------------------------------------------
// Free all allocated buffers.

void CommChannel::FreeBuffers()
{
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    delete[] send_buf_[n];
    send_buf_[n] = nullptr;
    delete[] recv_buf_[n];
    recv_buf_[n] = nullptr;
  }
}

//----------------------------------------------------------------------------------------
// Create persistent MPI send/recv requests for each off-rank neighbor.

void CommChannel::SetupPersistentMPI(const NeighborConnectivity& nc,
                                     int max_channel_id)
{
#ifdef MPI_PARALLEL
  MeshBlock* pmb = pmy_block_;

  // Compute tag bit widths from actual counts.
  // channel_bits must cover both ghost-exchange ids [0, max_channel_id] and
  // flux-correction ids [kFluxCorrTagChannelOffset, max_channel_id +
  // kFluxCorrTagChannelOffset] so that all MPI tags share the same bit layout
  // and cannot collide. bufid max is kMaxNeighbor - 1 = 55.
  const int channel_bits =
    BitsNeeded(max_channel_id + kFluxCorrTagChannelOffset);
  const int bufid_bits = BitsNeeded(kMaxNeighbor - 1);

  // Validate that tags fit within MPI implementation limits.
  int* tag_ub_ptr = nullptr;
  int flag        = 0;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &flag);
  const int tag_ub = (flag && tag_ub_ptr != nullptr) ? *tag_ub_ptr : 32767;
  // Use floor(log2(tag_ub)) to get the safe number of total tag bits.
  // BitsNeeded(x) returns ceil(log2(x+1)); subtracting 1 gives floor(log2(x)).
  const int total_bits = BitsNeeded(tag_ub) - 1;
  const int lid_bits   = total_bits - bufid_bits - channel_bits;
  if (lid_bits < 1 || pmb->lid >= (1 << lid_bits))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CommChannel::SetupPersistentMPI\n"
        << "MPI tag space overflow: lid=" << pmb->lid
        << " channel_id=" << channel_id_ << " tag_ub=" << tag_ub << std::endl;
    ATHENA_ERROR(msg);
  }

  // Buffers use sparse nb.bufid indexing; requests already use nb.bufid.
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.snb.rank == Globals::my_rank)
      continue;  // same-rank: no MPI

    // Buffer allocation size is the max over all level cases (same,
    // to-coarser, to-finer).  MPI message sizes may be smaller when the coarse
    // payload can be skipped for same-level neighbors in uniform-resolution
    // regions.
    //
    // Send size: skip coarse payload if the *receiver* has all same-level
    // neighbors
    //   (nb.neighbor_all_same_level) - it will never prolong, never read
    //   coarse data.
    // Recv size: skip coarse payload if *this block* has all same-level
    // neighbors
    //   (pmb->NeighborBlocksSameLevel()) - we will never prolong, never read
    //   coarse data.
    //
    // Cross-level sizes are unaffected (the flag only suppresses the
    // same-level coarse payload; to-coarser/to-finer paths are unchanged).
    int ssize = idx::ComputeMPIBufferSize(pmb,
                                          nb.ni,
                                          spec_.nvar,
                                          spec_.nghost,
                                          spec_.sampling,
                                          nb.neighbor_all_same_level);
    int rsize = idx::ComputeMPIBufferSize(pmb,
                                          nb.ni,
                                          spec_.nvar,
                                          spec_.nghost,
                                          spec_.sampling,
                                          pmb->NeighborBlocksSameLevel());

    // Send tag: receiver's (lid, targetid, channel_id).
    int tag =
      MakeTag(nb.snb.lid, nb.targetid, channel_id_, bufid_bits, channel_bits);
    if (req_send_[nb.bufid] != MPI_REQUEST_NULL)
      MPI_Request_free(&req_send_[nb.bufid]);
    MPI_Send_init(send_buf_[nb.bufid],
                  ssize,
                  MPI_ATHENA_REAL,
                  nb.snb.rank,
                  tag,
                  MPI_COMM_WORLD,
                  &req_send_[nb.bufid]);

    // Recv tag: this block's (lid, bufid, channel_id).
    tag = MakeTag(pmb->lid, nb.bufid, channel_id_, bufid_bits, channel_bits);
    if (req_recv_[nb.bufid] != MPI_REQUEST_NULL)
      MPI_Request_free(&req_recv_[nb.bufid]);
    MPI_Recv_init(recv_buf_[nb.bufid],
                  rsize,
                  MPI_ATHENA_REAL,
                  nb.snb.rank,
                  tag,
                  MPI_COMM_WORLD,
                  &req_recv_[nb.bufid]);
  }
#endif
}

//----------------------------------------------------------------------------------------
// Free all persistent MPI requests.

void CommChannel::FreeMPIRequests()
{
#ifdef MPI_PARALLEL
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    if (req_send_[n] != MPI_REQUEST_NULL)
    {
      MPI_Request_free(&req_send_[n]);
      req_send_[n] = MPI_REQUEST_NULL;
    }
    if (req_recv_[n] != MPI_REQUEST_NULL)
    {
      MPI_Request_free(&req_recv_[n]);
      req_recv_[n] = MPI_REQUEST_NULL;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
// Determine the effective CommTarget for a neighbor based on relative level.

static CommTarget NeighborTarget(int my_level, int nb_level)
{
  if (nb_level == my_level)
    return CommTarget::SameLevel;
  if (nb_level < my_level)
    return CommTarget::ToCoarser;
  return CommTarget::ToFiner;
}

//----------------------------------------------------------------------------------------
// Pack data from state array into send buffers, then initiate sends.
// For each neighbor, select the correct index range based on relative
// refinement level:
//   same-level  -> LoadSameLevel (fine + optional coarse payload)
//   to coarser  -> LoadToCoarser (from coarse buffer)
//   to finer    -> LoadToFiner (from fine array, half-block slab)

void CommChannel::PackAndSend(const NeighborConnectivity& nc,
                              CommTarget target_filter)
{
  MeshBlock* pmb       = pmy_block_;
  const int mylevel    = pmb->loc.level;
  const CommTarget eff = spec_.targets & target_filter;
  const int nvar       = spec_.nvar;
  const int ngh        = spec_.nghost;
  const Sampling samp  = spec_.sampling;

  // --- FC path: 3 face components packed sequentially ---
  if (samp == Sampling::FC)
  {
    AthenaArray<Real>* face[3] = { spec_.var_fc[0],
                                   spec_.var_fc[1],
                                   spec_.var_fc[2] };

    for (int n = 0; n < nc.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc.neighbor(n);
      CommTarget nbt          = NeighborTarget(mylevel, nb.snb.level);
      if (!HasTarget(eff, nbt))
        continue;

      // Zero-copy: for same-rank neighbors, pack directly into the target's
      // recv buffer, bypassing send_buf_ and the subsequent memcpy entirely.
      const bool same_rank = (nb.snb.rank == Globals::my_rank);
      Real* buf = same_rank ? ResolveTargetRecvBuffer(n) : send_buf_[nb.bufid];
      int p     = 0;

      if (nb.snb.level == mylevel)
      {
        // Same-level: pack 3 fine components
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D r =
            idx::LoadSameLevel(pmb, nb.ni, ngh, false, samp, sa);
          BufferUtility::PackData(
            *face[sa], buf, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
        }
        // Multilevel: also pack 3 coarse components.
        // Skip when the receiver has all same-level neighbors - it will never
        // prolong, so the coarse payload would go unread.
        if (pmb->pmy_mesh->multilevel && spec_.coarse_fc[0] != nullptr &&
            !nb.neighbor_all_same_level)
        {
          AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                          spec_.coarse_fc[1],
                                          spec_.coarse_fc[2] };
          for (int sa = 0; sa < 3; ++sa)
          {
            idx::IndexRange3D cr =
              idx::LoadSameLevel(pmb, nb.ni, ngh, true, samp, sa);
            BufferUtility::PackData(
              *cface[sa], buf, cr.si, cr.ei, cr.sj, cr.ej, cr.sk, cr.ek, p);
          }
        }
      }
      else if (nb.snb.level < mylevel)
      {
        // To coarser: pack 3 components from coarse buffer (pre-restricted)
        AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                        spec_.coarse_fc[1],
                                        spec_.coarse_fc[2] };
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D r = idx::LoadToCoarser(pmb, nb.ni, ngh, samp, sa);
          BufferUtility::PackData(
            *cface[sa], buf, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
        }
      }
      else
      {
        // To finer: pack 3 components from fine array (half-block slab)
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D r = idx::LoadToFiner(pmb, nb.ni, ngh, samp, sa);
          BufferUtility::PackData(
            *face[sa], buf, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
        }
      }

      // Send dispatch
      if (same_rank)
      {
        SetTargetRecvFlag(n);
      }
      else
      {
#ifdef MPI_PARALLEL
        int mpi_rc = MPI_Start(&req_send_[nb.bufid]);
        CheckMPIResult(mpi_rc,
                       "MPI_Start(SendBoundary)",
                       Globals::my_rank,
                       pmy_block_->gid,
                       nb.snb.rank,
                       nb.snb.gid,
                       nb.bufid,
                       channel_id_);
#endif
      }
      send_flag_[nb.bufid].store(BoundaryStatus::completed,
                                 std::memory_order_release);
    }
    return;
  }

  // --- CC/VC/CX path: single 4D array ---
  AthenaArray<Real>& var = *spec_.var;

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    CommTarget nbt          = NeighborTarget(mylevel, nb.snb.level);
    if (!HasTarget(eff, nbt))
      continue;

    // Zero-copy: for same-rank neighbors, pack directly into the target's
    // recv buffer, bypassing send_buf_ and the subsequent memcpy entirely.
    const bool same_rank = (nb.snb.rank == Globals::my_rank);
    Real* buf = same_rank ? ResolveTargetRecvBuffer(n) : send_buf_[nb.bufid];
    int p     = 0;

    if (nb.snb.level == mylevel)
    {
      // Same-level: pack fine interior slab.
      idx::IndexRange3D r = idx::LoadSameLevel(pmb, nb.ni, ngh, false, samp);
      BufferUtility::PackData(
        var, buf, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);

      // If multilevel, also pack the pre-restricted coarse payload so the
      // receiver can fill its coarse_buf ghost zones for prolongation.
      // Skip when the receiver has all same-level neighbors - it will never
      // prolong, so the coarse payload would go unread.
      if (pmb->pmy_mesh->multilevel && spec_.coarse_var != nullptr &&
          !nb.neighbor_all_same_level)
      {
        AthenaArray<Real>& cvar = *spec_.coarse_var;
        idx::IndexRange3D cr = idx::LoadSameLevel(pmb, nb.ni, ngh, true, samp);
        BufferUtility::PackData(
          cvar, buf, 0, nvar - 1, cr.si, cr.ei, cr.sj, cr.ej, cr.sk, cr.ek, p);
      }
    }
    else if (nb.snb.level < mylevel)
    {
      // To coarser: pack restricted data from the coarse buffer.
      AthenaArray<Real>& cvar = *spec_.coarse_var;
      idx::IndexRange3D r     = idx::LoadToCoarser(pmb, nb.ni, ngh, samp);
      BufferUtility::PackData(
        cvar, buf, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
    }
    else
    {
      // To finer: pack from fine array (half-block slab selected by fi1/fi2).
      idx::IndexRange3D r = idx::LoadToFiner(pmb, nb.ni, ngh, samp);
      BufferUtility::PackData(
        var, buf, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
    }

    // Initiate send: same-rank packs directly, off-rank uses persistent MPI.
    if (same_rank)
    {
      SetTargetRecvFlag(n);
    }
    else
    {
#ifdef MPI_PARALLEL
      int mpi_rc = MPI_Start(&req_send_[nb.bufid]);
      CheckMPIResult(mpi_rc,
                     "MPI_Start(SendBoundary)",
                     Globals::my_rank,
                     pmy_block_->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
#endif
    }
    send_flag_[nb.bufid].store(BoundaryStatus::completed,
                               std::memory_order_release);
  }
}

//----------------------------------------------------------------------------------------
// Poll for received data.  Returns true when all expected receives have
// arrived.

bool CommChannel::PollReceive(const NeighborConnectivity& nc,
                              CommTarget target_filter)
{
  MeshBlock* pmb       = pmy_block_;
  const int mylevel    = pmb->loc.level;
  const CommTarget eff = spec_.targets & target_filter;

  bool all_arrived = true;
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    CommTarget nbt          = NeighborTarget(mylevel, nb.snb.level);
    if (!HasTarget(eff, nbt))
      continue;

    // Already arrived?
    if (recv_flag_[nb.bufid].load(std::memory_order_acquire) ==
        BoundaryStatus::arrived)
      continue;

#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank)
    {
      int test_flag = 0;
      MPI_Status mpi_status;
      int mpi_rc = MPI_Test(&req_recv_[nb.bufid], &test_flag, &mpi_status);
      CheckMPIResult(mpi_rc,
                     "MPI_Test(PollReceive)",
                     Globals::my_rank,
                     pmy_block_->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
      if (test_flag)
        recv_flag_[nb.bufid].store(BoundaryStatus::arrived,
                                   std::memory_order_release);
    }
#endif

    if (recv_flag_[nb.bufid].load(std::memory_order_acquire) !=
        BoundaryStatus::arrived)
      all_arrived = false;
  }

  return all_arrived;
}

//----------------------------------------------------------------------------------------
// Unpack received data into the state array.
// For each neighbor, select the correct destination based on relative
// refinement level:
//   same-level  -> SetSameLevel (fine ghost zones + optional coarse
//   payload) from coarser-> SetFromCoarser (into coarse buffer for later
//   prolongation) from finer  -> SetFromFiner (restricted data into fine
//   ghost zones)
//
// When nb.polar is true, the j-index loop is reversed and per-component
// parity sign flips are applied - matching the old system's polar boundary
// handling exactly.
//
// VC additive unpack: vertex-centered data uses UnpackDataAdd for
// same-level (fine + coarse) and from-finer cases, because shared vertices
// accumulate contributions from multiple neighbors.  From-coarser uses
// plain UnpackData (overwrites coarse buffer). The caller
// (CommRegistry::SetBoundaries) handles ZeroGhosts before and
// ApplyDivision after, using the precomputed NodeMultiplicity.

// Helper: unpack buffer into array with reversed j-order and sign flips.
// Mirrors the old SetBoundarySameLevel polar branch.
static void UnpackPolar(Real* buf,
                        AthenaArray<Real>& var,
                        int nvar,
                        const std::vector<Real>& signs,
                        int si,
                        int ei,
                        int sj,
                        int ej,
                        int sk,
                        int ek,
                        int& p)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int nn = 0; nn < nvar; ++nn)
  {
    const Real sign = signs[nn];
    for (int k = sk; k <= ek; ++k)
    {
      for (int j = ej; j >= sj; --j)
      {  // reversed j
#pragma omp simd
        for (int i = si; i <= ei; ++i)
        {
          var(nn, k, j, i) = sign * buf[p + i - si];
        }
        p += ni;
      }
    }
  }
}

// Helper: unpack buffer into a 3D array with reversed j-order and sign
// flip. Used by FC polar unpack (one face component at a time).
static void UnpackPolarFC(Real* buf,
                          AthenaArray<Real>& arr,
                          Real sign,
                          int si,
                          int ei,
                          int sj,
                          int ej,
                          int sk,
                          int ek,
                          int& p)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int k = sk; k <= ek; ++k)
  {
    for (int j = ej; j >= sj; --j)
    {  // reversed j
#pragma omp simd
      for (int i = si; i <= ei; ++i)
      {
        arr(k, j, i) = sign * buf[p + i - si];
      }
      p += ni;
    }
  }
}

// Helper: copy the extra face in degenerate dimensions after FC unpack.
// For x2f when nx2==1: x2f(sk, sj+1, i) = x2f(sk, sj, i).
// For x3f when nx3==1: x3f(sk+1, j, i) = x3f(sk, j, i).
static void DegenerateCopyX2f(AthenaArray<Real>& x2f,
                              int si,
                              int ei,
                              int sj,
                              int sk)
{
#pragma omp simd
  for (int i = si; i <= ei; ++i)
    x2f(sk, sj + 1, i) = x2f(sk, sj, i);
}

static void DegenerateCopyX3f(AthenaArray<Real>& x3f,
                              int si,
                              int ei,
                              int sj,
                              int ej,
                              int sk)
{
  for (int j = sj; j <= ej; ++j)
  {
#pragma omp simd
    for (int i = si; i <= ei; ++i)
      x3f(sk + 1, j, i) = x3f(sk, j, i);
  }
}

void CommChannel::Unpack(const NeighborConnectivity& nc,
                         CommTarget target_filter)
{
  MeshBlock* pmb       = pmy_block_;
  const int mylevel    = pmb->loc.level;
  const CommTarget eff = spec_.targets & target_filter;
  const int nvar       = spec_.nvar;
  const int ngh        = spec_.nghost;
  const Sampling samp  = spec_.sampling;

  // --- FC path: 3 face components unpacked sequentially ---
  if (samp == Sampling::FC)
  {
    AthenaArray<Real>* face[3] = { spec_.var_fc[0],
                                   spec_.var_fc[1],
                                   spec_.var_fc[2] };

    // Per-component polar signs: B1 no flip, B2+B3 flip across pole.
    const Real fc_polar_sign[3] = { 1.0, -1.0, -1.0 };

    const bool nx2_degen = (pmb->block_size.nx2 == 1);
    const bool nx3_degen = (pmb->block_size.nx3 == 1);

    for (int n = 0; n < nc.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc.neighbor(n);
      CommTarget nbt          = NeighborTarget(mylevel, nb.snb.level);
      if (!HasTarget(eff, nbt))
        continue;

      int p = 0;

      if (nb.snb.level == mylevel)
      {
        // Same-level: unpack 3 fine face components into ghost zones.
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D r =
            idx::SetSameLevel(pmb, nb.ni, ngh, 1, samp, sa);
          if (nb.polar)
          {
            UnpackPolarFC(recv_buf_[nb.bufid],
                          *face[sa],
                          fc_polar_sign[sa],
                          r.si,
                          r.ei,
                          r.sj,
                          r.ej,
                          r.sk,
                          r.ek,
                          p);
          }
          else
          {
            switch (spec_.unpack_mode)
            {
              case UnpackMode::Min:
                BufferUtility::UnpackDataMin(recv_buf_[nb.bufid],
                                             *face[sa],
                                             r.si,
                                             r.ei,
                                             r.sj,
                                             r.ej,
                                             r.sk,
                                             r.ek,
                                             p);
                break;
              case UnpackMode::Max:
                BufferUtility::UnpackDataMax(recv_buf_[nb.bufid],
                                             *face[sa],
                                             r.si,
                                             r.ei,
                                             r.sj,
                                             r.ej,
                                             r.sk,
                                             r.ek,
                                             p);
                break;
              case UnpackMode::Average:
                BufferUtility::UnpackDataAvg(recv_buf_[nb.bufid],
                                             *face[sa],
                                             r.si,
                                             r.ei,
                                             r.sj,
                                             r.ej,
                                             r.sk,
                                             r.ek,
                                             p);
                break;
              default:  // UnpackMode::Default
                BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                          *face[sa],
                                          r.si,
                                          r.ei,
                                          r.sj,
                                          r.ej,
                                          r.sk,
                                          r.ek,
                                          p);
                break;
            }
          }
          // Degenerate dimension copies - always applied (even polar).
          if (sa == 1 && nx2_degen)
            DegenerateCopyX2f(*face[1], r.si, r.ei, r.sj, r.sk);
          if (sa == 2 && nx3_degen)
            DegenerateCopyX3f(*face[2], r.si, r.ei, r.sj, r.ej, r.sk);
        }

        // Multilevel: unpack 3 coarse face components into coarse_fc.
        // Coarse payload is NOT j-reversed (no polar reversal).
        // Skip when this block has all same-level neighbors - it will
        // never prolong, so coarse ghost data is not needed.
        if (pmb->pmy_mesh->multilevel && spec_.coarse_fc[0] != nullptr &&
            !pmb->NeighborBlocksSameLevel())
        {
          AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                          spec_.coarse_fc[1],
                                          spec_.coarse_fc[2] };
          for (int sa = 0; sa < 3; ++sa)
          {
            idx::IndexRange3D cr =
              idx::SetSameLevel(pmb, nb.ni, ngh, 2, samp, sa);
            BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                      *cface[sa],
                                      cr.si,
                                      cr.ei,
                                      cr.sj,
                                      cr.ej,
                                      cr.sk,
                                      cr.ek,
                                      p);
            if (sa == 1 && nx2_degen)
              DegenerateCopyX2f(*cface[1], cr.si, cr.ei, cr.sj, cr.sk);
            if (sa == 2 && nx3_degen)
              DegenerateCopyX3f(*cface[2], cr.si, cr.ei, cr.sj, cr.ej, cr.sk);
          }
        }
      }
      else if (nb.snb.level < mylevel)
      {
        // From coarser: unpack into coarse buffer for later prolongation.
        AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                        spec_.coarse_fc[1],
                                        spec_.coarse_fc[2] };
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D r = idx::SetFromCoarser(pmb, nb.ni, ngh, samp, sa);
          if (nb.polar)
          {
            UnpackPolarFC(recv_buf_[nb.bufid],
                          *cface[sa],
                          fc_polar_sign[sa],
                          r.si,
                          r.ei,
                          r.sj,
                          r.ej,
                          r.sk,
                          r.ek,
                          p);
          }
          else
          {
            BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                      *cface[sa],
                                      r.si,
                                      r.ei,
                                      r.sj,
                                      r.ej,
                                      r.sk,
                                      r.ek,
                                      p);
            // Degenerate copies only in non-polar branch (matching old
            // behavior).
            if (sa == 1 && nx2_degen)
              DegenerateCopyX2f(*cface[1], r.si, r.ei, r.sj, r.sk);
            if (sa == 2 && nx3_degen)
              DegenerateCopyX3f(*cface[2], r.si, r.ei, r.sj, r.ej, r.sk);
          }
        }
      }
      else
      {
        // From finer: unpack restricted data into fine face arrays.
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D r = idx::SetFromFiner(pmb, nb.ni, ngh, samp, sa);
          if (nb.polar)
          {
            UnpackPolarFC(recv_buf_[nb.bufid],
                          *face[sa],
                          fc_polar_sign[sa],
                          r.si,
                          r.ei,
                          r.sj,
                          r.ej,
                          r.sk,
                          r.ek,
                          p);
          }
          else
          {
            BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                      *face[sa],
                                      r.si,
                                      r.ei,
                                      r.sj,
                                      r.ej,
                                      r.sk,
                                      r.ek,
                                      p);
          }
          // Degenerate copies - always applied (even polar).
          if (sa == 1 && nx2_degen)
            DegenerateCopyX2f(*face[1], r.si, r.ei, r.sj, r.sk);
          if (sa == 2 && nx3_degen)
            DegenerateCopyX3f(*face[2], r.si, r.ei, r.sj, r.ej, r.sk);
        }
      }

      recv_flag_[nb.bufid].store(BoundaryStatus::completed,
                                 std::memory_order_release);
    }
    return;
  }

  // --- CC/VC/CX path: single 4D array ---
  // VC uses additive unpack for shared vertices (same-level + from-finer).
  const bool additive = (samp == Sampling::VC);

  AthenaArray<Real>& var = *spec_.var;

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    CommTarget nbt          = NeighborTarget(mylevel, nb.snb.level);
    if (!HasTarget(eff, nbt))
      continue;

    int p = 0;

    if (nb.snb.level == mylevel)
    {
      // Same-level: unpack fine data into ghost zones.
      idx::IndexRange3D r = idx::SetSameLevel(pmb, nb.ni, ngh, 1, samp);
      if (nb.polar)
      {
        // TODO: polar additive unpack for VC - not needed yet (Z4c VC has
        // no polar). When needed, add a sign-flipped UnpackDataAdd variant
        // here.
        UnpackPolar(recv_buf_[nb.bufid],
                    var,
                    nvar,
                    polar_signs_,
                    r.si,
                    r.ei,
                    r.sj,
                    r.ej,
                    r.sk,
                    r.ek,
                    p);
      }
      else if (additive)
      {
        BufferUtility::UnpackDataAdd(recv_buf_[nb.bufid],
                                     var,
                                     0,
                                     nvar - 1,
                                     r.si,
                                     r.ei,
                                     r.sj,
                                     r.ej,
                                     r.sk,
                                     r.ek,
                                     p);
      }
      else
      {
        BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                  var,
                                  0,
                                  nvar - 1,
                                  r.si,
                                  r.ei,
                                  r.sj,
                                  r.ej,
                                  r.sk,
                                  r.ek,
                                  p);
      }

      // If multilevel, unpack coarse payload into coarse_buf ghost zones.
      // Coarse payload is NOT j-reversed - the old system does not apply
      // polar reversal to the coarse payload in SetBoundarySameLevel. Skip
      // when this block has all same-level neighbors - it will never
      // prolong, so coarse ghost data is not needed.
      if (pmb->pmy_mesh->multilevel && spec_.coarse_var != nullptr &&
          !pmb->NeighborBlocksSameLevel())
      {
        AthenaArray<Real>& cvar = *spec_.coarse_var;
        idx::IndexRange3D cr    = idx::SetSameLevel(pmb, nb.ni, ngh, 2, samp);
        if (additive)
        {
          BufferUtility::UnpackDataAdd(recv_buf_[nb.bufid],
                                       cvar,
                                       0,
                                       nvar - 1,
                                       cr.si,
                                       cr.ei,
                                       cr.sj,
                                       cr.ej,
                                       cr.sk,
                                       cr.ek,
                                       p);
        }
        else
        {
          BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                    cvar,
                                    0,
                                    nvar - 1,
                                    cr.si,
                                    cr.ei,
                                    cr.sj,
                                    cr.ej,
                                    cr.sk,
                                    cr.ek,
                                    p);
        }
      }
    }
    else if (nb.snb.level < mylevel)
    {
      // From coarser: unpack into coarse buffer for later prolongation.
      // VC from-coarser is NOT additive (plain overwrite into coarse
      // buffer).
      AthenaArray<Real>& cvar = *spec_.coarse_var;
      idx::IndexRange3D r     = idx::SetFromCoarser(pmb, nb.ni, ngh, samp);
      if (nb.polar)
      {
        UnpackPolar(recv_buf_[nb.bufid],
                    cvar,
                    nvar,
                    polar_signs_,
                    r.si,
                    r.ei,
                    r.sj,
                    r.ej,
                    r.sk,
                    r.ek,
                    p);
      }
      else
      {
        BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                  cvar,
                                  0,
                                  nvar - 1,
                                  r.si,
                                  r.ei,
                                  r.sj,
                                  r.ej,
                                  r.sk,
                                  r.ek,
                                  p);
      }
    }
    else
    {
      // From finer: unpack restricted data into fine ghost zones.
      idx::IndexRange3D r = idx::SetFromFiner(pmb, nb.ni, ngh, samp);
      if (nb.polar)
      {
        // TODO: polar additive unpack for VC - not needed yet (Z4c VC has
        // no polar). When needed, add a sign-flipped UnpackDataAdd variant
        // here.
        UnpackPolar(recv_buf_[nb.bufid],
                    var,
                    nvar,
                    polar_signs_,
                    r.si,
                    r.ei,
                    r.sj,
                    r.ej,
                    r.sk,
                    r.ek,
                    p);
      }
      else if (additive)
      {
        BufferUtility::UnpackDataAdd(recv_buf_[nb.bufid],
                                     var,
                                     0,
                                     nvar - 1,
                                     r.si,
                                     r.ei,
                                     r.sj,
                                     r.ej,
                                     r.sk,
                                     r.ek,
                                     p);
      }
      else
      {
        BufferUtility::UnpackData(recv_buf_[nb.bufid],
                                  var,
                                  0,
                                  nvar - 1,
                                  r.si,
                                  r.ei,
                                  r.sj,
                                  r.ej,
                                  r.sk,
                                  r.ek,
                                  p);
      }
    }

    recv_flag_[nb.bufid].store(BoundaryStatus::completed,
                               std::memory_order_release);
  }
}

//----------------------------------------------------------------------------------------
// Wait on outstanding sends and reset all status flags.

bool CommChannel::Clear(const NeighborConnectivity& nc,
                        CommTarget target_filter,
                        bool wait)
{
  const int mylevel    = pmy_block_->loc.level;
  const CommTarget eff = spec_.targets & target_filter;

  // When wait=false (LTS non-blocking path), first check that all off-rank
  // sends have completed before touching any flags.  This avoids resetting
  // recv_flag/send_flag prematurely - a defensive measure that keeps the
  // code safe even if future DAG changes allow overlapping Clear with a
  // new communication cycle.  When wait=true (GTS blocking path), MPI_Wait
  // guarantees completion so we can reset flags in the same pass.
#ifdef MPI_PARALLEL
  if (!wait)
  {
    for (int n = 0; n < nc.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc.neighbor(n);
      CommTarget nbt          = NeighborTarget(mylevel, nb.snb.level);
      if (!HasTarget(eff, nbt))
        continue;
      if (nb.snb.rank != Globals::my_rank)
      {
        int flag;
        MPI_Status mpi_status;
        int mpi_rc = MPI_Test(&req_send_[nb.bufid], &flag, &mpi_status);
        CheckMPIResult(mpi_rc,
                       "MPI_Test(ClearBoundary)",
                       Globals::my_rank,
                       pmy_block_->gid,
                       nb.snb.rank,
                       nb.snb.gid,
                       nb.bufid,
                       channel_id_);
        if (!flag)
          return false;  // at least one send still pending - retry later
      }
    }
    // All sends confirmed complete - fall through to reset flags below.
  }
#endif

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    CommTarget nbt          = NeighborTarget(mylevel, nb.snb.level);
    if (!HasTarget(eff, nbt))
      continue;

    recv_flag_[nb.bufid].store(BoundaryStatus::waiting,
                               std::memory_order_relaxed);
    send_flag_[nb.bufid].store(BoundaryStatus::waiting,
                               std::memory_order_relaxed);

#ifdef MPI_PARALLEL
    if (wait && nb.snb.rank != Globals::my_rank)
    {
      MPI_Status mpi_status;
      int mpi_rc = MPI_Wait(&req_send_[nb.bufid], &mpi_status);
      CheckMPIResult(mpi_rc,
                     "MPI_Wait(ClearBoundary)",
                     Globals::my_rank,
                     pmy_block_->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
    }
#endif
  }
  return true;
}

//----------------------------------------------------------------------------------------
// Post persistent receive requests before the send/recv cycle.

void CommChannel::StartReceiving(const NeighborConnectivity& nc,
                                 CommTarget target_filter)
{
#ifdef MPI_PARALLEL
  const int mylevel    = pmy_block_->loc.level;
  const CommTarget eff = spec_.targets & target_filter;

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.snb.rank == Globals::my_rank)
      continue;

    CommTarget nbt = NeighborTarget(mylevel, nb.snb.level);
    if (!HasTarget(eff, nbt))
      continue;

    int mpi_rc = MPI_Start(&req_recv_[nb.bufid]);
    CheckMPIResult(mpi_rc,
                   "MPI_Start(StartReceive)",
                   Globals::my_rank,
                   pmy_block_->gid,
                   nb.snb.rank,
                   nb.snb.gid,
                   nb.bufid,
                   channel_id_);
  }
#endif
}

//----------------------------------------------------------------------------------------
// Same-rank direct copy: find the target block's CommChannel and memcpy
// into its recv buf. This avoids MPI overhead for intra-rank
// communication.  The target channel is found via
// CommRegistry::FindForBlock(), which uses a static map of all registries.

// --- Original CopyBufferSameProcess (commented out for reference) ---
// Superseded by zero-copy optimization: the sender now packs directly into
// the target's recv buffer via ResolveTargetRecvBuffer(), eliminating the
// intermediate send_buf copy + memcpy.
//
// void CommChannel::CopyBufferSameProcess(int nb_idx, int send_size) {
//   const NeighborBlock &nb = pmy_block_->nc().neighbor(nb_idx);
//
//   MeshBlock *ptarget = pmy_block_->pmy_mesh->FindMeshBlock(nb.snb.gid);
//   if (ptarget == nullptr) {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in CommChannel::CopyBufferSameProcess\n"
//         << "Target MeshBlock gid=" << nb.snb.gid << " not found on this
//         rank."
//         << std::endl;
//     ATHENA_ERROR(msg);
//   }
//
//   CommRegistry *target_reg = CommRegistry::FindForBlock(ptarget);
//   if (target_reg == nullptr) {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in CommChannel::CopyBufferSameProcess\n"
//         << "No CommRegistry found for target MeshBlock gid=" <<
//         nb.snb.gid
//         << std::endl;
//     ATHENA_ERROR(msg);
//   }
//
//   CommChannel &target_ch = target_reg->channel(channel_id_);
//   std::memcpy(target_ch.recv_buf_[nb.targetid], send_buf_[nb.bufid],
//               send_size * sizeof(Real));
//   target_ch.recv_flag_[nb.targetid].store(BoundaryStatus::arrived,
//                                           std::memory_order_release);
// }

//----------------------------------------------------------------------------------------
// Zero-copy helpers for same-rank communication.
//
// Instead of packing into send_buf_ and then memcpy-ing into the target's
// recv_buf_, the sender resolves the target's recv buffer pointer up front
// and packs directly into it.  This eliminates two of the three copies in
// the old same-rank path (pack->send_buf + memcpy->recv_buf), leaving only
// the single pack->recv_buf copy.

Real* CommChannel::ResolveTargetRecvBuffer(int nb_idx)
{
  const NeighborBlock& nb = pmy_block_->nc().neighbor(nb_idx);
  CommChannel* tgt        = target_channel_[nb.bufid];
  return tgt->recv_buf_[nb.targetid];
}

void CommChannel::SetTargetRecvFlag(int nb_idx)
{
  const NeighborBlock& nb = pmy_block_->nc().neighbor(nb_idx);
  CommChannel* tgt        = target_channel_[nb.bufid];
  tgt->recv_flag_[nb.targetid].store(BoundaryStatus::arrived,
                                     std::memory_order_release);
}

Real* CommChannel::ResolveTargetFluxCorrRecvBuffer(int nb_idx)
{
  const NeighborBlock& nb = pmy_block_->nc().neighbor(nb_idx);
  CommChannel* tgt        = target_channel_[nb.bufid];
  return tgt->flcor_recv_buf_[nb.targetid];
}

void CommChannel::SetTargetFluxCorrRecvFlag(int nb_idx)
{
  const NeighborBlock& nb = pmy_block_->nc().neighbor(nb_idx);
  CommChannel* tgt        = target_channel_[nb.bufid];
  tgt->flcor_recv_flag_[nb.targetid].store(BoundaryStatus::arrived,
                                           std::memory_order_release);
}

//========================================================================================
// Flux correction private helpers
//========================================================================================

//----------------------------------------------------------------------------------------
// Compute flux correction buffer size for one neighbor.
// Dispatches to the appropriate index_utilities function based on
// flcor_mode.

int CommChannel::ComputeFluxCorrBufferSize(const NeighborConnectivity& nc,
                                           int nb_idx) const
{
  const NeighborBlock& nb = nc.neighbor(nb_idx);
  if (spec_.flcor_mode == FluxCorrMode::OverwriteFromFiner)
    return idx::ComputeFluxCorrBufferSizeCC(pmy_block_, nb.ni, spec_.nvar);
  if (spec_.flcor_mode == FluxCorrMode::AccumulateAverage)
    return idx::ComputeFluxCorrBufferSizeFC(pmy_block_, nb.ni);
  return 0;
}

//----------------------------------------------------------------------------------------
// Populate edge_flag_[12] and nedge_fine_[12] by scanning the
// neighbor-level table. Determines whether each of the 12 edges has a
// finer neighbor touching it, and how many blocks contribute at the finest
// level.  Only meaningful for FC EMF (AccumulateAverage). Ported from
// FaceCenteredBoundaryVariable::CountFineEdges (bvals_fc.cpp:790-853).

void CommChannel::CountFineEdges(const NeighborConnectivity& nc)
{
  const int mylevel = pmy_block_->loc.level;
  int eid           = 0;

  // x1x2 edges (eid 0-3): exist when nx2 > 1
  if (pmy_block_->block_size.nx2 > 1)
  {
    for (int ox2 = -1; ox2 <= 1; ox2 += 2)
    {
      for (int ox1 = -1; ox1 <= 1; ox1 += 2)
      {
        int nis = std::max(ox1 - 1, -1), nie = std::min(ox1 + 1, 1);
        int njs = std::max(ox2 - 1, -1), nje = std::min(ox2 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nj = njs; nj <= nje; ++nj)
        {
          for (int ni = nis; ni <= nie; ++ni)
          {
            int lev = nc.neighbor_level(ni, nj, 0);
            if (lev > fl)
            {
              fl = lev;
              nf = 0;
            }
            if (lev == fl)
              ++nf;
          }
        }
        edge_flag_[eid]  = (fl == mylevel);
        nedge_fine_[eid] = nf;
        ++eid;
      }
    }
  }

  // x1x3 edges (eid 4-7): exist when nx3 > 1
  if (pmy_block_->block_size.nx3 > 1)
  {
    for (int ox3 = -1; ox3 <= 1; ox3 += 2)
    {
      for (int ox1 = -1; ox1 <= 1; ox1 += 2)
      {
        int nis = std::max(ox1 - 1, -1), nie = std::min(ox1 + 1, 1);
        int nks = std::max(ox3 - 1, -1), nke = std::min(ox3 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nk = nks; nk <= nke; ++nk)
        {
          for (int ni = nis; ni <= nie; ++ni)
          {
            int lev = nc.neighbor_level(ni, 0, nk);
            if (lev > fl)
            {
              fl = lev;
              nf = 0;
            }
            if (lev == fl)
              ++nf;
          }
        }
        edge_flag_[eid]  = (fl == mylevel);
        nedge_fine_[eid] = nf;
        ++eid;
      }
    }

    // x2x3 edges (eid 8-11): exist when nx3 > 1 (implies nx2 > 1)
    for (int ox3 = -1; ox3 <= 1; ox3 += 2)
    {
      for (int ox2 = -1; ox2 <= 1; ox2 += 2)
      {
        int njs = std::max(ox2 - 1, -1), nje = std::min(ox2 + 1, 1);
        int nks = std::max(ox3 - 1, -1), nke = std::min(ox3 + 1, 1);
        int nf = 0, fl = mylevel;
        for (int nk = nks; nk <= nke; ++nk)
        {
          for (int nj = njs; nj <= nje; ++nj)
          {
            int lev = nc.neighbor_level(0, nj, nk);
            if (lev > fl)
            {
              fl = lev;
              nf = 0;
            }
            if (lev == fl)
              ++nf;
          }
        }
        edge_flag_[eid]  = (fl == mylevel);
        nedge_fine_[eid] = nf;
        ++eid;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// Allocate flux correction send/recv buffers for participating neighbors.
// CC (OverwriteFromFiner): only face neighbors.
// FC (AccumulateAverage): face + edge neighbors (corners excluded).

void CommChannel::AllocateFluxCorrBuffers(const NeighborConnectivity& nc)
{
  // Sparse nb.bufid indexing - same rationale as AllocateBuffers.
  nbmax_flcor_ = 0;
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    int size                = ComputeFluxCorrBufferSize(nc, n);
    if (size <= 0)
      continue;  // corner (FC) or non-face (CC) - no buffer

    // Zero-copy: same-rank neighbors don't need flcor_send_buf_ because
    // PackAndSendFluxCorrCC/FC packs directly into the target's
    // flcor_recv_buf_.
    if (nb.snb.rank != Globals::my_rank)
    {
      flcor_send_buf_[nb.bufid] = new Real[size];
    }
    // flcor_send_buf_[nb.bufid] remains nullptr for same-rank (initialized
    // in ctor).
    flcor_recv_buf_[nb.bufid] = new Real[size];
    flcor_recv_flag_[nb.bufid].store(BoundaryStatus::waiting,
                                     std::memory_order_relaxed);
    flcor_send_flag_[nb.bufid].store(BoundaryStatus::waiting,
                                     std::memory_order_relaxed);
    nbmax_flcor_ = std::max(nbmax_flcor_, nb.bufid + 1);
  }
}

//----------------------------------------------------------------------------------------
// Free all flux correction buffers.

void CommChannel::FreeFluxCorrBuffers()
{
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    delete[] flcor_send_buf_[n];
    flcor_send_buf_[n] = nullptr;
    delete[] flcor_recv_buf_[n];
    flcor_recv_buf_[n] = nullptr;
  }
  nbmax_flcor_ = 0;
}

//----------------------------------------------------------------------------------------
// Create persistent MPI send/recv requests for flux correction buffers.
// The level-based filtering mirrors the old system:
//   CC: send only fine->coarse (nb.level < mylevel), recv only
//   coarse<-finer (nb.level > mylevel) FC: same-level face + same-level
//   edge (with edge_flag_) + fine->coarse + coarse<-finer
// Uses the same tag scheme as ghost exchange, with the flux correction
// channel_id offset by kFluxCorrTagChannelOffset to avoid tag collision
// with ghost exchange on the same channel.

void CommChannel::SetupFluxCorrMPI(const NeighborConnectivity& nc,
                                   int max_channel_id)
{
#ifdef MPI_PARALLEL
  MeshBlock* pmb    = pmy_block_;
  const int mylevel = pmb->loc.level;

  // Same tag bit layout as ghost exchange - both use
  // BitsNeeded(max_channel_id + kFluxCorrTagChannelOffset) for
  // channel_bits, ensuring no tag collisions between ghost-exchange and
  // flux-correction messages. Flux correction channel_ids are shifted by
  // kFluxCorrTagChannelOffset.
  const int effective_max_ch = max_channel_id + kFluxCorrTagChannelOffset;
  const int channel_bits     = BitsNeeded(effective_max_ch);
  const int bufid_bits       = BitsNeeded(kMaxNeighbor - 1);
  const int flcor_ch         = channel_id_ + kFluxCorrTagChannelOffset;

  // Validate tag space (same logic as ghost exchange).
  int* tag_ub_ptr = nullptr;
  int flag        = 0;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &flag);
  const int tag_ub     = (flag && tag_ub_ptr != nullptr) ? *tag_ub_ptr : 32767;
  const int total_bits = BitsNeeded(tag_ub) - 1;
  const int lid_bits   = total_bits - bufid_bits - channel_bits;
  if (lid_bits < 1 || pmb->lid >= (1 << lid_bits))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CommChannel::SetupFluxCorrMPI\n"
        << "MPI tag space overflow: lid=" << pmb->lid
        << " flcor_channel_id=" << flcor_ch << " tag_ub=" << tag_ub
        << std::endl;
    ATHENA_ERROR(msg);
  }

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.snb.rank == Globals::my_rank)
      continue;

    // Only set up MPI for neighbors that have an allocated flux correction
    // buffer.
    if (flcor_send_buf_[nb.bufid] == nullptr &&
        flcor_recv_buf_[nb.bufid] == nullptr)
      continue;

    int buf_size = ComputeFluxCorrBufferSize(nc, n);
    if (buf_size <= 0)
      continue;

    if (spec_.flcor_mode == FluxCorrMode::OverwriteFromFiner)
    {
      // CC: fine block sends to coarser neighbor; coarse block receives
      // from finer.
      if (nb.snb.level < mylevel)
      {
        // This block is finer -> send restricted flux to coarser neighbor.
        int tag =
          MakeTag(nb.snb.lid, nb.targetid, flcor_ch, bufid_bits, channel_bits);
        if (req_flcor_send_[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&req_flcor_send_[nb.bufid]);
        MPI_Send_init(flcor_send_buf_[nb.bufid],
                      buf_size,
                      MPI_ATHENA_REAL,
                      nb.snb.rank,
                      tag,
                      MPI_COMM_WORLD,
                      &req_flcor_send_[nb.bufid]);
      }
      else if (nb.snb.level > mylevel)
      {
        // This block is coarser -> receive restricted flux from finer
        // neighbor.
        int tag =
          MakeTag(pmb->lid, nb.bufid, flcor_ch, bufid_bits, channel_bits);
        if (req_flcor_recv_[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&req_flcor_recv_[nb.bufid]);
        MPI_Recv_init(flcor_recv_buf_[nb.bufid],
                      buf_size,
                      MPI_ATHENA_REAL,
                      nb.snb.rank,
                      tag,
                      MPI_COMM_WORLD,
                      &req_flcor_recv_[nb.bufid]);
      }
      // Same-level: CC flux correction has no same-level exchange.
    }
    else if (spec_.flcor_mode == FluxCorrMode::AccumulateAverage)
    {
      // FC EMF: same-level (face + qualifying edge) + fine->coarse +
      // coarse<-finer.
      if (nb.snb.level == mylevel)
      {
        // Same-level: face neighbors always participate.
        // Edge neighbors participate only if edge_flag_[eid] is true (no
        // finer neighbor at that edge - otherwise the finer block provides
        // the EMF).
        bool participates = (nb.ni.type == NeighborConnect::face);
        if (!participates && nb.ni.type == NeighborConnect::edge)
          participates = edge_flag_[nb.eid];
        if (participates)
        {
          // Send
          int tag = MakeTag(
            nb.snb.lid, nb.targetid, flcor_ch, bufid_bits, channel_bits);
          if (req_flcor_send_[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&req_flcor_send_[nb.bufid]);
          MPI_Send_init(flcor_send_buf_[nb.bufid],
                        buf_size,
                        MPI_ATHENA_REAL,
                        nb.snb.rank,
                        tag,
                        MPI_COMM_WORLD,
                        &req_flcor_send_[nb.bufid]);
          // Recv
          tag =
            MakeTag(pmb->lid, nb.bufid, flcor_ch, bufid_bits, channel_bits);
          if (req_flcor_recv_[nb.bufid] != MPI_REQUEST_NULL)
            MPI_Request_free(&req_flcor_recv_[nb.bufid]);
          MPI_Recv_init(flcor_recv_buf_[nb.bufid],
                        buf_size,
                        MPI_ATHENA_REAL,
                        nb.snb.rank,
                        tag,
                        MPI_COMM_WORLD,
                        &req_flcor_recv_[nb.bufid]);
        }
      }
      else if (nb.snb.level < mylevel)
      {
        // Fine -> coarser: send restricted EMF.
        // The old code computes a smaller f2csize for the restricted
        // buffer. Our ComputeFluxCorrBufferSizeFC returns the max
        // (same-level) size, which is safe for persistent requests
        // (over-allocates for to-coarser).
        int tag =
          MakeTag(nb.snb.lid, nb.targetid, flcor_ch, bufid_bits, channel_bits);
        if (req_flcor_send_[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&req_flcor_send_[nb.bufid]);
        MPI_Send_init(flcor_send_buf_[nb.bufid],
                      buf_size,
                      MPI_ATHENA_REAL,
                      nb.snb.rank,
                      tag,
                      MPI_COMM_WORLD,
                      &req_flcor_send_[nb.bufid]);
      }
      else
      {
        // Coarser <- finer: receive restricted EMF.
        int tag =
          MakeTag(pmb->lid, nb.bufid, flcor_ch, bufid_bits, channel_bits);
        if (req_flcor_recv_[nb.bufid] != MPI_REQUEST_NULL)
          MPI_Request_free(&req_flcor_recv_[nb.bufid]);
        MPI_Recv_init(flcor_recv_buf_[nb.bufid],
                      buf_size,
                      MPI_ATHENA_REAL,
                      nb.snb.rank,
                      tag,
                      MPI_COMM_WORLD,
                      &req_flcor_recv_[nb.bufid]);
      }
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
// Free all persistent MPI requests for flux correction.

void CommChannel::FreeFluxCorrMPIRequests()
{
#ifdef MPI_PARALLEL
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    if (req_flcor_send_[n] != MPI_REQUEST_NULL)
    {
      MPI_Request_free(&req_flcor_send_[n]);
      req_flcor_send_[n] = MPI_REQUEST_NULL;
    }
    if (req_flcor_recv_[n] != MPI_REQUEST_NULL)
    {
      MPI_Request_free(&req_flcor_recv_[n]);
      req_flcor_recv_[n] = MPI_REQUEST_NULL;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
// Same-rank direct copy for flux correction: write into target block's
// flcor recv buffer. Mirrors CopyBufferSameProcess but targets
// flcor_recv_buf_/flcor_recv_flag_.

// --- Original CopyFluxCorrBufferSameProcess (commented out for reference)
// --- Superseded by zero-copy optimization: the sender now packs directly
// into the target's flcor_recv_buf_ via ResolveTargetFluxCorrRecvBuffer().
//
// void CommChannel::CopyFluxCorrBufferSameProcess(int nb_idx, int
// send_size) {
//   const NeighborBlock &nb = pmy_block_->nc().neighbor(nb_idx);
//
//   MeshBlock *ptarget = pmy_block_->pmy_mesh->FindMeshBlock(nb.snb.gid);
//   if (ptarget == nullptr) {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in
//     CommChannel::CopyFluxCorrBufferSameProcess\n"
//         << "Target MeshBlock gid=" << nb.snb.gid << " not found on this
//         rank."
//         << std::endl;
//     ATHENA_ERROR(msg);
//   }
//
//   CommRegistry *target_reg = CommRegistry::FindForBlock(ptarget);
//   if (target_reg == nullptr) {
//     std::stringstream msg;
//     msg << "### FATAL ERROR in
//     CommChannel::CopyFluxCorrBufferSameProcess\n"
//         << "No CommRegistry found for target MeshBlock gid=" <<
//         nb.snb.gid
//         << std::endl;
//     ATHENA_ERROR(msg);
//   }
//
//   CommChannel &target_ch = target_reg->channel(channel_id_);
//   std::memcpy(target_ch.flcor_recv_buf_[nb.targetid],
//   flcor_send_buf_[nb.bufid],
//               send_size * sizeof(Real));
//   target_ch.flcor_recv_flag_[nb.targetid].store(BoundaryStatus::arrived,
//                                                  std::memory_order_release);
// }

//========================================================================================
// Flux correction public methods
//========================================================================================

//----------------------------------------------------------------------------------------
// Pack area-weighted (CC) or edge-length-weighted (FC) restricted fluxes
// into send buffers, then send to the appropriate neighbors. CC
// (OverwriteFromFiner): only face neighbors with level == mylevel-1. FC
// (AccumulateAverage): same-level face + qualifying edge, and to-coarser
// face
// + edge. Polar EMF is deferred.

void CommChannel::PackAndSendFluxCorr(const NeighborConnectivity& nc)
{
  if (spec_.flcor_mode == FluxCorrMode::None)
    return;

  if (spec_.flcor_mode == FluxCorrMode::OverwriteFromFiner)
  {
    PackAndSendFluxCorrCC(nc);
  }
  else
  {
    PackAndSendFluxCorrFC(nc);
  }
}

//----------------------------------------------------------------------------------------
// CC flux correction send: area-weighted restriction of fine fluxes to
// coarser neighbor. Ported from
// CellCenteredBoundaryVariable::SendFluxCorrection
// (flux_correction_cc.cpp).

void CommChannel::PackAndSendFluxCorrCC(const NeighborConnectivity& nc)
{
  MeshBlock* pmb    = pmy_block_;
  Coordinates* pco  = pmb->pcoord;
  const int mylevel = pmb->loc.level;
  const int nvar    = spec_.nvar;

  // Scratch arrays for vectorised face-area calls on x2/x3 faces.
  AthenaArray<Real>& sa0 = sarea_[0];
  AthenaArray<Real>& sa1 = sarea_[1];

  // Flux arrays for each coordinate direction.
  AthenaArray<Real>& x1flux = *spec_.flx_cc[0];
  AthenaArray<Real>& x2flux = *spec_.flx_cc[1];
  AthenaArray<Real>& x3flux = *spec_.flx_cc[2];

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    // CC flux correction is face-only, fine->coarse only.
    if (nb.ni.type != NeighborConnect::face)
      break;
    if (flcor_send_flag_[nb.bufid].load(std::memory_order_relaxed) ==
        BoundaryStatus::completed)
      continue;
    if (nb.snb.level != mylevel - 1)
      continue;

    int p = 0;
    // Zero-copy: for same-rank neighbors, pack directly into the target's
    // flcor_recv_buf_, bypassing flcor_send_buf_ and memcpy entirely.
    const bool same_rank = (nb.snb.rank == Globals::my_rank);
    Real* sbuf           = same_rank ? ResolveTargetFluxCorrRecvBuffer(n)
                                     : flcor_send_buf_[nb.bufid];

    // --- x1 face ---
    if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
    {
      int i = pmb->is + (pmb->ie - pmb->is + 1) * nb.fid;
      if (pmb->block_size.nx3 > 1)
      {  // 3D
        for (int nn = 0; nn < nvar; ++nn)
        {
          for (int k = pmb->ks; k <= pmb->ke; k += 2)
          {
            for (int j = pmb->js; j <= pmb->je; j += 2)
            {
              Real amm   = pco->GetFace1Area(k, j, i);
              Real amp   = pco->GetFace1Area(k, j + 1, i);
              Real apm   = pco->GetFace1Area(k + 1, j, i);
              Real app   = pco->GetFace1Area(k + 1, j + 1, i);
              Real tarea = amm + amp + apm + app;
              sbuf[p++] =
                (x1flux(nn, k, j, i) * amm + x1flux(nn, k, j + 1, i) * amp +
                 x1flux(nn, k + 1, j, i) * apm +
                 x1flux(nn, k + 1, j + 1, i) * app) /
                tarea;
            }
          }
        }
      }
      else if (pmb->block_size.nx2 > 1)
      {  // 2D
        int k = pmb->ks;
        for (int nn = 0; nn < nvar; ++nn)
        {
          for (int j = pmb->js; j <= pmb->je; j += 2)
          {
            Real am    = pco->GetFace1Area(k, j, i);
            Real ap    = pco->GetFace1Area(k, j + 1, i);
            Real tarea = am + ap;
            sbuf[p++] =
              (x1flux(nn, k, j, i) * am + x1flux(nn, k, j + 1, i) * ap) /
              tarea;
          }
        }
      }
      else
      {  // 1D
        int k = pmb->ks, j = pmb->js;
        for (int nn = 0; nn < nvar; ++nn)
          sbuf[p++] = x1flux(nn, k, j, i);
      }

      // --- x2 face ---
    }
    else if (nb.fid == BoundaryFace::inner_x2 ||
             nb.fid == BoundaryFace::outer_x2)
    {
      int j = pmb->js + (pmb->je - pmb->js + 1) * (nb.fid & 1);
      if (pmb->block_size.nx3 > 1)
      {  // 3D
        for (int nn = 0; nn < nvar; ++nn)
        {
          for (int k = pmb->ks; k <= pmb->ke; k += 2)
          {
            pco->Face2Area(k, j, pmb->is, pmb->ie, sa0);
            pco->Face2Area(k + 1, j, pmb->is, pmb->ie, sa1);
            for (int i = pmb->is; i <= pmb->ie; i += 2)
            {
              Real tarea = sa0(i) + sa0(i + 1) + sa1(i) + sa1(i + 1);
              sbuf[p++]  = (x2flux(nn, k, j, i) * sa0(i) +
                           x2flux(nn, k, j, i + 1) * sa0(i + 1) +
                           x2flux(nn, k + 1, j, i) * sa1(i) +
                           x2flux(nn, k + 1, j, i + 1) * sa1(i + 1)) /
                          tarea;
            }
          }
        }
      }
      else if (pmb->block_size.nx2 > 1)
      {  // 2D
        int k = pmb->ks;
        for (int nn = 0; nn < nvar; ++nn)
        {
          pco->Face2Area(0, j, pmb->is, pmb->ie, sa0);
          for (int i = pmb->is; i <= pmb->ie; i += 2)
          {
            Real tarea = sa0(i) + sa0(i + 1);
            sbuf[p++]  = (x2flux(nn, k, j, i) * sa0(i) +
                         x2flux(nn, k, j, i + 1) * sa0(i + 1)) /
                        tarea;
          }
        }
      }

      // --- x3 face ---
    }
    else if (nb.fid == BoundaryFace::inner_x3 ||
             nb.fid == BoundaryFace::outer_x3)
    {
      int k = pmb->ks + (pmb->ke - pmb->ks + 1) * (nb.fid & 1);
      for (int nn = 0; nn < nvar; ++nn)
      {
        for (int j = pmb->js; j <= pmb->je; j += 2)
        {
          pco->Face3Area(k, j, pmb->is, pmb->ie, sa0);
          pco->Face3Area(k, j + 1, pmb->is, pmb->ie, sa1);
          for (int i = pmb->is; i <= pmb->ie; i += 2)
          {
            Real tarea = sa0(i) + sa0(i + 1) + sa1(i) + sa1(i + 1);
            sbuf[p++]  = (x3flux(nn, k, j, i) * sa0(i) +
                         x3flux(nn, k, j, i + 1) * sa0(i + 1) +
                         x3flux(nn, k, j + 1, i) * sa1(i) +
                         x3flux(nn, k, j + 1, i + 1) * sa1(i + 1)) /
                        tarea;
          }
        }
      }
    }

    // Dispatch: same-rank already packed directly, off-rank uses MPI.
    if (same_rank)
    {
      SetTargetFluxCorrRecvFlag(n);
    }
#ifdef MPI_PARALLEL
    else
    {
      int mpi_rc = MPI_Start(&req_flcor_send_[nb.bufid]);
      CheckMPIResult(mpi_rc,
                     "MPI_Start(SendFluxCorrCC)",
                     Globals::my_rank,
                     pmy_block_->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
    }
#endif
    flcor_send_flag_[nb.bufid].store(BoundaryStatus::completed,
                                     std::memory_order_release);
  }
}

//----------------------------------------------------------------------------------------
// FC EMF flux correction send: raw EMFs (same-level) or
// edge-length-weighted restricted EMFs (to-coarser) for face + edge
// neighbors. Ported from FaceCenteredBoundaryVariable::SendFluxCorrection
// (flux_correction_fc.cpp). Polar EMF is deferred - only non-polar
// neighbors are handled here.

void CommChannel::PackAndSendFluxCorrFC(const NeighborConnectivity& nc)
{
  MeshBlock* pmb    = pmy_block_;
  Coordinates* pco  = pmb->pcoord;
  const int mylevel = pmb->loc.level;

  // EMF arrays: e1 = x1e, e2 = x2e, e3 = x3e.
  AthenaArray<Real>& e1 = *spec_.flx_fc[0];
  AthenaArray<Real>& e2 = *spec_.flx_fc[1];
  AthenaArray<Real>& e3 = *spec_.flx_fc[2];

  // Scratch for edge-length-weighted restriction on to-coarser path.
  AthenaArray<Real>& le1 = sarea_[0];
  AthenaArray<Real>& le2 = sarea_[1];

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    // Only face + edge participate; stop at corners.
    if (nb.ni.type != NeighborConnect::face &&
        nb.ni.type != NeighborConnect::edge)
      break;
    if (flcor_send_flag_[nb.bufid].load(std::memory_order_relaxed) ==
        BoundaryStatus::completed)
      continue;

    int p = 0;
    // Zero-copy: for same-rank neighbors, pack directly into the target's
    // flcor_recv_buf_, bypassing flcor_send_buf_ and memcpy entirely.
    const bool same_rank = (nb.snb.rank == Globals::my_rank);
    Real* sbuf           = same_rank ? ResolveTargetFluxCorrRecvBuffer(n)
                                     : flcor_send_buf_[nb.bufid];

    if (nb.snb.level == mylevel)
    {
      // Same-level: face always, edge only if edge_flag_ is true.
      if (nb.ni.type == NeighborConnect::face ||
          (nb.ni.type == NeighborConnect::edge && edge_flag_[nb.eid]))
      {
        p = LoadFluxBoundaryBufferSameLevel(sbuf, nb, e1, e2, e3);
      }
      else
      {
        continue;
      }
    }
    else if (nb.snb.level == mylevel - 1)
    {
      p = LoadFluxBoundaryBufferToCoarser(sbuf, nb, e1, e2, e3, pco, le1, le2);
    }
    else
    {
      continue;
    }

    if (same_rank)
    {
      SetTargetFluxCorrRecvFlag(n);
    }
#ifdef MPI_PARALLEL
    else
    {
      int mpi_rc = MPI_Start(&req_flcor_send_[nb.bufid]);
      CheckMPIResult(mpi_rc,
                     "MPI_Start(SendFluxCorrFC)",
                     Globals::my_rank,
                     pmy_block_->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
    }
#endif
    flcor_send_flag_[nb.bufid].store(BoundaryStatus::completed,
                                     std::memory_order_release);
  }
}

//----------------------------------------------------------------------------------------
// Pack raw EMF values for a same-level face or edge neighbor.
// Returns the number of Reals packed.

int CommChannel::LoadFluxBoundaryBufferSameLevel(Real* buf,
                                                 const NeighborBlock& nb,
                                                 const AthenaArray<Real>& e1,
                                                 const AthenaArray<Real>& e2,
                                                 const AthenaArray<Real>& e3)
{
  MeshBlock* pmb = pmy_block_;
  int p          = 0;

  if (nb.ni.type == NeighborConnect::face)
  {
    if (pmb->block_size.nx3 > 1)
    {  // 3D
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        for (int k = pmb->ks; k <= pmb->ke + 1; ++k)
          for (int j = pmb->js; j <= pmb->je; ++j)
            buf[p++] = e2(k, j, i);
        for (int k = pmb->ks; k <= pmb->ke; ++k)
          for (int j = pmb->js; j <= pmb->je + 1; ++j)
            buf[p++] = e3(k, j, i);
      }
      else if (nb.fid == BoundaryFace::inner_x2 ||
               nb.fid == BoundaryFace::outer_x2)
      {
        int j = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        for (int k = pmb->ks; k <= pmb->ke + 1; ++k)
          for (int i = pmb->is; i <= pmb->ie; ++i)
            buf[p++] = e1(k, j, i);
        for (int k = pmb->ks; k <= pmb->ke; ++k)
          for (int i = pmb->is; i <= pmb->ie + 1; ++i)
            buf[p++] = e3(k, j, i);
      }
      else
      {  // x3
        int k = (nb.fid == BoundaryFace::inner_x3) ? pmb->ks : pmb->ke + 1;
        for (int j = pmb->js; j <= pmb->je + 1; ++j)
          for (int i = pmb->is; i <= pmb->ie; ++i)
            buf[p++] = e1(k, j, i);
        for (int j = pmb->js; j <= pmb->je; ++j)
          for (int i = pmb->is; i <= pmb->ie + 1; ++i)
            buf[p++] = e2(k, j, i);
      }
    }
    else if (pmb->block_size.nx2 > 1)
    {  // 2D
      int k = pmb->ks;
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        for (int j = pmb->js; j <= pmb->je; ++j)
          buf[p++] = e2(k, j, i);
        for (int j = pmb->js; j <= pmb->je + 1; ++j)
          buf[p++] = e3(k, j, i);
      }
      else
      {  // x2
        int j = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        for (int i = pmb->is; i <= pmb->ie; ++i)
          buf[p++] = e1(k, j, i);
        for (int i = pmb->is; i <= pmb->ie + 1; ++i)
          buf[p++] = e3(k, j, i);
      }
    }
    else
    {  // 1D
      int j = pmb->js, k = pmb->ks;
      int i    = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
      buf[p++] = e2(k, j, i);
      buf[p++] = e3(k, j, i);
    }
  }
  else if (nb.ni.type == NeighborConnect::edge)
  {
    if (nb.eid >= 0 && nb.eid < 4)
    {
      // x1x2 edge: pack e3
      int i = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
      int j = ((nb.eid & 2) == 0) ? pmb->js : pmb->je + 1;
      for (int k = pmb->ks; k <= pmb->ke; ++k)
        buf[p++] = e3(k, j, i);
    }
    else if (nb.eid >= 4 && nb.eid < 8)
    {
      // x1x3 edge: pack e2
      int i = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
      int k = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
      for (int j = pmb->js; j <= pmb->je; ++j)
        buf[p++] = e2(k, j, i);
    }
    else
    {  // x2x3 edge: pack e1
      int j = ((nb.eid & 1) == 0) ? pmb->js : pmb->je + 1;
      int k = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
      for (int i = pmb->is; i <= pmb->ie; ++i)
        buf[p++] = e1(k, j, i);
    }
  }
  return p;
}

//----------------------------------------------------------------------------------------
// Pack edge-length-weighted restricted EMFs for a to-coarser face or edge
// neighbor. Returns the number of Reals packed.

int CommChannel::LoadFluxBoundaryBufferToCoarser(Real* buf,
                                                 const NeighborBlock& nb,
                                                 const AthenaArray<Real>& e1,
                                                 const AthenaArray<Real>& e2,
                                                 const AthenaArray<Real>& e3,
                                                 Coordinates* pco,
                                                 AthenaArray<Real>& le1,
                                                 AthenaArray<Real>& le2)
{
  MeshBlock* pmb = pmy_block_;
  int p          = 0;

  if (nb.ni.type == NeighborConnect::face)
  {
    if (pmb->block_size.nx3 > 1)
    {  // 3D
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        // restrict e2 by Edge2Length
        for (int k = pmb->ks; k <= pmb->ke + 1; k += 2)
        {
          for (int j = pmb->js; j <= pmb->je; j += 2)
          {
            Real el1 = pco->GetEdge2Length(k, j, i);
            Real el2 = pco->GetEdge2Length(k, j + 1, i);
            buf[p++] =
              (e2(k, j, i) * el1 + e2(k, j + 1, i) * el2) / (el1 + el2);
          }
        }
        // restrict e3 by Edge3Length (pole handling)
        for (int k = pmb->ks; k <= pmb->ke; k += 2)
        {
          for (int j = pmb->js; j <= pmb->je + 1; j += 2)
          {
            bool pole = pco->IsPole(j);
            Real el1, el2;
            if (!pole)
            {
              el1 = pco->GetEdge3Length(k, j, i);
              el2 = pco->GetEdge3Length(k + 1, j, i);
            }
            else
            {
              el1 = pco->dx3f(k);
              el2 = pco->dx3f(k + 1);
            }
            buf[p++] =
              (e3(k, j, i) * el1 + e3(k + 1, j, i) * el2) / (el1 + el2);
          }
        }
      }
      else if (nb.fid == BoundaryFace::inner_x2 ||
               nb.fid == BoundaryFace::outer_x2)
      {
        int j     = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        bool pole = pco->IsPole(j);
        // restrict e1 by Edge1Length
        for (int k = pmb->ks; k <= pmb->ke + 1; k += 2)
        {
          if (!pole || !GENERAL_RELATIVITY)
          {
            pco->Edge1Length(k, j, pmb->is, pmb->ie, le1);
          }
          else
          {
            for (int i = pmb->is; i <= pmb->ie + 1; i += 2)
            {
              le1(i)     = pco->dx1f(i);
              le1(i + 1) = pco->dx1f(i + 1);
            }
          }
          for (int i = pmb->is; i <= pmb->ie; i += 2)
            buf[p++] = (e1(k, j, i) * le1(i) + e1(k, j, i + 1) * le1(i + 1)) /
                       (le1(i) + le1(i + 1));
        }
        // restrict e3 by Edge3Length
        for (int k = pmb->ks; k <= pmb->ke; k += 2)
        {
          if (!pole)
          {
            pco->Edge3Length(k, j, pmb->is, pmb->ie + 1, le1);
            pco->Edge3Length(k + 1, j, pmb->is, pmb->ie + 1, le2);
          }
          else
          {
            for (int i = pmb->is; i <= pmb->ie + 1; i += 2)
            {
              le1(i) = pco->dx3f(k);
              le2(i) = pco->dx3f(k + 1);
            }
          }
          for (int i = pmb->is; i <= pmb->ie + 1; i += 2)
            buf[p++] = (e3(k, j, i) * le1(i) + e3(k + 1, j, i) * le2(i)) /
                       (le1(i) + le2(i));
        }
      }
      else
      {  // x3
        int k = (nb.fid == BoundaryFace::inner_x3) ? pmb->ks : pmb->ke + 1;
        // restrict e1 by Edge1Length
        for (int j = pmb->js; j <= pmb->je + 1; j += 2)
        {
          bool pole = pco->IsPole(j);
          if (!pole || !GENERAL_RELATIVITY)
          {
            pco->Edge1Length(k, j, pmb->is, pmb->ie, le1);
          }
          else
          {
            for (int i = pmb->is; i <= pmb->ie; i += 2)
            {
              le1(i)     = pco->dx1f(i);
              le1(i + 1) = pco->dx1f(i + 1);
            }
          }
          for (int i = pmb->is; i <= pmb->ie; i += 2)
            buf[p++] = (e1(k, j, i) * le1(i) + e1(k, j, i + 1) * le1(i + 1)) /
                       (le1(i) + le1(i + 1));
        }
        // restrict e2 by Edge2Length
        for (int j = pmb->js; j <= pmb->je; j += 2)
        {
          pco->Edge2Length(k, j, pmb->is, pmb->ie + 1, le1);
          pco->Edge2Length(k, j + 1, pmb->is, pmb->ie + 1, le2);
          for (int i = pmb->is; i <= pmb->ie + 1; i += 2)
            buf[p++] = (e2(k, j, i) * le1(i) + e2(k, j + 1, i) * le2(i)) /
                       (le1(i) + le2(i));
        }
      }
    }
    else if (pmb->block_size.nx2 > 1)
    {  // 2D
      int k = pmb->ks;
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        // restrict e2
        for (int j = pmb->js; j <= pmb->je; j += 2)
        {
          Real el1 = pco->GetEdge2Length(k, j, i);
          Real el2 = pco->GetEdge2Length(k, j + 1, i);
          buf[p++] = (e2(k, j, i) * el1 + e2(k, j + 1, i) * el2) / (el1 + el2);
        }
        // e3: no restriction in the degenerate k-direction
        for (int j = pmb->js; j <= pmb->je + 1; j += 2)
          buf[p++] = e3(k, j, i);
      }
      else
      {  // x2
        int j     = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        bool pole = pco->IsPole(j);
        // restrict e1
        if (!pole || !GENERAL_RELATIVITY)
        {
          pco->Edge1Length(k, j, pmb->is, pmb->ie, le1);
        }
        else
        {
          for (int i = pmb->is; i <= pmb->ie; i += 2)
          {
            le1(i)     = pco->dx1f(i);
            le1(i + 1) = pco->dx1f(i + 1);
          }
        }
        for (int i = pmb->is; i <= pmb->ie; i += 2)
          buf[p++] = (e1(k, j, i) * le1(i) + e1(k, j, i + 1) * le1(i + 1)) /
                     (le1(i) + le1(i + 1));
        // e3: no restriction in the degenerate k-direction
        for (int i = pmb->is; i <= pmb->ie + 1; i += 2)
          buf[p++] = e3(k, j, i);
      }
    }
    else
    {  // 1D
      int j = pmb->js, k = pmb->ks;
      int i    = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
      buf[p++] = e2(k, j, i);
      buf[p++] = e3(k, j, i);
    }
  }
  else if (nb.ni.type == NeighborConnect::edge)
  {
    if (pmb->block_size.nx3 > 1)
    {  // 3D
      if (nb.eid >= 0 && nb.eid < 4)
      {  // x1x2 edge: restrict e3
        int i     = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
        int j     = ((nb.eid & 2) == 0) ? pmb->js : pmb->je + 1;
        bool pole = pco->IsPole(j);
        for (int k = pmb->ks; k <= pmb->ke; k += 2)
        {
          Real el1, el2;
          if (!pole)
          {
            el1 = pco->GetEdge3Length(k, j, i);
            el2 = pco->GetEdge3Length(k + 1, j, i);
          }
          else
          {
            el1 = pco->dx3f(k);
            el2 = pco->dx3f(k + 1);
          }
          buf[p++] = (e3(k, j, i) * el1 + e3(k + 1, j, i) * el2) / (el1 + el2);
        }
      }
      else if (nb.eid >= 4 && nb.eid < 8)
      {  // x1x3 edge: restrict e2
        int i = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
        int k = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
        for (int j = pmb->js; j <= pmb->je; j += 2)
        {
          Real el1 = pco->GetEdge2Length(k, j, i);
          Real el2 = pco->GetEdge2Length(k, j + 1, i);
          buf[p++] = (e2(k, j, i) * el1 + e2(k, j + 1, i) * el2) / (el1 + el2);
        }
      }
      else
      {  // x2x3 edge: restrict e1
        int j     = ((nb.eid & 1) == 0) ? pmb->js : pmb->je + 1;
        int k     = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
        bool pole = pco->IsPole(j);
        if (!pole || !GENERAL_RELATIVITY)
        {
          pco->Edge1Length(k, j, pmb->is, pmb->ie, le1);
        }
        else
        {
          for (int i = pmb->is; i <= pmb->ie; i += 2)
          {
            le1(i)     = pco->dx1f(i);
            le1(i + 1) = pco->dx1f(i + 1);
          }
        }
        for (int i = pmb->is; i <= pmb->ie; i += 2)
          buf[p++] = (e1(k, j, i) * le1(i) + e1(k, j, i + 1) * le1(i + 1)) /
                     (le1(i) + le1(i + 1));
      }
    }
    else if (pmb->block_size.nx2 > 1)
    {  // 2D: only x1x2 edge, pack e3
      int i    = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
      int j    = ((nb.eid & 2) == 0) ? pmb->js : pmb->je + 1;
      buf[p++] = e3(pmb->ks, j, i);
    }
  }
  return p;
}

//----------------------------------------------------------------------------------------
// Poll for received flux correction data.
// CC (OverwriteFromFiner): poll only; unpack is done in UnpackFluxCorr.
// FC (AccumulateAverage): two-phase poll+unpack (same-level -> ClearCoarse
// -> from-finer
//   -> AverageFlux).  Each arrived buffer is unpacked immediately because
//   the two-phase protocol requires same-level unpack to finish before
//   ClearCoarseFluxBoundary.
// Returns true when all expected buffers have been received (and, for FC,
// fully processed).

bool CommChannel::PollReceiveFluxCorr(const NeighborConnectivity& nc)
{
  if (spec_.flcor_mode == FluxCorrMode::None)
    return true;

  if (spec_.flcor_mode == FluxCorrMode::OverwriteFromFiner)
    return PollReceiveFluxCorrCC(nc);

  return PollReceiveFluxCorrFC(nc);
}

//----------------------------------------------------------------------------------------
// CC flux correction poll: check face neighbors with level == mylevel+1.
// Does NOT unpack - that's done in UnpackFluxCorr.

bool CommChannel::PollReceiveFluxCorrCC(const NeighborConnectivity& nc)
{
  MeshBlock* pmb    = pmy_block_;
  const int mylevel = pmb->loc.level;
  bool all_arrived  = true;

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.ni.type != NeighborConnect::face)
      break;
    if (nb.snb.level != mylevel + 1)
      continue;
    if (flcor_recv_flag_[nb.bufid].load(std::memory_order_acquire) ==
        BoundaryStatus::completed)
      continue;

    if (flcor_recv_flag_[nb.bufid].load(std::memory_order_acquire) ==
        BoundaryStatus::arrived)
      continue;  // ready for unpack

    // Still waiting - probe MPI if cross-rank.
    if (nb.snb.rank == Globals::my_rank)
    {
      all_arrived = false;
      continue;
    }
#ifdef MPI_PARALLEL
    if (req_flcor_recv_[nb.bufid] != MPI_REQUEST_NULL)
    {
      int test = 0;
      MPI_Status mpi_status;
      int mpi_rc = MPI_Test(&req_flcor_recv_[nb.bufid], &test, &mpi_status);
      CheckMPIResult(mpi_rc,
                     "MPI_Test(PollRecvFluxCorr)",
                     Globals::my_rank,
                     pmy_block_->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
      if (test)
      {
        flcor_recv_flag_[nb.bufid].store(BoundaryStatus::arrived,
                                         std::memory_order_release);
      }
      else
      {
        all_arrived = false;
      }
    }
    else
    {
      all_arrived = false;
    }
#else
    all_arrived = false;
#endif
  }
  return all_arrived;
}

//----------------------------------------------------------------------------------------
// FC EMF two-phase poll+unpack.
// Phase 1 (recv_flx_same_lvl_ == true): poll same-level face + qualifying
// edge.
//   Unpack each arrived buffer immediately via SetFluxBoundarySameLevel.
//   When all same-level done -> ClearCoarseFluxBoundary -> transition to
//   phase 2.
// Phase 2: poll from-finer face + edge.
//   Unpack each arrived buffer via SetFluxBoundaryFromFiner.
//   When all from-finer done -> AverageFluxBoundary -> return true.

bool CommChannel::PollReceiveFluxCorrFC(const NeighborConnectivity& nc)
{
  MeshBlock* pmb    = pmy_block_;
  const int mylevel = pmb->loc.level;

  AthenaArray<Real>& e1 = *spec_.flx_fc[0];
  AthenaArray<Real>& e2 = *spec_.flx_fc[1];
  AthenaArray<Real>& e3 = *spec_.flx_fc[2];

  // --- Phase 1: same-level ---
  if (recv_flx_same_lvl_)
  {
    bool all_same = true;
    for (int n = 0; n < nc.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc.neighbor(n);
      if (nb.ni.type != NeighborConnect::face &&
          nb.ni.type != NeighborConnect::edge)
        break;
      if (nb.snb.level != mylevel)
        continue;
      // Only face + qualifying edge participate at same level.
      if (nb.ni.type == NeighborConnect::edge && !edge_flag_[nb.eid])
        continue;

      if (flcor_recv_flag_[nb.bufid].load(std::memory_order_acquire) ==
          BoundaryStatus::completed)
        continue;

      if (flcor_recv_flag_[nb.bufid].load(std::memory_order_acquire) ==
          BoundaryStatus::waiting)
      {
        if (nb.snb.rank == Globals::my_rank)
        {
          all_same = false;
          continue;
        }
#ifdef MPI_PARALLEL
        if (req_flcor_recv_[nb.bufid] != MPI_REQUEST_NULL)
        {
          int test = 0;
          MPI_Status mpi_status;
          int mpi_rc =
            MPI_Test(&req_flcor_recv_[nb.bufid], &test, &mpi_status);
          CheckMPIResult(mpi_rc,
                         "MPI_Test(PollRecvFluxCorrFC_SL)",
                         Globals::my_rank,
                         pmy_block_->gid,
                         nb.snb.rank,
                         nb.snb.gid,
                         nb.bufid,
                         channel_id_);
          if (test)
          {
            flcor_recv_flag_[nb.bufid].store(BoundaryStatus::arrived,
                                             std::memory_order_release);
          }
          else
          {
            all_same = false;
            continue;
          }
        }
        else
        {
          all_same = false;
          continue;
        }
#else
        all_same = false;
        continue;
#endif
      }

      // Buffer arrived - unpack immediately.
      SetFluxBoundarySameLevel(flcor_recv_buf_[nb.bufid], nb, e1, e2, e3);
      flcor_recv_flag_[nb.bufid].store(BoundaryStatus::completed,
                                       std::memory_order_release);
    }

    if (!all_same)
      return false;

    // All same-level received and unpacked.  Zero coarse-side EMFs before
    // from-finer.
    if (pmb->pmy_mesh->multilevel)
      ClearCoarseFluxBoundary(nc);
    recv_flx_same_lvl_ = false;
  }

  // --- Phase 2: from-finer ---
  if (!pmb->pmy_mesh->multilevel)
  {
    // Uniform grid: no from-finer phase.  Average and finish.
    AverageFluxBoundary(nc);
    return true;
  }

  bool all_finer = true;
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.ni.type != NeighborConnect::face &&
        nb.ni.type != NeighborConnect::edge)
      break;
    if (nb.snb.level != mylevel + 1)
      continue;

    if (flcor_recv_flag_[nb.bufid].load(std::memory_order_acquire) ==
        BoundaryStatus::completed)
      continue;

    if (flcor_recv_flag_[nb.bufid].load(std::memory_order_acquire) ==
        BoundaryStatus::waiting)
    {
      if (nb.snb.rank == Globals::my_rank)
      {
        all_finer = false;
        continue;
      }
#ifdef MPI_PARALLEL
      if (req_flcor_recv_[nb.bufid] != MPI_REQUEST_NULL)
      {
        int test = 0;
        MPI_Status mpi_status;
        int mpi_rc = MPI_Test(&req_flcor_recv_[nb.bufid], &test, &mpi_status);
        CheckMPIResult(mpi_rc,
                       "MPI_Test(PollRecvFluxCorrFC_FF)",
                       Globals::my_rank,
                       pmy_block_->gid,
                       nb.snb.rank,
                       nb.snb.gid,
                       nb.bufid,
                       channel_id_);
        if (test)
        {
          flcor_recv_flag_[nb.bufid].store(BoundaryStatus::arrived,
                                           std::memory_order_release);
        }
        else
        {
          all_finer = false;
          continue;
        }
      }
      else
      {
        all_finer = false;
        continue;
      }
#else
      all_finer = false;
      continue;
#endif
    }

    // Buffer arrived - unpack immediately.
    SetFluxBoundaryFromFiner(flcor_recv_buf_[nb.bufid], nb, e1, e2, e3);
    flcor_recv_flag_[nb.bufid].store(BoundaryStatus::completed,
                                     std::memory_order_release);
  }

  if (!all_finer)
    return false;

  // Both phases complete - divide accumulated EMFs by contributor count.
  AverageFluxBoundary(nc);
  return true;
}

//----------------------------------------------------------------------------------------
// Unpack received flux correction data.
// CC: overwrites coarse flux at the shared face (called after
// PollReceiveFluxCorr returns true). FC: no-op - FC unpack is done inside
// PollReceiveFluxCorrFC (two-phase protocol requires it).

void CommChannel::UnpackFluxCorr(const NeighborConnectivity& nc)
{
  if (spec_.flcor_mode != FluxCorrMode::OverwriteFromFiner)
    return;

  MeshBlock* pmb    = pmy_block_;
  const int mylevel = pmb->loc.level;

  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.ni.type != NeighborConnect::face)
      break;
    if (nb.snb.level != mylevel + 1)
      continue;
    if (flcor_recv_flag_[nb.bufid].load(std::memory_order_acquire) !=
        BoundaryStatus::arrived)
      continue;

    UnpackFluxCorrCC(flcor_recv_buf_[nb.bufid], nb);
    flcor_recv_flag_[nb.bufid].store(BoundaryStatus::completed,
                                     std::memory_order_release);
  }
}

//----------------------------------------------------------------------------------------
// CC flux correction unpack: overwrite coarse-side flux at the shared
// face. Quadrant selected by fi1/fi2 from the NeighborBlock. Ported from
// CellCenteredBoundaryVariable::ReceiveFluxCorrection
// (flux_correction_cc.cpp).

void CommChannel::UnpackFluxCorrCC(Real* buf, const NeighborBlock& nb)
{
  MeshBlock* pmb = pmy_block_;
  const int nvar = spec_.nvar;
  int p          = 0;

  AthenaArray<Real>& x1flux = *spec_.flx_cc[0];
  AthenaArray<Real>& x2flux = *spec_.flx_cc[1];
  AthenaArray<Real>& x3flux = *spec_.flx_cc[2];

  if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
  {
    int il = pmb->is + (pmb->ie - pmb->is) * nb.fid + nb.fid;
    int jl = pmb->js, ju = pmb->je, kl = pmb->ks, ku = pmb->ke;
    if (nb.ni.fi1 == 0)
      ju -= pmb->block_size.nx2 / 2;
    else
      jl += pmb->block_size.nx2 / 2;
    if (nb.ni.fi2 == 0)
      ku -= pmb->block_size.nx3 / 2;
    else
      kl += pmb->block_size.nx3 / 2;
    for (int nn = 0; nn < nvar; ++nn)
      for (int k = kl; k <= ku; ++k)
        for (int j = jl; j <= ju; ++j)
          x1flux(nn, k, j, il) = buf[p++];
  }
  else if (nb.fid == BoundaryFace::inner_x2 ||
           nb.fid == BoundaryFace::outer_x2)
  {
    int jl = pmb->js + (pmb->je - pmb->js) * (nb.fid & 1) + (nb.fid & 1);
    int il = pmb->is, iu = pmb->ie, kl = pmb->ks, ku = pmb->ke;
    if (nb.ni.fi1 == 0)
      iu -= pmb->block_size.nx1 / 2;
    else
      il += pmb->block_size.nx1 / 2;
    if (nb.ni.fi2 == 0)
      ku -= pmb->block_size.nx3 / 2;
    else
      kl += pmb->block_size.nx3 / 2;
    for (int nn = 0; nn < nvar; ++nn)
      for (int k = kl; k <= ku; ++k)
        for (int i = il; i <= iu; ++i)
          x2flux(nn, k, jl, i) = buf[p++];
  }
  else if (nb.fid == BoundaryFace::inner_x3 ||
           nb.fid == BoundaryFace::outer_x3)
  {
    int kl = pmb->ks + (pmb->ke - pmb->ks) * (nb.fid & 1) + (nb.fid & 1);
    int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je;
    if (nb.ni.fi1 == 0)
      iu -= pmb->block_size.nx1 / 2;
    else
      il += pmb->block_size.nx1 / 2;
    if (nb.ni.fi2 == 0)
      ju -= pmb->block_size.nx2 / 2;
    else
      jl += pmb->block_size.nx2 / 2;
    for (int nn = 0; nn < nvar; ++nn)
      for (int j = jl; j <= ju; ++j)
        for (int i = il; i <= iu; ++i)
          x3flux(nn, kl, j, i) = buf[p++];
  }
}

//----------------------------------------------------------------------------------------
// FC EMF unpack: accumulate (+= ) same-level EMF values.
// Ported from FaceCenteredBoundaryVariable::SetFluxBoundarySameLevel.

void CommChannel::SetFluxBoundarySameLevel(Real* buf,
                                           const NeighborBlock& nb,
                                           AthenaArray<Real>& e1,
                                           AthenaArray<Real>& e2,
                                           AthenaArray<Real>& e3)
{
  MeshBlock* pmb = pmy_block_;
  int p          = 0;

  if (nb.ni.type == NeighborConnect::face)
  {
    if (pmb->block_size.nx3 > 1)
    {  // 3D
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        for (int k = pmb->ks; k <= pmb->ke + 1; ++k)
          for (int j = pmb->js; j <= pmb->je; ++j)
            e2(k, j, i) += buf[p++];
        for (int k = pmb->ks; k <= pmb->ke; ++k)
          for (int j = pmb->js; j <= pmb->je + 1; ++j)
            e3(k, j, i) += buf[p++];
      }
      else if (nb.fid == BoundaryFace::inner_x2 ||
               nb.fid == BoundaryFace::outer_x2)
      {
        int j = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        // Polar sign handling for x2 face.
        Real sign_e1 = (nb.polar && flip_across_pole_field[IB1]) ? -1.0 : 1.0;
        for (int k = pmb->ks; k <= pmb->ke + 1; ++k)
          for (int i = pmb->is; i <= pmb->ie; ++i)
            e1(k, j, i) += sign_e1 * buf[p++];
        Real sign_e3 = (nb.polar && flip_across_pole_field[IB3]) ? -1.0 : 1.0;
        for (int k = pmb->ks; k <= pmb->ke; ++k)
          for (int i = pmb->is; i <= pmb->ie + 1; ++i)
            e3(k, j, i) += sign_e3 * buf[p++];
      }
      else
      {  // x3
        int k = (nb.fid == BoundaryFace::inner_x3) ? pmb->ks : pmb->ke + 1;
        for (int j = pmb->js; j <= pmb->je + 1; ++j)
          for (int i = pmb->is; i <= pmb->ie; ++i)
            e1(k, j, i) += buf[p++];
        for (int j = pmb->js; j <= pmb->je; ++j)
          for (int i = pmb->is; i <= pmb->ie + 1; ++i)
            e2(k, j, i) += buf[p++];
      }
    }
    else if (pmb->block_size.nx2 > 1)
    {  // 2D
      int k = pmb->ks;
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        // 2D: e2 written to both k and k+1 (degenerate k-direction).
        for (int j = pmb->js; j <= pmb->je; ++j)
        {
          e2(k + 1, j, i) += buf[p];
          e2(k, j, i) += buf[p++];
        }
        for (int j = pmb->js; j <= pmb->je + 1; ++j)
          e3(k, j, i) += buf[p++];
      }
      else
      {  // x2
        int j = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        for (int i = pmb->is; i <= pmb->ie; ++i)
        {
          e1(k + 1, j, i) += buf[p];
          e1(k, j, i) += buf[p++];
        }
        for (int i = pmb->is; i <= pmb->ie + 1; ++i)
          e3(k, j, i) += buf[p++];
      }
    }
    else
    {  // 1D
      int j = pmb->js, k = pmb->ks;
      int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
      e2(k + 1, j, i) += buf[p];
      e2(k, j, i) += buf[p++];
      e3(k, j + 1, i) += buf[p];
      e3(k, j, i) += buf[p++];
    }
  }
  else if (nb.ni.type == NeighborConnect::edge)
  {
    if (nb.eid >= 0 && nb.eid < 4)
    {  // x1x2 edge: e3
      int i     = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
      int j     = ((nb.eid & 2) == 0) ? pmb->js : pmb->je + 1;
      Real sign = (nb.polar && flip_across_pole_field[IB3]) ? -1.0 : 1.0;
      for (int k = pmb->ks; k <= pmb->ke; ++k)
        e3(k, j, i) += sign * buf[p++];
    }
    else if (nb.eid >= 4 && nb.eid < 8)
    {  // x1x3 edge: e2
      int i = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
      int k = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
      for (int j = pmb->js; j <= pmb->je; ++j)
        e2(k, j, i) += buf[p++];
    }
    else
    {  // x2x3 edge: e1
      int j     = ((nb.eid & 1) == 0) ? pmb->js : pmb->je + 1;
      int k     = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
      Real sign = (nb.polar && flip_across_pole_field[IB1]) ? -1.0 : 1.0;
      for (int i = pmb->is; i <= pmb->ie; ++i)
        e1(k, j, i) += sign * buf[p++];
    }
  }
}

//----------------------------------------------------------------------------------------
// FC EMF unpack: accumulate from-finer restricted EMFs with fi1/fi2
// quadrant narrowing. Ported from
// FaceCenteredBoundaryVariable::SetFluxBoundaryFromFiner.

void CommChannel::SetFluxBoundaryFromFiner(Real* buf,
                                           const NeighborBlock& nb,
                                           AthenaArray<Real>& e1,
                                           AthenaArray<Real>& e2,
                                           AthenaArray<Real>& e3)
{
  MeshBlock* pmb = pmy_block_;
  int p          = 0;

  if (nb.ni.type == NeighborConnect::face)
  {
    if (pmb->block_size.nx3 > 1)
    {  // 3D
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i  = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        int jl = pmb->js, ju = pmb->je, kl = pmb->ks, ku = pmb->ke;
        if (nb.ni.fi1 == 0)
          ju = pmb->js + pmb->block_size.nx2 / 2 - 1;
        else
          jl = pmb->js + pmb->block_size.nx2 / 2;
        if (nb.ni.fi2 == 0)
          ku = pmb->ks + pmb->block_size.nx3 / 2 - 1;
        else
          kl = pmb->ks + pmb->block_size.nx3 / 2;
        for (int k = kl; k <= ku + 1; ++k)
          for (int j = jl; j <= ju; ++j)
            e2(k, j, i) += buf[p++];
        for (int k = kl; k <= ku; ++k)
          for (int j = jl; j <= ju + 1; ++j)
            e3(k, j, i) += buf[p++];
      }
      else if (nb.fid == BoundaryFace::inner_x2 ||
               nb.fid == BoundaryFace::outer_x2)
      {
        int j  = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        int il = pmb->is, iu = pmb->ie, kl = pmb->ks, ku = pmb->ke;
        if (nb.ni.fi1 == 0)
          iu = pmb->is + pmb->block_size.nx1 / 2 - 1;
        else
          il = pmb->is + pmb->block_size.nx1 / 2;
        if (nb.ni.fi2 == 0)
          ku = pmb->ks + pmb->block_size.nx3 / 2 - 1;
        else
          kl = pmb->ks + pmb->block_size.nx3 / 2;
        Real sign_e1 = (nb.polar && flip_across_pole_field[IB1]) ? -1.0 : 1.0;
        for (int k = kl; k <= ku + 1; ++k)
          for (int i = il; i <= iu; ++i)
            e1(k, j, i) += sign_e1 * buf[p++];
        Real sign_e3 = (nb.polar && flip_across_pole_field[IB3]) ? -1.0 : 1.0;
        for (int k = kl; k <= ku; ++k)
          for (int i = il; i <= iu + 1; ++i)
            e3(k, j, i) += sign_e3 * buf[p++];
      }
      else
      {  // x3
        int k  = (nb.fid == BoundaryFace::inner_x3) ? pmb->ks : pmb->ke + 1;
        int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je;
        if (nb.ni.fi1 == 0)
          iu = pmb->is + pmb->block_size.nx1 / 2 - 1;
        else
          il = pmb->is + pmb->block_size.nx1 / 2;
        if (nb.ni.fi2 == 0)
          ju = pmb->js + pmb->block_size.nx2 / 2 - 1;
        else
          jl = pmb->js + pmb->block_size.nx2 / 2;
        for (int j = jl; j <= ju + 1; ++j)
          for (int i = il; i <= iu; ++i)
            e1(k, j, i) += buf[p++];
        for (int j = jl; j <= ju; ++j)
          for (int i = il; i <= iu + 1; ++i)
            e2(k, j, i) += buf[p++];
      }
    }
    else if (pmb->block_size.nx2 > 1)
    {  // 2D
      int k = pmb->ks;
      if (nb.fid == BoundaryFace::inner_x1 || nb.fid == BoundaryFace::outer_x1)
      {
        int i  = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
        int jl = pmb->js, ju = pmb->je;
        if (nb.ni.fi1 == 0)
          ju = pmb->js + pmb->block_size.nx2 / 2 - 1;
        else
          jl = pmb->js + pmb->block_size.nx2 / 2;
        for (int j = jl; j <= ju; ++j)
        {
          e2(k + 1, j, i) += buf[p];
          e2(k, j, i) += buf[p++];
        }
        for (int j = jl; j <= ju + 1; ++j)
          e3(k, j, i) += buf[p++];
      }
      else
      {  // x2
        int j  = (nb.fid == BoundaryFace::inner_x2) ? pmb->js : pmb->je + 1;
        int il = pmb->is, iu = pmb->ie;
        if (nb.ni.fi1 == 0)
          iu = pmb->is + pmb->block_size.nx1 / 2 - 1;
        else
          il = pmb->is + pmb->block_size.nx1 / 2;
        for (int i = il; i <= iu; ++i)
        {
          e1(k + 1, j, i) += buf[p];
          e1(k, j, i) += buf[p++];
        }
        for (int i = il; i <= iu + 1; ++i)
          e3(k, j, i) += buf[p++];
      }
    }
    else
    {  // 1D
      int j = pmb->js, k = pmb->ks;
      int i = (nb.fid == BoundaryFace::inner_x1) ? pmb->is : pmb->ie + 1;
      e2(k + 1, j, i) += buf[p];
      e2(k, j, i) += buf[p++];
      e3(k, j + 1, i) += buf[p];
      e3(k, j, i) += buf[p++];
    }
  }
  else if (nb.ni.type == NeighborConnect::edge)
  {
    if (pmb->block_size.nx3 > 1)
    {  // 3D
      if (nb.eid >= 0 && nb.eid < 4)
      {  // x1x2 edge: e3
        int i  = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
        int j  = ((nb.eid & 2) == 0) ? pmb->js : pmb->je + 1;
        int kl = pmb->ks, ku = pmb->ke;
        if (nb.ni.fi1 == 0)
          ku = pmb->ks + pmb->block_size.nx3 / 2 - 1;
        else
          kl = pmb->ks + pmb->block_size.nx3 / 2;
        Real sign = (nb.polar && flip_across_pole_field[IB3]) ? -1.0 : 1.0;
        for (int k = kl; k <= ku; ++k)
          e3(k, j, i) += sign * buf[p++];
      }
      else if (nb.eid >= 4 && nb.eid < 8)
      {  // x1x3 edge: e2
        int i  = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
        int k  = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
        int jl = pmb->js, ju = pmb->je;
        if (nb.ni.fi1 == 0)
          ju = pmb->js + pmb->block_size.nx2 / 2 - 1;
        else
          jl = pmb->js + pmb->block_size.nx2 / 2;
        for (int j = jl; j <= ju; ++j)
          e2(k, j, i) += buf[p++];
      }
      else
      {  // x2x3 edge: e1
        int j  = ((nb.eid & 1) == 0) ? pmb->js : pmb->je + 1;
        int k  = ((nb.eid & 2) == 0) ? pmb->ks : pmb->ke + 1;
        int il = pmb->is, iu = pmb->ie;
        if (nb.ni.fi1 == 0)
          iu = pmb->is + pmb->block_size.nx1 / 2 - 1;
        else
          il = pmb->is + pmb->block_size.nx1 / 2;
        Real sign = (nb.polar && flip_across_pole_field[IB1]) ? -1.0 : 1.0;
        for (int i = il; i <= iu; ++i)
          e1(k, j, i) += sign * buf[p++];
      }
    }
    else if (pmb->block_size.nx2 > 1)
    {  // 2D: only x1x2 edge, e3
      int i = ((nb.eid & 1) == 0) ? pmb->is : pmb->ie + 1;
      int j = ((nb.eid & 2) == 0) ? pmb->js : pmb->je + 1;
      e3(pmb->ks, j, i) += buf[p++];
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CommChannel::ClearCoarseFluxBoundary(const
//! NeighborConnectivity &nc)
//  \brief Zero EMFs at faces with finer neighbors and edges where finer
//  blocks contribute.
//
//  Called between the same-level and from-finer unpack phases in the
//  two-phase FC EMF protocol.  After same-level EMFs have been
//  accumulated, this zeros the EMF arrays at interfaces where finer blocks
//  will contribute, so the from-finer restricted EMFs start from a clean
//  slate.

void CommChannel::ClearCoarseFluxBoundary(const NeighborConnectivity& nc)
{
  MeshBlock* pmb        = pmy_block_;
  AthenaArray<Real>& e1 = *(spec_.flx_fc[0]);
  AthenaArray<Real>& e2 = *(spec_.flx_fc[1]);
  AthenaArray<Real>& e3 = *(spec_.flx_fc[2]);

  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;
  const int mylevel = pmb->loc.level;

  // Compute nface (number of active face directions, 2 per dimension)
  const int ndim =
    1 + (pmb->block_size.nx2 > 1 ? 1 : 0) + (pmb->block_size.nx3 > 1 ? 1 : 0);
  const int nface = 2 * ndim;

  // Compute nedge (number of edge types: 4 edges per pair of active
  // dimensions)
  int nedge = 0;
  if (pmb->block_size.nx2 > 1)
    nedge += 4;  // x1x2 edges
  if (pmb->block_size.nx3 > 1)
    nedge += 8;  // x1x3 + x2x3 edges

  // --- faces: zero interior EMFs at faces that have a finer neighbor ---
  for (int n = 0; n < nface; ++n)
  {
    if (n == BoundaryFace::inner_x1 || n == BoundaryFace::outer_x1)
    {
      int i = (n == BoundaryFace::inner_x1) ? is : ie + 1;
      // neighbor_level for x1 faces: inner -> (-1,0,0), outer -> (+1,0,0)
      int ox1 = (n == BoundaryFace::inner_x1) ? -1 : 1;
      int nl  = nc.neighbor_level(ox1, 0, 0);
      if (nl > mylevel)
      {
        if (pmb->block_size.nx3 > 1)
        {  // 3D
          for (int k = ks + 1; k <= ke; ++k)
            for (int j = js; j <= je; ++j)
              e2(k, j, i) = 0.0;
          for (int k = ks; k <= ke; ++k)
            for (int j = js + 1; j <= je; ++j)
              e3(k, j, i) = 0.0;
        }
        else if (pmb->block_size.nx2 > 1)
        {  // 2D
          for (int j = js; j <= je; ++j)
            e2(ks, j, i) = e2(ks + 1, j, i) = 0.0;
          for (int j = js + 1; j <= je; ++j)
            e3(ks, j, i) = 0.0;
        }
        else
        {  // 1D
          e2(ks, js, i) = e2(ks + 1, js, i) = 0.0;
          e3(ks, js, i) = e3(ks, js + 1, i) = 0.0;
        }
      }
    }
    if (n == BoundaryFace::inner_x2 || n == BoundaryFace::outer_x2)
    {
      int j   = (n == BoundaryFace::inner_x2) ? js : je + 1;
      int ox2 = (n == BoundaryFace::inner_x2) ? -1 : 1;
      int nl  = nc.neighbor_level(0, ox2, 0);
      if (nl > mylevel)
      {
        if (pmb->block_size.nx3 > 1)
        {  // 3D
          for (int k = ks + 1; k <= ke; ++k)
            for (int i = is; i <= ie; ++i)
              e1(k, j, i) = 0.0;
          for (int k = ks; k <= ke; ++k)
            for (int i = is + 1; i <= ie; ++i)
              e3(k, j, i) = 0.0;
        }
        else if (pmb->block_size.nx2 > 1)
        {  // 2D
          for (int i = is; i <= ie; ++i)
            e1(ks, j, i) = e1(ks + 1, j, i) = 0.0;
          for (int i = is + 1; i <= ie; ++i)
            e3(ks, j, i) = 0.0;
        }
      }
    }
    if (n == BoundaryFace::inner_x3 || n == BoundaryFace::outer_x3)
    {
      int k   = (n == BoundaryFace::inner_x3) ? ks : ke + 1;
      int ox3 = (n == BoundaryFace::inner_x3) ? -1 : 1;
      int nl  = nc.neighbor_level(0, 0, ox3);
      if (nl > mylevel)
      {
        // x3 faces only exist in 3D
        for (int j = js + 1; j <= je; ++j)
          for (int i = is; i <= ie; ++i)
            e1(k, j, i) = 0.0;
        for (int j = js; j <= je; ++j)
          for (int i = is + 1; i <= ie; ++i)
            e2(k, j, i) = 0.0;
      }
    }
  }

  // --- edges: zero EMFs at edges where finer neighbors contribute ---
  for (int n = 0; n < nedge; ++n)
  {
    // edge_flag_[n] == true means only same-level neighbors touch this
    // edge; skip those - only zero edges where finer blocks will
    // contribute
    if (edge_flag_[n])
      continue;

    if (n >= 0 && n < 4)
    {  // x1x2 edges: e3 component
      int i = ((n & 1) == 0) ? is : ie + 1;
      int j = ((n & 2) == 0) ? js : je + 1;
      for (int k = ks; k <= ke; ++k)
        e3(k, j, i) = 0.0;
    }
    else if (n >= 4 && n < 8)
    {  // x1x3 edges: e2 component
      int i = ((n & 1) == 0) ? is : ie + 1;
      int k = ((n & 2) == 0) ? ks : ke + 1;
      for (int j = js; j <= je; ++j)
        e2(k, j, i) = 0.0;
    }
    else if (n >= 8 && n < 12)
    {  // x2x3 edges: e1 component
      int j = ((n & 1) == 0) ? js : je + 1;
      int k = ((n & 2) == 0) ? ks : ke + 1;
      for (int i = is; i <= ie; ++i)
        e1(k, j, i) = 0.0;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CommChannel::AverageFluxBoundary(const NeighborConnectivity
//! &nc)
//  \brief Divide accumulated EMFs by contributor count after both unpack
//  phases complete.
//
//  At faces with same-level neighbors, each EMF was accumulated from both
//  sides -> /2. At faces with finer neighbors, the overlap seam (midpoint
//  of fine subfaces) was contributed by two fine blocks -> /2. At edges,
//  divide by nedge_fine_[eid] (the number of blocks at the finest level
//  touching that edge).

void CommChannel::AverageFluxBoundary(const NeighborConnectivity& nc)
{
  MeshBlock* pmb        = pmy_block_;
  AthenaArray<Real>& e1 = *(spec_.flx_fc[0]);
  AthenaArray<Real>& e2 = *(spec_.flx_fc[1]);
  AthenaArray<Real>& e3 = *(spec_.flx_fc[2]);

  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;
  const int mylevel = pmb->loc.level;

  const int ndim =
    1 + (pmb->block_size.nx2 > 1 ? 1 : 0) + (pmb->block_size.nx3 > 1 ? 1 : 0);
  const int nface = 2 * ndim;

  int nedge = 0;
  if (pmb->block_size.nx2 > 1)
    nedge += 4;  // x1x2 edges
  if (pmb->block_size.nx3 > 1)
    nedge += 8;  // x1x3 + x2x3 edges

  // --- faces ---
  for (int n = 0; n < nface; ++n)
  {
    // Skip physical (non-block) boundaries - no neighbor contributed EMFs
    // there
    BoundaryFlag bf = nc.boundary_flag(static_cast<BoundaryFace>(n));
    if (bf != BoundaryFlag::block && bf != BoundaryFlag::periodic &&
        bf != BoundaryFlag::polar)
      continue;

    if (n == BoundaryFace::inner_x1 || n == BoundaryFace::outer_x1)
    {
      int i   = (n == BoundaryFace::inner_x1) ? is : ie + 1;
      int ox1 = (n == BoundaryFace::inner_x1) ? -1 : 1;
      int nl  = nc.neighbor_level(ox1, 0, 0);

      if (nl == mylevel)
      {
        // Same-level: both blocks contributed the same EMFs -> divide by 2
        if (pmb->block_size.nx3 > 1)
        {  // 3D
          for (int k = ks + 1; k <= ke; ++k)
            for (int j = js; j <= je; ++j)
              e2(k, j, i) *= 0.5;
          for (int k = ks; k <= ke; ++k)
            for (int j = js + 1; j <= je; ++j)
              e3(k, j, i) *= 0.5;
        }
        else if (pmb->block_size.nx2 > 1)
        {  // 2D
          for (int j = js; j <= je; ++j)
            e2(ks, j, i) *= 0.5, e2(ks + 1, j, i) *= 0.5;
          for (int j = js + 1; j <= je; ++j)
            e3(ks, j, i) *= 0.5;
        }
        else
        {  // 1D
          e2(ks, js, i) *= 0.5;
          e2(ks + 1, js, i) *= 0.5;
          e3(ks, js, i) *= 0.5;
          e3(ks, js + 1, i) *= 0.5;
        }
      }
      else if (nl > mylevel)
      {
        // Finer: the midpoint seam between fine subfaces was
        // double-counted -> /2
        if (pmb->block_size.nx3 > 1)
        {  // 3D
          int k = ks + pmb->block_size.nx3 / 2;
          for (int j = js; j <= je; ++j)
            e2(k, j, i) *= 0.5;
        }
        if (pmb->block_size.nx2 > 1)
        {  // 2D or 3D
          int j = js + pmb->block_size.nx2 / 2;
          for (int k = ks; k <= ke; ++k)
            e3(k, j, i) *= 0.5;
        }
      }
    }

    if (n == BoundaryFace::inner_x2 || n == BoundaryFace::outer_x2)
    {
      int j   = (n == BoundaryFace::inner_x2) ? js : je + 1;
      int ox2 = (n == BoundaryFace::inner_x2) ? -1 : 1;
      int nl  = nc.neighbor_level(0, ox2, 0);

      if (nl == mylevel)
      {
        if (pmb->block_size.nx3 > 1)
        {  // 3D
          for (int k = ks + 1; k <= ke; ++k)
            for (int i = is; i <= ie; ++i)
              e1(k, j, i) *= 0.5;
          for (int k = ks; k <= ke; ++k)
            for (int i = is + 1; i <= ie; ++i)
              e3(k, j, i) *= 0.5;
        }
        else if (pmb->block_size.nx2 > 1)
        {  // 2D
          for (int i = is; i <= ie; ++i)
            e1(ks, j, i) *= 0.5, e1(ks + 1, j, i) *= 0.5;
          for (int i = is + 1; i <= ie; ++i)
            e3(ks, j, i) *= 0.5;
        }
      }
      else if (nl > mylevel)
      {
        if (pmb->block_size.nx3 > 1)
        {  // 3D
          int k = ks + pmb->block_size.nx3 / 2;
          for (int i = is; i <= ie; ++i)
            e1(k, j, i) *= 0.5;
        }
        if (pmb->block_size.nx2 > 1)
        {  // 2D or 3D
          int i = is + pmb->block_size.nx1 / 2;
          for (int k = ks; k <= ke; ++k)
            e3(k, j, i) *= 0.5;
        }
      }
    }

    if (n == BoundaryFace::inner_x3 || n == BoundaryFace::outer_x3)
    {
      int k   = (n == BoundaryFace::inner_x3) ? ks : ke + 1;
      int ox3 = (n == BoundaryFace::inner_x3) ? -1 : 1;
      int nl  = nc.neighbor_level(0, 0, ox3);

      if (nl == mylevel)
      {
        // x3 faces only exist in 3D
        for (int j = js + 1; j <= je; ++j)
          for (int i = is; i <= ie; ++i)
            e1(k, j, i) *= 0.5;
        for (int j = js; j <= je; ++j)
          for (int i = is + 1; i <= ie; ++i)
            e2(k, j, i) *= 0.5;
      }
      else if (nl > mylevel)
      {
        // Finer: overlap seams
        int j_fine = js + pmb->block_size.nx2 / 2;
        for (int i = is; i <= ie; ++i)
          e1(k, j_fine, i) *= 0.5;
        int i_fine = is + pmb->block_size.nx1 / 2;
        for (int j = js; j <= je; ++j)
          e2(k, j, i_fine) *= 0.5;
      }
    }
  }

  // --- edges: divide by number of contributors at the finest level ---
  for (int n = 0; n < nedge; ++n)
  {
    // nedge_fine_==1 means only one contributor; no averaging needed
    if (nedge_fine_[n] == 1)
      continue;
    Real div = 1.0 / static_cast<Real>(nedge_fine_[n]);

    if (n >= 0 && n < 4)
    {  // x1x2 edges: e3
      int i = ((n & 1) == 0) ? is : ie + 1;
      int j = ((n & 2) == 0) ? js : je + 1;
      for (int k = ks; k <= ke; ++k)
        e3(k, j, i) *= div;
    }
    else if (n >= 4 && n < 8)
    {  // x1x3 edges: e2
      int i = ((n & 1) == 0) ? is : ie + 1;
      int k = ((n & 2) == 0) ? ks : ke + 1;
      for (int j = js; j <= je; ++j)
        e2(k, j, i) *= div;
    }
    else if (n >= 8 && n < 12)
    {  // x2x3 edges: e1
      int j = ((n & 1) == 0) ? js : je + 1;
      int k = ((n & 2) == 0) ? ks : ke + 1;
      for (int i = is; i <= ie; ++i)
        e1(k, j, i) *= div;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CommChannel::StartReceivingFluxCorr(const NeighborConnectivity
//! &nc)
//  \brief Post persistent receive requests for flux correction.
//
//  CC (OverwriteFromFiner): post receives from face neighbors with level >
//  mylevel. FC (AccumulateAverage): post receives from same-level face +
//  edge neighbors (where applicable) AND from finer face + edge neighbors.
//  Also sets recv_flx_same_lvl_ = true to begin the same-level phase.

void CommChannel::StartReceivingFluxCorr(const NeighborConnectivity& nc)
{
  if (spec_.flcor_mode == FluxCorrMode::None)
    return;

  MeshBlock* pmb    = pmy_block_;
  const int mylevel = pmb->loc.level;

  if (spec_.flcor_mode == FluxCorrMode::AccumulateAverage)
  {
    // FC: begin in same-level phase
    recv_flx_same_lvl_ = true;
  }

#ifdef MPI_PARALLEL
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (nb.snb.rank == Globals::my_rank)
      continue;

    if (spec_.flcor_mode == FluxCorrMode::OverwriteFromFiner)
    {
      // CC: receive only from finer face neighbors
      if (nb.ni.type == NeighborConnect::face && nb.snb.level > mylevel)
      {
        int mpi_rc = MPI_Start(&req_flcor_recv_[nb.bufid]);
        CheckMPIResult(mpi_rc,
                       "MPI_Start(StartFluxCorrRecv)",
                       Globals::my_rank,
                       pmy_block_->gid,
                       nb.snb.rank,
                       nb.snb.gid,
                       nb.bufid,
                       channel_id_);
      }
    }
    else
    {
      // FC (AccumulateAverage): receive from face+edge neighbors that
      // participate
      if (nb.ni.type == NeighborConnect::face ||
          nb.ni.type == NeighborConnect::edge)
      {
        if (nb.snb.level > mylevel)
        {
          // From finer: always receive
          int mpi_rc = MPI_Start(&req_flcor_recv_[nb.bufid]);
          CheckMPIResult(mpi_rc,
                         "MPI_Start(StartFluxCorrRecv)",
                         Globals::my_rank,
                         pmy_block_->gid,
                         nb.snb.rank,
                         nb.snb.gid,
                         nb.bufid,
                         channel_id_);
        }
        else if (nb.snb.level == mylevel)
        {
          // Same level: face always, edge only if edge_flag_ says
          // same-level-only
          if (nb.ni.type == NeighborConnect::face ||
              (nb.ni.type == NeighborConnect::edge && edge_flag_[nb.eid]))
          {
            int mpi_rc = MPI_Start(&req_flcor_recv_[nb.bufid]);
            CheckMPIResult(mpi_rc,
                           "MPI_Start(StartFluxCorrRecv)",
                           Globals::my_rank,
                           pmy_block_->gid,
                           nb.snb.rank,
                           nb.snb.gid,
                           nb.bufid,
                           channel_id_);
          }
        }
      }
      // Corner neighbors do not participate in flux correction
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn bool CommChannel::ClearFluxCorr(const NeighborConnectivity &nc,
//! bool wait)
//  \brief Test (or wait on) outstanding flux correction sends and reset
//  flags.
//
//  Must be called after the flux correction cycle completes (both send and
//  receive sides finished).  Resets all flcor flags to waiting for the
//  next cycle.  When wait=false, uses MPI_Test instead of MPI_Wait and
//  returns false if any off-rank send is still pending.

bool CommChannel::ClearFluxCorr(const NeighborConnectivity& nc, bool wait)
{
  if (spec_.flcor_mode == FluxCorrMode::None)
    return true;

  MeshBlock* pmb    = pmy_block_;
  const int mylevel = pmb->loc.level;

  // Helper: returns true if this neighbor had a send that needs clearing.
  // CC (OverwriteFromFiner): only fine->coarse face sends.
  // FC (AccumulateAverage): fine->coarse + same-level face + qualifying
  // edges.
  auto NeighborHasSend = [&](const NeighborBlock& nb) -> bool
  {
    if (spec_.flcor_mode == FluxCorrMode::OverwriteFromFiner)
    {
      return nb.ni.type == NeighborConnect::face && nb.snb.level < mylevel;
    }
    else  // AccumulateAverage
    {
      if (nb.ni.type != NeighborConnect::face &&
          nb.ni.type != NeighborConnect::edge)
        return false;
      if (nb.snb.level < mylevel)
        return true;
      if (nb.snb.level == mylevel)
        return nb.ni.type == NeighborConnect::face ||
               (nb.ni.type == NeighborConnect::edge && edge_flag_[nb.eid]);
      return false;
    }
  };

  // Helper: returns true if this neighbor participates in flux correction
  // (both send and recv flags need resetting).
  auto NeighborParticipates = [&](const NeighborBlock& nb) -> bool
  {
    if (spec_.flcor_mode == FluxCorrMode::OverwriteFromFiner)
      return nb.ni.type == NeighborConnect::face;
    else
      return nb.ni.type == NeighborConnect::face ||
             nb.ni.type == NeighborConnect::edge;
  };

  // When wait=false, first check that all off-rank sends have completed
  // before touching any flags.  This avoids premature flag resets that
  // could cause subtle races if the DAG is ever restructured.
#ifdef MPI_PARALLEL
  if (!wait)
  {
    for (int n = 0; n < nc.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc.neighbor(n);
      if (nb.snb.rank == Globals::my_rank)
        continue;
      if (!NeighborHasSend(nb))
        continue;
      int flag;
      MPI_Status mpi_status;
      CheckMPIResult(MPI_Test(&req_flcor_send_[nb.bufid], &flag, &mpi_status),
                     "MPI_Test(ClearFluxCorr)",
                     Globals::my_rank,
                     pmb->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
      if (!flag)
        return false;  // at least one send still pending - retry later
    }
    // All sends confirmed complete - fall through to reset flags below.
  }
#endif

  // Reset all participating flags and (when wait=true) block on sends.
  for (int n = 0; n < nc.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc.neighbor(n);
    if (!NeighborParticipates(nb))
      continue;

    flcor_recv_flag_[nb.bufid].store(BoundaryStatus::waiting,
                                     std::memory_order_relaxed);
    flcor_send_flag_[nb.bufid].store(BoundaryStatus::waiting,
                                     std::memory_order_relaxed);

#ifdef MPI_PARALLEL
    if (wait && nb.snb.rank != Globals::my_rank && NeighborHasSend(nb))
    {
      MPI_Status mpi_status;
      CheckMPIResult(MPI_Wait(&req_flcor_send_[nb.bufid], &mpi_status),
                     "MPI_Wait(ClearFluxCorr)",
                     Globals::my_rank,
                     pmb->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     channel_id_);
    }
#endif
  }

  // Reset same-level phase flag for next cycle.
  if (spec_.flcor_mode == FluxCorrMode::AccumulateAverage)
  {
    recv_flx_same_lvl_ = true;
  }
  return true;
}

#ifdef DBG_FUSED_COMM
//========================================================================================
// Fused-comm helpers: called by CommRegistry for group-level buffer
// fusion. These methods pack/unpack a single neighbor's data into/from an
// external buffer at a given offset, without touching send/recv flags or
// MPI requests.
//========================================================================================

//----------------------------------------------------------------------------------------
//! \fn int CommChannel::PackInto(Real *buf, int offset,
//!                                const NeighborBlock &nb, int mylevel)
//!                                const
//  \brief Pack one neighbor's ghost exchange data into an external buffer.
//
//  Packs the same payload as PackAndSend() for the given neighbor,
//  starting at buf[offset].  Returns the new offset after packing (offset
//  + packed_size). Does NOT set any flags or start any MPI - the caller
//  (CommRegistry) handles that.
//
//  Preconditions:
//    - RestrictInterior has already been called if needed (handled by
//    CommRegistry).
//    - buf has enough space at [offset, ...).

int CommChannel::PackInto(Real* buf,
                          int offset,
                          const NeighborBlock& nb,
                          int mylevel) const
{
  MeshBlock* pmb      = pmy_block_;
  const int nvar      = spec_.nvar;
  const int ngh       = spec_.nghost;
  const Sampling samp = spec_.sampling;
  int p               = offset;

  if (samp == Sampling::FC)
  {
    // --- FC path: 3 face components packed sequentially ---
    AthenaArray<Real>* face[3] = { spec_.var_fc[0],
                                   spec_.var_fc[1],
                                   spec_.var_fc[2] };

    if (nb.snb.level == mylevel)
    {
      for (int sa = 0; sa < 3; ++sa)
      {
        idx::IndexRange3D r =
          idx::LoadSameLevel(pmb, nb.ni, ngh, false, samp, sa);
        BufferUtility::PackData(
          *face[sa], buf, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
      if (pmb->pmy_mesh->multilevel && spec_.coarse_fc[0] != nullptr &&
          !nb.neighbor_all_same_level)
      {
        AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                        spec_.coarse_fc[1],
                                        spec_.coarse_fc[2] };
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D cr =
            idx::LoadSameLevel(pmb, nb.ni, ngh, true, samp, sa);
          BufferUtility::PackData(
            *cface[sa], buf, cr.si, cr.ei, cr.sj, cr.ej, cr.sk, cr.ek, p);
        }
      }
    }
    else if (nb.snb.level < mylevel)
    {
      AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                      spec_.coarse_fc[1],
                                      spec_.coarse_fc[2] };
      for (int sa = 0; sa < 3; ++sa)
      {
        idx::IndexRange3D r = idx::LoadToCoarser(pmb, nb.ni, ngh, samp, sa);
        BufferUtility::PackData(
          *cface[sa], buf, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
    }
    else
    {
      for (int sa = 0; sa < 3; ++sa)
      {
        idx::IndexRange3D r = idx::LoadToFiner(pmb, nb.ni, ngh, samp, sa);
        BufferUtility::PackData(
          *face[sa], buf, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
    }
  }
  else
  {
    // --- CC/VC/CX path: single 4D array ---
    AthenaArray<Real>& var = *spec_.var;

    if (nb.snb.level == mylevel)
    {
      idx::IndexRange3D r = idx::LoadSameLevel(pmb, nb.ni, ngh, false, samp);
      BufferUtility::PackData(
        var, buf, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);

      if (pmb->pmy_mesh->multilevel && spec_.coarse_var != nullptr &&
          !nb.neighbor_all_same_level)
      {
        AthenaArray<Real>& cvar = *spec_.coarse_var;
        idx::IndexRange3D cr = idx::LoadSameLevel(pmb, nb.ni, ngh, true, samp);
        BufferUtility::PackData(
          cvar, buf, 0, nvar - 1, cr.si, cr.ei, cr.sj, cr.ej, cr.sk, cr.ek, p);
      }
    }
    else if (nb.snb.level < mylevel)
    {
      AthenaArray<Real>& cvar = *spec_.coarse_var;
      idx::IndexRange3D r     = idx::LoadToCoarser(pmb, nb.ni, ngh, samp);
      BufferUtility::PackData(
        cvar, buf, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
    }
    else
    {
      idx::IndexRange3D r = idx::LoadToFiner(pmb, nb.ni, ngh, samp);
      BufferUtility::PackData(
        var, buf, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
    }
  }

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CommChannel::UnpackFrom(Real *buf, int offset,
//!                                  const NeighborBlock &nb, int mylevel)
//  \brief Unpack one neighbor's ghost exchange data from an external
//  buffer.
//
//  Mirrors the Unpack() method's per-neighbor logic, reading from
//  buf[offset]. Returns the new offset after unpacking.  Does NOT set any
//  flags - the caller (CommRegistry) handles flag management.
//
//  VC additive unpack, polar reversal, and degenerate-dimension copies are
//  all handled exactly as in Unpack().

int CommChannel::UnpackFrom(Real* buf,
                            int offset,
                            const NeighborBlock& nb,
                            int mylevel)
{
  MeshBlock* pmb      = pmy_block_;
  const int nvar      = spec_.nvar;
  const int ngh       = spec_.nghost;
  const Sampling samp = spec_.sampling;
  int p               = offset;

  if (samp == Sampling::FC)
  {
    // --- FC path ---
    AthenaArray<Real>* face[3]  = { spec_.var_fc[0],
                                    spec_.var_fc[1],
                                    spec_.var_fc[2] };
    const Real fc_polar_sign[3] = { 1.0, -1.0, -1.0 };
    const bool nx2_degen        = (pmb->block_size.nx2 == 1);
    const bool nx3_degen        = (pmb->block_size.nx3 == 1);

    if (nb.snb.level == mylevel)
    {
      for (int sa = 0; sa < 3; ++sa)
      {
        idx::IndexRange3D r = idx::SetSameLevel(pmb, nb.ni, ngh, 1, samp, sa);
        if (nb.polar)
        {
          UnpackPolarFC(buf,
                        *face[sa],
                        fc_polar_sign[sa],
                        r.si,
                        r.ei,
                        r.sj,
                        r.ej,
                        r.sk,
                        r.ek,
                        p);
        }
        else
        {
          BufferUtility::UnpackData(
            buf, *face[sa], r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
        }
        if (sa == 1 && nx2_degen)
          DegenerateCopyX2f(*face[1], r.si, r.ei, r.sj, r.sk);
        if (sa == 2 && nx3_degen)
          DegenerateCopyX3f(*face[2], r.si, r.ei, r.sj, r.ej, r.sk);
      }
      if (pmb->pmy_mesh->multilevel && spec_.coarse_fc[0] != nullptr &&
          !pmb->NeighborBlocksSameLevel())
      {
        AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                        spec_.coarse_fc[1],
                                        spec_.coarse_fc[2] };
        for (int sa = 0; sa < 3; ++sa)
        {
          idx::IndexRange3D cr =
            idx::SetSameLevel(pmb, nb.ni, ngh, 2, samp, sa);
          BufferUtility::UnpackData(
            buf, *cface[sa], cr.si, cr.ei, cr.sj, cr.ej, cr.sk, cr.ek, p);
          if (sa == 1 && nx2_degen)
            DegenerateCopyX2f(*cface[1], cr.si, cr.ei, cr.sj, cr.sk);
          if (sa == 2 && nx3_degen)
            DegenerateCopyX3f(*cface[2], cr.si, cr.ei, cr.sj, cr.ej, cr.sk);
        }
      }
    }
    else if (nb.snb.level < mylevel)
    {
      AthenaArray<Real>* cface[3] = { spec_.coarse_fc[0],
                                      spec_.coarse_fc[1],
                                      spec_.coarse_fc[2] };
      for (int sa = 0; sa < 3; ++sa)
      {
        idx::IndexRange3D r = idx::SetFromCoarser(pmb, nb.ni, ngh, samp, sa);
        if (nb.polar)
        {
          UnpackPolarFC(buf,
                        *cface[sa],
                        fc_polar_sign[sa],
                        r.si,
                        r.ei,
                        r.sj,
                        r.ej,
                        r.sk,
                        r.ek,
                        p);
        }
        else
        {
          BufferUtility::UnpackData(
            buf, *cface[sa], r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
          if (sa == 1 && nx2_degen)
            DegenerateCopyX2f(*cface[1], r.si, r.ei, r.sj, r.sk);
          if (sa == 2 && nx3_degen)
            DegenerateCopyX3f(*cface[2], r.si, r.ei, r.sj, r.ej, r.sk);
        }
      }
    }
    else
    {
      for (int sa = 0; sa < 3; ++sa)
      {
        idx::IndexRange3D r = idx::SetFromFiner(pmb, nb.ni, ngh, samp, sa);
        if (nb.polar)
        {
          UnpackPolarFC(buf,
                        *face[sa],
                        fc_polar_sign[sa],
                        r.si,
                        r.ei,
                        r.sj,
                        r.ej,
                        r.sk,
                        r.ek,
                        p);
        }
        else
        {
          BufferUtility::UnpackData(
            buf, *face[sa], r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
        }
        if (sa == 1 && nx2_degen)
          DegenerateCopyX2f(*face[1], r.si, r.ei, r.sj, r.sk);
        if (sa == 2 && nx3_degen)
          DegenerateCopyX3f(*face[2], r.si, r.ei, r.sj, r.ej, r.sk);
      }
    }
  }
  else
  {
    // --- CC/VC/CX path ---
    const bool additive    = (samp == Sampling::VC);
    AthenaArray<Real>& var = *spec_.var;

    if (nb.snb.level == mylevel)
    {
      idx::IndexRange3D r = idx::SetSameLevel(pmb, nb.ni, ngh, 1, samp);
      if (nb.polar)
      {
        UnpackPolar(
          buf, var, nvar, polar_signs_, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
      else if (additive)
      {
        BufferUtility::UnpackDataAdd(
          buf, var, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
      else
      {
        BufferUtility::UnpackData(
          buf, var, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }

      if (pmb->pmy_mesh->multilevel && spec_.coarse_var != nullptr &&
          !pmb->NeighborBlocksSameLevel())
      {
        AthenaArray<Real>& cvar = *spec_.coarse_var;
        idx::IndexRange3D cr    = idx::SetSameLevel(pmb, nb.ni, ngh, 2, samp);
        if (additive)
        {
          BufferUtility::UnpackDataAdd(buf,
                                       cvar,
                                       0,
                                       nvar - 1,
                                       cr.si,
                                       cr.ei,
                                       cr.sj,
                                       cr.ej,
                                       cr.sk,
                                       cr.ek,
                                       p);
        }
        else
        {
          BufferUtility::UnpackData(buf,
                                    cvar,
                                    0,
                                    nvar - 1,
                                    cr.si,
                                    cr.ei,
                                    cr.sj,
                                    cr.ej,
                                    cr.sk,
                                    cr.ek,
                                    p);
        }
      }
    }
    else if (nb.snb.level < mylevel)
    {
      AthenaArray<Real>& cvar = *spec_.coarse_var;
      idx::IndexRange3D r     = idx::SetFromCoarser(pmb, nb.ni, ngh, samp);
      if (nb.polar)
      {
        UnpackPolar(buf,
                    cvar,
                    nvar,
                    polar_signs_,
                    r.si,
                    r.ei,
                    r.sj,
                    r.ej,
                    r.sk,
                    r.ek,
                    p);
      }
      else
      {
        BufferUtility::UnpackData(
          buf, cvar, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
    }
    else
    {
      idx::IndexRange3D r = idx::SetFromFiner(pmb, nb.ni, ngh, samp);
      if (nb.polar)
      {
        UnpackPolar(
          buf, var, nvar, polar_signs_, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
      else if (additive)
      {
        BufferUtility::UnpackDataAdd(
          buf, var, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
      else
      {
        BufferUtility::UnpackData(
          buf, var, 0, nvar - 1, r.si, r.ei, r.sj, r.ej, r.sk, r.ek, p);
      }
    }
  }

  return p;
}

//----------------------------------------------------------------------------------------
//! \fn int CommChannel::PackSizeForNeighbor(const NeighborBlock &nb,
//!                                          bool skip_coarse) const
//  \brief Compute the actual pack/unpack size (in Reals) for one neighbor.
//
//  When skip_coarse is false, returns the full allocation size (same as
//  ComputeBufferSizeFromRanges).  When skip_coarse is true, omits the
//  coarse payload for same-level neighbors (matching ComputeMPIBufferSize
//  semantics).

int CommChannel::PackSizeForNeighbor(const NeighborBlock& nb,
                                     bool skip_coarse) const
{
  return idx::ComputeMPIBufferSize(
    pmy_block_, nb.ni, spec_.nvar, spec_.nghost, spec_.sampling, skip_coarse);
}
#endif  // DBG_FUSED_COMM

}  // namespace comm
