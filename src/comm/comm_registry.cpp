//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file comm_registry.cpp
//  \brief CommRegistry implementation: registration, finalization, group-level
//  comm.

#include "comm_registry.hpp"

#include <algorithm>  // std::max
#include <cstdio>     // std::printf
#include <sstream>
#include <stdexcept>

#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "index_utilities.hpp"
#include "node_multiplicity.hpp"
#include "physical_bcs.hpp"
#include "refinement_ops.hpp"

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
                    int fused_group)
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
    "  nb_rank=%d nb_gid=%d bufid=%d fused_group=%d\n",
    call_site,
    my_rank,
    my_gid,
    err_class,
    err_len,
    err_str,
    nb_rank,
    nb_gid,
    bufid,
    fused_group);
  MPI_Abort(MPI_COMM_WORLD, rc);
}
}  // anonymous namespace
#endif

namespace comm
{

//----------------------------------------------------------------------------------------
// Static member: global registry map for same-process copy lookup.

std::unordered_map<MeshBlock*, CommRegistry*> CommRegistry::registry_map_;

//----------------------------------------------------------------------------------------
// Constructor.

CommRegistry::CommRegistry(MeshBlock* pmb)
    : pmy_block_(pmb), nc_(), finalized_(false)
{
  // Register this instance so same-process copy can find the target's
  // channels.
  registry_map_[pmb] = this;
}

//----------------------------------------------------------------------------------------
// Destructor.

CommRegistry::~CommRegistry()
{
#ifdef DBG_FUSED_COMM
  for (int g = 0; g < static_cast<int>(CommGroup::NumGroups); ++g)
  {
    if (fused_[g].active)
    {
      FreeFusedMPIRequests(g);
      FreeFusedBuffers(g);
    }
  }
#endif
  registry_map_.erase(pmy_block_);
}

//----------------------------------------------------------------------------------------
// Static lookup: find the CommRegistry for a given MeshBlock.

CommRegistry* CommRegistry::FindForBlock(MeshBlock* pmb)
{
  auto it = registry_map_.find(pmb);
  if (it != registry_map_.end())
    return it->second;
  return nullptr;
}

//----------------------------------------------------------------------------------------
// Register a variable.  Returns the channel_id.

int CommRegistry::Register(const CommSpec& spec)
{
  if (finalized_)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CommRegistry::Register\n"
        << "Cannot register channel '" << spec.label << "' after Finalize()."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // Check for duplicate labels.
  if (label_to_id_.count(spec.label) > 0)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CommRegistry::Register\n"
        << "Duplicate channel label '" << spec.label << "'." << std::endl;
    ATHENA_ERROR(msg);
  }

  int id = static_cast<int>(channels_.size());
  channels_.emplace_back(spec, pmy_block_, id);

  // Record primary group membership (ghost exchange).
  int g = static_cast<int>(spec.group);
  group_channels_[g].push_back(id);

  // Record secondary group membership (flux correction) if specified.
  // A channel with flux_group != NumGroups participates in two groups:
  // its primary group for ghost exchange, and its flux_group for flux
  // correction.
  if (spec.flux_group != CommGroup::NumGroups)
  {
    int fg = static_cast<int>(spec.flux_group);
    group_channels_[fg].push_back(id);
  }

  // Record label lookup.
  label_to_id_[spec.label] = id;

  return id;
}

//----------------------------------------------------------------------------------------
// Add an already-registered channel to an additional group.

void CommRegistry::AddToGroup(int channel_id, CommGroup group)
{
  if (finalized_)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CommRegistry::AddToGroup\n"
        << "Cannot add channel " << channel_id << " to group after Finalize()."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  int g = static_cast<int>(group);
  group_channels_[g].push_back(channel_id);
}

//----------------------------------------------------------------------------------------
// Finalize: build connectivity, allocate buffers, create MPI requests.

void CommRegistry::Finalize()
{
  RebuildConnectivity();
  // All channels must use the same MPI tag bit layout, so pass the max
  // channel_id.
  int max_ch_id = num_channels() > 0 ? num_channels() - 1 : 0;
  for (auto& ch : channels_)
  {
    ch.Finalize(nc_, max_ch_id);
  }
#ifdef DBG_FUSED_COMM
  FinalizeFused();
#endif
  finalized_ = true;
}

//----------------------------------------------------------------------------------------
// Reinitialize after regrid.

void CommRegistry::Reinitialize()
{
  // Must clear finalized_ so that the rebuild path in each channel can
  // re-enter Finalize logic.  Set it back to true at the end so post-regrid
  // comm works.
  finalized_ = false;
#ifdef DBG_FUSED_COMM
  ReinitializeFused();
#endif
  RebuildConnectivity();
  int max_ch_id = num_channels() > 0 ? num_channels() - 1 : 0;
  for (auto& ch : channels_)
  {
    ch.Reinitialize(nc_, max_ch_id);
  }
#ifdef DBG_FUSED_COMM
  FinalizeFused();
#endif
  finalized_ = true;
}

//----------------------------------------------------------------------------------------
// Copy the MeshBlock's NeighborConnectivity into the registry's local
// snapshot. Called before Finalize/Reinitialize so that channel setup sees
// current topology.

void CommRegistry::RebuildConnectivity()
{
  nc_ = pmy_block_->nc();
}

//----------------------------------------------------------------------------------------
// StartReceiving for a group.

void CommRegistry::StartReceiving(CommGroup group, CommTarget target_filter)
{
  int g = static_cast<int>(group);
#ifdef DBG_FUSED_COMM
  if (fused_[g].active)
  {
    StartReceivingFused(group, target_filter);
    return;
  }
#endif
  for (int id : group_channels_[g])
  {
    channels_[id].StartReceiving(nc_, target_filter);
  }
}

//----------------------------------------------------------------------------------------
// Send boundary buffers for a group.
// On a multilevel mesh, restrict the fine interior into the coarse buffer
// first so that PackAndSend has up-to-date coarse data for cross-level
// neighbors.
//
// Restriction can be skipped when this block AND all its same-level neighbors
// have only same-level neighbors (two-level check).  In that case:
//   - This block has no coarser neighbors, so no to-coarser send needs
//   restricted data.
//   - No same-level neighbor has coarser neighbors either, so none of them
//   will read
//     the coarse payload (it is skipped in PackAndSend by the same-level
//     guard).
// This is the same optimization as RestrictNonGhost in the old boundary
// system.

static bool NeedsRestriction(MeshBlock* pmb)
{
  // If this block has any cross-level neighbor, it needs restriction for sure.
  if (!pmb->NeighborBlocksSameLevel())
    return true;
  // Even if all our neighbors are same-level, check whether any of them have
  // a coarser neighbor - if so, they need the coarse payload from us.
  for (int n = 0; n < pmb->nc().num_neighbors(); n++)
  {
    if (!pmb->nc().neighbor(n).neighbor_all_same_level)
      return true;
  }
  return false;
}

void CommRegistry::SendBoundaryBuffers(CommGroup group,
                                       CommTarget target_filter)
{
  int g = static_cast<int>(group);

  // Restrict fine -> coarse interior for each channel that needs it.
  // Only necessary when cross-level neighbors exist (multilevel mesh).
  if (pmy_block_->pmy_mesh->multilevel && NeedsRestriction(pmy_block_))
  {
    for (int id : group_channels_[g])
    {
      RestrictInterior(pmy_block_, channels_[id].spec());
    }
  }

#ifdef DBG_FUSED_COMM
  if (fused_[g].active)
  {
    SendBoundaryBuffersFused(group, target_filter);
    return;
  }
#endif

  for (int id : group_channels_[g])
  {
    channels_[id].PackAndSend(nc_, target_filter);
  }
}

//----------------------------------------------------------------------------------------
// Receive boundary buffers for a group.  Returns true when all channels are
// done.

bool CommRegistry::ReceiveBoundaryBuffers(CommGroup group,
                                          CommTarget target_filter)
{
  int g = static_cast<int>(group);
#ifdef DBG_FUSED_COMM
  if (fused_[g].active)
  {
    return ReceiveBoundaryBuffersFused(group, target_filter);
  }
#endif
  bool all_done = true;
  for (int id : group_channels_[g])
  {
    if (!channels_[id].PollReceive(nc_, target_filter))
      all_done = false;
  }
  return all_done;
}

//----------------------------------------------------------------------------------------
// Set boundaries (unpack) for a group.
// VC channels require special protocol: ZeroGhosts -> Unpack -> ApplyDivision,
// because vertex-centered variables use additive unpack and need multiplicity
// correction.

void CommRegistry::SetBoundaries(CommGroup group, CommTarget target_filter)
{
  int g = static_cast<int>(group);
#ifdef DBG_FUSED_COMM
  if (fused_[g].active)
  {
    SetBoundariesFused(group, target_filter);
    return;
  }
#endif

  // Phase 1: zero VC ghost zones before additive unpack.
  // Note: zeroing and division (phase 3) are NOT filtered by target_filter
  // because the multiplicity counts are precomputed for the full neighbor
  // topology. VC channels should always be called with CommTarget::All.
  for (int id : group_channels_[g])
  {
    CommChannel& ch = channels_[id];
    if (ch.spec().sampling != Sampling::VC)
      continue;

    NodeMultiplicity& nm = ch.node_multiplicity();
    nm.ZeroGhosts(*ch.spec().var, pmy_block_, ch.spec().nvar);
    if (pmy_block_->pmy_mesh->multilevel && ch.spec().coarse_var != nullptr)
      nm.ZeroGhostsCoarse(*ch.spec().coarse_var, pmy_block_, ch.spec().nvar);
  }

  // Phase 2: unpack all channels (VC uses additive unpack internally).
  for (int id : group_channels_[g])
  {
    channels_[id].Unpack(nc_, target_filter);
  }

  // Phase 3: divide VC arrays by accumulated node multiplicity.
  for (int id : group_channels_[g])
  {
    CommChannel& ch = channels_[id];
    if (ch.spec().sampling != Sampling::VC)
      continue;

    const NodeMultiplicity& nm = ch.node_multiplicity();
    nm.ApplyDivision(*ch.spec().var, pmy_block_, ch.spec().nvar);
    if (pmy_block_->pmy_mesh->multilevel && ch.spec().coarse_var != nullptr)
      nm.ApplyDivisionCoarse(
        *ch.spec().coarse_var, pmy_block_, ch.spec().nvar);
  }
}

//----------------------------------------------------------------------------------------
// Clear boundary for a group.

bool CommRegistry::ClearBoundary(CommGroup group,
                                 CommTarget target_filter,
                                 bool wait)
{
  int g = static_cast<int>(group);
#ifdef DBG_FUSED_COMM
  if (fused_[g].active)
  {
    return ClearBoundaryFused(group, target_filter, wait);
  }
#endif
  bool all_clear = true;
  for (int id : group_channels_[g])
  {
    if (!channels_[id].Clear(nc_, target_filter, wait))
      all_clear = false;
  }
  return all_clear;
}

//----------------------------------------------------------------------------------------
// Find channel by label.

int CommRegistry::FindChannelByLabel(const std::string& label) const
{
  auto it = label_to_id_.find(label);
  if (it != label_to_id_.end())
    return it->second;
  return -1;
}

//----------------------------------------------------------------------------------------
// StartReceivingFluxCorr for a group.

void CommRegistry::StartReceivingFluxCorr(CommGroup group)
{
  int g = static_cast<int>(group);
  for (int id : group_channels_[g])
  {
    channels_[id].StartReceivingFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Send flux correction buffers for a group.

void CommRegistry::SendFluxCorrBuffers(CommGroup group)
{
  int g = static_cast<int>(group);
  for (int id : group_channels_[g])
  {
    channels_[id].PackAndSendFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Send flux correction buffers for a single channel.
// Allows per-variable send timing when channels in the same flux_group become
// ready at different points in the task DAG (e.g. CC fluxes before FC EMFs).

void CommRegistry::SendFluxCorrSingleChannel(int channel_id)
{
  channels_[channel_id].PackAndSendFluxCorr(nc_);
}

//----------------------------------------------------------------------------------------
// Post persistent receive for flux correction on a single channel.

void CommRegistry::StartReceivingFluxCorrSingleChannel(int channel_id)
{
  channels_[channel_id].StartReceivingFluxCorr(nc_);
}

//----------------------------------------------------------------------------------------
// Wait on flux correction sends and reset flags for a single channel.

bool CommRegistry::ClearFluxCorrSingleChannel(int channel_id, bool wait)
{
  return channels_[channel_id].ClearFluxCorr(nc_, wait);
}

//----------------------------------------------------------------------------------------
// Receive flux correction buffers for a group.
// Returns true when all channels in the group have completed receiving.
// For FC channels the two-phase protocol (same-level -> ClearCoarse ->
// from-finer -> Average) runs inside PollReceiveFluxCorr at the channel level.

bool CommRegistry::ReceiveFluxCorrBuffers(CommGroup group)
{
  int g         = static_cast<int>(group);
  bool all_done = true;
  for (int id : group_channels_[g])
  {
    if (!channels_[id].PollReceiveFluxCorr(nc_))
      all_done = false;
  }
  return all_done;
}

//----------------------------------------------------------------------------------------
// Set flux correction boundaries (unpack) for a group.
// CC channels overwrite coarse flux arrays; FC channels are no-ops because
// PollReceiveFluxCorrFC already unpacks inline during the two-phase receive.

void CommRegistry::SetFluxCorrBoundaries(CommGroup group)
{
  int g = static_cast<int>(group);
  for (int id : group_channels_[g])
  {
    channels_[id].UnpackFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Clear flux correction boundary for a group.

void CommRegistry::ClearFluxCorrBoundary(CommGroup group)
{
  int g = static_cast<int>(group);
  for (int id : group_channels_[g])
  {
    channels_[id].ClearFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Apply physical BCs for all channels in a group (fine-level).
// Dispatches CC vs FC per channel so callers don't need the sampling branch.

void CommRegistry::ApplyPhysicalBCs(CommGroup group, Real time, Real dt)
{
  int g = static_cast<int>(group);
  for (int id : group_channels_[g])
  {
    const CommSpec& spec = channels_[id].spec();
    if (spec.sampling == Sampling::FC)
      comm::ApplyPhysicalBCs_FC(pmy_block_, spec, time, dt);
    else
      comm::ApplyPhysicalBCs(pmy_block_, spec, time, dt);
  }
}

//----------------------------------------------------------------------------------------
// Apply coarse-level BCs + prolongation for all channels in a group.
// Caller passes coarse index bounds and ghost width from the physics module.

void CommRegistry::ProlongateAndApplyPhysicalBCs(CommGroup group,
                                                 Real time,
                                                 Real dt,
                                                 int cis,
                                                 int cie,
                                                 int cjs,
                                                 int cje,
                                                 int cks,
                                                 int cke,
                                                 int cng)
{
  int g = static_cast<int>(group);
  for (int id : group_channels_[g])
  {
    const CommSpec& spec = channels_[id].spec();
    if (spec.sampling == Sampling::FC)
    {
      // Re-restrict FC interior to refresh coarse_buf.
      // Without this, coarse_buf is stale after ReconcileSharedFacesFC
      // modifies fine-level shared boundary faces between the initial
      // SendBoundaryBuffers (which restricted) and the prolongation here.
      comm::RestrictInterior(pmy_block_, spec);
      comm::ApplyPhysicalBCsOnCoarseLevel_FC(
        pmy_block_, spec, time, dt, cis, cie, cjs, cje, cks, cke, cng);
    }
    else
    {
      comm::ApplyPhysicalBCsOnCoarseLevel(
        pmy_block_, spec, time, dt, cis, cie, cjs, cje, cks, cke, cng);
    }
    comm::ProlongateBoundaries(pmy_block_, spec, nc_);
  }
}

#ifdef DBG_FUSED_COMM
//========================================================================================
// Fused communication: group-level buffer fusion.
//
// All channels within a CommGroup are packed sequentially into a single
// contiguous buffer per neighbor.  A single MPI message is sent/received
// instead of one per channel, reducing MPI message count by ~3x for MainInt.
//
// Same-rank neighbors use zero-copy: the sender packs directly into the
// target's fused recv_buf[targetid], then atomically sets recv_flag[targetid]
// to arrived.
//========================================================================================

//----------------------------------------------------------------------------------------
// Tag construction helpers (duplicated from comm_channel.cpp's file-static
// functions because those are not visible outside that translation unit).

static int FusedBitsNeeded(int max_val)
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

static int FusedMakeTag(int lid,
                        int bufid,
                        int channel_id,
                        int bufid_bits,
                        int channel_bits)
{
  return (lid << (bufid_bits + channel_bits)) | (bufid << channel_bits) |
         channel_id;
}

// The offset added to channel_id for flux correction tags.
// Must match the value in comm_channel.cpp.
static constexpr int kFusedFluxCorrOffset = kMaxNeighbor;

//----------------------------------------------------------------------------------------
// Determine the effective CommTarget for a neighbor based on relative level.

static CommTarget FusedNeighborTarget(int my_level, int nb_level)
{
  if (nb_level == my_level)
    return CommTarget::SameLevel;
  if (nb_level < my_level)
    return CommTarget::ToCoarser;
  return CommTarget::ToFiner;
}

//----------------------------------------------------------------------------------------
// FreeFusedBuffers: release all fused send/recv buffers for one group.

void CommRegistry::FreeFusedBuffers(int g)
{
  FusedGroupState& fs = fused_[g];
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    delete[] fs.send_buf[n];
    fs.send_buf[n] = nullptr;
    delete[] fs.recv_buf[n];
    fs.recv_buf[n]       = nullptr;
    fs.buf_alloc_size[n] = 0;
    fs.target_fused[n]   = nullptr;
  }
  fs.active = false;
}

//----------------------------------------------------------------------------------------
// FreeFusedMPIRequests: release all fused persistent MPI requests for one
// group.

void CommRegistry::FreeFusedMPIRequests(int g)
{
#ifdef MPI_PARALLEL
  FusedGroupState& fs = fused_[g];
  for (int n = 0; n < kMaxNeighbor; ++n)
  {
    if (fs.req_send[n] != MPI_REQUEST_NULL)
    {
      MPI_Request_free(&fs.req_send[n]);
      fs.req_send[n] = MPI_REQUEST_NULL;
    }
    if (fs.req_recv[n] != MPI_REQUEST_NULL)
    {
      MPI_Request_free(&fs.req_recv[n]);
      fs.req_recv[n] = MPI_REQUEST_NULL;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
// ReinitializeFused: teardown fused state for all active groups before regrid
// rebuild.

void CommRegistry::ReinitializeFused()
{
  for (int g = 0; g < static_cast<int>(CommGroup::NumGroups); ++g)
  {
    if (fused_[g].active)
    {
      FreeFusedMPIRequests(g);
      FreeFusedBuffers(g);
    }
  }
}

//----------------------------------------------------------------------------------------
// FinalizeFused: allocate fused buffers, set up persistent MPI, resolve
// zero-copy.
//
// Called after all per-channel Finalize() calls are complete.  For each group
// with
// >= 2 channels, computes fused buffer sizes as the sum of per-channel sizes,
// then allocates buffers and creates persistent MPI requests.

void CommRegistry::FinalizeFused()
{
  MeshBlock* pmb      = pmy_block_;
  const int max_ch_id = num_channels() > 0 ? num_channels() - 1 : 0;

  for (int g = 0; g < static_cast<int>(CommGroup::NumGroups); ++g)
  {
    const std::vector<int>& chids = group_channels_[g];
    if (static_cast<int>(chids.size()) < 2)
    {
      fused_[g].active = false;
      continue;
    }

    FusedGroupState& fs = fused_[g];
    fs.active           = true;

    // --- Allocate fused buffers ---
    // Buffer allocation size per neighbor = sum of per-channel allocation
    // sizes. Same as per-channel: send_buf is only allocated for off-rank
    // neighbors.

    for (int n = 0; n < nc_.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc_.neighbor(n);
      int total_size          = 0;
      for (int id : chids)
      {
        total_size += channels_[id].ComputeBufferSize(nc_, n);
      }
      fs.buf_alloc_size[nb.bufid] = total_size;

      if (nb.snb.rank != Globals::my_rank)
      {
        fs.send_buf[nb.bufid] = new Real[total_size];
      }
      // send_buf remains nullptr for same-rank (zero-copy)
      fs.recv_buf[nb.bufid] = new Real[total_size];
      fs.recv_flag[nb.bufid].store(BoundaryStatus::waiting,
                                   std::memory_order_relaxed);
      fs.send_flag[nb.bufid].store(BoundaryStatus::waiting,
                                   std::memory_order_relaxed);
    }

    // --- Set up persistent MPI ---
#ifdef MPI_PARALLEL
    // Fused messages reuse the first channel's ID as the tag's channel_id
    // field. Safe because per-channel MPI for fused groups is never started
    // when DBG_FUSED_COMM is active.
    const int fused_ch_id  = chids[0];
    const int channel_bits = FusedBitsNeeded(max_ch_id + kFusedFluxCorrOffset);
    const int bufid_bits   = FusedBitsNeeded(kMaxNeighbor - 1);

    // Validate tag space.
    int* tag_ub_ptr = nullptr;
    int flag        = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &flag);
    const int tag_ub = (flag && tag_ub_ptr != nullptr) ? *tag_ub_ptr : 32767;
    const int total_bits = FusedBitsNeeded(tag_ub) - 1;
    const int lid_bits   = total_bits - bufid_bits - channel_bits;
    if (lid_bits < 1 || pmb->lid >= (1 << lid_bits))
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in CommRegistry::FinalizeFused\n"
          << "MPI tag space overflow: lid=" << pmb->lid
          << " fused_channel_id=" << fused_ch_id << " tag_ub=" << tag_ub
          << std::endl;
      ATHENA_ERROR(msg);
    }

    const int mylevel = pmb->loc.level;

    for (int n = 0; n < nc_.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc_.neighbor(n);
      if (nb.snb.rank == Globals::my_rank)
        continue;

      // Compute fused MPI message size (may be smaller than alloc size due to
      // skip_coarse optimization and per-channel target filtering).
      bool skip_coarse_send = nb.neighbor_all_same_level;
      bool skip_coarse_recv = pmb->NeighborBlocksSameLevel();
      CommTarget nbt        = FusedNeighborTarget(mylevel, nb.snb.level);

      int send_size = 0;
      int recv_size = 0;
      for (int id : chids)
      {
        if (!HasTarget(channels_[id].spec().targets, nbt))
          continue;
        send_size += channels_[id].PackSizeForNeighbor(nb, skip_coarse_send);
        recv_size += channels_[id].PackSizeForNeighbor(nb, skip_coarse_recv);
      }

      // Send tag: keyed on receiver's lid and targetid (matching per-channel
      // convention).
      int stag = FusedMakeTag(
        nb.snb.lid, nb.targetid, fused_ch_id, bufid_bits, channel_bits);
      // Recv tag: keyed on this block's lid and bufid.
      int rtag = FusedMakeTag(
        pmb->lid, nb.bufid, fused_ch_id, bufid_bits, channel_bits);
#ifdef MPI_NO_PERSIST
      fs.send_count[nb.bufid] = send_size;
      fs.send_tag[nb.bufid]   = stag;
      fs.recv_count[nb.bufid] = recv_size;
      fs.recv_tag[nb.bufid]   = rtag;
#else
      if (fs.req_send[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&fs.req_send[nb.bufid]);
      MPI_Send_init(fs.send_buf[nb.bufid],
                    send_size,
                    MPI_ATHENA_REAL,
                    nb.snb.rank,
                    stag,
                    MPI_COMM_WORLD,
                    &fs.req_send[nb.bufid]);

      if (fs.req_recv[nb.bufid] != MPI_REQUEST_NULL)
        MPI_Request_free(&fs.req_recv[nb.bufid]);
      MPI_Recv_init(fs.recv_buf[nb.bufid],
                    recv_size,
                    MPI_ATHENA_REAL,
                    nb.snb.rank,
                    rtag,
                    MPI_COMM_WORLD,
                    &fs.req_recv[nb.bufid]);
#endif  // MPI_NO_PERSIST
    }
#endif  // MPI_PARALLEL

    // --- Resolve zero-copy targets for same-rank neighbors ---
    for (int n = 0; n < nc_.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc_.neighbor(n);
      if (nb.snb.rank != Globals::my_rank)
        continue;

      MeshBlock* ptarget        = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
      CommRegistry* target_reg  = FindForBlock(ptarget);
      fs.target_fused[nb.bufid] = &target_reg->fused_[g];
    }
  }
}

//----------------------------------------------------------------------------------------
// SendBoundaryBuffersFused: pack all channels into fused buffer, then send.
//
// Restriction has already been called by SendBoundaryBuffers before reaching
// here. For each neighbor, packs all channels sequentially into the fused send
// buffer (or directly into the target's fused recv buffer for same-rank
// zero-copy).

void CommRegistry::SendBoundaryBuffersFused(CommGroup group,
                                            CommTarget target_filter)
{
  MeshBlock* pmb                = pmy_block_;
  const int mylevel             = pmb->loc.level;
  int g                         = static_cast<int>(group);
  FusedGroupState& fs           = fused_[g];
  const std::vector<int>& chids = group_channels_[g];

  // Compute effective target for each channel (intersection of spec targets
  // and filter). All channels in the group must agree on which neighbors to
  // communicate with; the target_filter narrows but cannot widen.

  for (int n = 0; n < nc_.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc_.neighbor(n);
    CommTarget nbt          = FusedNeighborTarget(mylevel, nb.snb.level);

    // Check if any channel in the group wants to communicate with this
    // neighbor. (All channels in a fused group should have compatible targets,
    // but be safe.)
    bool any_active = false;
    for (int id : chids)
    {
      CommTarget eff = channels_[id].spec().targets & target_filter;
      if (HasTarget(eff, nbt))
      {
        any_active = true;
        break;
      }
    }
    if (!any_active)
      continue;

    // Zero-copy: for same-rank, pack directly into target's fused recv buffer.
    const bool same_rank = (nb.snb.rank == Globals::my_rank);
    Real* buf = same_rank ? fs.target_fused[nb.bufid]->recv_buf[nb.targetid]
                          : fs.send_buf[nb.bufid];

    // Pack all channels sequentially into the fused buffer.
    int offset = 0;
    for (int id : chids)
    {
      CommTarget eff = channels_[id].spec().targets & target_filter;
      if (!HasTarget(eff, nbt))
        continue;
      offset = channels_[id].PackInto(buf, offset, nb, mylevel);
    }

    // Send dispatch
    if (same_rank)
    {
      fs.target_fused[nb.bufid]->recv_flag[nb.targetid].store(
        BoundaryStatus::arrived, std::memory_order_release);
    }
    else
    {
#ifdef MPI_PARALLEL
#ifdef MPI_NO_PERSIST
      CheckMPIResult(MPI_Isend(fs.send_buf[nb.bufid],
                               fs.send_count[nb.bufid],
                               MPI_ATHENA_REAL,
                               nb.snb.rank,
                               fs.send_tag[nb.bufid],
                               MPI_COMM_WORLD,
                               &fs.req_send[nb.bufid]),
#else
      CheckMPIResult(MPI_Start(&fs.req_send[nb.bufid]),
#endif  // MPI_NO_PERSIST
                     "MPI_Start(SendBoundaryFused)",
                     Globals::my_rank,
                     pmb->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     g);
#endif
    }
    fs.send_flag[nb.bufid].store(BoundaryStatus::completed,
                                 std::memory_order_release);
  }
}

//----------------------------------------------------------------------------------------
// ReceiveBoundaryBuffersFused: poll fused recv flags / MPI_Test.
// Returns true when all neighbors have arrived.

bool CommRegistry::ReceiveBoundaryBuffersFused(CommGroup group,
                                               CommTarget target_filter)
{
  MeshBlock* pmb      = pmy_block_;
  const int mylevel   = pmb->loc.level;
  int g               = static_cast<int>(group);
  FusedGroupState& fs = fused_[g];

  bool all_arrived = true;
  for (int n = 0; n < nc_.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc_.neighbor(n);
    CommTarget nbt          = FusedNeighborTarget(mylevel, nb.snb.level);

    // Must match the same filter logic as Send - only poll neighbors that were
    // expected.
    bool any_active = false;
    for (int id : group_channels_[g])
    {
      CommTarget eff = channels_[id].spec().targets & target_filter;
      if (HasTarget(eff, nbt))
      {
        any_active = true;
        break;
      }
    }
    if (!any_active)
      continue;

    if (fs.recv_flag[nb.bufid].load(std::memory_order_acquire) ==
        BoundaryStatus::arrived)
      continue;

#ifdef MPI_PARALLEL
    if (nb.snb.rank != Globals::my_rank)
    {
      int test_flag = 0;
      MPI_Status mpi_status;
      CheckMPIResult(MPI_Test(&fs.req_recv[nb.bufid], &test_flag, &mpi_status),
                     "MPI_Test(RecvBoundaryFused)",
                     Globals::my_rank,
                     pmb->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     g);
      if (test_flag)
        fs.recv_flag[nb.bufid].store(BoundaryStatus::arrived,
                                     std::memory_order_release);
    }
#endif

    if (fs.recv_flag[nb.bufid].load(std::memory_order_acquire) !=
        BoundaryStatus::arrived)
      all_arrived = false;
  }

  return all_arrived;
}

//----------------------------------------------------------------------------------------
// SetBoundariesFused: unpack per-channel from fused recv buffer.
// Handles VC zero-ghost / additive-unpack / divide protocol exactly as the
// unfused path.

void CommRegistry::SetBoundariesFused(CommGroup group,
                                      CommTarget target_filter)
{
  int g                         = static_cast<int>(group);
  MeshBlock* pmb                = pmy_block_;
  const int mylevel             = pmb->loc.level;
  FusedGroupState& fs           = fused_[g];
  const std::vector<int>& chids = group_channels_[g];

  // Phase 1: zero VC ghost zones before additive unpack (same as unfused
  // path).
  for (int id : chids)
  {
    CommChannel& ch = channels_[id];
    if (ch.spec().sampling != Sampling::VC)
      continue;

    NodeMultiplicity& nm = ch.node_multiplicity();
    nm.ZeroGhosts(*ch.spec().var, pmb, ch.spec().nvar);
    if (pmb->pmy_mesh->multilevel && ch.spec().coarse_var != nullptr)
      nm.ZeroGhostsCoarse(*ch.spec().coarse_var, pmb, ch.spec().nvar);
  }

  // Phase 2: for each neighbor, unpack all channels sequentially from the
  // fused buffer.
  for (int n = 0; n < nc_.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc_.neighbor(n);
    CommTarget nbt          = FusedNeighborTarget(mylevel, nb.snb.level);

    bool any_active = false;
    for (int id : chids)
    {
      CommTarget eff = channels_[id].spec().targets & target_filter;
      if (HasTarget(eff, nbt))
      {
        any_active = true;
        break;
      }
    }
    if (!any_active)
      continue;

    Real* buf  = fs.recv_buf[nb.bufid];
    int offset = 0;
    for (int id : chids)
    {
      CommTarget eff = channels_[id].spec().targets & target_filter;
      if (!HasTarget(eff, nbt))
        continue;
      offset = channels_[id].UnpackFrom(buf, offset, nb, mylevel);
    }

    fs.recv_flag[nb.bufid].store(BoundaryStatus::completed,
                                 std::memory_order_release);
  }

  // Phase 3: divide VC arrays by accumulated node multiplicity (same as
  // unfused path).
  for (int id : chids)
  {
    CommChannel& ch = channels_[id];
    if (ch.spec().sampling != Sampling::VC)
      continue;

    const NodeMultiplicity& nm = ch.node_multiplicity();
    nm.ApplyDivision(*ch.spec().var, pmb, ch.spec().nvar);
    if (pmb->pmy_mesh->multilevel && ch.spec().coarse_var != nullptr)
      nm.ApplyDivisionCoarse(*ch.spec().coarse_var, pmb, ch.spec().nvar);
  }
}

//----------------------------------------------------------------------------------------
// ClearBoundaryFused: test (or wait on) outstanding sends, reset fused flags.

bool CommRegistry::ClearBoundaryFused(CommGroup group,
                                      CommTarget target_filter,
                                      bool wait)
{
  MeshBlock* pmb      = pmy_block_;
  const int mylevel   = pmb->loc.level;
  int g               = static_cast<int>(group);
  FusedGroupState& fs = fused_[g];

  // Helper: returns true if this neighbor is active in any channel of the
  // group under the given target filter.
  auto IsNeighborActive = [&](const NeighborBlock& nb) -> bool
  {
    CommTarget nbt = FusedNeighborTarget(mylevel, nb.snb.level);
    for (int id : group_channels_[g])
    {
      CommTarget eff = channels_[id].spec().targets & target_filter;
      if (HasTarget(eff, nbt))
        return true;
    }
    return false;
  };

  // When wait=false, first check that all off-rank sends have completed
  // before touching any flags.  This avoids premature flag resets.
#ifdef MPI_PARALLEL
  if (!wait)
  {
    for (int n = 0; n < nc_.num_neighbors(); ++n)
    {
      const NeighborBlock& nb = nc_.neighbor(n);
      if (!IsNeighborActive(nb))
        continue;
      if (nb.snb.rank != Globals::my_rank)
      {
        int flag;
        MPI_Status mpi_status;
        CheckMPIResult(MPI_Test(&fs.req_send[nb.bufid], &flag, &mpi_status),
                       "MPI_Test(ClearBoundaryFused)",
                       Globals::my_rank,
                       pmb->gid,
                       nb.snb.rank,
                       nb.snb.gid,
                       nb.bufid,
                       g);
        if (!flag)
          return false;  // at least one send still pending - retry later
      }
    }
    // All sends confirmed complete - fall through to reset flags below.
  }
#endif

  // Reset all participating flags and (when wait=true) block on sends.
  for (int n = 0; n < nc_.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc_.neighbor(n);
    if (!IsNeighborActive(nb))
      continue;

    fs.recv_flag[nb.bufid].store(BoundaryStatus::waiting,
                                 std::memory_order_relaxed);
    fs.send_flag[nb.bufid].store(BoundaryStatus::waiting,
                                 std::memory_order_relaxed);

#ifdef MPI_PARALLEL
    if (wait && nb.snb.rank != Globals::my_rank)
    {
      MPI_Status mpi_status;
      CheckMPIResult(MPI_Wait(&fs.req_send[nb.bufid], &mpi_status),
                     "MPI_Wait(ClearBoundaryFused)",
                     Globals::my_rank,
                     pmb->gid,
                     nb.snb.rank,
                     nb.snb.gid,
                     nb.bufid,
                     g);
    }
#endif
  }
  return true;
}

//----------------------------------------------------------------------------------------
// StartReceivingFused: post persistent receive requests for fused buffers.

void CommRegistry::StartReceivingFused(CommGroup group,
                                       CommTarget target_filter)
{
#ifdef MPI_PARALLEL
  MeshBlock* pmb      = pmy_block_;
  const int mylevel   = pmb->loc.level;
  int g               = static_cast<int>(group);
  FusedGroupState& fs = fused_[g];

  for (int n = 0; n < nc_.num_neighbors(); ++n)
  {
    const NeighborBlock& nb = nc_.neighbor(n);
    if (nb.snb.rank == Globals::my_rank)
      continue;

    CommTarget nbt = FusedNeighborTarget(mylevel, nb.snb.level);

    bool any_active = false;
    for (int id : group_channels_[g])
    {
      CommTarget eff = channels_[id].spec().targets & target_filter;
      if (HasTarget(eff, nbt))
      {
        any_active = true;
        break;
      }
    }
    if (!any_active)
      continue;

#ifdef MPI_NO_PERSIST
    CheckMPIResult(MPI_Irecv(fs.recv_buf[nb.bufid],
                             fs.recv_count[nb.bufid],
                             MPI_ATHENA_REAL,
                             nb.snb.rank,
                             fs.recv_tag[nb.bufid],
                             MPI_COMM_WORLD,
                             &fs.req_recv[nb.bufid]),
#else
    CheckMPIResult(MPI_Start(&fs.req_recv[nb.bufid]),
#endif  // MPI_NO_PERSIST
                   "MPI_Start(StartRecvFused)",
                   Globals::my_rank,
                   pmb->gid,
                   nb.snb.rank,
                   nb.snb.gid,
                   nb.bufid,
                   g);
  }
#endif
}
#endif  // DBG_FUSED_COMM

}  // namespace comm
