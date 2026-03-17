//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file comm_registry.cpp
//  \brief CommRegistry implementation: registration, finalization, group-level comm.

#include "comm_registry.hpp"

#include <sstream>
#include <stdexcept>

#include "../mesh/mesh.hpp"
#include "node_multiplicity.hpp"
#include "physical_bcs.hpp"
#include "refinement_ops.hpp"

namespace comm {

//----------------------------------------------------------------------------------------
// Static member: global registry map for same-process copy lookup.

std::unordered_map<MeshBlock*, CommRegistry*> CommRegistry::registry_map_;

//----------------------------------------------------------------------------------------
// Constructor.

CommRegistry::CommRegistry(MeshBlock *pmb)
    : pmy_block_(pmb),
      nc_(),
      finalized_(false) {
  // Register this instance so same-process copy can find the target's channels.
  registry_map_[pmb] = this;
}

//----------------------------------------------------------------------------------------
// Destructor.

CommRegistry::~CommRegistry() {
  registry_map_.erase(pmy_block_);
}

//----------------------------------------------------------------------------------------
// Static lookup: find the CommRegistry for a given MeshBlock.

CommRegistry* CommRegistry::FindForBlock(MeshBlock *pmb) {
  auto it = registry_map_.find(pmb);
  if (it != registry_map_.end()) return it->second;
  return nullptr;
}

//----------------------------------------------------------------------------------------
// Register a variable.  Returns the channel_id.

int CommRegistry::Register(const CommSpec &spec) {
  if (finalized_) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CommRegistry::Register\n"
        << "Cannot register channel '" << spec.label
        << "' after Finalize()." << std::endl;
    ATHENA_ERROR(msg);
  }

  // Check for duplicate labels.
  if (label_to_id_.count(spec.label) > 0) {
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
  // its primary group for ghost exchange, and its flux_group for flux correction.
  if (spec.flux_group != CommGroup::NumGroups) {
    int fg = static_cast<int>(spec.flux_group);
    group_channels_[fg].push_back(id);
  }

  // Record label lookup.
  label_to_id_[spec.label] = id;

  return id;
}

//----------------------------------------------------------------------------------------
// Finalize: build connectivity, allocate buffers, create MPI requests.

void CommRegistry::Finalize() {
  RebuildConnectivity();
  // All channels must use the same MPI tag bit layout, so pass the max channel_id.
  int max_ch_id = num_channels() > 0 ? num_channels() - 1 : 0;
  for (auto &ch : channels_) {
    ch.Finalize(nc_, max_ch_id);
  }
  finalized_ = true;
}

//----------------------------------------------------------------------------------------
// Reinitialize after regrid.

void CommRegistry::Reinitialize() {
  // Must clear finalized_ so that the rebuild path in each channel can re-enter
  // Finalize logic.  Set it back to true at the end so post-regrid comm works.
  finalized_ = false;
  RebuildConnectivity();
  int max_ch_id = num_channels() > 0 ? num_channels() - 1 : 0;
  for (auto &ch : channels_) {
    ch.Reinitialize(nc_, max_ch_id);
  }
  finalized_ = true;
}

//----------------------------------------------------------------------------------------
// Copy the MeshBlock's NeighborConnectivity into the registry's local snapshot.
// Called before Finalize/Reinitialize so that channel setup sees current topology.

void CommRegistry::RebuildConnectivity() {
  nc_ = pmy_block_->nc();
}

//----------------------------------------------------------------------------------------
// StartReceiving for a group.

void CommRegistry::StartReceiving(CommGroup group, CommTarget target_filter) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    channels_[id].StartReceiving(nc_, target_filter);
  }
}

//----------------------------------------------------------------------------------------
// Send boundary buffers for a group.
// On a multilevel mesh, restrict the fine interior into the coarse buffer first so
// that PackAndSend has up-to-date coarse data for cross-level neighbors.
//
// Restriction can be skipped when this block AND all its same-level neighbors have
// only same-level neighbors (two-level check).  In that case:
//   - This block has no coarser neighbors, so no to-coarser send needs restricted data.
//   - No same-level neighbor has coarser neighbors either, so none of them will read
//     the coarse payload (it is skipped in PackAndSend by the same-level guard).
// This is the same optimization as RestrictNonGhost in the old boundary system.

static bool NeedsRestriction(MeshBlock *pmb) {
  // If this block has any cross-level neighbor, it needs restriction for sure.
  if (!pmb->NeighborBlocksSameLevel()) return true;
  // Even if all our neighbors are same-level, check whether any of them have
  // a coarser neighbor - if so, they need the coarse payload from us.
  for (int n = 0; n < pmb->nc().num_neighbors(); n++) {
    if (!pmb->nc().neighbor(n).neighbor_all_same_level)
      return true;
  }
  return false;
}

void CommRegistry::SendBoundaryBuffers(CommGroup group, CommTarget target_filter) {
  int g = static_cast<int>(group);

  // Restrict fine -> coarse interior for each channel that needs it.
  // Only necessary when cross-level neighbors exist (multilevel mesh).
  if (pmy_block_->pmy_mesh->multilevel && NeedsRestriction(pmy_block_)) {
    for (int id : group_channels_[g]) {
      RestrictInterior(pmy_block_, channels_[id].spec());
    }
  }

  for (int id : group_channels_[g]) {
    channels_[id].PackAndSend(nc_, target_filter);
  }
}

//----------------------------------------------------------------------------------------
// Receive boundary buffers for a group.  Returns true when all channels are done.

bool CommRegistry::ReceiveBoundaryBuffers(CommGroup group, CommTarget target_filter) {
  int g = static_cast<int>(group);
  bool all_done = true;
  for (int id : group_channels_[g]) {
    if (!channels_[id].PollReceive(nc_, target_filter))
      all_done = false;
  }
  return all_done;
}

//----------------------------------------------------------------------------------------
// Set boundaries (unpack) for a group.
// VC channels require special protocol: ZeroGhosts -> Unpack -> ApplyDivision,
// because vertex-centered variables use additive unpack and need multiplicity correction.

void CommRegistry::SetBoundaries(CommGroup group, CommTarget target_filter) {
  int g = static_cast<int>(group);

  // Phase 1: zero VC ghost zones before additive unpack.
  // Note: zeroing and division (phase 3) are NOT filtered by target_filter because
  // the multiplicity counts are precomputed for the full neighbor topology.
  // VC channels should always be called with CommTarget::All.
  for (int id : group_channels_[g]) {
    CommChannel &ch = channels_[id];
    if (ch.spec().sampling != Sampling::VC) continue;

    NodeMultiplicity &nm = ch.node_multiplicity();
    nm.ZeroGhosts(*ch.spec().var, pmy_block_, ch.spec().nvar);
    if (pmy_block_->pmy_mesh->multilevel && ch.spec().coarse_var != nullptr)
      nm.ZeroGhostsCoarse(*ch.spec().coarse_var, pmy_block_, ch.spec().nvar);
  }

  // Phase 2: unpack all channels (VC uses additive unpack internally).
  for (int id : group_channels_[g]) {
    channels_[id].Unpack(nc_, target_filter);
  }

  // Phase 3: divide VC arrays by accumulated node multiplicity.
  for (int id : group_channels_[g]) {
    CommChannel &ch = channels_[id];
    if (ch.spec().sampling != Sampling::VC) continue;

    const NodeMultiplicity &nm = ch.node_multiplicity();
    nm.ApplyDivision(*ch.spec().var, pmy_block_, ch.spec().nvar);
    if (pmy_block_->pmy_mesh->multilevel && ch.spec().coarse_var != nullptr)
      nm.ApplyDivisionCoarse(*ch.spec().coarse_var, pmy_block_, ch.spec().nvar);
  }
}

//----------------------------------------------------------------------------------------
// Clear boundary for a group.

void CommRegistry::ClearBoundary(CommGroup group, CommTarget target_filter) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    channels_[id].Clear(nc_, target_filter);
  }
}

//----------------------------------------------------------------------------------------
// Find channel by label.

int CommRegistry::FindChannelByLabel(const std::string &label) const {
  auto it = label_to_id_.find(label);
  if (it != label_to_id_.end()) return it->second;
  return -1;
}

//----------------------------------------------------------------------------------------
// StartReceivingFluxCorr for a group.

void CommRegistry::StartReceivingFluxCorr(CommGroup group) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    channels_[id].StartReceivingFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Send flux correction buffers for a group.

void CommRegistry::SendFluxCorrBuffers(CommGroup group) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    channels_[id].PackAndSendFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Send flux correction buffers for a single channel.
// Allows per-variable send timing when channels in the same flux_group become
// ready at different points in the task DAG (e.g. CC fluxes before FC EMFs).

void CommRegistry::SendFluxCorrSingleChannel(int channel_id) {
  channels_[channel_id].PackAndSendFluxCorr(nc_);
}

//----------------------------------------------------------------------------------------
// Post persistent receive for flux correction on a single channel.

void CommRegistry::StartReceivingFluxCorrSingleChannel(int channel_id) {
  channels_[channel_id].StartReceivingFluxCorr(nc_);
}

//----------------------------------------------------------------------------------------
// Wait on flux correction sends and reset flags for a single channel.

void CommRegistry::ClearFluxCorrSingleChannel(int channel_id) {
  channels_[channel_id].ClearFluxCorr(nc_);
}

//----------------------------------------------------------------------------------------
// Receive flux correction buffers for a group.
// Returns true when all channels in the group have completed receiving.
// For FC channels the two-phase protocol (same-level -> ClearCoarse -> from-finer ->
// Average) runs inside PollReceiveFluxCorr at the channel level.

bool CommRegistry::ReceiveFluxCorrBuffers(CommGroup group) {
  int g = static_cast<int>(group);
  bool all_done = true;
  for (int id : group_channels_[g]) {
    if (!channels_[id].PollReceiveFluxCorr(nc_))
      all_done = false;
  }
  return all_done;
}

//----------------------------------------------------------------------------------------
// Set flux correction boundaries (unpack) for a group.
// CC channels overwrite coarse flux arrays; FC channels are no-ops because
// PollReceiveFluxCorrFC already unpacks inline during the two-phase receive.

void CommRegistry::SetFluxCorrBoundaries(CommGroup group) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    channels_[id].UnpackFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Clear flux correction boundary for a group.

void CommRegistry::ClearFluxCorrBoundary(CommGroup group) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    channels_[id].ClearFluxCorr(nc_);
  }
}

//----------------------------------------------------------------------------------------
// Apply physical BCs for all channels in a group (fine-level).
// Dispatches CC vs FC per channel so callers don't need the sampling branch.

void CommRegistry::ApplyPhysicalBCs(CommGroup group, Real time, Real dt) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    const CommSpec &spec = channels_[id].spec();
    if (spec.sampling == Sampling::FC)
      comm::ApplyPhysicalBCs_FC(pmy_block_, spec, time, dt);
    else
      comm::ApplyPhysicalBCs(pmy_block_, spec, time, dt);
  }
}

//----------------------------------------------------------------------------------------
// Apply coarse-level BCs + prolongation for all channels in a group.
// Caller passes coarse index bounds and ghost width from the physics module.

void CommRegistry::ProlongateAndApplyPhysicalBCs(
    CommGroup group, Real time, Real dt,
    int cis, int cie, int cjs, int cje, int cks, int cke, int cng) {
  int g = static_cast<int>(group);
  for (int id : group_channels_[g]) {
    const CommSpec &spec = channels_[id].spec();
    if (spec.sampling == Sampling::FC) {
      comm::ApplyPhysicalBCsOnCoarseLevel_FC(
          pmy_block_, spec, time, dt, cis, cie, cjs, cje, cks, cke, cng);
    } else {
      comm::ApplyPhysicalBCsOnCoarseLevel(
          pmy_block_, spec, time, dt, cis, cie, cjs, cje, cks, cke, cng);
    }
    comm::ProlongateBoundaries(pmy_block_, spec, nc_);
  }
}

} // namespace comm
