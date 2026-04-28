//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file neighbor_connectivity.cpp
//  \brief NeighborConnectivity implementation: default construction and initialization.

#include "neighbor_connectivity.hpp"

#include <algorithm>  // std::fill

namespace comm {

//----------------------------------------------------------------------------------------
// Default constructor: no neighbors, all nblevel = -1, all BCs = undef.

NeighborConnectivity::NeighborConnectivity()
    : nneighbor_(0) {
  std::fill(&nblevel_[0][0][0], &nblevel_[0][0][0] + 27, -1);
  std::fill(block_bcs_, block_bcs_ + 6, BoundaryFlag::undef);
}

//----------------------------------------------------------------------------------------
// Set block_bcs_[6] from an input array (called by MeshBlock constructors).

void NeighborConnectivity::InitBoundaryFlags(const BoundaryFlag input_bcs[6]) {
  for (int f = 0; f < 6; ++f)
    block_bcs_[f] = input_bcs[f];
}

//----------------------------------------------------------------------------------------
// Reset neighbor state before SearchAndSetNeighbors populates it.
// Sets nneighbor to 0 and nblevel to -1, then stamps self-level at center.

void NeighborConnectivity::ResetForSearch(int my_level) {
  nneighbor_ = 0;
  std::fill(&nblevel_[0][0][0], &nblevel_[0][0][0] + 27, -1);
  nblevel_[1][1][1] = my_level;
}

} // namespace comm
