#ifndef COMM_COMM_ENUMS_HPP_
#define COMM_COMM_ENUMS_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file comm_enums.hpp
//  \brief Enums for the data-driven communication system.

#include <cstdint>

namespace comm {

//----------------------------------------------------------------------------------------
// Grid sampling: where a variable lives on the mesh.
// Determines index arithmetic, buffer sizes, and which prolong/restrict family to use.

enum class Sampling : int {
  CC = 0,  // cell-centered
  VC = 1,  // vertex-centered
  CX = 2,  // cell-centered extended (half-integer shift, wider ghost zone)
  FC = 3   // face-centered (three staggered components)
};

//----------------------------------------------------------------------------------------
// Communication targets: which neighbor relationships a channel participates in.
// Bitwise-OR combinable.  A channel registered with (SameLevel | ToCoarser) will only
// communicate with same-level and coarser neighbors; the call-site filter can only
// narrow this set, never widen it.

enum CommTarget : std::uint32_t {
  SameLevel   = 1u << 0,
  ToCoarser   = 1u << 1,  // this block sends restricted data to a coarser neighbor
  ToFiner     = 1u << 2,  // this block sends fine data to a finer neighbor
  All         = SameLevel | ToCoarser | ToFiner
};

inline constexpr CommTarget operator|(CommTarget a, CommTarget b) {
  return static_cast<CommTarget>(static_cast<std::uint32_t>(a) |
                                 static_cast<std::uint32_t>(b));
}
inline constexpr CommTarget operator&(CommTarget a, CommTarget b) {
  return static_cast<CommTarget>(static_cast<std::uint32_t>(a) &
                                 static_cast<std::uint32_t>(b));
}
inline constexpr bool HasTarget(CommTarget set, CommTarget flag) {
  return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(flag)) != 0;
}

//----------------------------------------------------------------------------------------
// Communication group: variables in the same group on the same MeshBlock are fused into
// one MPI message per neighbor.  Each group is a separate fused message.

enum class CommGroup : int {
  MainInt    = 0,  // main integrator variables (hydro cons, field, scalars)
  Z4c        = 1,  // Z4c main state (ubvar)
  Aux        = 2,  // Z4c Weyl auxiliary (abvar)
  AuxADM     = 3,  // Z4c ADM auxiliary derivatives (adm_abvar)
  M1         = 4,  // M1 radiation transport
  Iterated   = 5,  // iterated boundary comm (Z4c reference-BC iteration)
  FluxCorr   = 6,  // flux correction at coarse/fine interfaces
  Wave       = 7,  // wave equation
  NumGroups  = 8   // sentinel - number of groups
};

//----------------------------------------------------------------------------------------
// Prolongation operators: coarse-to-fine interpolation.
// Runtime-selectable per channel.

enum class ProlongOp : int {
  MinmodLinear         = 0,  // CC: minmod-limited piecewise linear
  LagrangeUniform      = 1,  // VC: symmetric Lagrange on uniform grid, inject+interp
  LagrangeChildren     = 2,  // CX: Lagrange children (interior order)
  LagrangeChildrenBC   = 3,  // CX: Lagrange children (boundary-compatible, lower order)
  FaceSharedMinmod     = 4,  // FC: minmod in transverse dirs at shared faces
  FaceDivPreserving    = 5,  // FC: divergence-preserving internal reconstruction
  None                 = 6   // no prolongation (uniform-only variable)
};

//----------------------------------------------------------------------------------------
// Restriction operators: fine-to-coarse averaging / injection.
// Runtime-selectable per channel.

enum class RestrictOp : int {
  VolumeWeighted       = 0,  // CC: volume-weighted average of 2^d fine cells
  Injection            = 1,  // VC: direct injection of coincident vertex
  LagrangeUniform      = 2,  // CX: symmetric Lagrange on uniform grid
  Barycentric          = 3,  // CX: Floater-Hormann barycentric rational (interior)
  AreaWeightedFace     = 4,  // FC: area-weighted average per face component
  None                 = 5,  // no restriction (uniform-only variable)
  LagrangeFull         = 6   // CX: symmetric Lagrange using ghost data (iterated BC)
};

//----------------------------------------------------------------------------------------
// Physical boundary conditions applied at domain faces.
// These are the conditions that require ghost-zone filling by function evaluation
// (as opposed to block-to-block or periodic, which are handled by communication).

enum class PhysicalBC : int {
  Reflect              = 0,
  Outflow              = 1,
  ExtrapolateOutflow   = 2,
  GRSommerfeld         = 3,
  PolarWedge           = 4,  // only valid on x2 faces
  User                 = 5,  // user-enrolled via Mesh::EnrollUserBoundaryFunction
  None                 = 6   // internal (block/periodic/polar) - no function call
};

//----------------------------------------------------------------------------------------
// Geometric type for parity specification.
// Used in component_groups to determine sign flips across coordinate boundaries (e.g.
// polar axis, reflecting walls).  Empty component_groups = all even parity (no flips).

enum class GeomType : int {
  Scalar    = 0,  // no sign flips
  Vector    = 1,  // 3 components, sign flips determined by face orientation
  SymTensor = 2   // 6 independent components, sign flips from tensor transformation
};

//----------------------------------------------------------------------------------------
// Flux correction mode: how received flux correction data is applied on the coarse side.
// Stored per-channel in CommSpec; determines unpack behaviour in CommChannel.

enum class FluxCorrMode : int {
  None                = 0,  // no flux correction for this channel
  OverwriteFromFiner  = 1,  // CC: fine-restricted flux overwrites coarse flux at face
  AccumulateAverage   = 2   // FC EMF: accumulate edge values, then average by count
};

} // namespace comm

#endif // COMM_COMM_ENUMS_HPP_
