#ifndef COMM_PARITY_HPP_
#define COMM_PARITY_HPP_
//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file parity.hpp
//  \brief Parity sign-flip module for boundary communication.
//
//  Computes per-component sign arrays from CommSpec::component_groups for:
//    - Reflecting BCs (flip the component normal to the wall)
//    - Polar inter-block communication (flip components odd under theta-reflection)
//    - PolarWedge BC (same flips as polar inter-block)
//
//  The sign arrays are computed once per channel at Finalize() time and reused.
//  Empty component_groups = all even parity (no flips anywhere).

#include <vector>

#include "../athena.hpp"
#include "../mesh/mesh_topology.hpp"  // BoundaryFace
#include "comm_enums.hpp"

namespace comm {

struct CommSpec;

//----------------------------------------------------------------------------------------
// Flip context: which geometric transformation the sign array is for.
//
// ReflectX1/X2/X3: reflecting wall - flip the component normal to the face.
//   For Vector: one component flips (the one aligned with the face normal).
//   For SymTensor: components with an odd count of the normal-direction index flip.
//
// Polar: theta = 0 or pi boundary - flip components odd under (theta, phi) -> (-theta, -phi).
//   For Vector: y(theta) and z(phi) components flip.
//   For SymTensor: off-diagonal components with an odd count of y or z indices flip.
//   This matches flip_across_pole_hydro/field in the old system.

enum class FlipContext : int {
  ReflectX1 = 0,
  ReflectX2 = 1,
  ReflectX3 = 2,
  Polar     = 3
};

//----------------------------------------------------------------------------------------
// Convert a BoundaryFace to the appropriate FlipContext for reflecting BCs.

inline FlipContext ReflectContext(BoundaryFace face) {
  switch (face) {
    case BoundaryFace::inner_x1:
    case BoundaryFace::outer_x1: return FlipContext::ReflectX1;
    case BoundaryFace::inner_x2:
    case BoundaryFace::outer_x2: return FlipContext::ReflectX2;
    case BoundaryFace::inner_x3:
    case BoundaryFace::outer_x3: return FlipContext::ReflectX3;
    default: return FlipContext::ReflectX1;  // unreachable
  }
}

//----------------------------------------------------------------------------------------
// Compute a per-component sign array from component_groups for a given context.
//
// Returns a vector of length nvar where each element is +1.0 or -1.0.
// For empty component_groups, returns all +1.0 (no flips).
//
// The sign rules for each GeomType:
//
//   Scalar: always +1.0 (no sign change under any transformation)
//
//   Vector (3 components indexed x=0, y=1, z=2):
//     ReflectX1: x flips  -> signs = {-1, +1, +1}
//     ReflectX2: y flips  -> signs = {+1, -1, +1}
//     ReflectX3: z flips  -> signs = {+1, +1, -1}
//     Polar:     y,z flip -> signs = {+1, -1, -1}
//
//   SymTensor (6 components in order: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5):
//     A component flips if it has an odd count of the "flip direction" index.
//     ReflectX1: flip if odd count of x -> xy(1), xz(2) flip
//     ReflectX2: flip if odd count of y -> xy(1), yz(4) flip
//     ReflectX3: flip if odd count of z -> xz(2), yz(4) flip
//     Polar:     flip if odd count of y or z combined (xor) -> xy(1), xz(2), yz(4) flip
//       This matches the old flip_across_pole pattern for tensor components.

std::vector<Real> ComputeSignArray(const CommSpec &spec, FlipContext ctx);

//----------------------------------------------------------------------------------------
// Apply parity sign flips in-place to an unpacked ghost-zone region.
// Multiplies var(n, k, j, i) by signs[n] for each component n in [0, nvar).
//
// This is a post-unpack pass: call after BufferUtility::UnpackData or after a
// reflecting/polar BC fill has written mirror-image data into ghost zones.
//
// Only touches cells in the specified index range (the ghost zone region).

void ApplyParitySigns(AthenaArray<Real> &var, int nvar,
                      const std::vector<Real> &signs,
                      int si, int ei, int sj, int ej, int sk, int ek);

} // namespace comm

#endif // COMM_PARITY_HPP_
