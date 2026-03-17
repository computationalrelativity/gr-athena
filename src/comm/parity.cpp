//========================================================================================
// GR-Athena++ communication layer
//========================================================================================
//! \file parity.cpp
//  \brief Parity sign-flip computation and application.
//
//  Translates ComponentGroup metadata (GeomType, count) into concrete
//  per-component sign arrays for each flip context (reflect x1/x2/x3, polar).
//  These arrays are then used as a post-processing pass after unpack or after
//  BC fill.

#include "parity.hpp"

#include <vector>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "comm_spec.hpp"

namespace comm {

//----------------------------------------------------------------------------------------
// Sign tables for each GeomType under each FlipContext.
//
// Vector: 3 components (x=0, y=1, z=2)
// SymTensor: 6 components in order (xx=0, xy=1, xz=2, yy=3, yz=4, zz=5)
//
// A component flips (sign = -1) if the transformation makes it change sign.
// For reflect: the component normal to the wall flips.
// For polar: components odd under the combined (theta,phi) -> (-theta,-phi)
// flip.

// Vector signs: +1.0 = no flip, -1.0 = flip
static const Real kVectorSigns[4][3] = {
    // ReflectX1: x flips
    {-1.0, 1.0, 1.0},
    // ReflectX2: y flips
    {1.0, -1.0, 1.0},
    // ReflectX3: z flips
    {1.0, 1.0, -1.0},
    // Polar: y and z flip (theta and phi components change sign across pole)
    {1.0, -1.0, -1.0}};

// SymTensor signs: indices {xx, xy, xz, yy, yz, zz}
// A component T_{ab} flips if it has an odd number of "flipping" indices.
static const Real kSymTensorSigns[4][6] = {
    // ReflectX1: odd count of x -> xy, xz flip
    {1.0, -1.0, -1.0, 1.0, 1.0, 1.0},
    // ReflectX2: odd count of y -> xy, yz flip
    {1.0, -1.0, 1.0, 1.0, -1.0, 1.0},
    // ReflectX3: odd count of z -> xz, yz flip
    {1.0, 1.0, -1.0, 1.0, -1.0, 1.0},
    // Polar: T_{ab} flips if odd count of theta/phi indices among (a,b)
    {1.0, -1.0, -1.0, 1.0, 1.0, 1.0}};

//----------------------------------------------------------------------------------------
// Compute the per-component sign array from component_groups.

std::vector<Real> ComputeSignArray(const CommSpec &spec, FlipContext ctx) {
  const int nvar = spec.nvar;
  std::vector<Real> signs(nvar, 1.0);

  // Empty component_groups = all scalar-like, no flips.
  if (spec.component_groups.empty())
    return signs;

  const int ictx = static_cast<int>(ctx);
  int offset = 0;

  for (const auto &grp : spec.component_groups) {
    switch (grp.type) {
    case GeomType::Scalar:
      // Scalars never flip. Advance offset by count.
      offset += grp.count;
      break;

    case GeomType::Vector: {
      // Each group of 3 components is one vector.
      // If count is not a multiple of 3, treat complete triplets only;
      // remaining components are treated as scalars (no flip).
      int nvec = grp.count / 3;
      int remainder = grp.count % 3;
      for (int v = 0; v < nvec; ++v) {
        for (int c = 0; c < 3; ++c) {
          signs[offset + v * 3 + c] = kVectorSigns[ictx][c];
        }
      }
      offset += nvec * 3 + remainder;
      break;
    }

    case GeomType::SymTensor: {
      // Each group of 6 components is one symmetric 3-tensor.
      int ntens = grp.count / 6;
      int remainder = grp.count % 6;
      for (int t = 0; t < ntens; ++t) {
        for (int c = 0; c < 6; ++c) {
          signs[offset + t * 6 + c] = kSymTensorSigns[ictx][c];
        }
      }
      offset += ntens * 6 + remainder;
      break;
    }
    }
  }

  return signs;
}

//----------------------------------------------------------------------------------------
// Apply sign flips in-place to an already-unpacked ghost-zone region.
// This is a simple element-wise multiply: var(n,k,j,i) *= signs[n].
// Only processes components whose sign is -1 (skip +1 for efficiency).

void ApplyParitySigns(AthenaArray<Real> &var, int nvar,
                      const std::vector<Real> &signs, int si, int ei, int sj,
                      int ej, int sk, int ek) {
  for (int n = 0; n < nvar; ++n) {
    if (signs[n] > 0.0)
      continue; // no flip needed
    for (int k = sk; k <= ek; ++k) {
      for (int j = sj; j <= ej; ++j) {
#pragma omp simd
        for (int i = si; i <= ei; ++i) {
          var(n, k, j, i) = -var(n, k, j, i);
        }
      }
    }
  }
}

} // namespace comm
