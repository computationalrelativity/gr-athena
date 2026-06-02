#ifndef Z4C_AUX_PARITY_HPP_
#define Z4C_AUX_PARITY_HPP_

//========================================================================================
// GR-Athena++
//========================================================================================
//! \file z4c_aux_parity.hpp
//  \brief Per-component parity signs for N_AUX=30 ADM derivative components
//  (dalpha_d, dbeta_du, dg_ddd) under Cartesian reflecting BCs.
//
//  These don't fit GeomType::Scalar/Vector/SymTensor because:
//    - dbeta_du is rank-(1,1) with asymmetric sign patterns
//    - dg_ddd is rank-3 where the sign depends on all three indices
//
//  Component layout (N_AUX=30):
//    0-2:   dalpha_d  - d_i alpha                       (co-vector)
//    3-11:  dbeta_du  - d_i beta^j, stored dbetaj_i     (rank-(1,1))
//    12-29: dg_ddd    - d_k g_ab, stored dgab_k         (rank-3 covariant)
//           ab in {xx,xy,xz,yy,yz,zz} order within each k-slice
//
//  Parity rule: (-1)^(count of flipping indices).
//    ReflectX1 = flip if odd count of x-indices
//    ReflectX2 = flip if odd count of y-indices
//    ReflectX3 = flip if odd count of z-indices
//
//  Polar (FlipContext 3) is NOT populated — those entries remain empty,
//  and the existing all-even path applies unchanged.

#include <vector>

#include "../athena.hpp"
#include "z4c.hpp"

inline void PopulateAuxParitySigns(std::vector<Real> (&signs)[4])
{
  for (int ctx = 0; ctx < 3; ++ctx)
    signs[ctx].assign(Z4c::N_AUX, 1.0);

  auto flips = [](int ctx, int idx) {
    return (ctx == 0 && idx == 0) || (ctx == 1 && idx == 1) ||
           (ctx == 2 && idx == 2);
  };

  for (int a = 0; a < 3; ++a) {
    const int c = Z4c::I_AUX_dalpha_x + a;
    for (int ctx = 0; ctx < 3; ++ctx)
      signs[ctx][c] = flips(ctx, a) ? -1.0 : 1.0;
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const int c = Z4c::I_AUX_dbetax_x + j + 3 * i;
      for (int ctx = 0; ctx < 3; ++ctx)
        signs[ctx][c] =
          ((flips(ctx, i) + flips(ctx, j)) % 2) ? -1.0 : 1.0;
    }
  static const int ab_idx[6][2] = {
    { 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 1 }, { 1, 2 }, { 2, 2 }
  };
  for (int k = 0; k < 3; ++k)
    for (int ab = 0; ab < 6; ++ab) {
      const int c = Z4c::I_AUX_dgxx_x + ab + 6 * k;
      for (int ctx = 0; ctx < 3; ++ctx)
        signs[ctx][c] =
          ((flips(ctx, k) + flips(ctx, ab_idx[ab][0]) +
            flips(ctx, ab_idx[ab][1])) %
           2)
            ? -1.0
            : 1.0;
    }
}

#endif  // Z4C_AUX_PARITY_HPP_
