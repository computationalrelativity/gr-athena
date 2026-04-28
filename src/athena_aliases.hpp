#ifndef ATHENA_ALIASES_HPP
#define ATHENA_ALIASES_HPP

// C++ headers

// External libraries

// Athena++ headers
#include <cstdint>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "athena_tensor.hpp"
#include "defs.hpp"
#include "globals.hpp"

// ----------------------------------------------------------------------------
// Collect some typedefs to make everything less awful
namespace gra::aliases
{

// for readability
inline constexpr int D = NDIM + 1;
inline constexpr int N = NDIM;

// Data structures ------------------------------------------------------------
typedef AthenaArray<Real> AA;
typedef AthenaArray<uint8_t> AA_B;

// BD: TODO - Replace AT_C_sca -> AT_N_sca
// scalars feature common treatment as \Sigma x {t*} so container at fixed t*
typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_C_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;

typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

// (V)ector slot, (S2)ymmetric pair, (T2) tensor pair
typedef AthenaTensor<Real, TensorSymm::NONE, N, 2> AT_N_T2;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 3> AT_N_VS2;
typedef AthenaTensor<Real, TensorSymm::ISYM2, N, 3> AT_N_S2V;

typedef AthenaTensor<Real, TensorSymm::SYM2, N, 4> AT_N_T2S2;
typedef AthenaTensor<Real, TensorSymm::SYM22, N, 4> AT_N_S2S2;

// For hydro(NHYDRO)-(magnetic field NWAVE)-variable vector
typedef AthenaTensor<Real, TensorSymm::NONE, NWAVE, 1> AT_H_vec;
// For passive scalars
typedef AthenaTensor<Real, TensorSymm::NONE, NSCALARS, 1> AT_S_vec;

// Ambient quantities
typedef AthenaTensor<Real, TensorSymm::NONE, D, 1> AT_D_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, D, 2> AT_D_sym;
typedef AthenaTensor<Real, TensorSymm::SYM2, D, 3> AT_D_VS2;

// Point-storage (stack-allocated single-point tensors) -----------------------
inline constexpr auto P = TensorStorage::Point;

typedef AthenaTensor<Real, TensorSymm::NONE, N, 0, P> ATP_N_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 1, P> ATP_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2, P> ATP_N_sym;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 2, P> ATP_N_T2;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 3, P> ATP_N_VS2;
typedef AthenaTensor<Real, TensorSymm::ISYM2, N, 3, P> ATP_N_S2V;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 4, P> ATP_N_T2S2;
typedef AthenaTensor<Real, TensorSymm::SYM22, N, 4, P> ATP_N_S2S2;

typedef AthenaTensor<Real, TensorSymm::NONE, D, 0, P> ATP_D_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, D, 1, P> ATP_D_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, D, 2, P> ATP_D_sym;
typedef AthenaTensor<Real, TensorSymm::SYM2, D, 3, P> ATP_D_VS2;

// (2-dim manifold) tensors: Grid and Point storage ---------------------------
inline constexpr int M = 2;

typedef AthenaTensor<Real, TensorSymm::NONE, M, 1> AT_M_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, M, 2> AT_M_sym;

typedef AthenaTensor<Real, TensorSymm::SYM2, M, 2, P> ATP_M_sym;
typedef AthenaTensor<Real, TensorSymm::SYM2, M, 3, P> ATP_M_VS2;

}  // namespace gra::aliases

// Looping constructs through macros ------------------------------------------

// Coordinate class internal loops for e.g. AddCoordTermsDivergence
#define CC_PCO_ILOOP1(i) _Pragma("omp simd") for (int i = il; i <= iu; ++i)

#define CC_ILOOP2(k, j)                    \
  for (int k = pmb->ks; k <= pmb->ke; ++k) \
    for (int j = pmb->js; j <= pmb->je; ++j)

#define CC_ILOOP1(i) \
  _Pragma("omp simd") for (int i = pmb->is; i <= pmb->ie; ++i)

#define CC_ILOOP3(k, j, i) \
  CC_ILOOP2(k, j)          \
  CC_ILOOP1(i)

#define CC_GLOOP2(k, j)                  \
  for (int k = 0; k < pmb->ncells3; ++k) \
    for (int j = 0; j < pmb->ncells2; ++j)

#define CC_GLOOP1(i) _Pragma("omp simd") for (int i = 0; i < pmb->ncells1; ++i)

#define CC_GLOOP3(k, j, i) \
  CC_GLOOP2(k, j)          \
  CC_GLOOP1(i)

// No-SIMD (NS) variants - use when the loop body contains scalar reductions
// or other patterns incompatible with #pragma omp simd.
#define CC_NS_ILOOP1(i) for (int i = pmb->is; i <= pmb->ie; ++i)

#define CC_NS_ILOOP3(k, j, i) \
  CC_ILOOP2(k, j)             \
  CC_NS_ILOOP1(i)

#define CC_NS_GLOOP1(i) for (int i = 0; i < pmb->ncells1; ++i)

#define CC_NS_GLOOP3(k, j, i) \
  CC_GLOOP2(k, j)             \
  CC_NS_GLOOP1(i)

#endif  // ATHENA_ALIASES_HPP

//
// :D
//
