#ifndef ATHENA_ALIASES_HPP
#define ATHENA_ALIASES_HPP

// C++ headers

// External libraries


// Athena++ headers
#include "athena.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "athena_arrays.hpp"
#include "athena_tensor.hpp"

// ----------------------------------------------------------------------------
// Collect some typedefs to make everything less awful
namespace gra::aliases {

// for readability
static const int D = NDIM + 1;
static const int N = NDIM;

// Data structures ------------------------------------------------------------
typedef AthenaArray< Real>                         AA;

// BD: TODO - Replace AT_C_sca -> AT_N_sca
// scalars feature common treatment as \Sigma x {t*} so container at fixed t*
typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_C_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;

typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

// (V)ector slot, (S2)ymmetric pair, (T2) tensor pair
typedef AthenaTensor<Real, TensorSymm::NONE,  N, 2> AT_N_T2;
typedef AthenaTensor<Real, TensorSymm::SYM2,  N, 3> AT_N_VS2;
typedef AthenaTensor<Real, TensorSymm::ISYM2, N, 3> AT_N_S2V;

typedef AthenaTensor<Real, TensorSymm::SYM2,  N, 4> AT_N_T2S2;
typedef AthenaTensor<Real, TensorSymm::SYM22, N, 4> AT_N_S2S2;

// For hydro(NHYDRO)-(magnetic field NWAVE)-variable vector
typedef AthenaTensor<Real, TensorSymm::NONE, NWAVE,    1> AT_H_vec;
// For passive scalars
typedef AthenaTensor<Real, TensorSymm::NONE, NSCALARS, 1> AT_S_vec;

// Ambient quantities
typedef AthenaTensor<Real, TensorSymm::NONE, D, 1> AT_D_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, D, 2> AT_D_sym;
typedef AthenaTensor<Real, TensorSymm::SYM2, D, 3> AT_D_VS2;

}  // namespace gra::aliases

// Looping constructs through macros ------------------------------------------

// Coordinate class internal loops for e.g. AddCoordTermsDivergence
#define CC_PCO_ILOOP1(i)				           \
  _Pragma("omp simd")		  		             \
  for (int i=il; i<=iu; ++i)


#define CC_ILOOP2(k,j)				               \
  for (int k=pmb->ks; k<=pmb->ke; ++k)       \
  for (int j=pmb->js; j<=pmb->je; ++j)

#define CC_ILOOP1(i)				                 \
  _Pragma("omp simd")				                 \
  for (int i=pmb->is; i<=pmb->ie; ++i)

#define CC_ILOOP3(k,j,i)				             \
  CC_ILOOP2(k,j)                             \
  CC_ILOOP1(i)

#endif // ATHENA_ALIASES_HPP

//
// :D
//
