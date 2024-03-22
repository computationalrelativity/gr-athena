#ifndef M1_UTILS_HPP
#define M1_UTILS_HPP

// c++
// ...

// Athena++ classes headers
// #include "../athena.hpp"
// #include "../athena_arrays.hpp"
// #include "../athena_tensor.hpp"
// #include "../mesh/mesh.hpp"
#include "../utils/linear_algebra.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"

// ============================================================================
namespace M1::Assemble {
// ============================================================================

inline void st_beta_u_(
  AT_D_vec _st_beta_u_,
  AT_N_vec _sp_beta_u,
  const int k, const int j,
  const int il, const int iu)
{
  for (int i=il; i<=iu; ++i)
  {
    _st_beta_u_(0,i) = 0;
  }


  for (int b=0; b<D-1; ++b)  // spatial ranges
  for (int i=il; i<=iu; ++i)
  {
    _st_beta_u_(b+1,i) = _sp_beta_u(b,k,j,i);
  }
}

// ============================================================================
}  // M1::Assemble
// ============================================================================

#endif // M1_UTILS_HPP

