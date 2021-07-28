#ifndef INTERP_INTERGRID_HPP_
#define INTERP_INTERGRID_HPP_
//! \file interp_intergrid.hpp
//  \brief prototypes of utility functions to pack/unpack buffers

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "interp_univariate.hpp"

namespace InterpIntergrid {

  void var_map_VC2CC(
    const AthenaArray<Real> & var_vc,
    AthenaArray<Real> & var_cc,
    const int dim,
    const int ll, const int lu,
    const int ml, const int mu,
    const int nl, const int nu
  );

  void var_map_CC2VC(
    const AthenaArray<Real> & var_cc,
    AthenaArray<Real> & var_vc,
    const int dim,
    const int ll, const int lu,
    const int ml, const int mu,
    const int nl, const int nu
  );

} // namespace InterpIntergrid
#endif // INTERP_INTERGRID_HPP_
