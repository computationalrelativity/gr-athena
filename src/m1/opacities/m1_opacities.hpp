#ifndef M1_OPACITIES_HPP
#define M1_OPACITIES_HPP

// c++
// ...

// Athena++ classes headers
#include "../m1.hpp"
#include "../m1_containers.hpp"
#include "../m1_macro.hpp"

// ============================================================================
namespace M1::Opacities {
// ============================================================================

inline void Zero(M1 * pm1)
{
  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    pm1->radmat.sc_eta_0(  ix_g,ix_s).ZeroClear();
    pm1->radmat.sc_kap_a_0(ix_g,ix_s).ZeroClear();

    pm1->radmat.sc_eta(  ix_g,ix_s).ZeroClear();
    pm1->radmat.sc_kap_a(ix_g,ix_s).ZeroClear();
    pm1->radmat.sc_kap_s(ix_g,ix_s).ZeroClear();
  }
}


// ============================================================================
}  // M1::Opacities
// ============================================================================

#endif // M1_OPACITIES_HPP

