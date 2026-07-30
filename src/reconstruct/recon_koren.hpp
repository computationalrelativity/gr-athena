#ifndef RECON_KOREN_HPP_
#define RECON_KOREN_HPP_

#include "../athena.hpp"

namespace reconstruction::koren {

#pragma omp declare simd
void ReconstructLR(const Real a,
                   const Real b,
                   const Real c,
                   Real& uL,
                   Real& uR);

}  // namespace reconstruction::koren

#endif  // RECON_KOREN_HPP_
