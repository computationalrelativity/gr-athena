#ifndef M1_CALC_FLUXES_HPP
#define M1_CALC_FLUXES_HPP

// c++
// ...

// Athena++ classes headers
// #include "../athena.hpp"
// #include "../athena_arrays.hpp"
// #include "../athena_tensor.hpp"
// #include "../mesh/mesh.hpp"
// #include "../utils/linear_algebra.hpp"
#include "m1.hpp"
// #include "m1_containers.hpp"

// ============================================================================
namespace M1::Fluxes {
// ============================================================================

// Build the HO/LO limited fluxes for scalar fields
void ReconstructLimitedFlux(M1 * pm1,
                            const int dir,
                            const AT_C_sca & q,
                            const AT_C_sca & F,
                            const AT_C_sca & kap_a,
                            const AT_C_sca & kap_s,
                            const AT_C_sca & lambda,
                            AT_C_sca & Flux);

// Build the HO/LO limited fluxes for vector fields
void ReconstructLimitedFlux(M1 * pm1,
                            const int dir,
                            const AT_N_vec & q,
                            const AT_N_vec & F,
                            const AT_C_sca & kap_a,
                            const AT_C_sca & kap_s,
                            const AT_C_sca & lambda,
                            AT_N_vec & Flux);

// Directional scalar field (or vector cpt.) reconstruction
void ReconstructLimitedFluxX1(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AT_C_sca & Flux);

void ReconstructLimitedFluxX2(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AT_C_sca & Flux);

void ReconstructLimitedFluxX3(M1 * pm1,
                              const AT_C_sca & q,
                              const AT_C_sca & F,
                              const AT_C_sca & kap_a,
                              const AT_C_sca & kap_s,
                              const AT_C_sca & lambda,
                              AT_C_sca & Flux);

// ============================================================================
} // namespace M1::Fluxes
// ============================================================================


#endif // M1_CALC_FLUXES_HPP

