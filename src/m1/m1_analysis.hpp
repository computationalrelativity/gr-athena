#ifndef M1_ANALYSIS_HPP
#define M1_ANALYSIS_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"

// ============================================================================
namespace M1::Analysis {
// ============================================================================

void CalcRadFlux(MeshBlock * pmb);
void CalcNeutrinoDiagnostics(MeshBlock * pmb);

// ============================================================================
} // namespace M1::Analysis
// ============================================================================


#endif // M1_ANALYSIS_HPP

