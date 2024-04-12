// c++
// ...

// Athena++ headers
#include "m1.hpp"
#include "opacities/m1_opacities.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Opacity function scheduled in the tasklist
void M1::CalcOpacity(Real const dt, AthenaArray<Real> & u)
{

  switch (opt.opacity_variety)
  {
    case (opt_opacity_variety::none):
    {
      return;
    }
    case (opt_opacity_variety::zero):
    {
      Opacities::Zero(this);
      break;
    }
    default:
    {
      assert(0);
      std::exit(0);
    }
  }

}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//