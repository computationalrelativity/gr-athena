#ifndef WEAKRATES_CONSTANTS_H
#define WEAKRATES_CONSTANTS_H

#include "../../../athena.hpp"
#include "../common/eos.hpp"

// Shared physical constants, helper functions, and bounds-check utility
// used by both the emission and opacity modules.

namespace M1::Opacities::WeakRates::Constants
{

// ---------------------------------------------------------------------------
// Physical constants (CGS / MeV)
// ---------------------------------------------------------------------------
inline constexpr Real clight     = 2.99792458e10;   // speed of light [cm/s]
inline constexpr Real mev_to_erg = 1.60217733e-6;   // MeV -> erg
inline constexpr Real hc_mevcm   = 1.23984172e-10;  // hc [MeV cm]
inline constexpr Real pi         = 3.14159265358979323846;

inline constexpr Real me_erg  = 8.187108692567103e-07; // electron mass [erg]
inline constexpr Real sigma_0 = 1.76e-44;         // weak cross-section [cm^2]
inline constexpr Real alpha   = 1.23e0;           // axial-vector coupling
inline constexpr Real Cv      = 0.5 + 2.0 * 0.23; // vector coupling
inline constexpr Real Ca      = 0.5;              // axial coupling
inline constexpr Real gamma_0 = 5.565e-2;         // plasmon parameter
inline constexpr Real fsc     = 1.0 / 137.036;    // fine-structure constant

// ---------------------------------------------------------------------------
// Inline helpers replacing WR_SQR / WR_CUBE / WR_QUAD macros
// ---------------------------------------------------------------------------
inline constexpr Real WR_POW2(Real x)
{
  return x * x;
}
inline constexpr Real WR_POW3(Real x)
{
  return x * x * x;
}
inline constexpr Real WR_POW4(Real x)
{
  return x * x * x * x;
}

// ---------------------------------------------------------------------------
// Common bounds-check + table-limit enforcement
//
// Returns:
//   0  : inputs are within bounds and table limits applied successfully
//  -2  : inputs below the density / temperature floor (caller should zero
//  outputs) -1  : EoS table-limit enforcement flagged an error
// ---------------------------------------------------------------------------
inline int CheckAndApplyBounds(Real& rho,
                               Real& temp,
                               Real& ye,
                               Real rho_min,
                               Real temp_min,
                               Common::EoS::EoSWrapper* EoS)
{
  if (rho < rho_min || temp < temp_min)
    return -2;
  bool err = EoS->ApplyTableLimits(rho, temp, ye);
  if (err)
    return -1;
  return 0;
}

}  // namespace M1::Opacities::WeakRates::Constants

#endif  // WEAKRATES_CONSTANTS_H
