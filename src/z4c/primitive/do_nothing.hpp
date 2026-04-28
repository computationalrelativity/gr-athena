#ifndef DO_NOTHING_HPP
#define DO_NOTHING_HPP
//! \file do_nothing.hpp
//  \brief Describes an error policy that basically does nothing.
//
//  Since this policy does nothing, it should really only be used for testing
//  purposes.

#include "../../athena.hpp"
#include "error_policy_interface.hpp"
#include "ps_error.hpp"

namespace Primitive
{

class DoNothing : public ErrorPolicyInterface
{
  protected:
  /// Constructor
  DoNothing();

  /// Floor for primitive variables
  bool PrimitiveFloor(Real& n, Real v[3], Real& T, Real* Y, int n_species);

  /// Floor for conserved variables
  bool ConservedFloor(Real& D,
                      Real Sd[3],
                      Real& tau,
                      Real* Y,
                      Real D_floor,
                      Real tau_floor,
                      Real tau_abs_floor,
                      int n_species);

  /// Response to excess magnetization
  Error MagnetizationResponse(Real& bsq, Real b_u[3]);

  /// Policy for resetting density
  bool DensityLimits(Real& n, Real n_min, Real n_max);

  /// Policy for resetting temperature
  bool TemperatureLimits(Real& T, Real T_min, Real T_max);

  /// Policy for resetting species fractions
  bool SpeciesLimits(Real* Y, Real* Y_min, Real* Y_max, int n_species);

  /// Policy for resetting pressure
  bool PressureLimits(Real& P, Real P_min, Real P_max);

  /// Policy for resetting energy density
  bool EnergyLimits(Real& e, Real e_min, Real e_max);

  /// Policy for dealing with failed points
  bool FailureResponse(Real prim[NPRIM], int n_species);

  /// Policy for capping momentum to a physical range
  bool MomentumLimits(Real Sd[3], Real Ssq, Real Ssq_max);
};

}  // namespace Primitive

#endif
