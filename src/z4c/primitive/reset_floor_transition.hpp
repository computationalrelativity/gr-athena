#ifndef RESET_FLOOR_TRANSITION_HPP
#define RESET_FLOOR_TRANSITION_HPP
//! \file reset_floor_transition.hpp
//  \brief Error floor for the two-table transition EOS.
//
//  Like ResetFloor (including its baryon-conserving c2p failure response),
//  but aware of the validity edges of the low-density (Helmholtz) and
//  high-density (compose) branches: PrimitiveFloor pushes states out of
//  the invalid wedge between the tables, and floors accumulate instead of
//  early-returning so several adjustments can apply in one call.

#include "../../athena.hpp"
#include "error_policy_interface.hpp"

namespace Primitive
{

enum class Error;

class ResetFloorTransition : public ErrorPolicyInterface
{
  protected:
  /// Constructor
  ResetFloorTransition();

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

  /// Cap momentum to enforce S^i S_i <= Ssq_max.
  bool MomentumLimits(Real Sd[3], Real Ssq, Real Ssq_max);

  public:
  /// Set the failure mode for conserved flooring
  inline void SetConservedFloorFailure(bool failure)
  {
    fail_conserved_floor = failure;
  }

  /// Set the failure mode for primitive flooring
  inline void SetPrimitiveFloorFailure(bool failure)
  {
    fail_primitive_floor = failure;
  }

  /// Set whether or not it's okay to adjust the conserved variables.
  inline void SetAdjustConserved(bool adjust)
  {
    adjust_conserved = adjust;
  }
  /// Set the individual boundaries for high and low density tables
  inline void SetTableBoundaries(Real ld_n, Real hd_n, Real ld_t, Real hd_t)
  {
    ld_n_max = ld_n;
    hd_n_min = hd_n;
    ld_t_max = ld_t;
    hd_t_min = hd_t;
  }

  private:
  Real ld_n_max, hd_n_min;
  Real ld_t_max, hd_t_min;
};

}  // namespace Primitive

#endif
