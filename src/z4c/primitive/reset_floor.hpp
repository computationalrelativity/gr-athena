#ifndef RESET_FLOOR_HPP
#define RESET_FLOOR_HPP
//! \file reset_floor.hpp
//  \brief Describes an error floor that simply resets nonphysical values.
//
//  If the density or pressure fall below the atmosphere, they get floored.
//  We impose similar limits for D and tau. If the density is floored,
//  the velocity is zeroed out and the pressure is also reset to the floor.
//  If the pressure is floored, all other quantities are ignored.
//  If the primitive solve fails, all points are set to floor.

#include "../../athena.hpp"
#include "error_policy_interface.hpp"

namespace Primitive {

enum class Error;

class ResetFloor : public ErrorPolicyInterface {
  protected:
    /// Constructor
    ResetFloor();

    /// Floor for primitive variables
    bool PrimitiveFloor(Real& n, Real v[3], Real& p, Real *Y, int n_species);

    /// Floor for conserved variables
    bool ConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y, Real D_floor, 
                        Real tau_floor, int n_species);

    /// Response to excess magnetization
    Error MagnetizationResponse(Real& bsq, Real b_u[3]);

    /// Policy for resetting density
    void DensityLimits(Real& n, Real n_min, Real n_max);

    /// Policy for resetting temperature
    void TemperatureLimits(Real& T, Real T_min, Real T_max);

    /// Policy for resetting species fractions
    void SpeciesLimits(Real* Y, Real* Y_min, Real* Y_max, int n_species);

    /// Policy for resetting pressure
    void PressureLimits(Real& P, Real P_min, Real P_max);

    /// Policy for resetting energy density
    void EnergyLimits(Real& e, Real e_min, Real e_max);

    /// Policy for dealing with failed points
    bool FailureResponse(Real prim[NPRIM]);

  public:
    /// Set the failure mode for conserved flooring
    inline void SetConservedFloorFailure(bool failure) {
      fail_conserved_floor = failure;
    }

    /// Set the failure mode for primitive flooring
    inline void SetPrimitiveFloorFailure(bool failure) {
      fail_primitive_floor = failure;
    }

    /// Set whether or not it's okay to adjust the conserved variables.
    inline void SetAdjustConserved(bool adjust) {
      adjust_conserved = adjust;
    }
};

} // namespace

#endif
