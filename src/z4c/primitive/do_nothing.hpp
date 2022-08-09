#ifndef DO_NOTHING_HPP
#define DO_NOTHING_HPP
//! \file do_nothing.hpp
//  \brief Describes an error policy that basically does nothing.
//
//  Since this policy does nothing, it should really only be used for testing
//  purposes.

#include "../../athena.hpp"
#include "ps_error.hpp"
#include "error_policy_interface.hpp"

namespace Primitive {

class DoNothing : public ErrorPolicyInterface {
  protected:
    DoNothing() {
      fail_conserved_floor = false;
      fail_primitive_floor = false;
      adjust_conserved = false;
    }

    bool PrimitiveFloor(Real& n, Real v[3], Real& p, Real *Y, int n_species) {return false;}
    bool ConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y, Real D_floor, 
                        Real tau_floor, int n_species) {return false;}
    Error MagnetizationResponse(Real& bsq, Real b_u[3]) {return Error::SUCCESS;}
    void DensityLimits(Real& n, Real n_min, Real n_max) {return;}
    void TemperatureLimits(Real& T, Real T_min, Real T_max) {return;}
    void SpeciesLimits(Real *Y, Real *Y_min, Real *Y_max, int n_species) {return;}
    void PressureLimits(Real& P, Real P_min, Real P_max) {return;}
    void EnergyLimits(Real& e, Real e_min, Real e_max) {return;}
    bool FailureResponse(Real prim[NPRIM]) {return false;}
};

} // namespace

#endif
