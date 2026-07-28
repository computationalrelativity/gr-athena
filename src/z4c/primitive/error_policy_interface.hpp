#ifndef ERROR_POLICY_INTERFACE_HPP
#define ERROR_POLICY_INTERFACE_HPP
//! \file error_policy_interface.hpp
//  \brief Defines a class that provides all the basic members
//         needed by an ErrorPolicy.
//
//  It cannot be instantiated and, in fact, has no purpose in
//  being instantiated. It literally just provides member
//  variables for an ErrorPolicy;

#include <limits>

#include "../../athena.hpp"

namespace Primitive {

class ErrorPolicyInterface {
  protected:
    ErrorPolicyInterface() = default;
    ~ErrorPolicyInterface() = default;

    // Default member initializers are the last line of defense: every
    // policy constructor is expected to set its flags explicitly, but a
    // policy that forgets one (as ResetFloorTransition did with
    // limit_momenta) must not read stack garbage.
    Real n_atm = 1e-10;
    Real n_threshold = 1.0;
    Real T_atm = 1e-10;
    Real Y_atm[MAX_SPECIES] = {0.0};
    Real v_max = 1.0 - 1e-15;
    Real max_bsq = std::numeric_limits<Real>::max();
    bool fail_conserved_floor = false;
    bool fail_primitive_floor = false;
    bool adjust_conserved = false;
    bool limit_momenta = false;
};

} // namespace

#endif
