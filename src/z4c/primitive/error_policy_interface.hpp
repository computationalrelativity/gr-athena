#ifndef ERROR_POLICY_INTERFACE_HPP
#define ERROR_POLICY_INTERFACE_HPP
//! \file error_policy_interface.hpp
//  \brief Defines a class that provides all the basic members
//         needed by an ErrorPolicy.
//
//  It cannot be instantiated and, in fact, has no purpose in
//  being instantiated. It literally just provides member
//  variables for an ErrorPolicy;

#include "../../athena.hpp"

namespace Primitive {

enum class RetainedFailureMode
{
  none,
  state,
  state_tau
};

enum class RetainedThermalMode
{
  preserved,
  projected,
  cold_fallback,
  invalid
};

struct RetainedThermalState
{
  Real T;
  Real e;
  Real P;
  RetainedThermalMode mode;
};

class ErrorPolicyInterface {
  protected:
    ErrorPolicyInterface() = default;
    ~ErrorPolicyInterface() = default;

    static constexpr RetainedFailureMode retained_failure_mode =
      RetainedFailureMode::none;

    bool RetainedFailureResponse(Real[NPRIM],
                                 Real,
                                 Real,
                                 const Real*,
                                 int)
    {
      return false;
    }

    Real n_atm;
    Real n_threshold;
    Real T_atm;
    Real Y_atm[MAX_SPECIES];
    Real v_max;
    Real max_bsq;
    bool fail_conserved_floor;
    bool fail_primitive_floor;
    bool adjust_conserved;
    bool limit_momenta;
};

} // namespace

#endif
