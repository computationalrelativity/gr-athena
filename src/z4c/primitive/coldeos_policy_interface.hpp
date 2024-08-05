#ifndef COLDEOS_POLICY_INTERFACE_HPP
#define COLDEOS_POLICY_INTERFACE_HPP

//! \file eos_policy_interface.hpp
//  \brief Defines a class that provides all the common class
//         member variables needed by an ColdEOSPolicy.

#include "../../athena.hpp"


namespace Primitive {

struct UnitSystem;

class ColdEOSPolicyInterface {
  protected:
    ColdEOSPolicyInterface() = default;
    ~ColdEOSPolicyInterface() = default;

    /// Number of particle species
    int n_species;
    /// Baryon mass
    Real mb;
    /// maximum number density
    Real max_n;
    /// minimum number density
    Real min_n;
    /// minimum enthalpy
    Real min_h;
    /// maximum enthalpy
    Real max_h;
    /// temperature of the slice
    Real T;
    /// Code unit system
    UnitSystem* code_units;
    /// ColdEOS unit system
    UnitSystem* eos_units;
};

} // namespace

#endif
