#ifndef COLDEOS_PWP_H
#define COLDEOS_PWP_H

//! \file eos_pwp.hpp
//  \brief Defines a piecewise polytropic ColdEOS class.

#include <cmath>

#include "../../athena.hpp"
#include "coldeos_policy_interface.hpp"
#include "piecewise_polytrope.hpp"

namespace Primitive {

  class ColdPiecewisePolytrope: public PiecewisePolytrope {

  protected:

    /// Constructor
    ColdPiecewisePolytrope();

    /// Destructor
    ~ColdPiecewisePolytrope();

    /// Calculate the pressure from the number density
    Real Pressure(Real n);

    /// Calculate the energy from the number density
    Real Energy(Real n);

    /// Calculate the derivative of the pressure wrt. the numberdensity from the number density
    Real dPdn(Real n);

    /// Calculate the specific internal energy from the number density
    Real SpecificInternalEnergy(Real n);

    /// Calculate the abundance of species iy from the number density
    Real Y(Real n, int iy);

    /// Calculate the specific enthalpy from the number density
    Real Enthalpy(Real n);

    /// temperature of the slice
    Real T;

  public:

    /// Set the Temperature
    inline void SetTemperature(Real t) {
      T = t;
    }

    /// Get the Temperature
    inline Real GetTemperature() {
      return T * eos_units->TemperatureConversion(*code_units);
    }
};

} // namespace Primitive

#endif
