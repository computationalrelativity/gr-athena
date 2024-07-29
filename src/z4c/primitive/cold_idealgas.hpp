#ifndef COLDEOS_POLYTROPE_H
#define COLDEOS_POLYTROPE_H

//! \file eos_polytrope.hpp
//  \brief Defines a polytrope for the initialization of an equation of state.

#include <cmath>
#include <limits>
#include <cstdio>

#include "../../athena.hpp"
#include "unit_system.hpp"

namespace Primitive {

class Polytrope {
  protected:
    /// Constructor
    Polytrope();

    /// Destructor
    ~Polytrope();

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

    /// Number of particle species
    int n_species;
    /// Baryon mass
    Real mb;
    /// maximum number density
    Real max_n;
    /// minimum number density
    Real min_n;
    /// temperature of the slice
    Real T;
    /// Code unit system
    UnitSystem* code_units;
    /// ColdEOS unit system
    UnitSystem* eos_units;

  public:
    /// Set the adiabatic index for the polytrope
    /// The range \f$1 < \gamma < 1\f$ is imposed. The lower
    /// constraint ensures that enthalpy is finite, and the upper
    /// bound keeps the sound speed causal.
    inline void SetGamma(Real g) {
      gamma = (g <= 1.0) ? 1.00001 : ((g >= 2.0) ? 1.99999 : g);
      gammam1 = gamma - 1.0;
    }

    /// Set the polytropic constant
    inline void SetK(Real k) {
      K = k;
    }

    /// Get the adiabatic index.
    inline Real GetGamma() const {
      return gamma;
    }

    /// Get the polytropic constant in EOS units.
    inline Real GetK() const {
      return K;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    void SetNSpecies(int n);

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return (K > 0) && (gamma > 1) && (gamma < 2);
    }

  private:
    /// Adiabatic index
    Real gamma;
    Real gammam1;
    // Polytropic constant
    Real K;
};

} // namespace Primitive

#endif
