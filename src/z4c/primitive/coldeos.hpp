#ifndef COLDEOS_HPP
#define COLDEOS_HPP

//! \file eos_cold.hpp
//  \brief Defines a cold slice equation of state.
//
//  EOS cold is effectively an interface that describes how to create a
//  cold slice equation of state.
//  It must be instantiated with an object implementing the following
//  protected functions:

//  And it must also have the following protected member variables
//  (available via EOSPolicyInterface):
//    const int n_species
//    Real mb
//    Real max_n
//    Real min_n
//    Real max_h
//    Real min_h
//    Real T
//    Real eos_units
//    Real code_untis

#include <cassert>
#include <cmath>

#include "../../athena.hpp"
#include "unit_system.hpp"

namespace Primitive {

enum class Error;

template <typename ColdEOSPolicy>
class ColdEOS : public ColdEOSPolicy {
  private:
    // ColdEOSPolicy member functions

    // EOSPolicy member variables
    // The number of particle species used by the EOS.
    using ColdEOSPolicy::n_species;
    // Maximum density
    using ColdEOSPolicy::max_n;
    // Minimum density
    using ColdEOSPolicy::min_n;
    // Temperature of the slice
    using ColdEOSPolicy::T;
    // Code unit system
    using ColdEOSPolicy::code_units;
    // EOS unit system
    using ColdEOSPolicy::eos_units;

    using ColdEOSPolicy::Pressure;
    using ColdEOSPolicy::Energy;
    using ColdEOSPolicy::dPdn;
    using ColdEOSPolicy::SpecificInternalEnergy;
    using ColdEOSPolicy::Y;
    using ColdEOSPolicy::Enthalpy;

    // density floor
    Real density_floor;

  public:
    // The baryon mass
    using ColdEOSPolicy::mb;

    //! \fn EOS()
    //  \brief Constructor for the EOS. It sets the code units.
    ColdEOS() {
      code_units = eos_units;
    }

    //! \fn Real GetPressure(Real rho)
    //
    //  \param[in] rho  The density in code units.
    //  \return The pressure according to the cold EOS slice.
    Real GetPressure(Real rho) {
      Real n = rho/GetBaryonMass();
      return Pressure(n) * eos_units->PressureConversion(*code_units);
    }

    //! \fn Real GetEnergy(Real rho)
    //
    //  \param[in] rho  The density
    //  \return The energy density according to the cold EOS slice.
    Real GetEnergy(Real rho) {
      Real n = rho/GetBaryonMass();
      return Energy(n) * eos_units->PressureConversion(*code_units);
    }

    //! \fn Real GetdPdrho(Real rho)
    //
    //  \param[in] rho  The density
    //  \return The derivative of the pressure wrt. the density according to the cold EOS slice.
    Real GetdPdrho(Real rho) {
      Real n = rho/GetBaryonMass();
      return dPdn(n)
        * eos_units->PressureConversion(*code_units)
        / GetBaryonMass();
    }

    //! \fn Real GetSpecificInternalEnergy(Real rho)
    //
    //  \param[in] rho  The density
    //  \return The specific internal energy according to the cold EOS slice.
    Real GetSpecificInternalEnergy(Real rho) {
      Real n = rho/GetBaryonMass();
      return SpecificInternalEnergy(n)
        * eos_units->EnergyConversion(*code_units)
        / eos_units->MassConversion(*code_units);
    }

    //! \fn Real GetY(Real rho)
    //
    //  \param[in] rho  The density
    //  \param[in] iy   The index of the species
    //  \return The electron fraction according to the cold EOS slice.
  Real GetY(Real rho, int iy) {
      Real n = rho/GetBaryonMass();
      return Y(n, iy);
    }

    //! \fn Real GetEnthalpy(Real rho)
    //
    //  \param[in] rho  The density
    //  \return The enthalpy according to the cold EOS slice.
    Real GetEnthalpy(Real rho) {
      Real n = rho/GetBaryonMass();
      return Enthalpy(n)/mb
        * eos_units->EnergyConversion(*code_units)
        / eos_units->MassConversion(*code_units);
    }

    //! \fn Real GetTemperature() const
    //  \brief Get the temperature of the slice.
    inline Real GetTemperature() const {
      return T*eos_units->TemperatureConversion(*code_units);
    }

    //! \brief Get the maximum number density (in EOS units) permitted by the EOS.
    inline Real GetMaximumDensity() const {
      return max_n;
    }

    //! \brief Get the minimum number density (in EOS units) permitted by the EOS.
    inline Real GetMinimumDensity() const {
      return min_n;
    }

    //! \fn Real GetBaryonMass() const
    //  \brief Get the baryon mass used by this EOS. Note that
    //         this factor also converts the density.
    inline Real GetBaryonMass() const {
      return mb*eos_units->MassConversion(*code_units)*eos_units->DensityConversion(*code_units);
    }

    inline void SetCodeUnitSystem(UnitSystem* units) {
      code_units = units;
    }

    inline UnitSystem* GetCodeUnitSystem() const {
      return code_units;
    }

    inline UnitSystem* GetEOSUnitSystem() const {
      return eos_units;
    }

  inline void SetDensityFloor(Real d) {
    density_floor = d;
    if (density_floor < min_n*GetBaryonMass()) {
      throw std::invalid_argument("Density floor is below the minimum density.");
    }
  }

  inline Real GetDensityFloor() const {
    return density_floor;
  }
};

} // namespace

#endif
