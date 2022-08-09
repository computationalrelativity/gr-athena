#ifndef EOS_HPP
#define EOS_HPP

//! \file eos.hpp
//  \brief Defines an equation of state.
//
//  EOS is effectively an interface that describes how to create an
//  equation of state. It must be instantiated with an object implementing
//  the following protected functions:
//    Real Temperature(Real n, Real e, Real *Y)
//    Real TemperatureFromP(Real n, Real p, Real *Y)
//    Real Energy(Real n, Real T, Real *Y)
//    Real Pressure(Real n, Real T, Real *Y)
//    Real Entropy(Real n, Real T, Real *Y)
//    Real Enthalpy(Real n, Real T, Real *Y)
//    Real MinimumEnthalpy()
//    Real SoundSpeed(Real n, Real T, Real *Y)
//    Real SpecificInternalEnergy(Real n, Real T, Real *Y)
//    Real MinimumPressure(Real n, Real *Y)
//    Real MaximumPressure(Real n, Real *Y)
//    Real MinimumEnergy(Real n, Real *Y)
//    Real MaximumEnergy(Real n, Real *Y)
//  And it must also have the following protected member variables
//  (available via EOSPolicyInterface):
//    const int n_species
//    Real mb
//    Real max_rho
//    Real min_rho
//
//  It must also take an object for an error policy that implements the
//  following functions:
//    bool PrimitiveFloor(Real& n, Real& v[3], Real& p)
//    bool ConservedFloor(Real& D, Real& Sd[3], Real& tau, Real& Bu[3])
//    void DensityLimits(Real& n, Real n_min, Real n_max);
//    void TemperatureLimits(Real& T, Real T_min, Real T_max);
//    void SpeciesLimits(Real* Y, Real* Y_min, Real* Y_max, int n_species);
//    void PressureLimits(Real& P, Real P_min, Real P_max);
//    void EnergyLimits(Real& e, Real e_min, Real e_max);
//    void FailureResponse(Real prim[NPRIM])
//  And the following protected variables (available via
//  ErrorPolicyInterface):
//    Real n_atm
//    Real p_atm
//    Real v_max
//    Real max_bsq_field
//    bool fail_conserved_floor
//    bool fail_primitive_floor
//    bool adjust_conserved

#include <limits>
#include <cassert>

#include "../../athena.hpp"
#include "unit_system.hpp"

namespace Primitive {

enum class Error;

template <typename EOSPolicy, typename ErrorPolicy>
class EOS : public EOSPolicy, public ErrorPolicy {
  private:
    // EOSPolicy member functions
    using EOSPolicy::TemperatureFromE;
    using EOSPolicy::TemperatureFromP;
    using EOSPolicy::Energy;
    using EOSPolicy::Pressure;
    using EOSPolicy::Entropy;
    using EOSPolicy::Enthalpy;
    using EOSPolicy::SoundSpeed;
    using EOSPolicy::SpecificInternalEnergy;
    using EOSPolicy::MinimumEnthalpy;
    using EOSPolicy::MinimumPressure;
    using EOSPolicy::MaximumPressure;
    using EOSPolicy::MinimumEnergy;
    using EOSPolicy::MaximumEnergy;

    // EOSPolicy member variables
    // The number of particle species used by the EOS.
    using EOSPolicy::n_species;
    // The baryon mass
    using EOSPolicy::mb;
    // Maximum density
    using EOSPolicy::max_n;
    // Minimum density
    using EOSPolicy::min_n;
    // Maximum temperature
    using EOSPolicy::max_T;
    // Minimum temperature
    using EOSPolicy::min_T;
    // Maximum Y
    using EOSPolicy::max_Y;
    // Minimum Y
    using EOSPolicy::min_Y;
    // Code unit system
    using EOSPolicy::code_units;
    // EOS unit system
    using EOSPolicy::eos_units;

    // ErrorPolicy member functions
    using ErrorPolicy::PrimitiveFloor;
    using ErrorPolicy::ConservedFloor;
    using ErrorPolicy::MagnetizationResponse;
    using ErrorPolicy::DensityLimits;
    using ErrorPolicy::TemperatureLimits;
    using ErrorPolicy::SpeciesLimits;
    using ErrorPolicy::PressureLimits;
    using ErrorPolicy::EnergyLimits;
    using ErrorPolicy::FailureResponse;

    // ErrorPolicy member variables
    using ErrorPolicy::n_atm;
    using ErrorPolicy::n_threshold;
    using ErrorPolicy::p_atm;
    using ErrorPolicy::v_max;
    using ErrorPolicy::fail_conserved_floor;
    using ErrorPolicy::fail_primitive_floor;
    using ErrorPolicy::adjust_conserved;
    using ErrorPolicy::max_bsq;

  public:
    //! \fn EOS()
    //  \brief Constructor for the EOS. It sets a default value for the floor.
    //
    //  n_atm gets fixed to 1e-10, and T_atm is set to 1.0. v_max is fixed to 
    //  1.0e - 1e15.
    EOS() {
      n_atm = 1e-10;
      n_threshold = 1.0;
      p_atm = 1e-10;
      v_max = 1.0 - 1e-15;
      max_bsq = std::numeric_limits<Real>::max();
      code_units = eos_units;
    }

    //! \fn Real GetTemperatureFromE(Real n, Real e, Real *Y)
    //  \brief Calculate the temperature from number density, energy density, and
    //         particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] e  The energy density
    //  \param[in] Y  An array of particle fractions, expected to be of size n_species.
    //  \return The temperature according to the EOS.
    inline Real GetTemperatureFromE(Real n, Real e, Real *Y) {
      return TemperatureFromE(n, e*code_units->PressureConversion(*eos_units), Y) *
             eos_units->TemperatureConversion(*code_units);
    }

    //! \fn Real GetTemperatureFromP(Real n, Real p, Real *Y)
    //  \brief Calculate the temperature from number density, pressure, and
    //         particle fractions.
    //  \param[in] n  The number density
    //  \param[in] p  The pressure
    //  \param[in] Y  An array of particle fractions, expected to be of size n_species.
    //  \return The temperature according to the EOS.
    inline Real GetTemperatureFromP(Real n, Real p, Real *Y) {
      return TemperatureFromP(n, p*code_units->PressureConversion(*eos_units), Y) *
             eos_units->TemperatureConversion(*code_units);
    }

    //! \fn Real GetEnergy(Real n, Real T, Real *Y)
    //  \brief Get the energy density from the number density, temperature, and
    //         particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The energy density according to the EOS.
    inline Real GetEnergy(Real n, Real T, Real *Y) {
      return Energy(n, T*code_units->TemperatureConversion(*eos_units), Y) *
             eos_units->PressureConversion(*code_units);
    }

    //! \fn Real GetPressure(Real n, Real T, Real *Y)
    //  \brief Get the pressure from the number density, temperature, and
    //         particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The pressure according to the EOS.
    inline Real GetPressure(Real n, Real T, Real *Y) {
      return Pressure(n, T*code_units->TemperatureConversion(*eos_units), Y) *
             eos_units->PressureConversion(*code_units);
    }

    //! \fn Real GetEntropy(Real n, Real T, Real *Y)
    //  \brief Get the entropy per mass from the number density, temperature,
    //         and particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The entropy per baryon for this EOS.
    inline Real GetEntropy(Real n, Real T, Real *Y) {
      return Entropy(n, T*code_units->TemperatureConversion(*eos_units), Y)/mb *
             eos_units->EntropyConversion(*code_units)/eos_units->MassConversion(*code_units);
    }

    //! \fn Real GetEnthalpy(Real n, Real T, Real *Y)
    //  \brief Get the enthalpy per mass from the number density, temperature,
    //         and particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The enthalpy per baryon for this EOS.
    inline Real GetEnthalpy(Real n, Real T, Real *Y) {
      return Enthalpy(n, T*code_units->TemperatureConversion(*eos_units), Y)/mb *
             (eos_units->EnergyConversion(*code_units)/eos_units->MassConversion(*code_units));
    }

    //! \fn Real GetMinimumEnthalpy()
    //  \brief Get the global minimum for enthalpy per mass from the EOS.
    //
    //  \return the minimum enthalpy per baryon.
    inline Real GetMinimumEnthalpy() {
      return MinimumEnthalpy()/mb *
             eos_units->EnergyConversion(*code_units)/eos_units->MassConversion(*code_units);
    }

    //! \fn Real GetSoundSpeed(Real n, Real T, Real *Y)
    //  \brief Get the sound speed from the number density, temperature, and 
    //         particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The sound speed for this EOS.
    inline Real GetSoundSpeed(Real n, Real T, Real *Y) {
      return SoundSpeed(n, T*code_units->TemperatureConversion(*eos_units), Y) *
             eos_units->VelocityConversion(*code_units);
    }

    //! \fn Real GetSpecificInternalEnergy(Real n, Real T, Real *Y)
    //  \brief Get the energy per mass from the number density, temperature,
    //         and particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The specific energy for the EOS.
    inline Real GetSpecificInternalEnergy(Real n, Real T, Real *Y) {
      return SpecificInternalEnergy(n, T*code_units->TemperatureConversion(*eos_units), Y) *
             eos_units->EnergyConversion(*code_units)/eos_units->MassConversion(*code_units);
    }

    //! \fn int Getn_species() const
    //  \brief Get the number of particle species in this EOS.
    inline int GetNSpecies() const {
      return n_species;
    }

    //! \fn Real GetBaryonMass() const
    //  \brief Get the baryon mass used by this EOS. Note that
    //         this factor also converts the density.
    inline Real GetBaryonMass() const {
      return mb*eos_units->MassConversion(*code_units)*eos_units->DensityConversion(*code_units);
    }

    //! \fn bool ApplyPrimitiveFloor(Real& n, Real& vu[3], Real& p, Real& T)
    //  \brief Apply the floor to the primitive variables (in code units).
    //
    //  \param[in,out] n   The number density
    //  \param[in,out] Wvu The velocity vector (contravariant)
    //  \param[in,out] p   The pressure
    //  \param[out]    T   The temperature
    //  \param[in]     Y   An array of size n_species of the particle fractions.
    //
    //  \return true if the primitives were adjusted, false otherwise.
    inline bool ApplyPrimitiveFloor(Real& n, Real Wvu[3], Real& p, Real& T, Real *Y) {
      bool result = PrimitiveFloor(n, Wvu, p);
      if (result) {
        T = TemperatureFromP(n, p*code_units->PressureConversion(*eos_units), Y) *
            eos_units->TemperatureConversion(*code_units);
      }
      return result;
    }

    //! \fn bool ApplyConservedFloor(Real& D, Real& Sd[3], Real& tau, Real& Bu[3])
    //  \brief Apply the floor to the conserved variables (in code units).
    //
    //  \param[in,out] D   The relativistic number density
    //  \param[in,out] Sd  The momentum density vector (covariant)
    //  \param[in,out] tau The tau variable (relativistic energy - D)
    //  \param[in,out] Bu  The magnetic field vector (contravariant)
    //  \param[in]     Y   An array of size_species of the particle fractions.
    //
    //  \return true if the conserved variables were adjusted, false otherwise.
    inline bool ApplyConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y) {
      return ConservedFloor(D, Sd, tau, n_atm*mb, GetTauFloor(D, Y));
    }

    //! \fn Real GetDensityFloor() const
    //  \brief Get the density floor used by the EOS ErrorPolicy.
    inline Real GetDensityFloor() const {
      return n_atm;
    }

    //! \fn Real GetPressureFloor() const
    //  \brief Get the pressure floor used by the EOS ErrorPolicy.
    inline Real GetPressureFloor() const {
      return p_atm;
    }

    //! \fn Real GetThreshold() const
    //  \brief Get the threshold factor used by the EOS ErrorPolicy.
    inline Real GetThreshold() const {
      return n_threshold;
    }

    //! \fn Real GetTauFloor() const
    //  \brief Get the tau floor used by the EOS ErrorPolicy based
    //         on the current particle composition.
    //
    //  \param[in] Y A n_species-sized array of particle fractions.
    inline Real GetTauFloor(Real D, Real *Y) {
      return GetEnergy(D/mb, min_T, Y)*eos_units->PressureConversion(*code_units) - D;
    }

    //! \fn void SetDensityFloor(Real floor)
    //  \brief Set the density floor used by the EOS ErrorPolicy.
    //         Also adjusts the pressure and tau floor to be consistent.
    inline void SetDensityFloor(Real floor) {
      n_atm = (floor >= 0.0) ? floor : 0.0;
    }

    //! \fn void SetPressureFloor(Real floor)
    //  \brief Set the pressure floor (in code units) used by the EOS ErrorPolicy.
    inline void SetPressureFloor(Real floor) {
      p_atm = (floor >= 0.0) ? floor : 0.0;
    }

    //! \fn void SetThreshold(Real threshold)
    //  \brief Set the threshold factor for the density floor.
    inline void SetThreshold(Real threshold) {
      threshold = (threshold >= 0.0) ? threshold : 0.0;
    }

    //! \fn Real GetMaxVelocity() const
    //  \brief Get the maximum velocity according to the ErrorPolicy.
    inline Real GetMaxVelocity() const {
      return v_max;
    }

    //! \fn void SetMaxVelocity(Real v)
    //  \brief Set the maximum velocity in the ErrorPolicy.
    //
    //  The velocity will be automatically restricted to the range [0,1 - 1e-15].
    //
    //  \param[in] v The maximum velocity
    inline void SetMaxVelocity(Real v) {
      v_max = (v >= 0) ? ((v <= 1.0-1e-15) ? v : 1.0e-15) : 0.0;
    }

    //! \brief Get the maximum number density (in EOS units) permitted by the EOS.
    inline Real GetMaximumDensity() const {
      return max_n;
    }

    //! \brief Get the minimum number density (in EOS units) permitted by the EOS.
    inline Real GetMinimumDensity() const {
      return min_n;
    }

    //! \brief Get the maximum temperature  (in EOS units) permitted by the EOS.
    inline Real GetMaximumTemperature() const {
      return max_T;
    }

    //! \brief Get the minimum temperature (in EOS units) permitted by the EOS.
    inline Real GetMinimumTemperature() const {
      return min_T;
    }

    //! \brief Get the minimum fraction permitted by the EOS.
    inline Real GetMinimumSpeciesFraction(int i) const {
      return min_Y[i];
    }

    //! \brief Get the maximum fraction permitted by the EOS.
    inline Real GetMaximumSpeciesFraction(int i) const {
      return max_Y[i];
    }

    //! \fn const bool IsConservedFlooringFailure() const
    //  \brief Find out if the EOSPolicy fails flooring the conserved variables.
    // 
    // \return true or false
    inline const bool IsConservedFlooringFailure() const {
      return fail_conserved_floor;
    }

    //! \fn const bool IsPrimitiveFlooringFailure() const
    //  \brief Find out if the EOSPolicy fails flooring the primitive variables.
    //
    //  \return true or false
    inline const bool IsPrimitiveFlooringFailure() const {
      return fail_primitive_floor;
    }

    //! \fn const bool KeepPrimAndConConsistent() const
    //  \brief Find out if the EOSPolicy wants the conserved variables to be
    //         adjusted to match the primitive variables.
    //  
    //  \return true or false
    inline const bool KeepPrimAndConConsistent() const {
      return adjust_conserved;
    }

    //! \brief Get the maximum squared magnetic field permitted by the ErrorPolicy
    inline Real GetMaximumMagnetization() const {
      return max_bsq;
    }

    //! \brief Set the maximum squared magnetic field permitted by the ErrorPolicy
    //         Adjusts the input to make sure it's nonnegative (does not
    //         return an error).
    inline void SetMaximumMagnetization(double bsq) {
      max_bsq = (bsq >= 0) ? bsq : 0.0;
    }

    //! \brief Respond to excess magnetization
    inline Error DoMagnetizationResponse(Real& bsq, Real b_u[3]) {
      return MagnetizationResponse(bsq, b_u);
    }

    //! \brief Limit the density to a physical range
    inline void ApplyDensityLimits(Real& n) {
      DensityLimits(n, min_n, max_n);
    }

    //! \brief Limit the temperature to a physical range
    inline void ApplyTemperatureLimits(Real& T) {
      Real T_eos = T*code_units->TemperatureConversion(*eos_units);
      TemperatureLimits(T_eos, min_T, max_T);
      T = T_eos*eos_units->TemperatureConversion(*code_units);
    }

    //! \brief Limit Y to a specified range
    inline void ApplySpeciesLimits(Real *Y) {
      SpeciesLimits(Y, min_Y, max_Y, n_species);
    }

    //! \brief Limit the pressure to a specified range at a given density and composition
    inline void ApplyPressureLimits(Real& P, Real n, Real* Y) {
      Real P_eos = P*code_units->PressureConversion(*eos_units);
      PressureLimits(P_eos, MinimumPressure(n, Y), MaximumPressure(n, Y));
      P = P_eos*eos_units->PressureConversion(*code_units);
    }

    //! \brief Limit the energy density to a specified range at a given density and composition
    inline void ApplyEnergyLimits(Real& e, Real n, Real* Y) {
      Real e_eos = e*code_units->PressureConversion(*eos_units);
      EnergyLimits(e_eos, MinimumEnergy(n, Y), MaximumEnergy(n, Y));
      e = e_eos*eos_units->PressureConversion(*code_units);
    }

    //! \brief Respond to a failed solve.
    inline bool DoFailureResponse(Real prim[NPRIM]) {
      bool result = FailureResponse(prim);
      if (result) {
        // Adjust the temperature to be consistent with the new primitive variables.
        prim[ITM] = TemperatureFromP(prim[IDN], 
                    prim[IPR]*code_units->PressureConversion(*eos_units), &prim[IYF]) * 
                    eos_units->TemperatureConversion(*code_units);
      }
      return result;
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
};

} // namespace

#endif
