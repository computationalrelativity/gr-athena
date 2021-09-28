#ifndef PS_EOS_HPP
#define PS_EOS_HPP

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
//    Real SpecificEnergy(Real n, Real T, Real *Y)
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
//    void EnergyLimits(Real& e, Real e_min, Real e_max);
//  And the following protected variables (available via
//  ErrorPolicyInterface):
//    Real n_atm
//    Real T_atm
//    Real v_max
//    Real max_bsq_field
//    bool fail_conserved_floor
//    bool fail_primitive_floor
//    bool adjust_conserved

#include <limits>

#include "../../athena.hpp"

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
    using EOSPolicy::SpecificEnergy;
    using EOSPolicy::MinimumEnthalpy;

    // EOSPolicy member variables
    // The number of particle species used by the EOS.
    using EOSPolicy::n_species;
    // The baryon mass
    using EOSPolicy::mb;
    // Maximum density
    using EOSPolicy::max_n;
    // Minimum density
    using EOSPolicy::min_n;
    // Maximum energy
    using EOSPolicy::max_e;
    // Minimum energy
    using EOSPolicy::min_e;

    // ErrorPolicy member functions
    using ErrorPolicy::PrimitiveFloor;
    using ErrorPolicy::ConservedFloor;
    using ErrorPolicy::MagnetizationResponse;
    using ErrorPolicy::DensityLimits;
    using ErrorPolicy::EnergyLimits;

    // ErrorPolicy member variables
    using ErrorPolicy::n_atm;
    using ErrorPolicy::T_atm;
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
      T_atm = 1.0;
      v_max = 1.0 - 1e-15;
      max_bsq = std::numeric_limits<Real>::max();
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
      return TemperatureFromE(n, e, Y);
    }

    //! \fn Real GetTemperatureFromP(Real n, Real p, Real *Y)
    //  \brief Calculate the temperature from number density, pressure, and
    //         particle fractions.
    //  \param[in] n  The number density
    //  \param[in] p  The pressure
    //  \param[in] Y  An array of particle fractions, expected to be of size n_species.
    //  \return The temperature according to the EOS.
    inline Real GetTemperatureFromP(Real n, Real p, Real *Y) {
      return TemperatureFromP(n, p, Y);
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
      return Energy(n, T, Y);
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
      return Pressure(n, T, Y);
    }

    //! \fn Real GetEntropy(Real n, Real T, Real *Y)
    //  \brief Get the entropy per baryon from the number density, temperature,
    //         and particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The entropy per baryon for this EOS.
    inline Real GetEntropy(Real n, Real T, Real *Y) {
      return Entropy(n, T, Y);
    }

    //! \fn Real GetEnthalpy(Real n, Real T, Real *Y)
    //  \brief Get the enthalpy per baryon from the number density, temperature,
    //         and particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The enthalpy per baryon for this EOS.
    inline Real GetEnthalpy(Real n, Real T, Real *Y) {
      return Enthalpy(n, T, Y);
    }

    //! \fn Real GetMinimumEnthalpy()
    //  \brief Get the global minimum for enthalpy per baryon from the EOS.
    //
    //  \return the minimum enthalpy per baryon.
    inline Real GetMinimumEnthalpy() {
      return MinimumEnthalpy();
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
      return SoundSpeed(n, T, Y);
    }

    //! \fn Real GetSpecificEnergy(Real n, Real T, Real *Y)
    //  \brief Get the energy per mass from the number density, temperature,
    //         and particle fractions.
    //
    //  \param[in] n  The number density
    //  \param[in] T  The temperature
    //  \param[in] Y  An array of size n_species of the particle fractions.
    //  \return The specific energy for the EOS.
    inline Real GetSpecificEnergy(Real n, Real T, Real *Y) {
      return SpecificEnergy(n, T, Y);
    }

    //! \fn int Getn_species() const
    //  \brief Get the number of particle species in this EOS.
    inline int GetNSpecies() const {
      return n_species;
    }

    //! \fn Real GetBaryonMass() const
    //  \brief Get the baryon mass used by this EOS.
    inline Real GetBaryonMass() const {
      return mb;
    }

    //! \fn bool ApplyPrimitiveFloor(Real& n, Real& vu[3], Real& p, Real& T)
    //  \brief Apply the floor to the primitive variables.
    //
    //  \param[in,out] n   The number density
    //  \param[in,out] Wvu The velocity vector (contravariant)
    //  \param[in,out] p   The pressure
    //  \param[out]    T   The temperature
    //  \param[in]     Y   An array of size n_species of the particle fractions.
    //
    //  \return true if the primitives were adjusted, false otherwise.
    bool ApplyPrimitiveFloor(Real& n, Real Wvu[3], Real& p, Real& T, Real *Y) {
      bool result = PrimitiveFloor(n, Wvu, T);
      if (result) {
        p = Pressure(n, T, Y);
      }
      return result;
    }

    //! \fn bool ApplyConservedFloor(Real& D, Real& Sd[3], Real& tau, Real& Bu[3])
    //  \brief Apply the floor to the conserved variables.
    //
    //  \param[in,out] D   The relativistic number density
    //  \param[in,out] Sd  The momentum density vector (covariant)
    //  \param[in,out] tau The tau variable (relativistic energy - D)
    //  \param[in,out] Bu  The magnetic field vector (contravariant)
    //  \param[in]     Y   An array of size_species of the particle fractions.
    //
    //  \return true if the conserved variables were adjusted, false otherwise.
    bool ApplyConservedFloor(Real& D, Real Sd[3], Real& tau, Real *Y) {
      return ConservedFloor(D, Sd, tau, n_atm*mb, GetTauFloor(Y));
    }

    //! \fn Real GetDensityFloor() const
    //  \brief Get the density floor used by the EOS ErrorPolicy.
    inline Real GetDensityFloor() const {
      return n_atm;
    }

    //! \fn Real GetPressureFloor(Real *Y) const
    //  \brief Get the pressure floor based on the current particle
    //         composition.
    //
    //  \param[in] Y A n_species-sized array of particle fractions.
    inline Real GetPressureFloor(Real *Y) {
      return GetPressure(n_atm, T_atm, Y);
    }

    //! \fn Real GetTemperatureFloor() const
    //  \brief Get the temperature floor used by the EOS ErrorPolicy.
    inline Real GetTemperatureFloor() const {
      return T_atm;
    }

    //! \fn Real GetTauFloor() const
    //  \brief Get the tau floor used by the EOS ErrorPolicy based
    //         on the current particle composition.
    //
    //  \param[in] Y A n_species-sized array of particle fractions.
    inline Real GetTauFloor(Real *Y) {
      return GetEnergy(n_atm, T_atm, Y) - mb*n_atm;
    }

    //! \fn Real SetDensityFloor(Real floor)
    //  \brief Set the density floor used by the EOS ErrorPolicy.
    //         Also adjusts the pressure and tau floor to be consistent.
    inline void SetDensityFloor(Real floor) {
      n_atm = (floor >= 0.0) ? floor : 0.0;
    }

    //! \fn Real SetPressureFloor(Real floor)
    //  \brief Set the pressure floor used by the EOS ErrorPolicy.
    //         Also adjusts the tau floor to be consistent.
    inline void SetTemperatureFloor(Real floor) {
      T_atm = (floor >= 0.0) ? floor : 0.0;
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

    //! \brief Get the maximum number density permitted by the EOS.
    inline Real GetMaximumDensity() const {
      return max_n;
    }

    //! \brief Get the minimum number density permitted by the EOS.
    inline Real GetMinimumDensity() const {
      return min_n;
    }

    //! \brief Get the maximum energy density permitted by the EOS.
    inline Real GetMaximumEnergy() const {
      return max_e;
    }

    //! \brief Get the minimum energy density permitted by the EOS.
    inline Real GetMinimumEnergy() const {
      return min_e;
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

    //! \brief Limit the energy to a physical range
    inline void ApplyEnergyLimits(Real& e) {
      EnergyLimits(e, min_e, max_e);
    }
};

} // namespace

#endif
