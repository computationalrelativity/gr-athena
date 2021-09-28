#ifndef PIECEWISE_POLYTROPE_H
#define PIECEWISE_POLYTROPE_H

//! \file piecewise_polytrope.hpp
//  \brief Defines a piecewise-polytropic equation of state.
//
//  Each individual piece satisfies the form
//  \f$P_\textrm{cold} = P_i \frac{\rho}{\rho_i}^{\gamma_i}\f$,
//  for some density \f$\rho > \rho_i\f$. There is an additional
//  finite-temperature portion added on top using the ideal gas
//  law:
//  \f$P_\textrm{therm} = nk_B T\f$

#include "../../athena.hpp"
#include "eos_policy_interface.hpp"

namespace Primitive {

class PiecewisePolytrope : public EOSPolicyInterface {
  private:
    /// Number of polytropes in the EOS
    int n_pieces;
    
    /// Parameters for the EOS
    Real *density_pieces;
    Real *a_pieces;
    Real *gamma_pieces;
    Real *pressure_pieces;
    Real gamma_thermal;
    bool initialized;

    /// Allocate memory for the different EOS pieces.
    void AllocateMemory();
  protected:
    /// Constructor
    PiecewisePolytrope();

    /// Destructor
    ~PiecewisePolytrope();

    /// Calculate the temperature using the ideal gas law.
    Real TemperatureFromE(Real n, Real e, Real *Y);

    /// Calculate the temperature using the ideal gas law.
    Real TemperatureFromP(Real n, Real p, Real *Y);

    /// Calculate the energy density using the ideal gas law.
    Real Energy(Real n, Real T, Real *Y);

    /// Calculate the pressure using the ideal gas law.
    Real Pressure(Real n, Real T, Real *Y);

    /// Calculate the entropy per baryon using the ideal gas law.
    [[ noreturn ]]
    Real Entropy(Real n, Real T, Real *Y);

    /// Calculate the enthalpy per baryon using the ideal gas law.
    Real Enthalpy(Real n, Real T, Real *Y);

    /// Get the minimum enthalpy per baryon according to the ideal gas law.
    Real MinimumEnthalpy();

    /// Calculate the sound speed for an ideal gas.
    Real SoundSpeed(Real n, Real T, Real *Y);

    /// Calculate the energy per baryon (NOT energy per mass!)
    Real SpecificEnergy(Real n, Real T, Real *Y);

    /// Calculate the maximum allowed energy at a given density.
    Real MaximumEnergyAtDensity(Real n);

    /// Calculate the minimum allowed energy at a given density.
    Real MinimumEnergyAtDensity(Real n);

  public:
    /// Load the EOS parameters from a file.
    bool ReadParametersFromFile(std::string fname);

    //! \brief Initialize PiecewisePolytrope from data.
    //  
    //  \param[in] densities The dividing densities
    //  \param[in] gammas    The adiabatic index for each polytrope
    //  \param[in] rho_min   The minimum density for the EOS
    //  \param[in] P0        The pressure at the first polytrope division
    //  \param[in] m         The baryon mass
    //  \param[in] n         The number of pieces in the EOS
    bool InitializeFromData(Real *densities, Real *gammas, 
                            Real rho_min, Real P0, Real m, int n);

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return initialized;
    }

    /// Find out how many polytropes are in the EOS.
    inline int GetNPieces() const {
      return n_pieces;
    }

    /// Get the adiabatic constant for a particular density.
    inline Real GetGamma(Real n) const {
      return gamma_pieces[FindPiece(n)];
    }

    /// Set the adiabatic constant for the thermal part.
    inline void SetThermalGamma(Real g) {
      gamma_thermal = (g <= 1.0) ? 1.00001 : ((g >= 2.0) ? 2.00001 : g);
    }

    /// Get the adiabatic constant for the thermal part.
    inline Real GetThermalGamma() const {
      return gamma_thermal;
    }

    /// Find the index of the piece that the density aligns with.
    int FindPiece(Real n) const;

    /// Polytropic Energy Density
    Real GetColdEnergy(Real n, int p);

    /// Polytropic Pressure
    Real GetColdPressure(Real n, int p);

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    void SetNSpecies(int n);
};

} // namespace

#endif
