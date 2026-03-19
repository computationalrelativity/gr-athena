#ifndef EOS_HELMHOLTZ_H
#define EOS_HELMHOLTZ_H

//! \file eos_helmholtz.hpp
//  \brief Defines EOSTable, which stores information for the helmholtz eos
//        .tabulated electron quantities

///  \warning This code assumes the table to be uniformly spaced in
///           log ne and log t

#include <cstddef>
#include <string>

#include "../../athena.hpp"
#include "eos_policy_interface.hpp"


namespace Primitive {

class EOSHelmholtz : public EOSPolicyInterface {
  friend class EOSTransition;
  public:
    enum TableVariables {
      ECLOGP   = 0,  //! logo of pressure / 1 MeV fm^-3
      ECENT    = 1,  //! entropy per baryon [kb]
      ECLOGEPS = 2,  //! log of specific internal energy
      ECETA    = 3,  //! electron degeneracy parameter
      ECDEPSDT = 4,
      ECDPDN   = 5,
      ECDPDT   = 6,
      ECNVARS  = 7
    };

  protected:
    /// Constructor
    EOSHelmholtz();

    /// Destructor
    ~EOSHelmholtz();

    /// Temperature from energy density
    Real TemperatureFromE(Real n, Real e, Real *Y);

    /// Calculate the temperature from the pressure
    Real TemperatureFromP(Real n, Real p, Real *Y);
    //
    /// Temperature from specific internal energy
    Real TemperatureFromEps(Real n, Real eps, Real *Y);

    /// Calculate the temperature from the entropy
    Real TemperatureFromEntropy(Real n, Real s, Real *Y);

    /// Calculate the energy density.
    Real Energy(Real n, Real T, Real *Y);

    /// Calculate the pressure.
    Real Pressure(Real n, Real T, Real *Y);

    /// Calculate the average baryon number per nucleus.
    Real Abar(Real n, Real T, Real *Y);

    /// Calculate the entropy per baryon.
    Real Entropy(Real n, Real T, Real *Y);

    /// Calculate the enthalpy per baryon.
    Real Enthalpy(Real n, Real T, Real *Y);

    /// Calculate the sound speed.
    Real SoundSpeed(Real n, Real T, Real *Y);

    /// Calculate the specific internal energy per unit mass
    Real SpecificInternalEnergy(Real n, Real T, Real *Y);

    /// Calculate the neutron chemical potential
    Real NeutronChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the proton chemical potential
    Real ProtonChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the baryon electron chemical potential
    Real ElectronChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the baryon chemical potential
    Real BaryonChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the charge chemical potential
    Real ChargeChemicalPotential(Real n, Real T, Real *Y);

    /// Calculate the electron-lepton chemical potential
    Real ElectronLeptonChemicalPotential(Real n, Real T, Real *Y);

    /// Get the minimum enthalpy per baryon.
    Real MinimumEnthalpy();

    /// Get the minimum pressure at a given density and composition
    Real MinimumPressure(Real n, Real *Y);

    /// Get the maximum pressure at a given density and composition
    Real MaximumPressure(Real n, Real *Y);

    /// Get the minimum energy at a given density and composition
    Real MinimumInternalEnergy(Real n, Real *Y);

    /// Get the maximum energy at a given density and composition
    Real MaximumInternalEnergy(Real n, Real *Y);

    /// Get the minimum entropy per baryon at a given density and composition
    Real MinimumEntropy(Real n, Real *Y);

    /// Get the maximum entropy per baryon at a given density and composition
    Real MaximumEntropy(Real n, Real *Y);

  public:
    /// Reads the table file.
    void ReadTableFromFile(std::string fname, Real min_Ye, Real max_Ye);

    /// Set the baryon mass.
    void SetBaryonMass(Real new_mb);

    /// Get the raw number density
    Real const * GetRawLogNumberDensity() const {
      return m_log_ne;
    }
    /// Get the raw number density
    Real const * GetRawLogTemperature() const {
      return m_log_t;
    }
    /// Get the raw table data
    Real const * GetRawTable() const {
      return m_table;
    }

    // Indexing used to access the data
    inline ptrdiff_t index(int iv, int in, int it) const {
      return it + m_nt*(in + m_nn*iv);
    }

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return m_initialized;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    void SetNSpecies(int n);

  private:
    inline Real inverse_abar(Real *Y) const {
      Real abar = Y[SCXN] + Y[SCXP] + Y[SCXA]/4 + ((Y[SCXH] > 0.0) ? Y[SCXH]/Y[SCAH]: 0.0);
      if (abar <= 0.0) {
        printf("EOSHelmholtz::inverse_abar: got invalid mass fractions, sum is %.5e\n", abar);
        return 1.0;
      }
      return abar;
    }

    /// Low level function, not intended for outside use
    Real temperature_from_var(int vi, Real var, Real n, Real *Y) const;
    /// Low level evaluation function, not intended for outside use
    Real eval_at_nty(int vi, Real n, Real T, Real *Y) const;
    /// Low level evaluation function, not intended for outside use
    Real eval_at_lnty(int vi, Real ln, Real lT) const;
    /// Low level function to add the analytic terms
    Real add_rad_ion(int vi, Real var, Real n, Real T, Real *Y) const;

    /// Evaluate interpolation weight for density
    void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const;
    /// Evaluate interpolation weight for temperature
    void weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const;

  private:
    // Inverse of table spacing
    Real m_id_log_ne, m_id_log_t;
    // Table size
    int m_nn, m_nt;
    // Minimum enthalpy per baryon
    const Real m_min_h = 0.0;

    // Table storage, care should be made to store these data on the GPU later
    // Static pointers used to share access to single instance of table in memory (per MPI process)
    static Real * m_log_ne;
    static Real * m_log_t;
    static Real * m_table;

    // bool to protect against access of uninitialised table, and prevent repeated reading of table
    static bool m_initialized;

    // Auxiliary static variables to share data only available when table is open to those threads that do not open it
    // variables from EOSHelmholtz
    static Real sm_id_log_ne, sm_id_log_t;
    static int sm_nn, sm_nt;

    // variables from EOSPolicy
    static Real s_mb, s_max_n, s_min_n, s_max_T, s_min_T;
    // these correspond to defined but unused vars in EOSPolicy
    // static Real s_max_P, s_min_P, s_max_e, s_min_e;
    static constexpr Real hbarc = 197.3269804; // MeV fm
    // const Real asol = 8.563456312967042e-08; // pi**2/(15*hbarc^3) (MeV fm)^-3
    static constexpr Real asol = M_PI*M_PI/(15.0*hbarc*hbarc*hbarc); // (MeV fm)^-3
    // const Real sac_const = 244654.27090035815; // h^2/(2*pi) in (MeV fm)^2
    static constexpr Real sac_const = hbarc*hbarc*2.0*M_PI; // (MeV fm)^2
    static constexpr Real me = 0.5109989461; // MeV
    static constexpr Real mn = 939.5654133; // MeV
    static constexpr Real mp = 938.2720813; // MeV
    static constexpr Real ma = 3727.379378; // MeV
    static constexpr int g_n = 2; // neutron spin degeneracy
    static constexpr int g_p = 2; // proton spin degeneracy
    static constexpr int g_a = 1; // alpha particle spin degeneracy set to 1 as in Just+ 2023
    static constexpr int g_h = 1; // heavy nuclei spin degeneracy set to 1 as in Just+ 2023
};

} // namespace Primitive

#endif
