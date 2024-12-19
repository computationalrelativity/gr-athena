#ifndef EOS_COMPOSE_TRANS_H
#define EOS_COMPOSE_TRANS_H

//! \file eos_compose.hpp
//  \brief Defines EOSTable, which stores information from a tabulated
//         equation of state in CompOSETransition format.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSETransition</a>

///  \warning This code assumes the table to be uniformly spaced in
///           log nb, log t, and yq

#include <cstddef>
#include <string>

#include <boost/math/tools/roots.hpp>

#include "../../athena.hpp"
#include "eos_policy_interface.hpp"


namespace Primitive {

struct UnitSystem;

class EOSCompOSETransition : public EOSPolicyInterface {
  public:
    enum TableVariables {
      ECLOGP  = 0,  //! log (pressure / 1 MeV fm^-3)
      ECENT   = 1,  //! entropy per baryon [kb]
      ECMUB   = 2,  //! baryon chemical potential [MeV]
      ECMUQ   = 3,  //! charge chemical potential [MeV]
      ECMUL   = 4,  //! lepton chemical potential [MeV]
      ECLOGE  = 5,  //! log(specific internal energy)
      ECCS    = 6,  //! sound speed [c]
      ECABAR  = 7,  //! average baryon number per nucleon
      ECNVARS = 8
    };

  protected:
    /// Constructor
    EOSCompOSETransition();

    /// Destructor
    ~EOSCompOSETransition();

    /// Temperature from energy density
    Real TemperatureFromE(Real n, Real e, Real *Y);
    Real TemperatureFromE(Real n, Real e, Real *Y, Real Tguess);

    /// Temperature from specific internal energy
    Real TemperatureFromEps(Real n, Real eps, Real *Y);
    Real TemperatureFromEps(Real n, Real eps, Real *Y, Real Tguess);

    /// Calculate the temperature using.
    Real TemperatureFromP(Real n, Real p, Real *Y);
    Real TemperatureFromP(Real n, Real p, Real *Y, Real Tguess);

    /// Calculate the energy density using.
    Real Energy(Real n, Real T, Real *Y);

    /// Calculate the pressure using.
    Real Pressure(Real n, Real T, Real *Y);

    /// Calculate the entropy per baryon using.
    Real Entropy(Real n, Real T, Real *Y);

    /// Calculate the enthalpy per baryon using.
    Real Enthalpy(Real n, Real T, Real *Y);

    /// Calculate the sound speed.
    Real SoundSpeed(Real n, Real T, Real *Y);

    /// Calculate the average baryon number per nucleon.
    Real Abar(Real n, Real T, Real *Y);

    /// Calculate the specific internal energy per unit mass
    Real SpecificInternalEnergy(Real n, Real T, Real *Y);

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

    /// Get the minimum specific internal energy at a given density and composition
    Real MinimumSpecificInternalEnergy(Real n, Real *Y);

    /// Get the maximum specific internal energy at a given density and composition
    Real MaximumSpecificInternalEnergy(Real n, Real *Y);

    /// Get the minimum energy at a given density and composition
    Real MinimumEnergy(Real n, Real *Y);

    /// Get the maximum energy at a given density and composition
    Real MaximumEnergy(Real n, Real *Y);

  public:
    /// Reads the table files and clalculates transition parameters.
    void InitializeTables(std::string fname, std::string helm_fname);

    /// Some setters for parameters
    void SetTransition(Real n_start, Real n_end, Real T_start, Real T_end);
    void SetMaxIteration(int iter_max);
    void SetTemperatureTolerance(Real tol);

    /// Get the transition parameters
    void PrintParameters();

    /// Get the raw number density
    Real const * GetRawLogNumberDensity() const {
      return m_log_nb;
    }
    Real const * GetRawYq() const {
      return m_yq;
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
    inline ptrdiff_t index(int iv, int in, int iy, int it) const {
      return it + m_nt*(iy + m_ny*(in + m_nn*iv));
    }

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return m_initialized;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    void SetNSpecies(int n);

  private:
    /// Reads the compose table.
    void read_compose_table(std::string fname);
    /// Reads the helmholtz table.
    void read_helmholtz_table(std::string fname);
    /// Lowers the reference baryon mass to ensure eps > 0.
    void update_baryon_mass();
    void update_bounds();


    /// Low level inversion function, not intended for outside use
    Real eval_at_nty(int iv, Real ln, Real lT, Real Yq, Real Abar) const;
    Real eval_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar) const;
    Real eval_compose_at_lnty(int iv, Real ln, Real lT, Real Yq) const;
    Real eval_helm_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar) const;
    Real temperature_from_var(int iv, Real var, Real n, Real Yq, Real Abar) const;
    Real temperature_from_var_with_guess(int iv, Real var, Real n, Real Yq, Real Abar, Real Tguess) const;

    /// Evaluate interpolation weight for density
    void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const;
    /// Evaluate interpolation weight for composition
    void weight_idx_yq(Real *w0, Real *w1, int *iy, Real yq) const;
    /// Evaluate interpolation weight for temperature
    void weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const;

  protected:
    boost::uintmax_t max_iter;  // Maximum iterations for root finding
    Real T_tol;  // Tolerance for temperature root finding
    static Real trans_T_start, trans_T_end, trans_ln_start, trans_ln_end; // Transition parameters

    // these are redefinitions to static from the eos_policy_interface
    static Real mb, max_n, min_n, max_T, min_T, max_Y[MAX_SPECIES], min_Y[MAX_SPECIES];

  private:
    // Inverse of table spacing
    static Real m_id_log_nb, m_id_log_t, m_id_yq;
    // Table size
    static int m_nn, m_nt, m_ny;
    // Minimum enthalpy per baryon
    static Real m_min_h;

    // Transitions width
    static Real m_trans_T_width, m_trans_ln_width;

    static Real * m_log_nb;
    static Real * m_log_t;
    static Real * m_yq;
    static Real * m_table;

    // bool to protect against access of uninitialised table, and prevent repeated reading of table
    static bool m_initialized;
};


  extern "C" {
    void read_helm_table(const char * helmTablePath, const int * str_len);

    void helm_eos_wrap(
      const Real * const rho,             // density in g / cm^3
      const Real * const temp,            // temperature in K
      const Real * const Abar,            // average mass number A
      const Real * const Zbar,            // average atomic number Z
      // output specific internal electron and photon energy in erg / g
      Real * const etot,
      // output pressure in erg / cm^3
      Real * const ptot,
      // output entropy in erg / g / K
      Real * const stot,
      // output electron degeneracy parameter
      Real * const etaele,
      // output ion degeneracy parameter
      Real * const etaion,
      // output speed of sound in cm / s
      Real * const cs,
      bool * const success_flag  // true if succeeded
    );

    void check_bounds(
      const Real * rho_trans,
      const Real * temp_trans,
      const Real * ye_min,
      const Real * ye_max,
      bool * success
    );

    void get_bounds(
      Real * ye_min,
      Real * rho_min,
      Real * rho_max,
      Real * temp_min,
      Real * temp_max
    );

    void set_mb(Real * const mb);

    void test_print();
  }
} // namespace Primitive

#endif
