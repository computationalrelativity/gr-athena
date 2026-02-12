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

    enum ScalarVariables {
      SCYE    = 0,  //! electron fraction
      SCABAR  = 1,  //! average baryon number per nucleon
      SCEB    = 2,  //! tracked Binding energy per baryon
      SCPENT  = 3,  //! tracked past entropy per baryon
      SCPTAU  = 4,  //! tracked past expansion timescale
      SCPYE   = 5,  //! tracked past electron fraction
      SCPTFO  = 6,  //! tracked past temperature
      SCNVARS = 7
    };

    enum HeatingParameters {
       HA = 0,
       HALPHA = 1,
       HSIGMA = 2,
       HT0 = 3,
       HNVARS = 4
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
    //
    /// Temperature from entropy per baryon
    Real TemperatureFromEntropy(Real n, Real eps, Real *Y);

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

    /// Species fractions (not implented, will throw an exception if called)
    Real FrYn(Real n, Real T, Real *Y);
    Real FrYp(Real n, Real T, Real *Y);
    Real FrYh(Real n, Real T, Real *Y);
    Real AN(Real n, Real T, Real *Y);
    Real ZN(Real n, Real T, Real *Y);

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
    void InitializeTables(std::string fname, std::string helm_fname, std::string heating_fname, Real baryon_mass);

    /// Evaluate the heating rate fits a t given tau, ye, s and the current time
    Real HeatingRate(Real tau, Real ye, Real s, Real t);

    /// Some setters for parameters
    void SetTransition(Real n_start, Real n_end, Real T_start, Real T_end);
    void SetMaxIteration(int iter_max);
    void SetTemperatureTolerance(Real tol);

    /// Get the NSE value of the binding energy per baryon
    Real GetBindingEnergy(Real n, Real T, Real *Y);

    /// Get the transition parameters
    void PrintParameters();

    /// Get the factor to transition between the two EOSs, as a function of density and temperature
    Real TransitionFactor(Real n, Real T) const;

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

    Real const GetTempTransStart() const {
      return trans_T_start;
    }

    Real const GetDensTransStart() const {
      return exp(trans_ln_start);
    }

    // Indexing used to access the data
    inline ptrdiff_t index(int iv, int in, int iy, int it) const {
      return it + m_nt*(iy + m_ny*(in + m_nn*iv));
    }

    // Indexing used to access the heating data
    inline ptrdiff_t h_index(int iv, int it, int iy, int is) const {
      return is + m_h_ns*(iy + m_h_ny*(it + m_h_nt*iv));
    }

    /// Check if the EOS has been initialized properly.
    inline bool IsInitialized() const {
      return m_initialized;
    }

    /// Set the number of species. Throw an exception if
    /// the number of species is invalid.
    void SetNSpecies(int n);

    /// Set the maxium density.
    /// Values higher than the max of the table will lead to extrapolation
    void SetMaximumDensity(Real n_max) {
      max_n = n_max;
    }

  private:
    /// Reads the compose table.
    void read_compose_table(std::string fname);
    /// Reads the helmholtz table.
    void read_helmholtz_table(std::string fname);
    /// Reads the table of heating rate fits.
    void read_heating_table(std::string fname);
    /// Lowers the reference baryon mass to ensure eps > 0.
    void update_baryon_mass(Real new_mb);
    void update_bounds();


    /// Low level inversion function, not intended for outside use
    Real eval_at_nty(int iv, Real ln, Real lT, Real Yq, Real Abar, Real Eb) const;
    Real eval_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar, Real Eb) const;
    Real eval_compose_at_lnty(int iv, Real ln, Real lT, Real Yq) const;
    Real eval_helm_at_lnty(int iv, Real ln, Real lT, Real Yq, Real Abar, Real Eb) const;
    Real eval_heating_at_ltys(int iv, Real lt, Real ye, Real ls) const;
    Real temperature_from_var(int iv, Real var, Real n, Real Yq, Real Abar, Real Eb) const;
    Real temperature_from_var_with_guess(int iv, Real var, Real n, Real Yq, Real Abar, Real Eb, Real Tguess) const;

    /// Evaluate interpolation weight for density
    void weight_idx_ln(Real *w0, Real *w1, int *in, Real log_n) const;
    /// Evaluate interpolation weight for composition
    void weight_idx_yq(Real *w0, Real *w1, int *iy, Real yq) const;
    /// Evaluate interpolation weight for temperature
    void weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const;

    /// Evaluate interpolation weight for past tau
    void h_weight_idx_lt(Real *w0, Real *w1, int *it, Real log_t) const;
    /// Evaluate interpolation weight for past ye
    void h_weight_idx_ye(Real *w0, Real *w1, int *iy, Real yq) const;
    /// Evaluate interpolation weight for past entroy
    void h_weight_idx_ls(Real *w0, Real *w1, int *is, Real log_s) const;

  protected:
    boost::uintmax_t max_iter;  // Maximum iterations for root finding
    Real T_tol;  // Tolerance for temperature root finding
    static Real trans_T_start, trans_T_end, trans_ln_start, trans_ln_end; // Transition parameters

    // these are redefinitions to static from the eos_policy_interface
    static Real mb, max_n, min_n, max_T, min_T, max_Y[MAX_SPECIES], min_Y[MAX_SPECIES];

    static Real h_min_s, h_max_s;
    static Real h_min_y, h_max_y;
    static Real h_min_t, h_max_t;

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

    // Inverse of heating table spacing
    static Real m_h_id_lt, m_h_id_y, m_h_id_ls;
    static int m_h_ns, m_h_nt, m_h_ny;
    static Real * m_heating_ls;
    static Real * m_heating_ye;
    static Real * m_heating_ltau;
    static Real * m_heating_table;

    // bool to protect against access of uninitialised table, and prevent repeated reading of table
    static bool m_initialized;

    static Real helm_ln_max;
    static Real helm_lt_max;
    static Real comp_ln_min;
    static Real comp_lt_min;
};


  extern "C" {
    void read_helm_table(const char * helmTablePath, const int * str_len);

    void helm_etot(
      const Real * const rho,             // density in g / cm^3
      const Real * const temp,            // temperature in K
      const Real * const Abar,            // average mass number A
      const Real * const Zbar,            // average atomic number Z
      // output specific internal electron and photon energy in erg / g
      Real * const etot,
      // output pressure in erg / cm^3
      bool * const success_flag  // true if succeeded
    );

    void helm_ptot(
      const Real * const rho,             // density in g / cm^3
      const Real * const temp,            // temperature in K
      const Real * const Abar,            // average mass number A
      const Real * const Zbar,            // average atomic number Z
      // output pressure in erg / cm^3
      Real * const ptot,
      bool * const success_flag  // true if succeeded
    );

    void helm_cs(
      const Real * const rho,             // density in g / cm^3
      const Real * const temp,            // temperature in K
      const Real * const Abar,            // average mass number A
      const Real * const Zbar,            // average atomic number Z
      // output speed of sound in cm / s
      Real * const cs,
      bool * const success_flag  // true if succeeded
    );

    void pacz_etot(
      const Real * const rho,             // density in g / cm^3
      const Real * const temp,            // temperature in K
      const Real * const Abar,            // average mass number A
      const Real * const Zbar,            // average atomic number Z
      Real * const etot,
      bool * const success_flag  // true if succeeded
    );

    void pacz_ptot(
      const Real * const rho,             // density in g / cm^3
      const Real * const temp,            // temperature in K
      const Real * const Abar,            // average mass number A
      const Real * const Zbar,            // average atomic number Z
      Real * const ptot,
      bool * const success_flag  // true if succeeded
    );

    void pacz_cs(
      const Real * const rho,             // density in g / cm^3
      const Real * const temp,            // temperature in K
      const Real * const Abar,            // average mass number A
      const Real * const Zbar,            // average atomic number Z
      Real * const cs,
      bool * const success_flag  // true if succeeded
    );

    void check_bounds(
      const Real * rho_trans,
      const Real * temp_trans,
      const Real * ye_max,
      bool * success
    );

    void get_bounds(
      Real * ye_min,
      Real * ye_max,
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
