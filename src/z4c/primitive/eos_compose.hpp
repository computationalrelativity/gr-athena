#ifndef EOS_COMPOSE_H
#define EOS_COMPOSE_H

//! \file eos_compose.hpp
//  \brief Defines EOSTable, which stores information from a tabulated
//         equation of state in CompOSE format.
//
//  Tables should be generated using
//  <a href="https://bitbucket.org/dradice/pycompose">PyCompOSE</a>

///  \warning This code assumes the table to be uniformly spaced in
///           log nb, log t, and yq

#include <cstddef>
#include <string>

#include "../../athena.hpp"
#include "../../globals.hpp"
#include "eos_policy_interface.hpp"

namespace Primitive
{

class EOSCompOSE : public EOSPolicyInterface
{
  friend class EOSTransition;

  public:
  enum TableVariables
  {
    ECLOGP  = 0,   //! log (pressure / 1 MeV fm^-3)
    ECENT   = 1,   //! entropy per baryon [kb]
    ECMUB   = 2,   //! baryon chemical potential [MeV]
    ECMUQ   = 3,   //! charge chemical potential [MeV]
    ECMUL   = 4,   //! lepton chemical potential [MeV]
    ECLOGE  = 5,   //! log (total energy density / 1 MeV fm^-3)
    ECCS    = 6,   //! sound speed [c]
    ECYN    = 7,   //! Y[n]
    ECYP    = 8,   //! Y[p]
    ECXA    = 9,   //! X[He4]
    ECXH    = 10,  //! X_h = A_N * Y[N], heavy-nucleus mass fraction
    ECAN    = 11,  //! A[N]
    ECZN    = 12,  //! Z[N]
    ECDU    = 13,  //! effective nucleon potential difference dU [MeV]
    ECNVARS = 14
  };

  protected:
  /// Constructor
  EOSCompOSE();

  /// Destructor
  ~EOSCompOSE();

  /// Temperature from energy density
  Real TemperatureFromE(Real n, Real e, Real* Y);

  /// Calculate the temperature from the pressure
  Real TemperatureFromP(Real n, Real p, Real* Y);

  /// Temperature from specific internal energy
  Real TemperatureFromEps(Real n, Real e, Real* Y);

  /// Calculate the temperature from the entropy
  Real TemperatureFromEntropy(Real n, Real s, Real* Y);

  /// Calculate the energy density using.
  Real Energy(Real n, Real T, Real* Y);

  /// Calculate the pressure using.
  Real Pressure(Real n, Real T, Real* Y);

  /// Calculate the entropy per baryon using.
  Real Entropy(Real n, Real T, Real* Y);

  /// Calculate the enthalpy per baryon using.
  Real Enthalpy(Real n, Real T, Real* Y);

  /// Fused temperature + pressure + enthalpy from energy: avoiding redundant
  /// lookups
  void TemperaturePressureAndEnthalpyFromE(Real n,
                                           Real e,
                                           Real* Y,
                                           Real* T,
                                           Real* P,
                                           Real* h,
                                           int* guess_it = nullptr);

  void PressureAndEnthalpyFromE(Real n,
                                Real e,
                                Real* Y,
                                Real* P,
                                Real* h,
                                int* guess_it = nullptr);

  /// Fused pressure + enthalpy: single weight computation for both P and h.
  void PressureAndEnthalpy(Real n, Real T, Real* Y, Real* P, Real* h);

  /// Calculate the sound speed.
  Real SoundSpeed(Real n, Real T, Real* Y);

  // Returns neutron number fraction Y_n = n_n / n_b.
  Real FrYn(Real n, Real T, Real* Y);
  // Returns proton number fraction Y_p = n_p / n_b.
  Real FrYp(Real n, Real T, Real* Y);
  // The following are mass fraction, not a number fraction.
  // Returns alpha-particle mass fraction X_a = 4 * Y_a.
  Real FrXa(Real n, Real T, Real* Y);
  // Returns heavy-nucleus mass fraction X_h = A_N * Y_N.
  Real FrXh(Real n, Real T, Real* Y);

  Real AN(Real n, Real T, Real* Y);
  Real ZN(Real n, Real T, Real* Y);

  /// Calculate the specific internal energy per unit mass
  Real SpecificInternalEnergy(Real n, Real T, Real* Y);

  /// Calculate the baryon chemical potential
  Real BaryonChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the effective nucleon potential difference
  Real InteractionPotentialDifference(Real n, Real T, Real* Y);

  /// Calculate the charge chemical potential
  Real ChargeChemicalPotential(Real n, Real T, Real* Y);

  /// Calculate the electron-lepton chemical potential
  Real ElectronLeptonChemicalPotential(Real n, Real T, Real* Y);

  /// Get the minimum enthalpy per baryon.
  Real MinimumEnthalpy();

  /// Get the minimum specific internal energy at a given density and
  /// composition
  Real MinimumSpecificInternalEnergy(Real n, Real* Y);

  /// Get the maximum specific internal energy at a given density and
  /// composition
  Real MaximumSpecificInternalEnergy(Real n, Real* Y);

  /// Get the minimum pressure at a given density and composition
  Real MinimumPressure(Real n, Real* Y);

  /// Get the maximum pressure at a given density and composition
  Real MaximumPressure(Real n, Real* Y);

  /// Get the minimum energy at a given density and composition
  Real MinimumEnergy(Real n, Real* Y);

  /// Get the maximum energy at a given density and composition
  Real MaximumEnergy(Real n, Real* Y);

  /// Get the minimum entropy per baryon at a given density and composition
  Real MinimumEntropy(Real n, Real* Y);

  /// Get the maximum entropy per baryon at a given density and composition
  Real MaximumEntropy(Real n, Real* Y);

  public:
  /// Reads the table file.
  void ReadTableFromFile(std::string fname);

  /// Set the baryon mass.
  /// Updating the table is not necessary because it stores the total energy
  /// and the baryon number density
  void SetBaryonMass(Real new_mb)
  {
    mb = new_mb;
  }

  /// Get the raw number density
  Real const* GetRawLogNumberDensity() const
  {
    return m_log_nb;
  }
  Real const* GetRawYq() const
  {
    return m_yq;
  }
  /// Get the raw number density
  Real const* GetRawLogTemperature() const
  {
    return m_log_t;
  }
  /// Get the raw table data
  Real const* GetRawTable() const
  {
    return m_table;
  }

  // Indexing used to access the data
  inline ptrdiff_t index(int iv, int in, int iy, int it) const
  {
    return it + m_nt * (iy + m_ny * (in + m_nn * iv));
  }

  /// Check if the EOS has been initialized properly.
  inline bool IsInitialized() const
  {
    return m_initialized;
  }

  /// Set the number of species. Throw an exception if
  /// the number of species is invalid.
  void SetNSpecies(int n);

  /// Set the maxium density.
  /// Values higher than the max of the table will lead to extrapolation
  void SetMaximumDensity(Real n_max)
  {
    max_n = n_max;
  }

  /// Set the maxium termperature.
  /// Values higher than the max of the table will lead to extrapolation
  void SetMaximumTemperature(Real T_max)
  {
    max_T = T_max;
  }

  // N.B. non-converted
  Real GetTableProtonMass()
  {
    return s_mp;
  }

  Real GetTableNeutronMass()
  {
    return s_mn;
  }

  private:
  /// Low level function, not intended for outside use
  Real temperature_from_var(int vi, Real var, Real n, Real Yq) const;
  /// Low level function with pre-computed density/composition weights
  Real temperature_from_var_precomp(Real var_min,
                                    Real var_max,
                                    int iv,
                                    Real var,
                                    Real wn0,
                                    Real wn1,
                                    int in,
                                    Real wy0,
                                    Real wy1,
                                    int iy) const;
  /// Low level evaluation function, not intended for outside use
  Real eval_at_nty(int vi, Real n, Real T, Real Yq) const;
  /// Low level evaluation function, not intended for outside use
  Real eval_at_lnty(int vi, Real ln, Real lT, Real Yq) const;

  /// Evaluate interpolation weight for density
  void weight_idx_ln(Real* w0, Real* w1, int* in, Real log_n) const;
  /// Evaluate interpolation weight for composition
  void weight_idx_yq(Real* w0, Real* w1, int* iy, Real yq) const;
  /// Evaluate interpolation weight for temperature
  void weight_idx_lt(Real* w0, Real* w1, int* it, Real log_t) const;

  /// Shared root-search used by Pressure/Temperature variants of
  /// *AndEnthalpyFromE.  Given (n, e, Y) finds the log-T bracket and
  /// returns all interpolation weights and indices plus the final
  /// log-T estimate.  If the energy lies outside the table's [e_min,
  /// e_max] at this (n, Y) slab the routine sets one of the
  /// boundary_{lo,hi} flags and the caller must delegate to
  /// PressureAndEnthalpy(n, min_T/max_T, ...).  guess_it is the usual
  /// in/out temperature hunt hint (may be nullptr).
  void FindTBracketAndWeights(Real n,
                              Real e,
                              Real* Y,
                              int* guess_it,
                              int& in,
                              int& iy,
                              int& it,
                              Real& wn0,
                              Real& wn1,
                              Real& wy0,
                              Real& wy1,
                              Real& wt0,
                              Real& wt1,
                              Real& lt,
                              bool& boundary_lo,
                              bool& boundary_hi) const;

  /// Evaluate table variable at a specific temperature index with
  /// pre-computed density/composition weights (bilinear, 4 lookups).
  inline Real eval_at_it(int iv,
                         Real wn0,
                         Real wn1,
                         int in,
                         Real wy0,
                         Real wy1,
                         int iy,
                         int it) const
  {
    ptrdiff_t const b00 = index(iv, in, iy, it);
    ptrdiff_t const b01 = index(iv, in, iy + 1, it);
    ptrdiff_t const b10 = index(iv, in + 1, iy, it);
    ptrdiff_t const b11 = index(iv, in + 1, iy + 1, it);
    return wn0 * (wy0 * m_table[b00] + wy1 * m_table[b01]) +
           wn1 * (wy0 * m_table[b10] + wy1 * m_table[b11]);
  }

  private:
  // Inverse of table spacing
  Real m_id_log_nb, m_id_log_t, m_id_yq;
  // Table size
  int m_nn, m_nt, m_ny;
  // Minimum enthalpy per baryon
  Real m_min_h;

  // Table storage, care should be made to store these data on the GPU later
  // Static pointers used to share access to single instance of table in memory
  // (per MPI process)
  static Real* m_log_nb;
  static Real* m_log_t;
  static Real* m_yq;
  static Real* m_table;

  // bool to protect against access of uninitialised table, and prevent
  // repeated reading of table
  static bool m_initialized;

  // Whether the optional dU dataset was present in the loaded HDF5 table.
  // When false, InteractionPotentialDifference asserts on call.
  bool m_has_dU = false;

  // Auxiliary static variables to share data only available when table is open
  // to those threads that do not open it variables from EOSCompOSE
  static Real sm_id_log_nb, sm_id_log_t, sm_id_yq;
  static int sm_nn, sm_nt, sm_ny;
  static Real sm_min_h;

  // variables from EOSPolicy
  static Real s_mb, s_max_n, s_min_n, s_max_T, s_min_T, s_max_Y[MAX_SPECIES],
    s_min_Y[MAX_SPECIES];
  // these correspond to defined but unused vars in EOSPolicy
  // static Real s_max_P, s_min_P, s_max_e, s_min_e;

  // storage for neutron and proton mass resp.
  static Real s_mn, s_mp;
};

}  // namespace Primitive

#endif
