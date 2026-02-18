#ifndef M1_OPACITIES_COMMON_EOS_HPP_
#define M1_OPACITIES_COMMON_EOS_HPP_

//! \file eos.hpp
//  \brief Unified EOS wrapper for neutrino opacity calculations.
//
//  Provides a single interface to the Primitive Solver EOS table,
//  handling unit conversions between code units (GeometricSolar),
//  weak-rates units (CGS + MeV), EOS units (Nuclear: fm + MeV),
//  and nurates units (NGS: nm + MeV).

#include <cassert>
#include <cmath>
#include <limits>

#include "../../../athena.hpp"
#include "../../../defs.hpp"
#include "units.hpp"
#include "utils.hpp"

namespace M1::Opacities::Common::EoS
{

namespace Units = ::M1::Opacities::Common::Units;

class EoSWrapper
{
  public:
  // Type alias for the shared options struct
  using Opt = ::M1::Opacities::Common::OpacityUtils::Opt;

  // Constructor - reads M1_opacities settings from the shared opt struct.
  //
  // Flooring logic follows a 3-step pattern:
  //   1. Start from opt.min_*/max_* default values (code units -> CGS+MeV)
  //   2. if (opt.limits_from_table)  -> override from PS_EoS table
  //   3. if (opt.min_rho_usefloor)   -> override rho_min from EOS floor
  //      if (opt.min_t_usefloor)     -> override temp_min from EOS floor
  EoSWrapper(
    const Opt& opt,
    Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS)
      : opt_(opt), PS_EoS(PS_EoS)
  {
    // Set the units
    // wr_units ......... CGS + MeV
    // nurates_units .... CGS + MeV + nm
    // code_units ....... Geometric + Solar masses
    // eos_units ........ fm + MeV + fm
    wr_units      = &Units::WeakRatesUnits;
    nurates_units = &Units::NGS;
    code_units    = &Units::GeometricSolar;
    eos_units     = &Units::Nuclear;

    // Unit conversion factors (code units -> CGS + MeV)
    Real conv_rho_mass = code_units->MassDensityConversion(*wr_units);
    Real conv_rho      = PS_EoS->GetBaryonMass() * conv_rho_mass;
    Real conv_temp     = code_units->TemperatureConversion(*wr_units);

    // Step 1: EOS limits (code units -> CGS + MeV)
    // opt.min_rho etc. are mass densities in code units
    eos_rho_min  = opt.min_rho * conv_rho_mass;
    eos_rho_max  = opt.max_rho * conv_rho_mass;
    eos_temp_min = opt.min_t * conv_temp;
    eos_temp_max = opt.max_t * conv_temp;
    eos_ye_min   = opt.min_ye;
    eos_ye_max   = opt.max_ye;

    // Step 2: override from EOS table if requested
    // GetMinimumDensity()/GetMaximumDensity() return number density in
    // EOS units; conv_rho includes GetBaryonMass() to convert from
    // EOS-unit number density -> code-unit mass density -> CGS mass density.
    if (opt.limits_from_table)
    {
      eos_rho_min  = conv_rho * PS_EoS->GetMinimumDensity();
      eos_rho_max  = conv_rho * PS_EoS->GetMaximumDensity();
      eos_temp_min = conv_temp * PS_EoS->GetMinimumTemperature();
      eos_temp_max = conv_temp * PS_EoS->GetMaximumTemperature();
      eos_ye_min   = PS_EoS->GetMinimumSpeciesFraction(0);
      eos_ye_max   = PS_EoS->GetMaximumSpeciesFraction(0);
    }

    // Step 3: override minimums from EOS atmosphere floors if requested
    // GetDensityFloor() also returns number density in EOS units.
    if (opt.min_rho_usefloor)
      eos_rho_min = conv_rho * PS_EoS->GetDensityFloor();

    if (opt.min_t_usefloor)
      eos_temp_min = conv_temp * PS_EoS->GetTemperatureFloor();

    // mb [g]
    atomic_mass =
      PS_EoS->GetRawBaryonMass() * code_units->MassConversion(*wr_units);
  };

  ~EoSWrapper() {};

  // -----------------------------------------------------------------------
  // Chemical potentials
  // -----------------------------------------------------------------------

  // Chemical potentials calculation (EOS-units interface).
  // in:    nb_eos  [eos_units number density]
  //        T       [MeV]
  //        Ye      [-]
  // out:   mus     [MeV]
  void ChemicalPotentials_npe(Real nb_eos,
                              Real T,
                              Real Ye,
                              Real& mu_n,
                              Real& mu_p,
                              Real& mu_e)
  {
    Real Y[1] = { Ye };
    Real mu_b = PS_EoS->GetBaryonChemicalPotential(nb_eos, T, Y);
    Real mu_q = PS_EoS->GetChargeChemicalPotential(nb_eos, T, Y);
    Real mu_l = PS_EoS->GetElectronLeptonChemicalPotential(nb_eos, T, Y);
    mu_n      = mu_b;
    mu_p      = mu_b + mu_q;
    mu_e      = mu_l - mu_q;
  }

  // Chemical potentials in CGS + MeV.
  //
  // in:   rho   [g/cm^3]   (wr_units mass density)
  //       temp  [MeV]      (wr_units temperature)
  //       Ye    [-]        (electron fraction, dimensionless)
  // out:  mu_n  [MeV]      (neutron chemical potential)
  //       mu_p  [MeV]      (proton chemical potential)
  //       mu_e  [MeV]      (electron chemical potential)
  //
  // Internally converts rho -> nb [eos_units] before calling
  // ChemicalPotentials_npe; applies ChemicalPotentialConversion
  // on output for unit-system clarity.
  void ChemicalPotentials_npe_cgs(Real rho,
                                  Real temp,
                                  Real Ye,
                                  Real& mu_n,
                                  Real& mu_p,
                                  Real& mu_e)
  {
    ChemicalPotentials_npe(
      rho / atomic_mass * wr_units->NumberDensityConversion(*eos_units),
      temp,
      Ye,
      mu_n,
      mu_p,
      mu_e);
    Real conv_mu = code_units->ChemicalPotentialConversion(*wr_units);
    mu_n *= conv_mu;
    mu_p *= conv_mu;
    mu_e *= conv_mu;
  }

  //! Unified chemical potential interface for the common solver.
  //  Forwards to ChemicalPotentials_npe_cgs.
  //
  // in:   rho   [g/cm^3]   (wr_units mass density)
  //       temp  [MeV]
  //       Ye    [-]
  // out:  mu_n  [MeV], mu_p [MeV], mu_e [MeV]
  void ChemicalPotentials_cgs(Real rho,
                              Real temp,
                              Real Ye,
                              Real& mu_n,
                              Real& mu_p,
                              Real& mu_e)
  {
    ChemicalPotentials_npe_cgs(rho, temp, Ye, mu_n, mu_p, mu_e);
  }

  // -----------------------------------------------------------------------
  // Particle fractions from EOS table
  // -----------------------------------------------------------------------

  // Neutron fraction from EOS (code-units interface).
  // in:    nb      [code_units number density]
  //        T       [MeV]  (= code_units temperature for GeometricSolar)
  //        Ye      [-]
  // out:   Yn      [-]
  //
  // When tabulated_particle_fractions is false, returns the free-particle
  // approximation Yn = 1 - Ye.
  Real GetNeutronFraction(Real nb, Real T, Real Ye)
  {
    if (!opt_.tabulated_particle_fractions)
      return 1.0 - Ye;
    Real Y[1]   = { Ye };
    Real nb_eos = nb * code_units->NumberDensityConversion(*eos_units);
    return PS_EoS->GetYn(nb_eos, T, Y);
  }

  // Proton fraction from EOS (code-units interface).
  // in:    nb      [code_units number density]
  //        T       [MeV]  (= code_units temperature for GeometricSolar)
  //        Ye      [-]
  // out:   Yp      [-]
  //
  // When tabulated_particle_fractions is false, returns the free-particle
  // approximation Yp = Ye.
  Real GetProtonFraction(Real nb, Real T, Real Ye)
  {
    if (!opt_.tabulated_particle_fractions)
      return Ye;
    Real Y[1]   = { Ye };
    Real nb_eos = nb * code_units->NumberDensityConversion(*eos_units);
    return PS_EoS->GetYp(nb_eos, T, Y);
  }

  // Particle fractions from EOS (CGS interface).
  //
  // in:   rho   [g/cm^3]   (wr_units mass density)
  //       temp  [MeV]
  //       ye    [-]
  // out:  xn    [-]  neutron fraction
  //       xp    [-]  proton fraction
  //       xh    [-]  heavy-nucleus fraction
  //       Ab    [-]  average mass number of heavy nuclei
  //       Zb    [-]  average charge number of heavy nuclei
  void GetFracs(Real rho,
                Real temp,
                Real ye,
                Real& xn,
                Real& xp,
                Real& xh,
                Real& Ab,
                Real& Zb)
  {
    if (opt_.tabulated_particle_fractions)
    {
      // Input:
      // rho: CGS
      // temp: MeV
      const Real rho_conv_factor =
        (wr_units->MassDensityConversion(*code_units));
      const Real nb = rho * rho_conv_factor / PS_EoS->GetBaryonMass();

      Real Y[1] = { ye };

      xp = PS_EoS->GetYp(nb, temp, Y);
      xn = PS_EoS->GetYn(nb, temp, Y);
      xh = PS_EoS->GetYh(nb, temp, Y);

      // The following suppresses coherent neutrinos nucleus scattering
      // i.e. dodging a zero-division.
      //
      // It appears that stellarcollapse tables handle this by setting 1 in the
      // values; PyCompOSE tables don't have this.
      if (xh == 0.0)
      {
        Ab = 1;
        Zb = 1;
      }
      else
      {
        Ab = PS_EoS->GetAN(nb, temp, Y);
        Zb = PS_EoS->GetZN(nb, temp, Y);
      }
    }
    else
    {
      xp = ye;
      xn = 1 - ye;
      Ab = 1.0;
      Zb = 1.0;
      xh = 0.0;
    }
  }

  // -----------------------------------------------------------------------
  // Degeneracy parameters and eta coefficients
  // -----------------------------------------------------------------------

  // Neutron-proton mass difference [MeV].
  //
  // When tabulated_degeneracy_parameter is true, uses table values.
  // Otherwise uses the THC value (1.293333 MeV).
  inline Real GetDegeneracyParameter()
  {
    Real Qnp = std::numeric_limits<Real>::quiet_NaN();

    if (opt_.tabulated_degeneracy_parameter)
    {
      Qnp = PS_EoS->GetTableNeutronMass() - PS_EoS->GetTableProtonMass();
    }
    else
    {
      // CompOSE value:
      Qnp = 1.2933399999999438;

      // THC value:
      Qnp = 1.293333;
    }

    return Qnp;
  }

  // Compute neutrino degeneracy parameters and nucleon blocking factors.
  //
  // in:   rho   [g/cm^3]   (wr_units mass density)
  //       temp  [MeV]
  //       ye    [-]
  // out:  eta_nue, eta_nua, eta_nux  [-]  neutrino degeneracy parameters
  //       eta_e                      [-]  electron degeneracy parameter
  //       eta_np, eta_pn             [cm^-3]  nucleon blocking factors
  void GetEtas(Real rho,
               Real temp,
               Real ye,
               Real& eta_nue,
               Real& eta_nua,
               Real& eta_nux,
               Real& eta_e,
               Real& eta_np,
               Real& eta_pn)
  {
    // !Density is assumed to be in cgs units and
    // !the temperature in MeV

    // !Compute the baryon number density (mass_fact is given in MeV)
    Real nb = rho / AtomicMassImpl();

    Real mu_n, mu_p, mu_e;
    ChemicalPotentials_npe_cgs(rho, temp, ye, mu_n, mu_p, mu_e);

    Real xn, xp, xh;
    Real abar, zbar;

    GetFracs(rho, temp, ye, xn, xp, xh, abar, zbar);

    /*
    !Compute the neutrino degeneracy assuming that neutrons and
    !protons chemical potentials includes the rest mass density
    !This is the correct formula for stellarcollapse.org tables
    */

    eta_nue = (mu_p + mu_e - mu_n) / temp;
    eta_nua = -eta_nue;
    eta_nux = 0.0;
    eta_e   = mu_e / temp;

    // Neutron and proton degeneracy
    Real eta_n = (mu_n) / temp;
    Real eta_p = (mu_p) / temp;

    // Difference in the degeneracy parameters without
    // neutron-proton rest mass difference
    Real Qnp     = GetDegeneracyParameter();
    Real eta_hat = eta_n - eta_p - Qnp / temp;

    // !Janka takes into account the Pauli blocking effect for
    // !degenerate nucleons as in Bruenn (1985). Ruffert et al. Eq. (A8)
    // !xp = xp / (1.0d0 + 2.0d0 / 3.0d0 * (max(eta_p, 0.0d0)))
    // !xn = xn / (1.0d0 + 2.0d0 / 3.0d0 * (max(eta_n, 0.0d0)))

    // !Consistency check on the fractions
    xp   = std::max(0.0, xp);
    xn   = std::max(0.0, xn);
    xh   = std::max(0.0, xh);
    abar = std::max(0.0, abar);
    zbar = std::max(0.0, zbar);

    // eta takes into account the nucleon final state blocking
    // (at high density)

    eta_np = nb * (xp - xn) / (std::exp(-eta_hat) - 1.0);
    eta_pn = nb * (xn - xp) / (std::exp(eta_hat) - 1.0);

    // !There is no significant defferences between Rosswog (prev. formula)
    // !and Janka's prescriptions
    // ! eta_np = nb * ((2.0d0 * ye-1.0d0) / (exp(eta_p-eta_n) - 1.0d0))
    // ! eta_pn = eta_np * exp(eta_p-eta_n)

    // !See Bruenn (ApJSS 58 1985) formula 3.1, non degenerate matter limit.
    if (rho < 2.0e11)
    {
      eta_pn = nb * xp;
      eta_np = nb * xn;
    }  // endif

    // !Consistency Eqs (A9) (Rosswog's paper) they should be positive
    eta_pn = std::max(eta_pn, 0.0);
    eta_np = std::max(eta_np, 0.0);
  }

  // -----------------------------------------------------------------------
  // Energy density
  // -----------------------------------------------------------------------

  // Compute specific internal energy density from the EOS table.
  //
  // in:   rho   [g/cm^3]   (wr_units mass density)
  //       temp  [MeV]      (wr_units temperature)
  //       ye    [-]        (electron fraction)
  // out:  return [erg/cm^3] (wr_units energy density)
  //
  // Internally converts rho,temp -> code_units for the EOS call, then
  // converts the result back to wr_units.
  Real GetEnergyDensity(Real rho, Real temp, Real ye)
  {
    Real mb_code = PS_EoS->GetBaryonMass();
    Real nb_code =
      rho * wr_units->MassDensityConversion(*code_units) / mb_code;
    Real temp_code = temp * wr_units->TemperatureConversion(*code_units);
    Real Y[1]      = { ye };
    Real e         = PS_EoS->GetEnergy(nb_code, temp_code, Y) *
             code_units->EnergyDensityConversion(*wr_units);
    return e;
  }

  // Compute minimum energy density at the table's temperature floor.
  //
  // in:   rho   [g/cm^3]   (wr_units mass density)
  //       ye    [-]        (electron fraction)
  // out:  return [erg/cm^3] (wr_units energy density)
  //
  // Uses eos_temp_min (CGS+MeV, set during construction).
  Real GetMinimumEnergyDensity(Real rho, Real ye)
  {
    Real mb_code = PS_EoS->GetBaryonMass();
    Real nb_code =
      rho * wr_units->MassDensityConversion(*code_units) / mb_code;
    Real temp_code =
      eos_temp_min * wr_units->TemperatureConversion(*code_units);
    Real Y[1] = { ye };
    Real e    = PS_EoS->GetEnergy(nb_code, temp_code, Y) *
             code_units->EnergyDensityConversion(*wr_units);
    return e;
  }

  // -----------------------------------------------------------------------
  // EOS table limits
  // -----------------------------------------------------------------------

  // Clamp (rho, temp, ye) to the EOS table bounds.
  // All limit values are stored in wr_units (CGS + MeV):
  //   rho  [g/cm^3], temp [MeV], ye [-].
  //
  // Returns true if any value was clamped AND enforced_limits_fail is set.
  bool ApplyTableLimits(Real& rho, Real& temp, Real& ye)
  {
    bool limits_applied = false;

    if (!opt_.apply_table_limits_internally)
    {
      return limits_applied;
    }

    if (rho < eos_rho_min)
    {
      rho            = eos_rho_min;
      limits_applied = true;
    }
    else if (rho > eos_rho_max)
    {
      rho            = eos_rho_max;
      limits_applied = true;
    }

    if (temp < eos_temp_min)
    {
      temp           = eos_temp_min;
      limits_applied = true;
    }
    else if (temp > eos_temp_max)
    {
      temp           = eos_temp_max;
      limits_applied = true;
    }

    if (ye < eos_ye_min)
    {
      ye             = eos_ye_min;
      limits_applied = true;
    }
    else if (ye > eos_ye_max)
    {
      ye             = eos_ye_max;
      limits_applied = true;
    }

    if (!opt_.enforced_limits_fail)
      limits_applied = false;

    return limits_applied;
  }

  // Clamp the total lepton fraction yl to the floor value (min_eql_yl).
  bool ApplyLeptonLimits(Real& yl)
  {
    bool limits_applied = false;
    if (yl < opt_.min_eql_yl)
    {
      limits_applied = true;
      yl             = std::max(yl, opt_.min_eql_yl);
    }
    return limits_applied;
  }

  void GetTableLimits(Real& rho_min,
                      Real& rho_max,
                      Real& temp_min,
                      Real& temp_max,
                      Real& ye_min,
                      Real& ye_max)
  {
    rho_min  = eos_rho_min;
    rho_max  = eos_rho_max;
    temp_min = eos_temp_min;
    temp_max = eos_temp_max;
    ye_min   = eos_ye_min;
    ye_max   = eos_ye_max;
  }

  void GetTableLimitsYe(Real& ye_min, Real& ye_max)
  {
    ye_min = eos_ye_min;
    ye_max = eos_ye_max;
  }

  // -----------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------

  Real AtomicMassImpl() const
  {
    return atomic_mass;
  }
  Real GetRawBaryonMass() const
  {
    return PS_EoS->GetRawBaryonMass();
  }

  Units::UnitSystem* GetWrUnits()
  {
    return wr_units;
  }
  Units::UnitSystem* GetNuratesUnits()
  {
    return nurates_units;
  }
  Units::UnitSystem* GetCodeUnits()
  {
    return code_units;
  }
  Units::UnitSystem* GetEosUnits()
  {
    return eos_units;
  }

  private:
  const Opt& opt_;
  Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS;

  Units::UnitSystem* wr_units;
  Units::UnitSystem* nurates_units;
  Units::UnitSystem* code_units;
  Units::UnitSystem* eos_units;

  Real atomic_mass;

  Real eos_rho_min;
  Real eos_rho_max;
  Real eos_temp_min;
  Real eos_temp_max;
  Real eos_ye_min;
  Real eos_ye_max;
};

}  // namespace M1::Opacities::Common::EoS

#endif  // M1_OPACITIES_COMMON_EOS_HPP_
