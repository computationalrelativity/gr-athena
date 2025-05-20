#ifndef WEAKRATES_EOS_H
#define WEAKRATES_EOS_H

#include <iostream>
#include <cassert>

#include "../../../athena.hpp"
#include "../../../defs.hpp"
#include "../../../eos/eos.hpp"

#include "units.hpp"


namespace M1::Opacities::WeakRates::WeakRates_EoS {

class WeakEoSMod {
  public:
    // Constructor
    WeakEoSMod(bool apply_table_limits_internally,
               bool enforced_limits_fail,
               bool wr_use_eos_dfloor,
               bool wr_use_eos_tfloor,
               Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS)
      : apply_table_limits_internally(apply_table_limits_internally),
        enforced_limits_fail(enforced_limits_fail),
        wr_use_eos_dfloor(wr_use_eos_dfloor),
        wr_use_eos_tfloor(wr_use_eos_tfloor),
        PS_EoS(PS_EoS)
    {
      my_units = &WeakRates_Units::WeakRatesUnits;
      code_units = &WeakRates_Units::GeometricSolar;

      // set up internal settings for EOS (unit conversions & limits)
      Real conv_rho = PS_EoS->GetBaryonMass()*code_units->MassDensityConversion(*my_units);
      Real conv_temp = code_units->TemperatureConversion(*my_units);

      if (wr_use_eos_dfloor)
      {
        table_rho_min = conv_rho * PS_EoS->GetDensityFloor();
      }
      else
      {
        table_rho_min = conv_rho * PS_EoS->GetMinimumDensity();
      }
      table_rho_max = PS_EoS->GetMaximumDensity() * conv_rho;

      if (wr_use_eos_tfloor)
      {
        table_temp_min = conv_temp * PS_EoS->GetTemperatureFloor();
      }
      else
      {
        table_temp_min = conv_temp * PS_EoS->GetMinimumTemperature();
      }
      table_temp_max = conv_temp * PS_EoS->GetMaximumTemperature();

      table_ye_min = PS_EoS->GetMinimumSpeciesFraction(0);
      table_ye_max = PS_EoS->GetMaximumSpeciesFraction(0);
    };

    // Destructor
    ~WeakEoSMod() {};

    // functions implemented by Weak Rates
    // Average atomic mass of nuclei
    int NucleiAbarImpl(Real rho, Real temp, Real ye, Real& abar) {
      std::cout<<"NucleiAbarImpl not implemented"<<std::endl;
      return -1;
    }

    // Average mass of baryons
    Real AtomicMassImpl() {
      Real mb_code = PS_EoS->GetRawBaryonMass();
      Real mb_cgs  = mb_code * code_units->MassConversion(*my_units);
      return mb_cgs;
    }

    // EoS calls required by Weak Rates and others
    Real GetNeutronChemicalPotential(Real rho, Real temp, Real ye) {
      Real mb_code = PS_EoS->GetBaryonMass();
      Real nb_code = rho * my_units->MassDensityConversion(*code_units)/mb_code;
      Real temp_code = temp * my_units->TemperatureConversion(*code_units);
      Real Y[1] = {ye};
      Real mu_b = PS_EoS->GetBaryonChemicalPotential(nb_code, temp_code, Y) * code_units->ChemicalPotentialConversion(*my_units);

      Real mu_n = mu_b;
      return mu_n;
    }

    Real GetProtonChemicalPotential(Real rho, Real temp, Real ye) {
      Real mb_code = PS_EoS->GetBaryonMass();
      Real nb_code = rho * my_units->MassDensityConversion(*code_units)/mb_code;
      Real temp_code = temp * my_units->TemperatureConversion(*code_units);
      Real Y[1] = {ye};
      Real mu_b = PS_EoS->GetBaryonChemicalPotential(nb_code, temp_code, Y) * code_units->ChemicalPotentialConversion(*my_units);
      Real mu_q = PS_EoS->GetChargeChemicalPotential(nb_code, temp_code, Y) * code_units->ChemicalPotentialConversion(*my_units);

      Real mu_p = mu_b + mu_q;
      return mu_p;
    }

    Real GetElectronChemicalPotential(Real rho, Real temp, Real ye) {
      Real mb_code = PS_EoS->GetBaryonMass();
      Real nb_code = rho * my_units->MassDensityConversion(*code_units)/mb_code;
      Real temp_code = temp * my_units->TemperatureConversion(*code_units);
      Real Y[1] = {ye};
      Real mu_q = PS_EoS->GetChargeChemicalPotential(nb_code, temp_code, Y) * code_units->ChemicalPotentialConversion(*my_units);
      Real mu_l = PS_EoS->GetElectronLeptonChemicalPotential(nb_code, temp_code, Y) * code_units->ChemicalPotentialConversion(*my_units);

      Real mu_e = mu_l - mu_q;
      return mu_e;
    }

    Real GetEnergyDensity(Real rho, Real temp, Real ye) {
      Real mb_code = PS_EoS->GetBaryonMass();
      Real nb_code = rho * my_units->MassDensityConversion(*code_units)/mb_code;
      Real temp_code = temp * my_units->TemperatureConversion(*code_units);
      Real Y[1] = {ye};
      Real e = PS_EoS->GetEnergy(nb_code, temp_code, Y) * code_units->EnergyDensityConversion(*my_units);

      return e;
    }

    Real GetMinimumEnergyDensity(Real rho, Real ye) {
      Real mb_code = PS_EoS->GetBaryonMass();
      Real nb_code = rho * my_units->MassDensityConversion(*code_units)/mb_code;
      Real temp_code = table_temp_min * my_units->TemperatureConversion(*code_units); // TODO: Is this right?
      Real Y[1] = {ye};
      Real e = PS_EoS->GetEnergy(nb_code, temp_code, Y) * code_units->EnergyDensityConversion(*my_units);

      return e;
    }

    void GetEtas(Real rho, Real temp, Real ye, Real& eta_nue, Real& eta_nua, Real& eta_nux, Real& eta_e, Real& eta_np, Real& eta_pn) {
      // Turning include into function
      /*
      #ifndef WEAK_RATES_ITS_ME
      #error "This file should not be included by the end user!"
      #endif
      */
    
      // !Density is assumed to be in cgs units and
      // !the temperature in MeV

      // lrho  = log10(rho)
      // ltemp = log10(temp)

      // !Compute the baryon number density (mass_fact is given in MeV)
      // mass_fact_cgs = mass_fact * mev_to_erg / (clight*clight)
      // nb = rho / mass_fact_cgs
      Real nb = rho/AtomicMassImpl();

      // !Interpolate the chemical potentials (stored in MeV in the table)
      // mu_n = lkLinearInterpolation3d(lrho, ltemp, ye, MU_N)
      // mu_p = lkLinearInterpolation3d(lrho, ltemp, ye, MU_P)
      // mu_e = lkLinearInterpolation3d(lrho, ltemp, ye, MU_E)
      Real mu_n = GetNeutronChemicalPotential(rho, temp, ye);
      Real mu_p = GetProtonChemicalPotential(rho, temp, ye);
      Real mu_e = GetElectronChemicalPotential(rho, temp, ye);

      // !Interpolate the fractions
      /* TODO? Our EoS does not currently implement nuclei fractions, and we
               don't intend to use this opacity lib anyway, so I'm not 
               bothering to implement them, and just setting the heavy 
               nuclei to zero and the proton/neutron fractions with ye.
  
      abar = lkLinearInterpolation3d(lrho, ltemp, ye, ABAR)
      zbar = lkLinearInterpolation3d(lrho, ltemp, ye, ZBAR)
      xp   = lkLinearInterpolation3d(lrho, ltemp, ye, XP)
      xn   = lkLinearInterpolation3d(lrho, ltemp, ye, XN)
      xh   = lkLinearInterpolation3d(lrho, ltemp, ye, XH)
      */

      Real abar = 1.0;
      Real zbar = 1.0;
      Real xp = ye;
      Real xn = 1-ye;
      Real xh = 0.0;

      /*
      !Compute the neutrino degeneracy assuming that neutrons and
      !protons chemical potentials DO NOT include the rest mass density
      ! eta_nue = (mu_p + mu_e - mu_n - Qnp) / temp
      */

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
      Real eta_n   = mu_n / temp;
      Real eta_p   = mu_p / temp;
          
      // Difference in the degeneracy parameters without
      // neutron-proton rest mass difference
      Real Qnp = 1.293333; // neutron-proton mass difference in MeV - in principle should come from the EoS
      Real eta_hat = eta_n - eta_p - Qnp / temp;

      // !Janka takes into account the Pauli blocking effect for
      // !degenerate nucleons as in Bruenn (1985). Ruffert et al. Eq. (A8)
      // !xp = xp / (1.0d0 + 2.0d0 / 3.0d0 * (max(eta_p, 0.0d0)))
      // !xn = xn / (1.0d0 + 2.0d0 / 3.0d0 * (max(eta_n, 0.0d0)))

      // !Consistency check on the fractions
      xp = std::max(0.0, xp);
      xn = std::max(0.0, xn);
      xh = std::max(0.0, xh);
      abar = std::max(0.0, abar);
      zbar = std::max(0.0, zbar);

      // eta takes into account the nucleon final state blocking
      // (at high density)
      eta_np = nb * (xp-xn) / (exp(-eta_hat) - 1.0);
      eta_pn = nb * (xn-xp) / (exp(eta_hat) - 1.0);

      // !There is no significant defferences between Rosswog (prev. formula)
      // !and Janka's prescriptions
      // ! eta_np = nb * ((2.0d0 * ye-1.0d0) / (exp(eta_p-eta_n) - 1.0d0))
      // ! eta_pn = eta_np * exp(eta_p-eta_n)

      // !See Bruenn (ApJSS 58 1985) formula 3.1, non degenerate matter limit.
      if  (rho < 2.0e11) {
        eta_pn = nb * xp;
        eta_np = nb * xn;
      } // endif

      // !Consistency Eqs (A9) (Rosswog's paper) they should be positive
      eta_pn = std::max(eta_pn, 0.0);
      eta_np = std::max(eta_np, 0.0);

}

    void GetTableLimits(Real& rho_min, Real& rho_max, Real& temp_min, Real& temp_max, Real& ye_min, Real& ye_max) {
      rho_min = table_rho_min;
      rho_max = table_rho_max;

      temp_min = table_temp_min;
      temp_max = table_temp_max;

      ye_min = table_ye_min;
      ye_max = table_ye_max;
    }

    void GetTableLimitsYe(Real& ye_min, Real& ye_max) {
      ye_min = table_ye_min;
      ye_max = table_ye_max;
    }

    bool ApplyTableLimits(Real& rho, Real& temp, Real& ye)
    {
      bool limits_applied = false;

      if (!apply_table_limits_internally)
      {
        return limits_applied;
      }

      if (rho < table_rho_min) {
        rho = table_rho_min;
        limits_applied = true;
      } else if (rho > table_rho_max) {
        rho = table_rho_max;
        limits_applied = true;
      }

      if (temp < table_temp_min) {
        temp = table_temp_min;
        limits_applied = true;
      } else if (temp > table_temp_max) {
        temp = table_temp_max;
        limits_applied = true;
      }

      if (ye < table_ye_min) {
        ye = table_ye_min;
        limits_applied = true;
      } else if (ye > table_ye_max) {
        ye = table_ye_max;
        limits_applied = true;
      }

      if (!enforced_limits_fail)
        limits_applied = false;

      return limits_applied;
    }

  private:
    const bool apply_table_limits_internally;
    const bool enforced_limits_fail;
    const bool wr_use_eos_dfloor;
    const bool wr_use_eos_tfloor;

    Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS;

    WeakRates_Units::UnitSystem* my_units;
    WeakRates_Units::UnitSystem* code_units;

    Real table_rho_min;
    Real table_rho_max;
    Real table_temp_min;
    Real table_temp_max;
    Real table_ye_min;
    Real table_ye_max;

};

}

#endif