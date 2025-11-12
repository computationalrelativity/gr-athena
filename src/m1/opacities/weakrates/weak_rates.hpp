#ifndef WEAKRATES_H
#define WEAKRATES_H

#include <cassert>

// Athena++ headers
#include "../../m1.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../../eos/eos.hpp"
#include "../../../athena.hpp"
#include "../../../defs.hpp"

// Weakrates headers
#include "weak_eos.hpp"
#include "weak_emission.hpp"
#include "weak_opacity.hpp"
#include "weak_equilibrium.hpp"
#include "units.hpp"

namespace M1::Opacities::WeakRates::WeakRatesNeutrinos{

// TODO unit conversion takes place here
class WeakRates {
  public:
  // Constructor
    WeakRates(ParameterInput *pin,
              Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS)
      : pin(pin),
        PS_EoS(PS_EoS),
        apply_table_limits_internally(pin->GetOrAddBoolean("M1_opacities", "apply_table_limits_internally", true)),
        enforced_limits_fail(pin->GetOrAddBoolean("M1_opacities", "enforced_limits_fail", true)),
        wr_zero_limits(pin->GetOrAddBoolean("M1_opacities", "wr_zero_limits", true)),
        wr_use_eos_dfloor(pin->GetOrAddBoolean("M1_opacities", "wr_use_eos_dfloor", false)),
        wr_use_eos_tfloor(pin->GetOrAddBoolean("M1_opacities", "wr_use_eos_tfloor", false)),
        WR_EoS {apply_table_limits_internally,
                enforced_limits_fail,
                wr_use_eos_dfloor,
                wr_use_eos_tfloor,
                pin->GetOrAddBoolean("M1_opacities", "tabulated_particle_fractions", true),
                PS_EoS}
    {
      // my_units (of WeakRates) vs code_units (of GR(M)HD)
      my_units   = &WeakRates_Units::WeakRatesUnits;
      code_units = &WeakRates_Units::GeometricSolar;

      // set up internal settings for EOS (unit conversions & limits);
      // target is cgs, MeV resp.
      Real conv_rho = PS_EoS->GetBaryonMass()*code_units->MassDensityConversion(*my_units);
      Real conv_temp = code_units->TemperatureConversion(*my_units);

      if (wr_use_eos_dfloor)
      {
        // this returns n_atm, but the conversion takes care of it
        rho_min = conv_rho * PS_EoS->GetDensityFloor();
      }
      else
      {
        // not clear that this should be 0
        if (wr_zero_limits)
        {
          rho_min = 0;
        }
        else
        {
          rho_min = conv_rho * PS_EoS->GetMinimumDensity();
        }
      }

      rho_min_code_units = rho_min / conv_rho;

      if (wr_use_eos_tfloor)
      {
        temp_min = conv_temp * PS_EoS->GetTemperatureFloor();
      }
      else
      {
        // not clear that this should be 0
        if (wr_zero_limits)
        {
          temp_min = 0;
        }
        else
        {
          temp_min = conv_temp * PS_EoS->GetMinimumTemperature();
        }
      }

      WR_Emission.SetBounds(rho_min, temp_min);
      WR_Opacity.SetBounds(rho_min, temp_min);
      WR_Equilibrium.SetBounds(rho_min, temp_min);

      // internal Emission, Opacity, Equilibrium
      atomic_mass = WR_EoS.AtomicMassImpl();

      WR_Emission.SetEos(&WR_EoS);
      WR_Opacity.SetEos(&WR_EoS);
      WR_Equilibrium.SetEos(&WR_EoS);
    }

    // Destructor
    ~WeakRates() {
    }

    /*
    I have adopted the following naming convention for the transport coefficient variables:
    <variable>_<type>_<species>

    <variable> = [emi, abs, sct] for emission, absorption opacity, and scattering opacty respectively
    <type> = [n, e] for number and energy transport respectively
    <species> = [nue, nua, nux] for electron-, electron-anti-, and heavy-lepton neutrinos respectively


    For the densities and energies themselves I have used:
    <type>_<species>(_eq)

    for the same type and species above, and the (_eq) suffix denotes an equilibriated variable.
    */

    // Calculate the energy and number emission rates
    int NeutrinoEmission(Real rho, Real temp, Real ye, Real& emi_n_nue, Real& emi_n_nua, Real& emi_n_nux, Real& emi_e_nue, Real& emi_e_nua, Real& emi_e_nux) {
      int ierr = WR_Emission.NeutrinoEmissionImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units), 
        ye, // dimensionless 
        emi_n_nue, 
        emi_n_nua, 
        emi_n_nux, 
        emi_e_nue, 
        emi_e_nua, 
        emi_e_nux);
      // TODO output units like this?
      Real n_rate_conv = my_units->NumberRateConversion(*code_units);
      Real e_rate_conv = my_units->EnergyRateConversion(*code_units);
      emi_n_nue = emi_n_nue * n_rate_conv;
      emi_n_nua = emi_n_nua * n_rate_conv;
      emi_n_nux = emi_n_nux * n_rate_conv;
      emi_e_nue = emi_e_nue * e_rate_conv;
      emi_e_nua = emi_e_nua * e_rate_conv;
      emi_e_nux = emi_e_nux * e_rate_conv;

      return ierr;
    }

    // Calculate the absorption opacities of the neutrinos
    int NeutrinoAbsorptionOpacity(Real rho, Real temp, Real ye, Real& abs_n_nue, Real& abs_n_nua, Real& abs_n_nux, Real& abs_e_nue, Real& abs_e_nua, Real& abs_e_nux) {
      int ierr = WR_Opacity.NeutrinoAbsorptionOpacityImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
        ye,
        abs_n_nue,
        abs_n_nua,
        abs_n_nux,
        abs_e_nue,
        abs_e_nua,
        abs_e_nux);

      Real k_conv = my_units->OpacityConversion(*code_units);
      abs_n_nue = abs_n_nue*k_conv;
      abs_n_nua = abs_n_nua*k_conv;
      abs_n_nux = abs_n_nux*k_conv;
      abs_e_nue = abs_e_nue*k_conv;
      abs_e_nua = abs_e_nua*k_conv;
      abs_e_nux = abs_e_nux*k_conv;

      return ierr;
    }

    // Calculate the scattering opacities of the neutrinos
    int NeutrinoScatteringOpacity(Real rho, Real temp, Real ye, Real& sct_n_nue, Real& sct_n_nua, Real& sct_n_nux, Real& sct_e_nue, Real& sct_e_nua, Real& sct_e_nux) {
      int ierr = WR_Opacity.NeutrinoScatteringOpacityImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
        ye,
        sct_n_nue,
        sct_n_nua,
        sct_n_nux,
        sct_e_nue,
        sct_e_nua,
        sct_e_nux);

      Real k_conv = my_units->OpacityConversion(*code_units);
      sct_n_nue = sct_n_nue*k_conv;
      sct_n_nua = sct_n_nua*k_conv;
      sct_n_nux = sct_n_nux*k_conv;
      sct_e_nue = sct_e_nue*k_conv;
      sct_e_nua = sct_e_nua*k_conv;
      sct_e_nux = sct_e_nux*k_conv;

      return ierr;
    }

    // Calculate the neutrino number and energy densities assuming equilibrium with the fluid (mu_nue = -mu_n + mu_p + mu_e).
    int NeutrinoDensity(Real rho, Real temp, Real ye, Real& n_nue, Real& n_nua, Real& n_nux, Real& e_nue, Real& e_nua, Real& e_nux) {
      int ierr = WR_Equilibrium.NeutrinoDensityImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
        ye,
        n_nue,
        n_nua,
        n_nux,
        e_nue,
        e_nua,
        e_nux);
      // TODO output units?
      Real n_conv = my_units->NumberDensityConversion(*code_units);
      Real e_conv = my_units->EnergyDensityConversion(*code_units);
      n_nue = n_nue*n_conv;
      n_nua = n_nua*n_conv; 
      n_nux = n_nux*n_conv; 
      e_nue = e_nue*e_conv; 
      e_nua = e_nua*e_conv; 
      e_nux = e_nux*e_conv;

      return ierr;
    }

    // Calculate the equilibrium fluid temperature and electron fraction, and neutrino number and energy densities assuming energy and lepton number conservation.
    int WeakEquilibrium(Real rho, Real temp, Real ye, Real n_nue, Real n_nua, Real n_nux, Real e_nue, Real e_nua, Real e_nux, Real& temp_eq, Real& ye_eq, Real& n_nue_eq, Real& n_nua_eq, Real& n_nux_eq, Real& e_nue_eq, Real& e_nua_eq, Real& e_nux_eq) {
      Real n_conv = code_units->NumberDensityConversion(*my_units);
      Real e_conv = code_units->EnergyDensityConversion(*my_units);

      int ierr = WR_Equilibrium.WeakEquilibriumImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
        ye,
        n_nue*n_conv,
        n_nua*n_conv,
        n_nux*n_conv,
        e_nue*e_conv,
        e_nua*e_conv,
        e_nux*e_conv,
        temp_eq,
        ye_eq,
        n_nue_eq,
        n_nua_eq,
        n_nux_eq,
        e_nue_eq,
        e_nua_eq,
        e_nux_eq);
        
      temp_eq = temp_eq*my_units->TemperatureConversion(*code_units);
      // ye_eq = ye_eq;
      n_conv = my_units->NumberDensityConversion(*code_units);
      e_conv = my_units->EnergyDensityConversion(*code_units);
      n_nue_eq = n_nue_eq*n_conv;
      n_nua_eq = n_nua_eq*n_conv; 
      n_nux_eq = n_nux_eq*n_conv; 
      e_nue_eq = e_nue_eq*e_conv; 
      e_nua_eq = e_nua_eq*e_conv; 
      e_nux_eq = e_nux_eq*e_conv;

      return ierr;
    }
    
    int NucleiAbar(Real rho, Real temp, Real ye, Real& abar) {
      int ierr = WR_EoS.NucleiAbarImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
        ye,
        abar);
      // TODO output units?

      return ierr;
    }

    Real AverageBaryonMass() {
      Real atomic_mass = WR_EoS.AtomicMassImpl();
      atomic_mass = atomic_mass * my_units->MassConversion(*code_units);
      return atomic_mass;
    }

    int NeutrinoOpacity(Real rho, Real temp, Real ye, Real& kappa_0_nue, Real& kappa_0_nua, Real& kappa_0_nux, Real& kappa_1_nue, Real& kappa_1_nua, Real& kappa_1_nux) {
      int ierr = WR_Opacity.NeutrinoOpacityImpl(
        rho*code_units->MassDensityConversion(*my_units), 
        temp*code_units->TemperatureConversion(*my_units), 
        ye, 
        kappa_0_nue, 
        kappa_0_nua, 
        kappa_0_nux, 
        kappa_1_nue, 
        kappa_1_nua, 
        kappa_1_nux);
      
      Real k_conv = my_units->OpacityConversion(*code_units);
      kappa_0_nue = kappa_0_nue*k_conv;
      kappa_0_nua = kappa_0_nua*k_conv; 
      kappa_0_nux = kappa_0_nux*k_conv; 
      kappa_1_nue = kappa_1_nue*k_conv; 
      kappa_1_nua = kappa_1_nua*k_conv; 
      kappa_1_nux = kappa_1_nux*k_conv;

      return ierr;
    }

    int NeutrinoAbsorptionRate(Real rho, Real temp, Real ye, Real& abs_0_nue, Real& abs_0_nua, Real& abs_0_nux, Real& abs_1_nue, Real& abs_1_nua, Real& abs_1_nux) {
      int ierr = WR_Opacity.NeutrinoAbsorptionRateImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
        ye,
        abs_0_nue,
        abs_0_nua,
        abs_0_nux,
        abs_1_nue,
        abs_1_nua,
        abs_1_nux);
      // TODO output units? Called "rate" here but looks like opacity inside the function.
      Real k_conv = my_units->OpacityConversion(*code_units);
      abs_0_nue = abs_0_nue*k_conv;
      abs_0_nua = abs_0_nua*k_conv; 
      abs_0_nux = abs_0_nux*k_conv; 
      abs_1_nue = abs_1_nue*k_conv; 
      abs_1_nua = abs_1_nua*k_conv; 
      abs_1_nux = abs_1_nux*k_conv;

      return ierr;
    }

    inline void SetCodeUnitSystem(WeakRates_Units::UnitSystem* units) {
      code_units = units;
    }

    inline WeakRates_Units::UnitSystem* GetCodeUnitSystem() const {
      return code_units;
    }

    inline WeakRates_Units::UnitSystem* GetWRUnitSystem() const {
      return my_units;
    }

    Real get_rho_min_cgs() {return rho_min;}
    Real get_temp_min_mev() {return temp_min;}

    Real get_rho_min()
    {
      return rho_min_code_units;
    }

  private:
    ParameterInput *pin;
    const bool apply_table_limits_internally;
    const bool enforced_limits_fail;
    const bool wr_zero_limits;
    const bool wr_use_eos_dfloor;
    const bool wr_use_eos_tfloor;
    Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS;

    WeakRates_Units::UnitSystem* my_units;
    WeakRates_Units::UnitSystem* code_units;

    WeakRates_EoS::WeakEoSMod WR_EoS;
    WeakRates_Emission::WeakEmissionMod WR_Emission;
    WeakRates_Opacity::WeakOpacityMod WR_Opacity;
    WeakRates_Equilibrium::WeakEquilibriumMod WR_Equilibrium;

  public:
    Real rho_min;
    Real temp_min;
    Real atomic_mass;

    Real rho_min_code_units;
};

} // namespace Neutrinos_WeakRates

#endif
