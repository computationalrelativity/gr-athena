#ifndef UNIT_CONVERT_H
#define UNIT_CONVERT_H

#include <cassert>

// Athena++ headers
#include "../../../athena_aliases.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../../eos/eos.hpp"

// Weakrates headers
#include "weak_rates.hpp"
#include "units.hpp"

// ----------------------------------------------------------------------------

using namespace M1::Opacities::WeakRates;

namespace UnitConvert {

struct unit_base {
  Real number_density       = -1;
  Real number_rate          = -1;
  Real energy               = -1;
  Real energy_density       = -1;
  Real energy_rate          = -1;
  Real mass_density         = -1;
  Real mass                 = -1;
  Real pressure             = -1;
  Real temperature          = -1;
  Real opacity              = -1;

  std::string name;
  bool have_printed = false;

  void print_me()
  {
    #pragma omp critical
    {
      std::printf("-------------------------------------------------------\n");
      std::printf("@unit_base: [* src->tar %s]\n", name.c_str());
      std::printf("%-22s % .16e\n", "number_density:",      number_density);
      std::printf("%-22s % .16e\n", "number_rate:",         number_rate);
      std::printf("%-22s % .16e\n", "energy:",              energy);
      std::printf("%-22s % .16e\n", "energy_density:",      energy_density);
      std::printf("%-22s % .16e\n", "energy_rate:",         energy_rate);
      std::printf("%-22s % .16e\n", "mass_density:",        mass_density);
      std::printf("%-22s % .16e\n", "mass:",                mass);
      std::printf("%-22s % .16e\n", "pressure:",            pressure);
      std::printf("%-22s % .16e\n", "temperature:",         temperature);
      std::printf("%-22s % .16e\n", "opacity:",             opacity);
      std::printf("-------------------------------------------------------\n");
    }
  }
};

class UnitConvert {

  // variables ----------------------------------------------------------------
  public:
    Mesh *pm;
    ParameterInput *pin;
    MeshBlock *pmb;

    // Store the unit conversions
    unit_base gra_to_gra_wr_impl;
    unit_base gra_wr_impl_to_gra;

    unit_base thc_to_thc_wr_impl;
    unit_base thc_wr_impl_to_thc;

    unit_base thc_wr_impl_to_gra_wr_impl;
    unit_base gra_wr_impl_to_thc_wr_impl;

    unit_base gra_to_thc;
    unit_base thc_to_gra;

    // Weak rates units config.
    Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>* PS_EoS;
    WeakRates_Units::UnitSystem* wr_my_units;
    WeakRates_Units::UnitSystem* wr_code_units;

  // functions ----------------------------------------------------------------
  public:
    struct
    {
      // Conversion factors
      Real cactus2cgsRho    = 6.176269145886163e+17;
      Real cactus2cgsEps    = 8.98755178736817e+20;

      Real cgs2cactusRho    = 1.619100425158886e-18;
      Real cgs2cactusPress  = 1.8014921788094724e-39;
      Real cgs2cactusEps    = 1.112650056053618e-21;
      Real cgs2cactusMass   = 5.0278543128934301e-34;
      Real cgs2cactusEnergy = 5.5942423830703013e-55;
      Real cgs2cactusTime   = 203012.91587112966;
      Real cgs2cactusLength = 6.7717819596091924e-06;

      // additional scaling (enters e.g. mb)
      Real normfact = 1e50;

      // Fundamental constants
      Real pi = 3.14159265358979;
      Real mev_to_erg = 1.60217733e-6;
      Real erg_to_mev = 6.24150636e5;
      Real amu_cgs = 1.66053873e-24; // !Atomic mass unit in g
      Real amu_mev = 931.49432; // !Atomic mass unit in MeV
      Real kb_erg = 1.380658e-16; // !Boltzmann constant in erg
      Real kb_mev = 8.61738568e-11; // !Boltzmann constant in MeV
      Real temp_mev_to_kelvin = 1.1604447522806e10;
      Real fermi_cubed_cgs = 1.e-39;
      Real clight = 2.99792458e10;
      Real h_MeVs = 4.1356943e-21; // !Planck constant in MeV
      Real me_mev = 0.510998910; // !mass of the electron in MeV
      Real me_erg = 8.187108692567103e-07; // !mass of the electron in erg
      Real sigma_0 = 1.76e-44; // !cross section in unit of cm^2
      Real alpha = 1.23; // !dimensionless
      Real multipl_nuea = 1; // !multiplicity factor for nu_e and anti nu_e
      Real multipl_nux = 2; // !multiplicity factor for nu_x
      Real Qnp = 1.293333; // !neutron-proton mass difference in MeV
      Real hc_mevcm = 1.23984172e-10; // !hc in units of MeV*cm
      Real hc_ergvcm = 1.986445683269303e-16; // !hc in units of erg*cm/s
      Real Cv = 0.5 + 2.0*0.23; // !vector  const. dimensionless
      Real Ca = 0.5; // !axial const. dimensionless
      Real gamma_0 = 5.565e-2; // ! dimensionless
      Real fsc = 1.0/137.036; // !fine structure constant, dimensionless
      Real planck = 6.626176e-27; // !Planck constant erg s
      Real avo = 6.0221367e23; // !Avogadro's number mol^-1
      Real Ggrav = 6.6742e-8;
      Real solarMass = 1.9891e33;
    } thc_constants;

  private:

    // Make the unit conversions
    void make_product_system(
      unit_base & tar,
      unit_base & src_A,
      unit_base & src_B)
    {
      tar.number_density = src_A.number_density * src_B.number_density;
      tar.number_rate = src_A.number_rate * src_B.number_rate;

      tar.energy = src_A.energy * src_B.energy;
      tar.energy_density = src_A.energy_density * src_B.energy_density;
      tar.energy_rate = src_A.energy_rate * src_B.energy_rate;

      tar.mass_density = src_A.mass_density * src_B.mass_density;
      tar.mass = src_A.mass * src_B.mass;

      tar.pressure = src_A.pressure * src_B.pressure;
      tar.temperature = src_A.temperature * src_B.temperature;
      tar.opacity = src_A.opacity * src_B.opacity;
    }

    void make_inverse_system(
      unit_base & tar,
      unit_base & src
    )
    {
      tar.number_density = 1.0 / src.number_density;
      tar.number_rate = 1.0 / src.number_rate;

      tar.energy = 1.0 / src.energy;
      tar.energy_density = 1.0 / src.energy_density;
      tar.energy_rate = 1.0 / src.energy_rate;

      tar.mass_density = 1.0 / src.mass_density;
      tar.mass = 1.0 / src.mass;

      tar.pressure = 1.0 / src.pressure;
      tar.temperature = 1.0 / src.temperature;
      tar.opacity = 1.0 / src.opacity;
    }

    void make_gra_to_gra_wr_impl()
    {
      gra_to_gra_wr_impl.number_density = (
        wr_code_units->NumberDensityConversion(*wr_my_units)
      );
      gra_to_gra_wr_impl.number_rate = (
        wr_code_units->NumberRateConversion(*wr_my_units)
      );

      gra_to_gra_wr_impl.energy = (
        wr_code_units->EnergyConversion(*wr_my_units)
      );
      gra_to_gra_wr_impl.energy_density = (
        wr_code_units->EnergyDensityConversion(*wr_my_units)
      );
      gra_to_gra_wr_impl.energy_rate = (
        wr_code_units->EnergyRateConversion(*wr_my_units)
      );

      gra_to_gra_wr_impl.opacity = (
        wr_code_units->OpacityConversion(*wr_my_units)
      );

      gra_to_gra_wr_impl.mass = (
        wr_code_units->MassConversion(*wr_my_units)
      );
      gra_to_gra_wr_impl.mass_density = (
        wr_code_units->MassDensityConversion(*wr_my_units)
      );

      gra_to_gra_wr_impl.pressure = (
        wr_code_units->PressureConversion(*wr_my_units)
      );

      gra_to_gra_wr_impl.temperature = (
        wr_code_units->TemperatureConversion(*wr_my_units)
      );

      gra_to_gra_wr_impl.name = "gra_to_gra_wr_impl";
    };

    void make_gra_wr_to_gra_impl()
    {
      gra_wr_impl_to_gra.number_density = (
        wr_my_units->NumberDensityConversion(*wr_code_units)
      );
      gra_wr_impl_to_gra.number_rate = (
        wr_my_units->NumberRateConversion(*wr_code_units)
      );

      gra_wr_impl_to_gra.energy = (
        wr_my_units->EnergyConversion(*wr_code_units)
      );
      gra_wr_impl_to_gra.energy_density = (
        wr_my_units->EnergyDensityConversion(*wr_code_units)
      );
      gra_wr_impl_to_gra.energy_rate = (
        wr_my_units->EnergyRateConversion(*wr_code_units)
      );

      gra_wr_impl_to_gra.opacity = (
        wr_my_units->OpacityConversion(*wr_code_units)
      );

      gra_wr_impl_to_gra.mass = (
        wr_my_units->MassConversion(*wr_code_units)
      );
      gra_wr_impl_to_gra.mass_density = (
        wr_my_units->MassDensityConversion(*wr_code_units)
      );

      gra_wr_impl_to_gra.pressure = (
        wr_my_units->PressureConversion(*wr_code_units)
      );

      gra_wr_impl_to_gra.temperature = (
        wr_my_units->TemperatureConversion(*wr_code_units)
      );

      gra_wr_impl_to_gra.name = "gra_wr_impl_to_gra";
    };

    // THC and its weakrates implementation -----------------------------------
    void make_thc_wr_impl_to_thc()
    {
      // Constants from:
      //   thcsupport/Units/src/units.F90
      //   WeakRates/src/utils/weakrates_calc_rates.F90
      //   WeakRates/src/weakrates_impl.F90

      const Real r_cgs2cactus = 1.0 / (
        thc_constants.cgs2cactusTime *
        std::pow(thc_constants.cgs2cactusLength, 3.0)
      );
      const Real q_cgs2cactus = (
        thc_constants.mev_to_erg * thc_constants.cgs2cactusEnergy
      ) / (
        thc_constants.cgs2cactusTime *
        std::pow(thc_constants.cgs2cactusLength, 3.0)
      );
      const Real kappa_cgs2cactus = 1.0 / thc_constants.cgs2cactusLength;

      // Cf. NeutrinoDensityImpl:
      thc_wr_impl_to_thc.number_density = 1.0 / (
        std::pow(thc_constants.cgs2cactusLength, 3.0) *
        thc_constants.normfact
      );

      // Cf. NeutrinoEmissionImpl:
      thc_wr_impl_to_thc.number_rate = r_cgs2cactus / thc_constants.normfact;

      // Cf. WeakRates/src/utils/weakrates_calc_rates.F90
      thc_wr_impl_to_thc.energy = thc_constants.cgs2cactusEps;

      // Cf. NeutrinoDensityImpl:
      thc_wr_impl_to_thc.energy_density = (
        thc_constants.cgs2cactusEnergy
      ) / (
        std::pow(thc_constants.cgs2cactusLength, 3.0)
      );

      // Cf. NeutrinoEmissionImpl:
      // Note that "Emissions_cgs" outputs in MeV thus we add the the factor
      // below as we assume that wr internals are cgs
      thc_wr_impl_to_thc.energy_rate = (
        1.0 / thc_constants.mev_to_erg *  // see above
        q_cgs2cactus
      );

      // Cf. NeutrinoOpacityImpl:
      thc_wr_impl_to_thc.mass_density = 1.0 / thc_constants.cactus2cgsRho;
      // thc_wr_impl_to_thc.mass                 = -1;

      // Cf. WeakRates/src/utils/weakrates_calc_rates.F90
      thc_wr_impl_to_thc.pressure = thc_constants.cgs2cactusPress;
      thc_wr_impl_to_thc.temperature = 1;

      // Cf. NeutrinoOpacityImpl:
      thc_wr_impl_to_thc.opacity = kappa_cgs2cactus;

      thc_wr_impl_to_thc.name = "thc_wr_impl_to_thc";
    };

    void make_thc_to_thc_wr_impl()
    {
      // Cf. NeutrinoOpacityImpl:
      thc_to_thc_wr_impl.mass_density = thc_constants.cactus2cgsRho;
      thc_to_thc_wr_impl.temperature  = 1;
      thc_to_thc_wr_impl.name = "thc_to_thc_wr_impl";
    };

    void fill_missing_thc_wr_impl()
    {
      // populate the missing factors in the above
      thc_to_thc_wr_impl.number_density = (
        1.0 / thc_wr_impl_to_thc.number_density
      );
      thc_to_thc_wr_impl.number_rate = 1.0 / thc_wr_impl_to_thc.number_rate;

      thc_to_thc_wr_impl.energy = 1.0 / thc_wr_impl_to_thc.energy;
      thc_to_thc_wr_impl.energy_density = (
        1.0 / thc_wr_impl_to_thc.energy_density
      );
      thc_to_thc_wr_impl.energy_rate = 1.0 / thc_wr_impl_to_thc.energy_rate;

      thc_to_thc_wr_impl.mass = 1.0 / thc_wr_impl_to_thc.mass;
      thc_to_thc_wr_impl.pressure = 1.0 / thc_wr_impl_to_thc.pressure;

      thc_to_thc_wr_impl.opacity = 1.0 / thc_wr_impl_to_thc.opacity;
    }
    // ------------------------------------------------------------------------

    void make_thc_wr_impl_to_gra_wr_impl()
    {
      // Can only differ from unity if there is a mismatch in definition of
      // Fundamental constants etc.

      thc_wr_impl_to_gra_wr_impl.number_density = 1;
      thc_wr_impl_to_gra_wr_impl.number_rate    = 1;
      thc_wr_impl_to_gra_wr_impl.energy         = 1;
      thc_wr_impl_to_gra_wr_impl.energy_density = 1;
      thc_wr_impl_to_gra_wr_impl.energy_rate    = 1;
      thc_wr_impl_to_gra_wr_impl.mass_density   = 1;
      thc_wr_impl_to_gra_wr_impl.mass           = 1;
      thc_wr_impl_to_gra_wr_impl.pressure       = 1;
      thc_wr_impl_to_gra_wr_impl.temperature    = 1;
      thc_wr_impl_to_gra_wr_impl.opacity        = 1;

      thc_wr_impl_to_gra_wr_impl.name = "thc_wr_impl_to_gra_wr_impl";
    }

    void make_gra_wr_impl_to_thc_wr_impl()
    {
      make_inverse_system(
        gra_wr_impl_to_thc_wr_impl,
        thc_wr_impl_to_gra_wr_impl
      );

      gra_wr_impl_to_thc_wr_impl.name = "gra_wr_impl_to_thc_wr_impl";
    }

    void make_gra_to_thc()
    {
      unit_base prod;

      make_product_system(
        prod,
        gra_to_gra_wr_impl,
        gra_wr_impl_to_thc_wr_impl
      );

      make_product_system(
        gra_to_thc,
        prod,
        thc_wr_impl_to_thc
      );

      gra_to_thc.name = "gra_to_thc";
    }

    void make_thc_to_gra()
    {
      make_inverse_system(thc_to_gra, gra_to_thc);
      thc_to_gra.name = "thc_to_gra";
    }

    // void make_newX_to_newY()
    // {
    //   // struct [fill here]:
    //   // name.number_density = -1;
    //   // name.number_rate    = -1;
    //   // name.energy         = -1;
    //   // name.energy_density = -1;
    //   // name.energy_rate    = -1;
    //   // name.mass_density   = -1;
    //   // name.mass           = -1;
    //   // name.pressure       = -1;
    //   // name.temperature    = -1;
    //   // name.opacity        = -1;

    //   // name.name = "name_of_this_system";
    // }

  public:
    UnitConvert(ParameterInput * pin, MeshBlock * pmb)
      : pin(pin),
        pmb(pmb),
        pm(pmb->pmy_mesh)
    {
      // Weak rates config.
      PS_EoS = &pmb->peos->GetEOS();
      wr_code_units = &WeakRates_Units::GeometricSolar;
      wr_my_units = &WeakRates_Units::WeakRatesUnits;

      // Set up the unit conversions.
      make_gra_to_gra_wr_impl();
      make_gra_wr_to_gra_impl();

      make_thc_wr_impl_to_thc();
      make_thc_to_thc_wr_impl();
      fill_missing_thc_wr_impl();

      make_thc_wr_impl_to_gra_wr_impl();
      make_gra_wr_impl_to_thc_wr_impl();

      make_gra_to_thc();
      make_thc_to_gra();

      // checks (should yield unity)
      /*
      unit_base thc_wr;
      make_product_system(thc_wr, thc_to_thc_wr_impl, thc_wr_impl_to_thc);
      thc_wr.name = "thc_consistency";
      thc_wr.print_me();

      unit_base gra_wr;
      make_product_system(gra_wr, gra_to_gra_wr_impl, gra_wr_impl_to_gra);
      gra_wr.name = "gra_consistency";
      gra_wr.print_me();
      */
    }

    ~UnitConvert() {};

  public:
    void print_once(unit_base * ubase)
    {
      if (ubase->have_printed)
        return;

      ubase->print_me();
      ubase->have_printed = true;
    }

};

} // namespace UnitConvert

#endif // UNIT_CONVERT_H

//
// :(
//