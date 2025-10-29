#ifndef BNSNURATES_UNITS_HPP
#define BNSNURATES_UNITS_HPP

//! \file units.hpp
//  \brief contains unit definitions, conversion, and constants for BNSNuRates.
//  
//  Each unit system is defined as its own struct inside the namespace.
//  This is mostly a copy of the units from Primitive Solver
//  This is a copy of the units from WeakRates

#include "../../../athena.hpp"

namespace M1::Opacities::BNSNuRates::BNSNuRates_Units {

struct UnitSystem {
  const Real c;    //! Speed of light
  const Real G;    //! Gravitational constant
  const Real kb;   //! Boltzmann constant
  const Real Msun; //! Solar mass
  const Real MeV;  //! 10^6 electronvolt

  const Real length;            //! Length unit
  const Real time;              //! Time unit
  const Real density;           //! Number density unit
  const Real mass;              //! Mass unit
  const Real energy;            //! Energy unit
  const Real pressure;          //! Pressure unit
  const Real temperature;       //! Temperature unit
  const Real chemicalPotential; //! Chemical potential unit

  //! \defgroup conversiongroup Conversion Methods
  //  A collection of methods for getting unit
  //  conversions from the original system to the
  //  specified system.
  //  \{
  inline constexpr Real LengthConversion(UnitSystem& b) const {
    return b.length/length;
  }

  inline constexpr Real OpacityConversion(UnitSystem& b) const { // number per length
    return length/b.length;
  }

  inline constexpr Real TimeConversion(UnitSystem& b) const {
    return b.time/time;
  }

  inline constexpr Real VelocityConversion(UnitSystem& b) const {
    return b.length/length * time/b.time;
  }

  inline constexpr Real MassConversion(UnitSystem& b) const {
    return b.mass/mass;
  }

  inline constexpr Real MassDensityConversion(UnitSystem& b) const {
    return (b.mass*POW3(length))/(mass*POW3(b.length));
  }

  inline constexpr Real NumberDensityConversion(UnitSystem& b) const { // number per unit volume
    return POW3(length)/POW3(b.length);
  }

  inline constexpr Real EnergyConversion(UnitSystem& b) const {
    return b.energy/energy;
  }

  inline constexpr Real EnergyDensityConversion(UnitSystem& b) const { // energy per unit volume
    return (b.energy*POW3(length))/(energy*POW3(b.length));
  }

  inline constexpr Real EntropyConversion(UnitSystem& b) const {
    return b.kb/kb;
  }

  inline constexpr Real PressureConversion(UnitSystem& b) const {
    return b.pressure/pressure;
  }

  inline constexpr Real TemperatureConversion(UnitSystem& b) const {
    return b.temperature/temperature;
  }

  inline constexpr Real ChemicalPotentialConversion(UnitSystem& b) const {
    return b.chemicalPotential/chemicalPotential;
  }

  inline constexpr Real NumberRateConversion(UnitSystem& b) const { // number per unit time per unit volume
    return (time*POW3(length))/(b.time*POW3(b.length));
  }

  inline constexpr Real EnergyRateConversion(UnitSystem& b) const { // energy per unit time per unit volume
    return (b.energy*time*POW3(length))/(energy*b.time*POW3(b.length));
  }
};
  
// Global static objects for a particular unit system.

//! CGS units
//
//  Fundamental constants are defined using the 2014
//  CODATA values to be consistent with CompOSE. Solar
//  mass is derived from the solar mass parameter given
//  in the 2021 Astronomer's Almanac:
//  GM_S = 1.32712442099e26 cm^3 s^-2
static UnitSystem CGS{
  2.99792458e10, // c, cm/s
  6.67408e-8, // G, cm^3 g^-1 s^-2
  1.38064852e-16, // kb, erg K^-1
  1.98848e33, // Msun, g
  1.6021766208e-6, // MeV, erg

  1.0, // length, cm
  1.0, // time, s
  1.0, // number density, cm^-3
  1.0, // mass, g
  1.0, // energy, erg
  1.0, // pressure, erg/cm^3
  1.0, // temperature, K
  1.0, // chemical potential, erg
};
//! Geometric units with length in solar masses
static UnitSystem GeometricSolar{
  1.0, // c
  1.0, // G
  1.0, // kb
  1.0, // Msun
  CGS.MeV / (CGS.c*CGS.c), // MeV, Msun

  (CGS.c*CGS.c)/(CGS.G * CGS.Msun), // length, Msun
  POW3( CGS.c)/(CGS.G * CGS.Msun), // time, Msun
  POW3( (CGS.G * CGS.Msun)/(CGS.c*CGS.c) ), // number density, Msun^-3 
  1.0 / CGS.Msun, // mass, Msun
  1.0 / (CGS.Msun * CGS.c*CGS.c), // energy, Msun
  POW3( CGS.G/(CGS.c*CGS.c) ) * SQR( CGS.Msun/(CGS.c) ), // pressure, Msun^-2
  CGS.kb/CGS.MeV, // temperature, MeV
  CGS.kb/CGS.MeV, // chemical potential, MeV
};
//! Nuclear units
static UnitSystem Nuclear{
  1.0, // c
  CGS.G * CGS.MeV/(CGS.c*CGS.c*CGS.c*CGS.c)*1e13, // G, fm
  1.0, // kb
  CGS.Msun * (CGS.c*CGS.c) / CGS.MeV, // Msun, MeV
  1.0, // MeV

  1e13, // length, fm
  CGS.c * 1e13, // time, fm
  1e-39, // number density, fm^-3
  (CGS.c*CGS.c) / CGS.MeV, // mass, MeV
  1.0/CGS.MeV, // energy, MeV
  1e-39/CGS.MeV, // pressure, MeV/fm^3
  CGS.kb/CGS.MeV, // temperature, MeV
  CGS.kb/CGS.MeV, // chemical potential, MeV
};
//! Units used in neutrino calculations: mostly CGS but with 
//  temperature and chemical potential in MeV
static UnitSystem WeakRatesUnits{
  CGS.c, // c, cm/s
  CGS.G, // G, cm^3 g^-1 s^-2
  CGS.kb, // kb, erg K^-1
  CGS.Msun, // Msun, g
  CGS.MeV, // MeV, erg

  CGS.length, // length, cm
  CGS.time, // time, s
  CGS.density, // number density, cm^-3
  CGS.mass, // mass, g
  CGS.energy, // energy, erg
  CGS.pressure, // pressure, erg/cm^3
  CGS.kb/CGS.MeV, // temperature, MeV
  CGS.kb/CGS.MeV, // chemical potential, MeV
};
//! NGS unit system (nanometer, gram, second, used by the bns_nurates library)
static UnitSystem NGS{
  CGS.c * 1e7,                                              // c, nm/s
  CGS.G * CGS.MeV / (CGS.c * CGS.c * CGS.c * CGS.c) * 1e7,  // G, nm
  1.0,                                                      // kb
  CGS.Msun * (CGS.c * CGS.c) / CGS.MeV,                     // Msun, MeV
  1.0,                                                      // MeV

  1e7,               // length, nm
  1.0,               // time, s
  1e-21,             // number density nm^-3
  //1e-21 * CGS.density, // mass density, g nm^-3
  1.0,               // mass, g
  1.0 / CGS.MeV,     // energy, MeV
  1e-21 / CGS.MeV,   // pressure, MeV/nm^3
  CGS.kb / CGS.MeV,  // temperature, MeV
  1.0 / CGS.MeV,     // chemical potential, MeV
};

}
#endif
