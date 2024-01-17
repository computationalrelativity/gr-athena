#ifndef WEAKRATES_EMIS_H
#define WEAKRATES_EMIS_H

#include "../../athena.hpp"

#include "weak_eos.hpp"

namespace WeakRates_Emission {

class WeakEmissionMod {
  public:
    // Constructor
    WeakEmissionMod() {}

    // Destructor
    ~WeakEmissionMod() {}

    int NeutrinoEmissionImpl(Real rho, Real temp, Real ye, Real& emi_n_nue, Real& emi_n_nua, Real& emi_n_nux, Real& emi_e_nue, Real& emi_e_nua, Real& emi_e_nux);

   void inline SetEos(WeakRates_EoS::WeakEoSMod* WR_EoS) {
      EoS = WR_EoS;
      atomic_mass = EoS->AtomicMassImpl();
    }

    void inline SetBounds(Real rho_min_cgs, Real temp_min_mev) {
      rho_min = rho_min_cgs;
      temp_min = temp_min_mev;
    }

  private:
    int Emissions_cgs(Real rho, Real temp, Real ye, Real& emi_n_nue, Real& emi_n_nua, Real& emi_n_nux, Real& emi_e_nue, Real& emi_e_nua, Real& emi_e_nux);

    WeakRates_EoS::WeakEoSMod* EoS;
    Real atomic_mass;
    Real rho_min;
    Real temp_min;

    //.....from units.F90
    const Real clight = 2.99792458e10;
    const Real mev_to_erg = 1.60217733e-6;     // conversion from MeV to erg
    const Real hc_mevcm = 1.23984172e-10;      // hc in units of MeV*cm
    const Real pi    = 3.14159265358979323846; // pi
    Real me_erg = 8.187108692567103e-07; // mass of the electron in erg
    Real sigma_0 = 1.76e-44; // cross section in unit of cm^2
    Real alpha = 1.23e0; // dimensionless
    Real Cv = 0.5 + 2.0*0.23; // vector  const. dimensionless
    Real Ca = 0.5; //axial const. dimensionless
    Real gamma_0 = 5.565e-2; // dimensionless
    Real fsc = 1.0/137.036; // fine structure constant, dimensionless
    

};

} // namespace WeakRates_Emission

#endif