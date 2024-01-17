#ifndef WEAKRATES_OPAC_H
#define WEAKRATES_OPAC_H

#include "../../athena.hpp"
#include "weak_eos.hpp"

namespace WeakRates_Opacity {

class WeakOpacityMod {
  public:
    // Constructor
    WeakOpacityMod() {}

    // Destructor
    ~WeakOpacityMod() {}

    int NeutrinoAbsorptionOpacityImpl(Real rho, Real temp, Real ye, Real& abs_n_nue, Real& abs_n_nua, Real& abs_n_nux, Real& abs_e_nue, Real& abs_e_nua, Real& abs_e_nux);

    int NeutrinoScatteringOpacityImpl(Real rho, Real temp, Real ye, Real& sct_n_nue, Real& sct_n_nua, Real& sct_n_nux, Real& sct_e_nue, Real& sct_e_nua, Real& sct_e_nux);

    // Old functions
    int NeutrinoOpacityImpl(Real rho, Real temp, Real ye, Real& kappa_0_nue, Real& kappa_0_nua, Real& kappa_0_nux, Real& kappa_1_nue, Real& kappa_1_nua, Real& kappa_1_nux);

    int NeutrinoAbsorptionRateImpl(Real rho, Real temp, Real ye, Real& abs_0_nue, Real& abs_0_nua, Real& abs_0_nux, Real& abs_1_nue, Real& abs_1_nua, Real& abs_1_nux);

    void inline SetEos(WeakRates_EoS::WeakEoSMod* WR_EoS) {
      EoS = WR_EoS;
      atomic_mass = EoS->AtomicMassImpl();
    }

    void inline SetBounds(Real rho_min_cgs, Real temp_min_mev) {
      rho_min = rho_min_cgs;
      temp_min = temp_min_mev;
    }

  private:
    int Opacities_cgs(Real rho, Real temp, Real ye, Real& kappa_0_nue, Real& kappa_0_nua, Real& kappa_0_nux, Real& kappa_1_nue, Real& kappa_1_nua, Real& kappa_1_nux);
    int Absorption_cgs(Real rho, Real temp, Real ye, Real& kappa_0_nue_abs, Real& kappa_0_nua_abs, Real& kappa_0_nux_abs, Real& kappa_1_nue_abs, Real& kappa_1_nua_abs, Real& kappa_1_nux_abs);
    int Scattering_cgs(Real rho, Real temp, Real ye, Real& kappa_0_nue_sct, Real& kappa_0_nua_sct, Real& kappa_0_nux_sct, Real& kappa_1_nue_sct, Real& kappa_1_nua_sct, Real& kappa_1_nux_sct);
    // Moved to EoS
    // void GetEtas(Real rho, Real temp, Real ye, Real& eta_nue, Real& eta_nua, Real& eta_nux, Real& eta_e, Real& eta_np, Real& eta_pn);

    // from Units.F90
    Real mev_to_erg = 1.60217733e-6;
    Real me_erg = 8.187108692567103e-07; // mass of the electron in erg
    Real sigma_0 = 1.76e-44; // cross section in unit of cm^2
    Real alpha = 1.23e0; // dimensionless
    Real Cv = 0.5 + 2.0*0.23; // vector  const. dimensionless

    WeakRates_EoS::WeakEoSMod* EoS;
    Real atomic_mass;
    Real rho_min;
    Real temp_min;
};

} // namespace WeakRates_Emission

#endif