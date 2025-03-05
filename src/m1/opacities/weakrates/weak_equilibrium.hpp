#ifndef WEAKRATES_EQUI_H
#define WEAKRATES_EQUI_H

#include <cmath>

#include "../../../athena.hpp"
#include "weak_eos.hpp"

namespace M1::Opacities::WeakRates::WeakRates_Equilibrium {

class WeakEquilibriumMod {
  public:
    // Constructor
    WeakEquilibriumMod() {}

    // Destructor
    ~WeakEquilibriumMod() {}

    int WeakEquilibriumImpl(Real rho, Real temp, Real ye, Real n_nue, Real n_nua, Real n_nux, Real e_nue, Real e_nua, Real e_nux, Real& temp_eq, Real& ye_eq, Real& n_nue_eq, Real& n_nua_eq, Real& n_nux_eq, Real& e_nue_eq, Real& e_nua_eq, Real& e_nux_eq);

    int NeutrinoDensityImpl(Real rho, Real temp, Real ye, Real& n_nue, Real& n_nua, Real& n_nux, Real& e_nue, Real& e_nua, Real& e_nux);

    void inline SetEos(WeakRates_EoS::WeakEoSMod* WR_EoS) {
      EoS = WR_EoS;
      atomic_mass = EoS->AtomicMassImpl();

      EoS->GetTableLimits(eos_rhomin, eos_rhomax, eos_tempmin, eos_tempmax, eos_yemin, eos_yemax);
    }

    void inline SetBounds(Real rho_min_cgs, Real temp_min_mev) {
      rho_min = rho_min_cgs;
      temp_min = temp_min_mev;
    }

  private:
    WeakRates_EoS::WeakEoSMod* EoS; // pointer to allow EoS calls
    Real rho_min;                  // density below which nothing is done in g cm-3
    Real temp_min;                 // temperature below which nothing is done in g cm-3
    Real atomic_mass;              // atomic mass in g (to convert mass density to number density)


    //.....some parameters later used in the calculations....................
    const Real eps_lim  = 1.e-7; // standard tollerance in 2D NR
    const int n_cut_max = 8;     // number of bisections of dx
    const int n_max     = 100;   // Newton-Raphson max number of iterations
    static const int n_at      = 16;    // number of independent initial guesses

    //.....deltas to compute numerical derivatives in the EOS tables.........
    const Real delta_ye = 0.005;
    const Real delta_t  = 0.01;

    //.....from units.F90
    const Real clight = 2.99792458e10;
    const Real mev_to_erg = 1.60217733e-6;     // conversion from MeV to erg
    const Real hc_mevcm = 1.23984172e-10;      // hc in units of MeV*cm
    const Real pi    = 3.14159265358979323846; // pi

    //.....TODO: these should come from the EoS  // TODO: mass_fact is set in the table read
    const Real mass_fact = 9.223158894119980e2;
    Real eos_rhomin;
    Real eos_rhomax;
    Real eos_tempmin;
    Real eos_tempmax;
    Real eos_yemin;
    Real eos_yemax;

#define WR_SQR(x) ((x)*(x))
#define WR_CUBE(x) ((x)*(x)*(x))
#define WR_QUAD(x) ((x)*(x)*(x)*(x))

    //.....some constants....................................................
    const Real pi2   = WR_SQR(pi);                          // pi**2 [-]
    const Real pref1 = 4.0/3.0*pi/WR_CUBE(hc_mevcm);        // 4/3 *pi/(hc)**3 [MeV^3/cm^3]
    const Real pref2 = 4.0*pi*mev_to_erg/WR_CUBE(hc_mevcm); // 4*pi/(hc)**3 [erg/MeV^4/cm^3]
    const Real cnst1 = 7.0*WR_QUAD(pi)/20.0;                // 7*pi**4/20 [-]
    const Real cnst5 = 7.0*WR_QUAD(pi)/60.0;                // 7*pi**4/60 [-]
    const Real cnst6 = 7.0*WR_QUAD(pi)/30.0;                // 7*pi**4/30 [-]
    const Real cnst2 = 7.0*WR_QUAD(pi)/5.0;                 // 7*pi**4/5 [-]
    const Real cnst3 = 7.0*WR_QUAD(pi)/15.0;                // 7*pi**4/15 [-]
    const Real cnst4 = 14.0*WR_QUAD(pi)/15.0;               // 14*pi**4/15 [-]

#undef WR_SQR
#undef WR_CUBE
#undef WR_QUAD

    //.....variable to switch between analytic and numerical solutions of Fermi integrals
    // const bool fermi_analytics = true;
#define WR_FERMI_ANALYTIC 1

    /*    
    // The following are used for exact calculation of fermi integrals, 
    // which are not implemented (yet?).
    //.....number of points in Gauss-Legendre integration....................
    static const int ngl = 64;
    const Real gl_eps=3.0e-14;

    // if I remember correctly, gaulag needs to be in real*8. Thus, I
    // have hardcoded it here (to be checked if compiles). Since we are
    // probably not using numerical Fermi integrals, it could be that
    // it doesn't matter
    Real xgl[ngl];
    Real wgl[ngl];
    bool gl_init = false; // FORTRAN "save" attribute?
    */

    // Primamry implementations
    void weak_equil_wnu(Real rho, Real T, Real y_in[4], Real e_in[4], Real& T_eq, Real y_eq[4], Real e_eq[4], int& na, int& ierr);
    int NeutrinoDens_cgs(Real rho, Real temp, Real ye, Real& n_nue, Real& n_nua, Real& n_nux, Real& en_nue, Real& en_nua, Real& en_nux);

    // aux functions for weak_equil_wnu
    void new_raph_2dim(Real rho, Real u, Real yl, Real x0[2], Real x1[2], int& ierr);
    void func_eq_weak(Real rho, Real u, Real yl, Real x[2], Real y[2]);
    void error_func_eq_weak(Real yl, Real u, Real y[2], Real& err);
    void jacobi_eq_weak(Real rho, Real u, Real yl, Real x[2], Real J[2][2], int& ierr);
    void eta_e_gradient(Real rho, Real t, Real ye, Real eta, Real& detadt, Real& detadye, Real& dedt, Real& dedye, int& ierr);
    void inv_jacobi(Real det, Real J[2][2], Real invJ[2][2]);
    void nu_deg_param_trap(Real temp_m, Real chem_pot[2], Real eta[3]);
    void dens_nu_trap(Real temp_m, Real eta_nu[3], Real nu_dens[3]);
    void edens_nu_trap(Real temp_m, Real eta_nu[3], Real enu_dens[3]);

    // The following two are hard coded not to be used
    //     void f2_analytic(Real eta, Real *f2);
    //     void f3_analytic(Real eta, Real *f3);

    // The following four are used for exact calculation of the fermi 
    // integrals, and unused by default, so they have not been implemented
    //     void fermiint(Real k, Real eta, Real *f);
    //    void kernel(Real k, Real eta, Real x, Real *fcompx);
    //    Real fermi(Real arg);
    //    void gauleg(Real x1, Real x2);

    // aux functions for NeutrinoDens_cgs

}; // class WeakEquilibriumMod

} // namespace WeakRatesNeutrinos

#endif
