#include "weak_equilibrium.hpp"
#include "fermi.hpp"
#include "error_codes.hpp"

using namespace std;
using namespace M1::Opacities::WeakRates::WeakRatesFermi;
using namespace M1::Opacities::WeakRates::WeakRates_Equilibrium;

#define NO_THC_NRG_DENS_FLOOR

int WeakEquilibriumMod::NeutrinoDensityImpl(
  Real rho,     // [g/cm^3] 
  Real temp,    // [MeV]
  Real ye,      // [-]
  Real& n_nue,  // [1/cm^3]
  Real& n_nua,  // [1/cm^3] 
  Real& n_nux,  // [1/cm^3] 
  Real& e_nue, // [erg/cm^3]
  Real& e_nua, // [erg/cm^3] 
  Real& e_nux  // [erg/cm^3]
  ) 
  
{
  int iout = 0; //NeutrinoDensityImpl = 0
  
  // We will already be in cgs units here
  /*
  ! Conversion to cgs units
  rho0  = rho*cactus2cgsRho
  temp0 = temp
  ye0   = ye
  */

  Real rho0 = rho;
  Real temp0 = temp;
  Real ye0 = ye;

  if ((rho0<rho_min) || (temp0<temp_min)) {
      n_nue = 0.0;
      n_nua = 0.0;
      n_nux = 0.0;
      e_nue = 0.0;
      e_nua = 0.0;
      e_nux = 0.0;
    return iout;
  } // end if

  // TODO done by the EoS interface
  // boundsErr = enforceTableBounds(rho_cgs,temp0,ye0)
  // int boundsErr = 0;

  // if (boundsErr==-1) {
  //   iout = -1;
  //   return iout;
  // } // end if

  // Call CGS backend
  iout = NeutrinoDens_cgs(rho0, temp0, ye0, n_nue, n_nua, n_nux, e_nue, e_nua, e_nux);

  /* Now done elsewhere
  ! Conversion from CGS units
  n_nue = n_nue / (cgs2cactusLength**3 * normfact)
  n_nua = n_nua / (cgs2cactusLength**3 * normfact)
  n_nux = n_nux / (cgs2cactusLength**3 * normfact)
  */
  e_nue = e_nue * mev_to_erg; // * (cgs2cactusEnergy / cgs2cactusLength**3)
  e_nua = e_nua * mev_to_erg; // * (cgs2cactusEnergy / cgs2cactusLength**3)
  e_nux = e_nux * mev_to_erg; // * (cgs2cactusEnergy / cgs2cactusLength**3)
  

  return iout;
} // END FUNCTION NeutrinoDensityImpl

int WeakEquilibriumMod::WeakEquilibriumImpl(
  Real rho,        // [g/cm^3]
  Real temp,       // [MeV]
  Real ye,         // [-]
  Real n_nue,      // [1/cm^3] 
  Real n_nua,      // [1/cm^3] 
  Real n_nux,      // [1/cm^3] 
  Real e_nue,     // [erg/cm^3] 
  Real e_nua,     // [erg/cm^3] 
  Real e_nux,     // [erg/cm^3]
  Real& temp_eq,   // [MeV]
  Real& ye_eq,     // [-] 
  Real& n_nue_eq,  // [1/cm^3]
  Real& n_nua_eq,  // [1/cm^3]
  Real& n_nux_eq,  // [1/cm^3]
  Real& e_nue_eq, // [erg/cm^3]
  Real& e_nua_eq, // [erg/cm^3]
  Real& e_nux_eq  // [erg/cm^3]
  ) 
{
  int iout = 0;

  // We will already be in cgs units here
  /*
  ! Conversion to cgs units
  rho0  = rho*cactus2cgsRho
  temp0 = temp
  ye0   = ye
  */
  
  Real rho0 = rho;
  Real temp0 = temp;
  Real ye0 = ye;
  
  // Do not do anything outside of this range
  if ((rho0<rho_min) || (temp0<temp_min)) {
    n_nue_eq  = 0.0;
    n_nua_eq  = 0.0;
    n_nux_eq  = 0.0;
    e_nue_eq = 0.0;
    e_nua_eq = 0.0;
    e_nux_eq = 0.0;
    // WeakEquilibriumImpl = 0
    iout = 0;
    return iout;
  } // end if

    // Enforce table bounds
    // TODO something something EoS
    // boundsErr = enforceTableBounds(rho0, temp0, ye0)

    // if (boundsErr.eq.-1) then
    //   WeakEquilibriumImpl = -1
    //   return
    // end if

    // Compute baryon number density in cgs.
    Real nb = rho0/atomic_mass;

    // Compute fractions
    Real y_in[4] = {0.0};
    y_in[0] = ye0;
    y_in[1] = n_nue/nb;
    y_in[2] = n_nua/nb;
    y_in[3] = 0.25*n_nux/nb;

    // Compute energy (note that tab3d_eps works in Cactus units)
    // eps0 = tab3d_eps(rho0/cactus2cgsRho, temp, ye)*cactus2cgsEps
    // TODO get eps from EoS
    Real e_in[4] = {0.0};
    e_in[0] = EoS->GetEnergyDensity(rho0, temp0, ye0);
    e_in[1] = e_nue;
    e_in[2] = e_nua;
    e_in[3] = e_nux;

    // Compute weak equilibrium
    Real y_eq[4] = {0.0};
    Real e_eq[4] = {0.0};
    int na=0;
    weak_equil_wnu(rho0, temp0, y_in, e_in, temp_eq, y_eq, e_eq, na, iout);
    ye_eq = y_eq[0];

    // Convert results to Cactus units
    // Conversion no longer here, just split output arrays from weak_equil_wnu
    n_nue_eq  = nb*y_eq[1];
    n_nua_eq  = nb*y_eq[2];
    n_nux_eq  = 4.0*nb*y_eq[3];
    e_nue_eq = e_eq[1];
    e_nua_eq = e_eq[2];
    e_nux_eq = e_eq[3];

    return iout;
} // END FUNCTION WeakEquilibriumImpl


/*======================================================================

      subroutine: weak_equil_wnu

      This subroutine ...

======================================================================*/

void WeakEquilibriumMod::weak_equil_wnu(Real rho, Real T, Real y_in[4], Real e_in[4], Real& T_eq, Real y_eq[4], Real e_eq[4], int& na, int& ierr) {
/*=====================================================================
!
!     input:
!
!     rho  ... fluid density [g/cm^3]
!     T    ... fluid temperature [MeV]
!     y_in ... incoming abundances
!              y_in(1) ... initial electron fraction               [#/baryon]
!              y_in(2) ... initial electron neutrino fraction      [#/baryon]
!              y_in(3) ... initial electron antineutrino fraction  [#/baryon]
!              y_in(4) ... initial heavy flavor neutrino fraction  [#/baryon]
!                          The total one would be 0, so we are
!                          assuming this to be each of the single ones.
!                          Anyway, this value is useless for our
!                          calculations. We could also assume it to be
!                          the total and set it to 0
!     e_eq ... incoming energies
!              e_in(1) ... initial fluid energy, incl rest mass    [erg/cm^3]
!              e_in(2) ... initial electron neutrino energy        [erg/cm^3]
!              e_in(3) ... initial electron antineutrino energy    [erg/cm^3]
!              e_in(4) ... total initial heavy flavor neutrino energy    [erg/cm^3]
!                          This is assumed to be 4 times the energy of
!                          each single heavy flavor neutrino species
!
!     output:
!
!     T_eq ... equilibrium temperature   [MeV]
!     y_eq ... equilibrium abundances    [#/baryons]
!              y_eq(1) ... equilibrium electron fraction              [#/baryon]
!              y_eq(2) ... equilibrium electron neutrino fraction     [#/baryon]
!              y_eq(3) ... equilibrium electron antineutrino fraction [#/baryon]
!              y_eq(4) ... equilibrium heavy flavor neutrino fraction [#/baryon]
!                          see explanation above and change if necessary
!     e_eq ... equilibrium energies
!              e_eq(1) ... equilibrium fluid energy                   [erg/cm^3]
!              e_eq(2) ... equilibrium electron neutrino energy       [erg/cm^3]
!              e_eq(3) ... equilibrium electron antineutrino energy   [erg/cm^3]
!              e_eq(4) ... total equilibrium heavy flavor neutrino energy   [erg/cm^3]
!                          see explanation above and change if necessary
!     na   ... number of attempts in 2D Newton-Raphson
!     ierr ... 0 success in Newton-Raphson
!              1 failure in Newton-Raphson
!
!=====================================================================*/


/*
!.....guesses for the 2D Newton-Raphson.................................
      CCTK_REAL, dimension(2)      :: x0,x1
      CCTK_REAL, dimension(n_at,2) :: vec_guess

      CCTK_REAL, dimension(3) :: mus
      CCTK_REAL, dimension(3) :: eta
      CCTK_REAL, dimension(3) :: nu_dens

      CCTK_REAL :: lrho
      CCTK_REAL :: ltemp
      CCTK_REAL :: mu_n
      CCTK_REAL :: mu_p
      CCTK_REAL :: mu_e
      CCTK_REAL :: nb
      CCTK_REAL :: mass_fact_cgs

      CCTK_REAL :: yl  ! total lepton mumber
      CCTK_REAL :: u   ! total internal energy (fluid + radiation)
*/

//.....compute the total lepton fraction and internal energy
  Real yl = y_in[0] + y_in[1] - y_in[2];           // [#/baryon]
  Real u  = e_in[0] + e_in[1] + e_in[2] + e_in[3]; // [erg/cm^3]

/*
!.....vector with the coefficients for the different guesses............
!     at the moment, to solve the 2D NR we assign guesses for the
!     equilibrium ye and T close to the incoming ones. This array
!     quantifies this closeness. Different guesses are used, one after
!     the other, until a solution is found. Hopefully, the first one
!     works already in most of the cases. The other ones are used as
!     backups
*/
  Real vec_guess[n_at][2] = { 
    {1.00e0, 1.00e0},
    {0.90e0, 1.25e0},
    {0.90e0, 1.10e0},
    {0.90e0, 1.00e0},
    {0.90e0, 0.90e0},
    {0.90e0, 0.75e0},
    {0.75e0, 1.25e0},
    {0.75e0, 1.10e0},
    {0.75e0, 1.00e0},
    {0.75e0, 0.90e0},
    {0.75e0, 0.75e0},
    {0.50e0, 1.25e0},
    {0.50e0, 1.10e0},
    {0.50e0, 1.00e0},
    {0.50e0, 0.90e0},
    {0.50e0, 0.75e0},
  };

  na = 0; // counter for the number of attempts

/*
! ierr is the variable that check if equilibrium has been found:
! ierr = 0   equilibrium found
! ierr = 1   equilibrium not found
*/
  ierr = 1;

/*
! here we try different guesses x0 = [T,Ye]*vec_guess[i,:], one after 
! the other, until success is obtained
*/
  Real x0[2]; // Guess for T,Ye
  Real x1[2]; // Result for T,Ye

  while (ierr!=0 && na<n_at){

//....make an initial guess............................................
    x0[0] = vec_guess[na][0]*T;       // T guess  [MeV]
    x0[1] = vec_guess[na][1]*y_in[0]; // ye guess [#/baryon]

//....call the 2d Newton-Raphson........................................
    new_raph_2dim(rho,u,yl,x0,x1,ierr);

    na += 1;
  } // end while


  //....assign the output.................................................
  if (ierr==0) {
    // calculations worked
    T_eq = x1[0];
    y_eq[0] = x1[1];
  } else {
    // calculations did not work
    /*
    ! write(6,*)'2D Newton-Raphson search did not work!'
    ! write(6,*)'Point log10 density [g/cm^3]: ',log10(rho)
    ! write(6,*)'Point temperature [MeV]: ',T
    ! write(6,*)'Point yl [#/baryon]: ',yl
    ! write(6,*)'Point log10 total energy [erg/cm^3]: ',log10(u)
    */

//....as backup plan, we assign the initial values to all outputs.......
    T_eq = T;    // [MeV]
    for (int i=0;i<4;i++) {
      y_eq[i] = y_in[i]; // [#/baryon]
      e_eq[i] = e_in[i]; // [erg/cm^3]
    }

    ierr = WE_FAIL_INI_ASSIGN;
    return;

// 25    format(3es14.6)
//        close(6)

  } // end if

  // Here we want to compute the total energy and fractions in the
  // equilibrated state

  // Interpolate the chemical potentials (stored in MeV in the table)
  // TODO: Replace with EoS calls
  // lrho  = log10(rho)
  // ltemp = log10(T_eq)
  // mu_n = lkLinearInterpolation3d(lrho, ltemp, y_eq(1), MU_N)
  // mu_p = lkLinearInterpolation3d(lrho, ltemp, y_eq(1), MU_P)
  // mu_e = lkLinearInterpolation3d(lrho, ltemp, y_eq(1), MU_E)
  Real mu_n = EoS->GetNeutronChemicalPotential(rho, T_eq, y_eq[0]);
  Real mu_p = EoS->GetProtonChemicalPotential(rho, T_eq, y_eq[0]);
  Real mu_e = EoS->GetElectronChemicalPotential(rho, T_eq, y_eq[0]);

  // in the original verison of this function mus has size 3, whereas 
  // later it is 2, and in nu_deg_param_trap/dens_nu_trap/edens_nu_trap 
  // it is also 2, so we go with 2
  Real mus[2]     = {0.0}; // Chemical potentials for calculating etas
  Real eta[3]     = {0.0}; // Neutrino degeneracy parameters
  Real nu_dens[3] = {0.0}; // Neutrino number densities

  mus[0] = mu_e;        // electron chem pot including rest mass [MeV]
  mus[1] = mu_n - mu_p; // n-p chem pot including rest masses [MeV]

  // compute the degeneracy parameters
  nu_deg_param_trap(T_eq,mus,eta);

  // compute the density of the trapped neutrinos
  dens_nu_trap(T_eq,eta,nu_dens);

  // Compute the baryon number density (mass_fact is given in MeV)
  // Real mass_fact_cgs = mass_fact * mev_to_erg / (clight*clight);
  Real nb = rho / atomic_mass; // [#/cm^3]

  y_eq[1] = nu_dens[0]/nb;          // electron neutrino
  y_eq[2] = nu_dens[1]/nb;          // electron anti-neutrino
  y_eq[3] = nu_dens[2]/nb;          // heavy-lepton neutrino
  y_eq[0] = yl - y_eq[1] + y_eq[2]; // fluid electron fraction

  // compute the energy density of the trapped neutrinos
  edens_nu_trap(T_eq,eta,nu_dens);

  e_eq[1] = nu_dens[0]*mev_to_erg;           // electron neutrino energy density [erg/cm^3]
  e_eq[2] = nu_dens[1]*mev_to_erg;           // electron anti-neutrino energy density [erg/cm^3]
  e_eq[3] = 4.0*nu_dens[2]*mev_to_erg;       // heavy-lepton neutrino energy density [erg/cm^3]
  e_eq[0] = u - e_eq[1] - e_eq[2] - e_eq[3]; // fluid energy density [erg/cm^3]

  // TODO these checks should probably be done by the EoS?
  // check that the energy is positive
  // For tabulated eos we should check that the energy is above the minimum?

  // Note [DD2]:
  // nb*atomic_mass*clight*clight < EoS->GetMinimumEnergyDensity(rho, y_eq[0])
  // 2.387e+35 < 2.514e+35
#ifdef THC_NRG_DENS_FLOOR
  if (e_eq[0]<nb*atomic_mass*clight*clight) {
#else
  Real e_min = EoS->GetMinimumEnergyDensity(rho, y_eq[0]);
  if (e_eq[0]<e_min) {
#endif // THC_NRG_DENS_FLOOR
    ierr = WE_FAIL_INI_ASSIGN_NRG;
    T_eq = T;
    for (int i=0;i<4;i++) {
      y_eq[i] = y_in[i]; // [#/baryon]
      e_eq[i] = e_in[i]; // [erg/cm^3]
    }
    return;
  } // end if

  // check that Y_e is within the range
  Real table_ye_min, table_ye_max;
  EoS->GetTableLimitsYe(table_ye_min, table_ye_max);
  if (y_eq[0]<table_ye_min || y_eq[0]>table_ye_max) {
    ierr = WE_FAIL_INI_ASSIGN_Y_E;
    T_eq = T;
    for (int i=0;i<4;i++) {
      y_eq[i] = y_in[i]; // [#/baryon]
      e_eq[i] = e_in[i]; // [erg/cm^3]
    }
    return;
  } // end if

  return;

} // end subroutine weak_equil_wnu

//======================================================================

/*======================================================================
!
!     subroutine: new_raph_2dim
!
!     This subroutine ...
!
!=====================================================================*/

void WeakEquilibriumMod::new_raph_2dim(Real rho, Real u, Real yl, Real x0[2], Real x1[2], int& ierr){
/*======================================================================
!
!     input:
!     rho ... density               [g/cm^3]
!     u   ... total internal energy [erg/cm^3]
!     yl  ... lepton number         [erg/cm^3]
!     x0  ... T and ye guess
!        x0(1) ... T               [MeV]
!        x0(2) ... ye              [#/baryon]
!
!     output:
!     x1 ... T and ye at equilibrium
!        x1(1) ... T               [MeV]
!        x1(2) ... ye              [#/baryon]
!     ierr  ...
!
!=====================================================================*/

/*
      integer :: n_iter,n_cut
      CCTK_REAL :: err,err_old
      CCTK_REAL :: det
      CCTK_REAL, dimension(2)   :: y
      CCTK_REAL, dimension(2)   :: dx1
      CCTK_REAL, dimension(2)   :: x1_tmp
      CCTK_REAL, dimension(2,2) :: J
      CCTK_REAL, dimension(2,2) :: invJ

      INTEGER :: enforceTableBounds
      INTEGER :: tabBoundsFlag
*/

/*
  ! If true then we satisfy the Karush-Kuhn-Tucker conditions.
  ! This means that the equilibrium is out of the table and we have the best possible result.
  LOGICAL :: KKT
  ! Normal to the domain
  CCTK_REAL, dimension(2) :: norm
  CCTK_REAL :: scal
  ! Active component of the gradient
  CCTK_REAL, dimension(2) :: dxa
*/

  // initialize the solution
  x1[0] = x0[0];
  x1[1] = x0[1];
  bool KKT = false;

  //compute the initial residuals
  Real y[2] = {0.0};
  func_eq_weak(rho,u,yl,x1,y);

  // compute the error from the residuals
  Real err = 0.0;
  error_func_eq_weak(yl,u,y,err);

  // initialize the iteration variables
  int n_iter = 0;
  Real J[2][2] = {0.0};
  Real invJ[2][2] = {0.0};
  Real dx1[2] = {0.0};
  Real dxa[2] = {0.0};
  Real norm[2] = {0.0};
  Real x1_tmp[2] = {0.0};

  // loop until a low enough residual is found or until  a too
  // large number of steps has been performed
  while (err>eps_lim && n_iter<=n_max && !KKT) {

    // compute the Jacobian
    jacobi_eq_weak(rho,u,yl,x1,J,ierr);
    if (ierr != 0) {
      return;
    } // end if
    
    // compute and check the determinant of the Jacobian
    Real det = J[0][0]*J[1][1] - J[0][1]*J[1][0];
    if (det==0.0) {
      ierr = 1;
      return;
      // write(6,*)'Singular determinant in weak equilibrium!'
      // stop
    } // end if

    // invert the Jacobian
    inv_jacobi(det,J,invJ);

    // compute the next step
    dx1[0] = - (invJ[0][0]*y[0] + invJ[0][1]*y[1]);
    dx1[1] = - (invJ[1][0]*y[0] + invJ[1][1]*y[1]);

    // check if we are the boundary of the table
    // TODO: Replace with EoS calls
    if (x1[0] == eos_tempmin) {
      norm[0] = -1.0;
    } else if (x1[0] == eos_tempmax) {
      norm[0] = 1.0;
    } else { 
      norm[0] = 0.0;
    } // endif

    if (x1[1] == eos_yemin) {
      norm[1] = -1.0;
    } else if (x1[1] == eos_yemax) {
      norm[1] = 1.0;
    } else {
      norm[1] = 0.0;
    } // endif

    // Take the part of the gradient that is active (pointing within the eos domain)
    Real scal = norm[0]*norm[0] + norm[1]*norm[1];
    if (scal <= 0.5) { // this can only happen if norm = (0, 0)
      scal = 1.0;
    } // endif
    dxa[0] = dx1[0] - (dx1[0]*norm[0] + dx1[1]*norm[1])*norm[0]/scal;
    dxa[1] = dx1[1] - (dx1[0]*norm[0] + dx1[1]*norm[1])*norm[1]/scal;

    if ((dxa[0]*dxa[0] + dxa[1]*dxa[1]) < (eps_lim*eps_lim * (dx1[0]*dx1[0] + dx1[1]*dx1[1]))) {
      KKT = true;
      ierr = 2;
      return;
    } // endif

    int n_cut = 0;
    Real fac_cut = 1.0;
    Real err_old = err;

    while (n_cut <= n_cut_max && err >= err_old) {
      // the variation of x1 is divided by an powers of 2 if the
      // error is not decreasing along the gradient direction
      
      x1_tmp[0] = x1[0] + (dx1[0]*fac_cut);
      x1_tmp[1] = x1[1] + (dx1[1]*fac_cut);

      // check if the next step calculation had problems
      if (isnan(x1_tmp[0])) {
        ierr = 1;
        return;
        // write(*,*)'x1_tmp NaN',x1_tmp(1)
        // write(*,*)'x1',x1(1)
        // write(*,*)'dx1',dx1(1)
        // write(*,*)'J',J
        // stop
      } // end if

      // TODO this should be done by the EoS
      // tabBoundsFlag = enforceTableBounds(rho, x1_tmp[0], x1_tmp[1]);
      bool tabBoundsFlag = EoS->ApplyTableLimits(rho, x1_tmp[0], x1_tmp[1]);

      // assign the new point
      x1[0] = x1_tmp[0];
      x1[1] = x1_tmp[1];

      // compute the residuals for the new point
      func_eq_weak(rho,u,yl,x1,y);

      // compute the error
      error_func_eq_weak(yl,u,y,err);

      // update the bisection cut along the gradient
      n_cut += 1;
      fac_cut *= 0.5;

      } // end do

    // update the iteration
    n_iter += 1;

  } //end do

  // if equilibrium has been found, set ierr=0 and return
  // if too many attempts have been performed, set ierr=1
  if (n_iter <= n_max) {
    ierr = 0;
  } else {
    ierr = 1;
  } // end if
  
  return;

} // end subroutine new_raph_2dim

//======================================================================

/*======================================================================
!
!     function: func_eq_weak
!
!     This function ...
!
!=====================================================================*/

void WeakEquilibriumMod::func_eq_weak(Real rho, Real u, Real yl, Real x[2], Real y[2]) {
/*----------------------------------------------------------------------
!
!     Input:
!
!     rho ... density                                  [g/cm^3]
!     u   ... total (fluid+radiation) internal energy  [erg/cm^3]
!     yl  ... lepton number                            [#/baryon]
!     x   ...  array with the temperature and ye
!        x(1) ... T                                  [MeV]
!        x(2) ... ye                                 [#/baryon]
!
!     Output:
!
!     y ... array with the function whose zeros we are searching for
!
!---------------------------------------------------------------------*/

/*
      CCTK_REAL :: lrho
      CCTK_REAL :: ltemp
      CCTK_REAL :: ye
      CCTK_REAL :: mu_n
      CCTK_REAL :: mu_e
      CCTK_REAL :: mu_p
      CCTK_REAL :: rho_cu
      CCTK_REAL :: eps_cu
      CCTK_REAL :: e

      CCTK_REAL               :: nb      ! baryon density             [baryon/cm^3]
      CCTK_REAL, dimension(2) :: mus     ! chemical potential array   [MeV]
      CCTK_REAL, dimension(3) :: eta_vec ! degeneracy parameter array [-]

      CCTK_REAL               :: mass_fact_cgs
      CCTK_REAL               :: eta,eta2
*/

  // Compute the baryon number density (mass_fact is given in MeV)
  // Real mass_fact_cgs = mass_fact * mev_to_erg / (clight*clight);
  // Real nb = rho / mass_fact_cgs; // [#/cm^3]
  Real nb = rho / atomic_mass; // [#/cm^3]

  // Interpolate the chemical potentials (stored in MeV in the table)
  // TODO: Replace with EoS calls
  // lrho  = log10(rho)
  // ltemp = log10(x(1))
  // ye = x(2)
  // mu_n = lkLinearInterpolation3d(lrho, ltemp, ye, MU_N)
  // mu_p = lkLinearInterpolation3d(lrho, ltemp, ye, MU_P)
  // mu_e = lkLinearInterpolation3d(lrho, ltemp, ye, MU_E)
  Real mu_n = EoS->GetNeutronChemicalPotential(rho, x[0], x[1]);
  Real mu_p = EoS->GetProtonChemicalPotential(rho, x[0], x[1]);
  Real mu_e = EoS->GetElectronChemicalPotential(rho, x[0], x[1]);

  Real mus[2] = {0.0};
  mus[0] = mu_e;
  mus[1] = mu_n - mu_p;

  // Call the EOS
  // TODO: Replace with EoS call
  // rho_cu = rho*cgs2cactusRho
  // eps_cu = tab3d_eps(rho_cu, x(1), ye)
  // e = rho*(clight**2 + eps_cu*cactus2cgsEps)
  Real e = EoS->GetEnergyDensity(rho, x[0], x[1]);

//....compute the neutrino degeneracy paramater at equilibrium..........
  Real eta_vec[2] = {0.0};
  nu_deg_param_trap(x[0],mus,eta_vec);
  Real eta = eta_vec[0]; // [-]
  Real eta2 = eta*eta;   // [-]

//....compute the function..............................................
  Real t3 = x[0]*x[0]*x[0];
  Real t4 = t3*x[0];
  y[0] = x[1] + pref1*t3*eta*(pi2 + eta2)/nb - yl;
  y[1] = (e+pref2*t4*((cnst5+0.5*eta2*(pi2+0.5*eta2))+cnst6))/u - 1.0;

  return;

} // end subroutine func_eq_weak

//======================================================================

/*======================================================================
!
!     function: error_func_eq_weak
!
!     This function ...
!
!=====================================================================*/

void WeakEquilibriumMod::error_func_eq_weak(Real yl, Real u, Real y[2], Real &err) {
/*----------------------------------------------------------------------
!
!     Input:
!
!     yl ... lepton number                            [#/baryon]
!     u  ... total (fluid+radiation) internal energy  [erg/cm^3]
!     y  ... array with residuals                     [-]
!
!     Output:
!
!     err ... error associated with the residuals     [-]
!
!---------------------------------------------------------------------*/

//....since the first equation is has yl as constant, we normalized the error to it.
//    since the second equation was normalized wrt u, we divide it by 1.
//    the modulus of the two contributions are then summed

  err = abs(y[0]/yl) + abs(y[1]/1.0);
  return;

} // end subroutine error_func_eq_weak

//======================================================================

/*======================================================================
!
!     subroutine: jacobi_eq_weak
!
!=====================================================================*/

void WeakEquilibriumMod::jacobi_eq_weak(Real rho, Real u, Real yl, Real x[2], Real J[2][2], int &ierr) {
/*----------------------------------------------------------------------
!
!     Input:
!
!     rho ... density                   [g/cm^3]
!     u   ... total energy density      [erg/cm^3]
!     yl  ... lepton fraction           [#/baryon]
!     x   ... array with T and ye
!        x(1) ... T                     [MeV]
!        x(2) ... ye                    [#/baryon]
!
!     Output:
!
!     J ... Jacobian for the 2D Newton-Raphson
!
!---------------------------------------------------------------------*/

/*
      CCTK_REAL :: mass_fact_cgs
      CCTK_REAL :: nb,t,ye
      CCTK_REAL :: eta,eta2
      CCTK_REAL :: dedt,dedye
      CCTK_REAL :: detadt,detadye
      CCTK_REAL, dimension(3) :: eta_vec

      CCTK_REAL :: lrho,ltemp
      CCTK_REAL :: mu_e,mu_p,mu_n

      !integer :: ierr
      !CCTK_REAL :: x1,x2
      CCTK_REAL, dimension(2) :: mus
*/

  // Interpolate the chemical potentials (stored in MeV in the table)
  // TODO: Replace with EoS calls
  // lrho  = log10(rho)
  // ltemp = log10(x(1))
  Real t = x[0];
  Real ye = x[1];
  // mu_n = lkLinearInterpolation3d(lrho, ltemp, ye, MU_N)
  // mu_p = lkLinearInterpolation3d(lrho, ltemp, ye, MU_P)
  // mu_e = lkLinearInterpolation3d(lrho, ltemp, ye, MU_E)
  Real mu_n = EoS->GetNeutronChemicalPotential(rho, t, ye);
  Real mu_p = EoS->GetProtonChemicalPotential(rho, t, ye);
  Real mu_e = EoS->GetElectronChemicalPotential(rho, t, ye);

  Real mus[2] = {0.0};
  mus[0] = mu_e;        // electron chemical potential (w rest mass) [MeV]
  mus[1] = mu_n - mu_p; // n minus p chemical potential (w rest mass) [MeV]
  
  Real eta_vec[3] = {0.0};
  // compute the degeneracy parameters
  nu_deg_param_trap(t,mus,eta_vec);
  Real eta = eta_vec[0]; // electron neutrinos degeneracy parameter
  Real eta2 = eta*eta;

//....compute the gradients of eta and of the internal energy...........
  Real detadt,detadye,dedt,dedye;
  eta_e_gradient(rho,t,ye,eta,detadt,detadye,dedt,dedye,ierr);
  if (ierr != 0) {
    return;
  } // end if

   // Compute the baryon number density (mass_fact is given in MeV)
  // Real mass_fact_cgs = mass_fact * mev_to_erg / (clight*clight);
  // Real nb = rho / mass_fact_cgs; // [#/cm^3]
  Real nb = rho / atomic_mass; // [#/cm^3]

/*
!     J[0,0]: df1/dt
!     J[0,1]: df1/dye
!     J[1,0]: df2/dt
!     J[1,1]: df2/dye
*/

//....compute the Jacobian.............................................
  Real t2 = t*t;
  Real t3 = t2*t;
  Real t4 = t3*t;
  J[0][0] = pref1/nb*t2*(3.e0*eta*(pi2+eta2)+t*(pi2+3.e0*eta2)*detadt);
  J[0][1] = 1.e0+pref1/nb*t3*(pi2+3.e0*eta2)*detadye;

  J[1][0] = (dedt+pref2*t3*(cnst3+cnst4+2.e0*eta2*(pi2+0.5*eta2)+eta*t*(pi2+eta2)*detadt))/u;
  J[1][1] = (dedye+pref2*t4*eta*(pi2+eta2)*detadye)/u;

//....check on the degeneracy parameters and temperature................
  if (isnan(eta)) {
    ierr = 1;
    return;
    // write(*,*)'eta',eta
  } // end if

  if (isnan(detadt)) {
    ierr = 1;
    return;
    // write(*,*)'detadt',detadt
  } // end if

  if (isnan(t)) {
    ierr = 1;
    return;
    // write(*,*)'t',t
  } // end if

  ierr = 0;

  return;

} // end subroutine jacobi_eq_weak

//======================================================================

/*======================================================================
!
!     subroutine: eta_e_gradient
!
!     this subroutine computes the gradient of the degeneracy parameter
!     and of the fluid internal energy with respect to temperature and ye
!
!=====================================================================*/

void WeakEquilibriumMod::eta_e_gradient(Real rho, Real t, Real ye, Real eta, Real& detadt, Real& detadye, Real& dedt, Real& dedye, int& ierr) {
/*----------------------------------------------------------------------
!
!     Input:
!     rho ... density             [g/cm^3]
!     t   ... temperature         [MeV]
!     ye  ... electron fraction   [#/baryon]
!     eta ... electron neutrino degeneracy parameter at equilibrium [-]
!
!     Output:
!     detadt  ... derivative of eta wrt T (for ye and rho fixed)             [1/MeV]
!     detadye ... derivative of eta wrt ye (for T and rho fixed)             [-]
!     dedt  ... derivative of internal energy wrt T (for ye and rho fixed) [erg/cm^3/MeV]
!     dedye ... derivative of internal energy wrt ye (for T and rho fixed) [erg/cm^3]
!
!---------------------------------------------------------------------*/

/*
      CCTK_REAL, dimension(2) :: mus1, mus2

      CCTK_REAL :: ye1, ye2  ! values used for the derivatives
      CCTK_REAL :: t1, t2    ! values used for the derivatives
      CCTK_REAL :: dmuedt,dmuedye
      CCTK_REAL :: dmuhatdt,dmuhatdye
      CCTK_REAL :: x1,x2

      CCTK_REAL :: lrho,ltemp,tv,yev
      CCTK_REAL :: rho_cu, eps_cu
      CCTK_REAL :: mu_e,mu_p,mu_n
      CCTK_REAL :: e1,e2
*/

/*
!.....gradients are computed numerically. To do it, we consider small
!     variations in ye and temperature, and we compute the detivative
!     using finite differencing. The real limitation is that this way
!     relies on the EOS table interpolation procedure

!     the goal of this part is to obtain chemical potentials (mus1 and
!     mus2) in two points close to the point we are considering, first
!     varying wrt ye and then wrt T
*/

//....vary the electron fraction........................................

/*
  ! these are the two calls to the EOS. The goal here is to get
  ! the fluid internal energy and the chemical potential for
  ! electrons and the difference between neutron and proton
  ! chemical potential (usually called mu_hat) for two points with
  ! slightly different ye
*/
  // TODO: Reaplace with Eos calls
  // lrho  = log10(rho)
  // ltemp = log10(t)

  //  first, for ye slightly smaller
  Real ye1 = max(ye - delta_ye, eos_yemin);
  Real yev = ye1;
  // mu_n = lkLinearInterpolation3d(lrho, ltemp, yev, MU_N)
  // mu_p = lkLinearInterpolation3d(lrho, ltemp, yev, MU_P)
  // mu_e = lkLinearInterpolation3d(lrho, ltemp, yev, MU_E)
  Real mu_n = EoS->GetNeutronChemicalPotential(rho, t, yev);
  Real mu_p = EoS->GetProtonChemicalPotential(rho, t, yev);
  Real mu_e = EoS->GetElectronChemicalPotential(rho, t, yev);

  // rho_cu = rho*cgs2cactusRho
  // eps_cu = tab3d_eps(rho_cu, t, yev)
  //e1 = rho*(clight**2 + eps_cu*cactus2cgsEps)
  // Real e1 = rho*(1.0 + 1.0e-3)*clight*clight;
  Real e1 = EoS->GetEnergyDensity(rho, t, yev);

  Real mus1[2] = {mu_e, mu_n - mu_p};
  // mus1(1) = mu_e
  // mus1(2) = mu_n - mu_p

  // second, for ye slightly larger
  Real ye2 = min(ye + delta_ye, eos_yemax);
  yev = ye2;
  // mu_n = lkLinearInterpolation3d(lrho, ltemp, yev, MU_N)
  // mu_p = lkLinearInterpolation3d(lrho, ltemp, yev, MU_P)
  // mu_e = lkLinearInterpolation3d(lrho, ltemp, yev, MU_E)
  mu_n = EoS->GetNeutronChemicalPotential(rho, t, yev);
  mu_p = EoS->GetProtonChemicalPotential(rho, t, yev);
  mu_e = EoS->GetElectronChemicalPotential(rho, t, yev);

  // eps_cu = tab3d_eps(rho_cu, t, yev)
  // e2 = rho*(clight**2 + eps_cu*cactus2cgsEps)
  // Real e2 = rho*(1.0 + 1.0e-3)*clight*clight;
  Real e2 = EoS->GetEnergyDensity(rho, t, yev);

  Real mus2[2] = {mu_e, mu_n - mu_p};
  // mus2(1) = mu_e
  // mus2(2) = mu_n - mu_p

//....compute numerical derivaties......................................
  Real dmuedye   = (mus2[0]-mus1[0])/(ye2 - ye1);
  Real dmuhatdye = (mus2[1]-mus1[1])/(ye2 - ye1);
  dedye          = (e2-e1)/(ye2 - ye1);

//....vary the temperature..............................................
  Real t1 = max(t - delta_t, eos_tempmin);
  Real t2 = min(t + delta_t, eos_tempmax);

  // these are the two other calls to the EOS. The goal here is to get
  // the fluid internal energy and the chemical potential for
  // electrons and the difference between neutron and proton
  // chemical potential (usually called mu_hat) for two points with
  // slightly different t

  // ye is the original one

  // first, for t slightly smaller
  Real tv = t1;
  // ltemp = log10(t-delta_t)
  // mu_n = lkLinearInterpolation3d(lrho, ltemp, ye, MU_N)
  // mu_p = lkLinearInterpolation3d(lrho, ltemp, ye, MU_P)
  // mu_e = lkLinearInterpolation3d(lrho, ltemp, ye, MU_E)
  mu_n = EoS->GetNeutronChemicalPotential(rho, tv, ye);
  mu_p = EoS->GetProtonChemicalPotential(rho, tv, ye);
  mu_e = EoS->GetElectronChemicalPotential(rho, tv, ye);

  // eps_cu = tab3d_eps(rho_cu, tv, ye)
  // e1 = rho*(clight**2 + eps_cu*cactus2cgsEps)
  // e1 = rho*(1.0 + 1.0e-3)*clight*clight;
  e1 = EoS->GetEnergyDensity(rho, t1, ye);

  mus1[0] = mu_e; mus1[1] = mu_n - mu_p;
  // mus1(1) = mu_e
  // mus1(2) = mu_n - mu_p

   // second, for t slightly larger
  tv = t2;
  // ltemp = log10(t+delta_t)
  // mu_n = lkLinearInterpolation3d(lrho, ltemp, ye, MU_N)
  // mu_p = lkLinearInterpolation3d(lrho, ltemp, ye, MU_P)
  // mu_e = lkLinearInterpolation3d(lrho, ltemp, ye, MU_E)
  mu_n = EoS->GetNeutronChemicalPotential(rho, tv, ye);
  mu_p = EoS->GetProtonChemicalPotential(rho, tv, ye);
  mu_e = EoS->GetElectronChemicalPotential(rho, tv, ye);

  // eps_cu = tab3d_eps(rho_cu, tv, ye)
  // e2 = rho*(clight**2 + eps_cu*cactus2cgsEps)
  // e2 = rho*(1.0 + 1.0e-3)*clight*clight;
  e2 = EoS->GetEnergyDensity(rho, tv, ye);


  mus2[0] = mu_e; mus2[1] = mu_n - mu_p;
  //mus2(1) = mu_e
  // mus2(2) = mu_n - mu_p

//....compute the derivatives wrt temperature...........................
  Real dmuedt   = (mus2[0] - mus1[0])/(t2 - t1);
  Real dmuhatdt = (mus2[1] - mus1[1])/(t2 - t1);
  dedt          = (e2   - e1  )/(t2 - t1);
//....combine the eta derivatives.......................................
  detadt  = (-eta + dmuedt - dmuhatdt)/t; // [1/MeV]
  detadye = (dmuedye - dmuhatdye)/t;      // [-]

//....check if the derivative has a problem.............................
  if (isnan(detadt)) {
    ierr = 1;
    return; 
    // write(*,*)'problem with eta: ',eta
    // write(*,*)'mue1: ',mus1(1)
    // write(*,*)'mue2: ',mus2(1)
    // write(*,*)'dmuedt: ',dmuedt
    // write(*,*)'dmuhatdt: ',dmuhatdt
    // write(*,*)'rho: ',rho
    // write(*,*)'ye: ',ye
    // write(*,*)'t: ',t
  } // end if

  ierr = 0;

  return;

} // end subroutine eta_e_gradient

//======================================================================

/*======================================================================
!
!     subroutine: inv_jacobi
!
!     This subroutine inverts the Jacobian matrix, assuming it to be a
!     2x2 matrix
!
!=====================================================================*/

void WeakEquilibriumMod::inv_jacobi(Real det, Real J[2][2], Real invJ[2][2]) {
/*======================================================================
!
!     Input:
!     det ... determinant of the Jacobian matrix
!     J   ... Jacobian matrix
!
!     Output:
!     invJ ... inverse of the Jacobian matrix
!
!=====================================================================*/
  Real inv_det = 1.0/det;
  invJ[0][0] =  J[1][1]*inv_det;
  invJ[1][1] =  J[0][0]*inv_det;
  invJ[0][1] = -J[0][1]*inv_det;
  invJ[1][0] = -J[1][0]*inv_det;

} // end subroutine inv_jacobi

//======================================================================

/*======================================================================
!
!     subroutine: nu_deg_param_trap
!
!     In this subroutine, we compute the neutrino degeneracy parameters
!     assuming weak and thermal equilibrium, i.e. using as input the
!     local thermodynamical properties
!
!=====================================================================*/

void WeakEquilibriumMod::nu_deg_param_trap(Real temp_m, Real chem_pot[2], Real eta[3]) {
/*----------------------------------------------------------------------
!
!     Input:
!     temp_m   ----> local matter temperature [MeV]
!     chem_pot ----> matter chemical potential [MeV]
!                    chem_pot(1): electron chemical potential (w rest mass)
!                    chem_pot(2): n minus p chemical potential (w/o rest mass)
!
!     Output:
!     eta      ----> neutrino degeneracy parameters [-]
!                    eta(1): electron neutrino
!                    eta(2): electron antineutrino
!                    eta(3): mu and tau neutrinos
!
!---------------------------------------------------------------------*/

  if (temp_m>0.0) {
    eta[0] = (chem_pot[0] - chem_pot[1])/temp_m; // [-]
    eta[1] = - eta[0];                       // [-]
    eta[2] = 0.0;                            // [-]
  } else {
    // write(*,*)'Problem with the temperature in computing eta_nu'
    // write(*,*)'temp',temp_m
    eta[0] = 0.0; // [-]
    eta[1] = 0.0; // [-]
    eta[2] = 0.0; // [-]
  } // end if

} // end subroutine nu_deg_param_trap

//======================================================================

/*======================================================================
!
!     subroutine: dens_nu_trap
!
!=====================================================================*/

void WeakEquilibriumMod::dens_nu_trap(Real temp_m, Real eta_nu[3], Real nu_dens[3]) {
/*----------------------------------------------------------------------
!     In this subroutine, we compute the neutrino densities in equilibrium
!     conditions, using as input the local thermodynamical properties
!
!     Input:
!     temp_m   ----> local matter temperature [MeV]
!     eta_nu   ----> neutrino degeneracy parameter [-]
!
!     Output:
!     nu_dens ----> neutrino density [particles/cm^3]
!
!---------------------------------------------------------------------*/

  const Real pref=4.0*pi/(hc_mevcm*hc_mevcm*hc_mevcm); // [1/MeV^3/cm^3]
  Real temp_m3 = (temp_m*temp_m*temp_m);               // [MeV^3]

  for (int it=0; it<3; it++) {
#if WR_FERMI_ANALYTIC
    Real f2 = fermi2(eta_nu[it]);
    // call f2_analytic(eta_nu(it),f2)
#else
    Real f2 = 0.0;
    fermiint(2.0,eta_nu[it],f2);
#endif
    nu_dens[it] = pref * temp_m3 * f2; // [#/cm^3]
  } // end do

} // end subroutine dens_nu_trap

//======================================================================

/*======================================================================
!
!     subroutine: edens_nu_trap
!
!=====================================================================*/

void WeakEquilibriumMod::edens_nu_trap(Real temp_m, Real eta_nu[3], Real enu_dens[3]) {
/*----------------------------------------------------------------------
!     In this subroutine, we compute the neutrino densities in equilibrium
!     conditions, using as input the local thermodynamical properties
!
!     Input:
!     temp_m   ----> local matter temperature [MeV]
!     eta_nu   ----> neutrino degeneracy parameter [-]
!
!     Output:
!     enu_dens ----> neutrino density [MeV/cm^3]
!
!---------------------------------------------------------------------*/

  const Real pref=4.0*pi/(hc_mevcm*hc_mevcm*hc_mevcm); // [1/MeV^3/cm^3]
  Real temp_m4 = temp_m*temp_m*temp_m*temp_m;          // [MeV^3]

  for (int it=0; it<3; it++) {
#if WR_FERMI_ANALYTIC
    Real f3 = fermi3(eta_nu[it]);
    // call f3_analytic(eta_nu(it),f3)
#else
    Real f3 = 0.0;
    fermiint(3.0,eta_nu[it],f3);
#endif
    enu_dens[it] = pref * temp_m4 * f3;
  } //end do

} // end subroutine edens_nu_trap

//======================================================================

/*
PH: These appear to be unused versions of the f2 and f3 functions.
I leave them here in case we need them in the future, but they are still
in F90.

!=======================================================================
!
!     Subroutine: f2_analytic
!
!=======================================================================

      subroutine f2_analytic(eta,f2)

      implicit none

      CCTK_REAL, intent(in)  :: eta
      CCTK_REAL, intent(out) :: f2

      if (eta.gt.1.e-3) then
        f2 = (eta**3/3.e0 + 3.2899*eta)/(1.-exp(-1.8246e0*eta))
      else
        f2 = 2.e0*exp(eta)/(1.e0+0.1092*exp(0.8908e0*eta))
      end if

      end subroutine f2_analytic

!=======================================================================

!=======================================================================
!
!     Subroutine: f3_analytic
!
!=======================================================================

      subroutine f3_analytic(eta,f3)

      implicit none

      CCTK_REAL, intent(in)  :: eta
      CCTK_REAL, intent(out) :: f3

      CCTK_REAL :: eta2,eta4

      if (eta.gt.1.e-3) then
        eta2 = eta*eta
        eta4 = eta2*eta2
        f3 = (eta4/4.e0 + 4.9348e0*eta2 + 1.13644e0)/(1.+exp(-1.9039e0*eta))
      else
        f3 = 6.e0*exp(eta)/(1.e0+0.0559e0*exp(0.9069e0*eta))
      end if

      end subroutine f3_analytic

!=======================================================================
*/

/*
PH: These are the exact fermi integral calculations. In the original 
code they are switched off by default, so I have put them in here, but 
they are not translated from F90.

!=======================================================================
!
!     Fermi integral calculation
!
!=======================================================================

      subroutine fermiint(k,eta,f)

!=======================================================================
! This subroutine calculates the Fermi integral function, once the order,
! the point and the order of Gauss-Legendre integration have been
! specified
!=======================================================================

      implicit none

      CCTK_REAL, intent(in) :: k
      CCTK_REAL, intent(in) :: eta

      CCTK_REAL, intent(out) :: f

!.......................................................................
!     Input variables:
!     k     ----> order of the Fermi integral
!     eta   ----> point where to evaluate the Fermi integral function
!
!     Output variables:
!     f     ----> F_k (eta)
!.......................................................................

      integer :: i
      CCTK_REAL :: fxi

!.....initialize function to 0
      f = 0.e0
      if (.not.gl_init) then
        call gauleg(0.d0,1.d0)
        gl_init = .true.
      end if
      do i=1,ngl
         call kernel(k,eta,xgl(i),fxi)
         f = f + wgl(i) * fxi
      end do

      end subroutine fermiint

!=======================================================================

!=======================================================================

      subroutine kernel(k,eta,x,fcompx)
!....................................................................
! This subroutine calculate the kernel for the calculation of the Fermi
! integral of order k, shift factor eta, in the point x, for the
! numerical integration.
!
!     Input:
!     x ------> abscissa
!     k ------> order of the Fermi integral
!     eta ----> shift parameter of the Fermi integral
!
!     Output:
!     fcompx -----> function
!....................................................................

       implicit none
       CCTK_REAL, intent(in) :: x
       CCTK_REAL, intent(in) :: k
       CCTK_REAL, intent(in) :: eta
       CCTK_REAL, intent(out) :: fcompx

       CCTK_REAL :: f
       CCTK_REAL :: t
       CCTK_REAL :: s, a

!....................................................................
!      the parameter a describes how the nodes should be sampled
!      (i.e. instead of [0,1] and [1,infty], we use the intervals
!      [0,eta] and [eta,infty] here), except eta<1.0
!....................................................................
       a = max(1.e0,eta)

       f = a * (x*a)**k * fermi(x*a-eta)
       t = a/x
       s = a * t**k * fermi(t-eta)/(x*x)
       fcompx = f + s

       end subroutine kernel

!=======================================================================

!=======================================================================
!
!
!
!=======================================================================

      function fermi(arg)

      implicit none

      CCTK_REAL, intent(in) :: arg
      CCTK_REAL :: fermi

!-----------------------------------------------------------------------

      CCTK_REAL :: tmp

      if (arg.gt.0.e0) then
        tmp = exp(-arg)
        fermi = tmp/(tmp+1.e0)
      else
        tmp = exp(arg)
        fermi = 1./(tmp+1.e0)
      endif

      end function fermi

!=======================================================================

!=======================================================================
!
!     Subroutine: gauleg
!
!=======================================================================

      subroutine gauleg(x1,x2)

      implicit none

!     Note: iof I remember correctly, this subroutine needs to be in
!     double. I am forcing it to be real*8

      real*8, intent(in) :: x1,x2

      integer :: i,j,m
      real*8 :: p1,p2,p3,pp,xl,xm,z,z1

      m = (ngl+1)/2
      xm = 0.5d0*(x2+x1)
      xl = 0.5d0*(x2-x1)
      do i=1,m
         z = dcos(pi*(dble(i)-0.25d0)/(dble(ngl)+0.5d0))
         z1 = 0.0
         do while(abs(z-z1).gt.gl_eps)
            p1 = 1.0d0
            p2 = 0.0d0
            do j=1,ngl
               p3 = p2
               p2 = p1
               p1 = ((2.0d0*dble(j)-1.0d0)*z*p2-(dble(j)-1.0d0)*p3)   &
     &              / dble(j)
            end do
            pp = dble(ngl)*(z*p1-p2)/(z*z-1.0d0)
            z1 = z
            z = z1 - p1/pp
         end do
         xgl(i) = xm - xl*z
         xgl(ngl+1-i) = xm + xl*z
         wgl(i) = (2.0d0*xl)/((1.0d0-z*z)*pp*pp)
         wgl(ngl+1-i) = wgl(i)
      end do

      end subroutine gauleg

!=======================================================================
*/

// end module weak_equilibrium_mod

// *********************************************************************

int WeakEquilibriumMod::NeutrinoDens_cgs(Real rho, Real temp, Real ye, Real& n_nue, Real& n_nua, Real& n_nux, Real& en_nue, Real& en_nua, Real& en_nux) {
  int iout = 0; // NeutrinoDens_cgs = 0

// Below is the full include. It seems that much of it is not needed, so it has been commented out.
// #define WEAK_RATES_ITS_ME
// #include "inc/weak_rates_guts.inc"

// #ifndef WEAK_RATES_ITS_ME
// #error "This file should not be included by the end user!"
// #endif

  // Density is assumed to be in cgs units and
  // the temperature in MeV

  /* TODO: EoS calls 
  lrho  = log10(rho)
  ltemp = log10(temp)

  // Compute the baryon number density (mass_fact is given in MeV)
  mass_fact_cgs = mass_fact * mev_to_erg / (clight*clight)
  nb = rho / mass_fact_cgs

  // Interpolate the chemical potentials (stored in MeV in the table)
  mu_n = lkLinearInterpolation3d(lrho, ltemp, ye, MU_N)
  mu_p = lkLinearInterpolation3d(lrho, ltemp, ye, MU_P)
  mu_e = lkLinearInterpolation3d(lrho, ltemp, ye, MU_E)

  ! Interpolate the fractions
  abar = lkLinearInterpolation3d(lrho, ltemp, ye, ABAR)
  zbar = lkLinearInterpolation3d(lrho, ltemp, ye, ZBAR)
  xp   = lkLinearInterpolation3d(lrho, ltemp, ye, XP)
  xn   = lkLinearInterpolation3d(lrho, ltemp, ye, XN)
  xh   = lkLinearInterpolation3d(lrho, ltemp, ye, XH)
*/
  Real mu_n = EoS->GetNeutronChemicalPotential(rho, temp, ye);
  Real mu_p = EoS->GetProtonChemicalPotential(rho, temp, ye);
  Real mu_e = EoS->GetElectronChemicalPotential(rho, temp, ye);

/*
  ! Compute the neutrino degeneracy assuming that neutrons and
  ! protons chemical potentials DO NOT include the rest mass density
  ! eta_nue = (mu_p + mu_e - mu_n - Qnp) / temp
*/
      
  // Compute the neutrino degeneracy assuming that neutrons and
  // protons chemical potentials includes the rest mass density
  // This is the correct formula for stellarcollapse.org tables
  Real eta_nue = (mu_p + mu_e - mu_n) / temp;
  Real eta_nua = -eta_nue;
  Real eta_nux = 0.0;
  Real eta_e   = mu_e / temp;

  // Neutron and proton degeneracy
  // Real eta_n   = mu_n / temp
  // Real eta_p   = mu_p / temp
  // Difference in the degeneracy parameters without
  // neutron-proton rest mass difference
  // eta_hat = eta_n - eta_p - Qnp / temp

  // Janka takes into account the Pauli blocking effect for
  // degenerate nucleons as in Bruenn (1985). Ruffert et al. Eq. (A8)
  // xp = xp / (1.0d0 + 2.0d0 / 3.0d0 * (max(eta_p, 0.0d0)))
  // xn = xn / (1.0d0 + 2.0d0 / 3.0d0 * (max(eta_n, 0.0d0)))

  // Consistency check on the fractions
  // xp = max(0.0d0, xp)
  // xn = max(0.0d0, xn)
  // xh = max(0.0d0, xh)
  // abar = max(0.0d0, abar)
  // zbar = max(0.0d0, zbar)

  // eta takes into account the nucleon final state blocking
  // (at high density)
  // eta_np = nb * (xp-xn) / (exp(-eta_hat) - 1.0d0)
  // eta_pn = nb * (xn-xp) / (exp(eta_hat) - 1.0d0)

  // There is no significant defferences between Rosswog (prev. formula)
  // and Janka's prescriptions
  // eta_np = nb * ((2.0d0 * ye-1.0d0) / (exp(eta_p-eta_n) - 1.0d0))
  // eta_pn = eta_np * exp(eta_p-eta_n)

  // See Bruenn (ApJSS 58 1985) formula 3.1, non degenerate matter limit.
  // if  (rho < 2.0d11) then
  //   eta_pn = nb * xp
  //   eta_np = nb * xn
  // endif

  // Consistency Eqs (A9) (Rosswog's paper) they should be positive
  // eta_pn = max(eta_pn, 0.0d0)
  // eta_np = max(eta_np, 0.0d0)

// #include ends

// #undef WEAK_RATES_ITS_ME

  Real hc_mevcm3 = hc_mevcm*hc_mevcm*hc_mevcm;
  Real temp3 = temp*temp*temp;
  Real temp4 = temp3*temp;

  n_nue = 4.0 * pi / hc_mevcm3 * temp3 * fermi2(eta_nue);
  n_nua = 4.0 * pi / hc_mevcm3 * temp3 * fermi2(eta_nua);
  n_nux = 16.0 * pi / hc_mevcm3 * temp3 * fermi2(eta_nux);

  en_nue = 4.0 * pi / hc_mevcm3 * temp4 * fermi3(eta_nue);
  en_nua = 4.0 * pi / hc_mevcm3 * temp4 * fermi3(eta_nua);
  en_nux = 16.0 * pi / hc_mevcm3 * temp4 * fermi3(eta_nux);

// Not sure if we want this
// #ifndef FORTRAN_DISABLE_IEEE_ARITHMETIC
  if (!isfinite(n_nue)) {
    // write(*,*) "NeutrinoDens_cgs: NaN/Inf in n_nue", rho, temp, ye
    iout = WE_ND_NONFINITE;
  } // endif

  if (!isfinite(n_nua)) {
    // write(*,*) "NeutrinoDens_cgs: NaN/Inf in n_nua", rho, temp, ye
    iout = WE_ND_NONFINITE;
  } // endif

  if (!isfinite(n_nux)) {
    // write(*,*) "NeutrinoDens_cgs: NaN/Inf in n_nux", rho, temp, ye
    iout = WE_ND_NONFINITE;
  } // endif

  if (!isfinite(en_nue)) {
    // write(*,*) "NeutrinoDens_cgs: NaN/Inf in en_nue", rho, temp, ye
    iout = WE_ND_NONFINITE;
  } // endif

  if (!isfinite(en_nua)) {
    // write(*,*) "NeutrinoDens_cgs: NaN/Inf in en_nua", rho, temp, ye
    iout = WE_ND_NONFINITE;
  } // endif

  if (!isfinite(en_nux)) {
    // write(*,*) "NeutrinoDens_cgs: NaN/Inf in en_nux", rho, temp, ye
    iout = WE_ND_NONFINITE;
  } // endif
// #endif
  return iout;
} // END FUNCTION NeutrinoDens_cgs