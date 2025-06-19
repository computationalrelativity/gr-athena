#include <limits>

// Athena++ classes headers
#include "../../../athena.hpp"
#include "../../../eos/eos.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../m1.hpp"

#include "bnsnurates.hpp"

// Assert isfinite for opacities and other stuff?
#define ASSERT_OPAC_ISFINITE (1)
#define ASSERT_NDEN_ISFINITE (1)

// Variable to switch between analytic and numerical solutions of Fermi integrals
#define FERMI_ANALYTIC (1)


namespace M1::Opacities::BNSNuRates {

  
  //! \fn void bns_nurates(Real &nb, Real Real &temp, Real &ye, Real &mu_n, Real &mu_p,
  //!                      Real &mu_e, Real &n_nue, Real &j_nue, Real &chi_nue,
  //!                      Real &n_anue, Real &j_anue, Real &chi_anue, Real &n_nux,
  //!                      Real &j_nux, Real &chi_nux, Real &n_anux, Real &j_anux,
  //!                      Real &chi_anux, Real &R_nue, Real &R_anue, Real &R_nux,
  //!                      Real &R_anux, Real &Q_nue, Real &Q_anue, Real &Q_nux,
  //!                      Real &Q_anux, Real &sigma_0_nue, Real &sigma_0_anue,
  //!                      Real &sigma_0_nux, Real &sigma_0_anux, Real &sigma_1_nue,
  //!                      Real &sigma_1_anue, Real &sigma_1_nux, Real &sigma_1_anux,
  //!                      Real &scat_0_nue, Real &scat_0_anue, Real &scat_0_nux,
  //!                      Real &scat_0_anux, Real &scat_1_nue, Real &scat_1_anue,
  //!                      Real &scat_1_nux, Real &scat_1_anux)
  //   \brief Computes the rates given the M1 quantities
  //
  //   \note  All input and output quantities are in code units
  //
  //   \param[in] nb              baryon number density
  //   \param[in] temp            temperature 
  //   \param[in] ye              electron fraction
  //   \param[in] mu_n            neutron chemical potential
  //   \param[in] mu_p            proton chemical potential
  //   \param[in] mu_e            electron chemical potential
  //   \param[in] n_nue           number density electron neutrinos
  //   \param[in] j_nue           energy density electron neutrinos
  //   \param[in] chi_nue         eddington factor electron neutrinos
  //   \param[in] n_anue          number density electron anti-neutrinos
  //   \param[in] j_anue          energy density electron anti-neutrinos
  //   \param[in] chi_anue        eddington factor electron anti-neutrinos
  //   \param[in] n_nux           number density mu/tau neutrinos
  //   \param[in] j_nux           energy density mu/tau neutrinos
  //   \param[in] chi_nux         eddington factor mu/tau neutrinos
  //   \param[in] n_anux          number density mu/tau neutrinos
  //   \param[in] j_anux          energy density mu/tau neutrinos
  //   \param[in] chi_anux        eddington factor mu/tau neutrinos
  //
  //   \param[out] R_nue          number emissivity electron neutrinos
  //   \param[out] R_anue         number emissivity electron anti-neutrinos
  //   \param[out] R_nux          number emissivity mu/tau neutrinos
  //   \param[out] R_anux         number emissivity mu/tau anti-neutrinos
  //   \param[out] Q_nue          energy emissivity electron neutrinos
  //   \param[out] Q_anue         energy emissivity electron anti-neutrinos
  //   \param[out] Q_nux          energy emissivity mu/tau neutrinos
  //   \param[out] Q_anux         energy emissivity mu/tau anti-neutrinos
  //   \param[out] sigma_0_nue    number inv mean-free path electron neutrinos
  //   \param[out] sigma_0_anue   number inv mean-free path electron anti-neutrinos
  //   \param[out] sigma_0_nux    number inv mean-free path mu/tau neutrinos
  //   \param[out] sigma_0_anux   number inv mean-free path mu/tau anti-neutrinos
  //   \param[out] sigma_1_nue    energy inv mean-free path electron neutrinos
  //   \param[out] sigma_1_anue   energy inv mean-free path electron anti-neutrinos
  //   \param[out] sigma_1_nux    energy inv mean-free path mu/tau neutrinos
  //   \param[out] sigma_1_anux   energy inv mean-free path mu/tau neutrinos
  //   \param[out] scat_0_nue     number scatt coeff electron neutrinos
  //   \param[out] scat_0_anue    number scatt coeff electron anti-neutrinos
  //   \param[out] scat_0_nux     number scatt coeff mu/tau neutrinos
  //   \param[out] scat_0_anux    number scatt coeff mu/tau anti-neutrinos
  //   \param[out] scat_1_nue     energy scatt coeff electron neutrinos
  //   \param[out] scat_1_anue    energy scatt coeff electron ant-neutrinos
  //   \param[out] scat_1_nux     energy scatt coeff mu/tau neutrinos
  //   \param[out] scat_1_anux    energy scatt coeff mu/tau anti-neutrinos
  //   \param[in]  nurates_params params for nurates
  int BNSNuRates::bns_nurates_wrapper(Real &nb, Real &temp, Real &ye,
				      Real &mu_n, Real &mu_p, Real &mu_e,
				      Real &n_nue, Real &j_nue, Real &chi_nue,
				      Real &n_anue, Real &j_anue, Real &chi_anue,
				      Real &n_nux, Real &j_nux, Real &chi_nux,
				      Real &n_anux, Real &j_anux, Real &chi_anux,
				      Real &R_nue, Real &R_anue, Real &R_nux, Real &R_anux,
				      Real &Q_nue, Real &Q_anue, Real &Q_nux, Real &Q_anux,
				      Real &sigma_0_nue, Real &sigma_0_anue, Real &sigma_0_nux, Real &sigma_0_anux,
				      Real &sigma_1_nue, Real &sigma_1_anue, Real &sigma_1_nux, Real &sigma_1_anux,
				      Real &scat_0_nue, Real &scat_0_anue, Real &scat_0_nux, Real &scat_0_anux,
				      Real &scat_1_nue, Real &scat_1_anue, Real &scat_1_nux, Real &scat_1_anux,
				      const NuratesParams nurates_params)
  
  {
    
    int ierr = 0;

    // Some conversions factors for opacities from CGS+MeV to code units
    // from cm^-3 to code units
    const Real cgs2code_n = my_units->NumberDensityConversion(*code_units);
    // from cm^-3 s^-1 to code units
    const Real cgs2code_R = cgs2code_n / my_units->TimeConversion(*code_units);
    // from MeV cm^-3 to code units
    const Real cgs2code_j = MEV_TO_ERG * my_units->EnergyDensityConversion(*code_units); 
    // from MeV cm^-3 s^-1 to code units
    const Real cgs2code_Q = cgs2code_j / my_units->TimeConversion(*code_units);
    // from cm^-1 to code units
    const Real cgs2code_kappa = 1.0 / my_units->LengthConversion(*code_units);
    
    // code units to MeV 
    const Real MeV = code_units->TemperatureConversion(*my_units); 

    // Convert input to CGS+MeV
    const Real nb_cgs = nb / cgs2code_n; // [baryon/cm^-3]
    const Real temp_mev = temp * MeV; 
    const Real mu_n_mev = mu_n * MeV;
    const Real mu_p_mev = mu_p * MeV;
    const Real mu_e_mev = mu_e * MeV;
    
    if ( (nb_cgs < nurates_params.nb_min_cgs) ||
         (temp_mev < nurates_params.temp_min_mev) ) {
      R_nue = 0.;
      R_anue = 0.;
      R_nux = 0.;
      R_anux = 0.;
      Q_nue = 0.;
      Q_anue = 0.;
      Q_nux = 0.;
      Q_anux = 0.;
      sigma_0_nue = 0.;
      sigma_0_anue = 0.;
      sigma_0_nux = 0.;
      sigma_0_anux = 0.;
      sigma_1_nue = 0.;
      sigma_1_anue = 0.;
      sigma_1_nux = 0.;
      sigma_1_anux = 0.;
      scat_0_nue = 0.;
      scat_0_anue = 0.;
      scat_0_nux = 0.;
      scat_0_anux = 0.;
      scat_1_nue = 0.;
      scat_1_anue = 0.;
      scat_1_nux = 0.;
      scat_1_anux = 0.;
      return ierr;
    }

    // bns_nurates requires CGS + MeV + nm units 
    // this needs a further rescaling from cm to nm
    
    // convert neutrino quantities from code units to CGS + nm units (and adjust for NORMFACT)
    const Real n_nue_nmunits = n_nue / (cgs2code_n / NORMFACT) * 1e-21;    // [nm^-3]
    const Real n_anue_nmunits = n_anue / (cgs2code_n / NORMFACT) * 1e-21;  // [nm^-3]
    const Real n_nux_nmunits = n_nux / (cgs2code_n / NORMFACT) * 1e-21;    // [nm^-3]
    const Real n_anux_nmunits = n_anux / (cgs2code_n / NORMFACT) * 1e-21;  // [nm^-3]
    
    const Real j_nue_nmunits = j_nue / cgs2code_j * 1e-21 * kBS_MeV;    // [g s^-2 nm^-1]
    const Real j_anue_nmunits = j_anue / cgs2code_j * 1e-21 * kBS_MeV;  // [g s^-2 nm^-1]
    const Real j_nux_nmunits = j_nux / cgs2code_j * 1e-21 * kBS_MeV;    // [g s^-2 nm^-1]
    const Real j_anux_nmunits = j_anux / cgs2code_j * 1e-21 * kBS_MeV;  // [g s^-2 nm^-1]

    // convert also baryon density
    const Real nb_nmunits = nb / my_units->NumberDensityConversion(*code_units) * 1e-21; // [baryon/nm^-3]
    
    // opacity params structure
    GreyOpacityParams grey_op_params{};
    
    // reaction flags
    //grey_op_params.opacity_flags = opacity_flags_default_none; //TODO  opacity_flags_default_none = ?
    grey_op_params.opacity_flags.use_abs_em = nurates_params.use_abs_em;
    grey_op_params.opacity_flags.use_brem = nurates_params.use_brem;
    grey_op_params.opacity_flags.use_pair = nurates_params.use_pair;
    grey_op_params.opacity_flags.use_iso = nurates_params.use_iso;
    grey_op_params.opacity_flags.use_inelastic_scatt =
      nurates_params.use_inelastic_scatt;
    
    // other flags
    //grey_op_params.opacity_pars = opacity_params_default_none; //TODO
    grey_op_params.opacity_pars.use_WM_ab = nurates_params.use_WM_ab;
    grey_op_params.opacity_pars.use_WM_sc = nurates_params.use_WM_sc;
    grey_op_params.opacity_pars.use_dU = nurates_params.use_dU;
    grey_op_params.opacity_pars.use_dm_eff = nurates_params.use_dm_eff;
    grey_op_params.opacity_pars.use_NN_medium_corr = nurates_params.use_NN_medium_corr;
    grey_op_params.opacity_pars.neglect_blocking = nurates_params.neglect_blocking;
    grey_op_params.opacity_pars.use_decay = nurates_params.use_decay;
    grey_op_params.opacity_pars.use_BRT_brem = nurates_params.use_BRT_brem;

    //NB logic for nurates_params.use_dU not implemented.
    // See THC WeakRates2.
    
    // populate EOS quantities
    grey_op_params.eos_pars.nb = nb_nmunits;  // [baryon/nm^3]
    grey_op_params.eos_pars.temp = temp_mev;  // [MeV]
    grey_op_params.eos_pars.yp = ye;          // [dimensionless]
    grey_op_params.eos_pars.yn = 1 - ye;      // [dimensionless]
    grey_op_params.eos_pars.mu_e = mu_e_mev;  // [MeV]
    grey_op_params.eos_pars.mu_p = mu_p_mev;  // [MeV]
    grey_op_params.eos_pars.mu_n = mu_n_mev;  // [MeV]

    grey_op_params.eos_pars.dU = nurates_params.dU;
    grey_op_params.eos_pars.dm_eff = nurates_params.dm_eff;
    
    // populate M1 quantities
    // The factors of 1/2 come from the fact that bns_nurates and THC weight the
    // heavy neutrinos differently. THC weights them with a factor of 2 (because
    // "nux" means "mu AND tau"), bns_nurates with a factor of 1 (because "nux"
    // means "mu OR tau").
    // GR-Athena++ uses same treatment as THC.
    
    grey_op_params.m1_pars.n[id_nue] = n_nue_nmunits;  // [nm^-3]
    grey_op_params.m1_pars.J[id_nue] = j_nue_nmunits;  // [g s^-2 nm^-1]
    grey_op_params.m1_pars.chi[id_nue] = chi_nue;
    grey_op_params.m1_pars.n[id_anue] = n_anue_nmunits;  // [nm^-3]
    grey_op_params.m1_pars.J[id_anue] = j_anue_nmunits;  // [g s^-2 nm^-1]
    grey_op_params.m1_pars.chi[id_anue] = chi_anue;
    grey_op_params.m1_pars.n[id_nux] = n_nux_nmunits * 0.5;  // [nm^-3]
    grey_op_params.m1_pars.J[id_nux] = j_nux_nmunits * 0.5;  // [g s^-2 nm^-1]
    grey_op_params.m1_pars.chi[id_nux] = chi_nux;
    grey_op_params.m1_pars.n[id_anux] = n_anux_nmunits * 0.5;  // [nm^-3]
    grey_op_params.m1_pars.J[id_anux] = j_anux_nmunits * 0.5;  // [g s^-2 nm^-1]
    grey_op_params.m1_pars.chi[id_anux] = chi_anux;
    
    // reconstruct distribution function
    if (!nurates_params.use_equilibrium_distribution) {
      grey_op_params.distr_pars =
        CalculateDistrParamsFromM1(&grey_op_params.m1_pars,
                                   &grey_op_params.eos_pars);
    } else {
      grey_op_params.distr_pars =
        NuEquilibriumParams(&grey_op_params.eos_pars);
      
      // compute neutrino number and energy densities
      ComputeM1DensitiesEq(&grey_op_params.eos_pars,
                           &grey_op_params.distr_pars,
                           &grey_op_params.m1_pars);
      
      // populate M1 quantities
      grey_op_params.m1_pars.chi[id_nue] = 0.333333333333333333333333333;
      grey_op_params.m1_pars.chi[id_anue] = 0.333333333333333333333333333;
      grey_op_params.m1_pars.chi[id_nux] = 0.333333333333333333333333333;
      grey_op_params.m1_pars.chi[id_anux] = 0.333333333333333333333333333;
      
      // convert neutrino energy density to mixed MeV and cgs as requested by bns_nurates
      grey_op_params.m1_pars.J[id_nue] *= kBS_MeV; 
      grey_op_params.m1_pars.J[id_anue] *= kBS_MeV;
      grey_op_params.m1_pars.J[id_nux] *= kBS_MeV;
      grey_op_params.m1_pars.J[id_anux] *= kBS_MeV;
    }
    
    // compute opacities
    M1Opacities opacities = ComputeM1Opacities(&nurates_params.quadrature,
                                               &nurates_params.quadrature,
                                               &grey_op_params);
    
    // Similar to the comment above, the factors of 2 come from the fact that
    // bns_nurates and THC weight the heavy neutrinos differently. THC weights
    // them with a factor of 2 (because "nux" means "mu AND tau"), bns_nurates
    // with a factor of 1 (because "nux" means "mu OR tau").
    // GR-Athena++ uses same treatment as THC.
    
    // extract emissivities
    R_nue = opacities.eta_0[id_nue];
    R_anue = opacities.eta_0[id_anue];
    R_nux = opacities.eta_0[id_nux] * 2.;
    R_anux = opacities.eta_0[id_anux] * 2.;
    Q_nue = opacities.eta[id_nue];
    Q_anue = opacities.eta[id_anue];
    Q_nux = opacities.eta[id_nux] * 2.;
    Q_anux = opacities.eta[id_anux] * 2.;
    
    // extract absorption inverse mean-free path
    sigma_0_nue = opacities.kappa_0_a[id_nue];
    sigma_0_anue = opacities.kappa_0_a[id_anue];
    sigma_0_nux = opacities.kappa_0_a[id_nux] * 2.;
    sigma_0_anux = opacities.kappa_0_a[id_anux] * 2.;
    sigma_1_nue = opacities.kappa_a[id_nue];
    sigma_1_anue = opacities.kappa_a[id_anue];
    sigma_1_nux = opacities.kappa_a[id_nux] * 2.;
    sigma_1_anux = opacities.kappa_a[id_anux] * 2.;
    
    // extract scattering inverse mean-free path
    scat_0_nue = 0;
    scat_0_anue = 0;
    scat_0_nux = 0;
    scat_0_anux = 0;
    scat_1_nue = opacities.kappa_s[id_nue];
    scat_1_anue = opacities.kappa_s[id_anue];
    scat_1_nux = opacities.kappa_s[id_nux] * 2.;
    scat_1_anux = opacities.kappa_s[id_anux] * 2.;
    
    // Check for NaNs/Infs
#if (ASSERT_OPAC_ISFINITE)
    assert(isfinite(R_nue));
    assert(isfinite(R_anue));
    assert(isfinite(R_nux));
    assert(isfinite(R_anux));
    assert(isfinite(Q_nue));
    assert(isfinite(Q_anue));
    assert(isfinite(Q_nux));
    assert(isfinite(Q_anux));
    assert(isfinite(sigma_0_nue));
    assert(isfinite(sigma_0_anue));
    assert(isfinite(sigma_0_nux));
    assert(isfinite(sigma_0_anux));
    assert(isfinite(sigma_1_nue));
    assert(isfinite(sigma_1_anue));
    assert(isfinite(sigma_1_nux));
    assert(isfinite(sigma_1_anux));
    assert(isfinite(scat_0_nue));
    assert(isfinite(scat_0_anue));
    assert(isfinite(scat_0_nux));
    assert(isfinite(scat_0_anux));
    assert(isfinite(scat_1_nue));
    assert(isfinite(scat_1_anue));
    assert(isfinite(scat_1_nux));
    assert(isfinite(scat_1_anux));
#endif

    // Catch these and return ierr code 1...24
    // (returns int for the number of failures.)
    if (!isfinite(R_nue)) ierr++; 
    if (!isfinite(R_anue)) ierr++;
    if (!isfinite(R_nux)) ierr++;
    if (!isfinite(R_anux)) ierr++;
    if (!isfinite(Q_nue)) ierr++;
    if (!isfinite(Q_anue)) ierr++;
    if (!isfinite(Q_nux)) ierr++;
    if (!isfinite(Q_anux)) ierr++;
    if (!isfinite(sigma_0_nue)) ierr++;
    if (!isfinite(sigma_0_anue)) ierr++;
    if (!isfinite(sigma_0_nux)) ierr++;
    if (!isfinite(sigma_0_anux)) ierr++;
    if (!isfinite(sigma_1_nue)) ierr++;
    if (!isfinite(sigma_1_anue)) ierr++;
    if (!isfinite(sigma_1_nux)) ierr++;
    if (!isfinite(sigma_1_anux)) ierr++;
    if (!isfinite(scat_0_nue)) ierr++;
    if (!isfinite(scat_0_anue)) ierr++;
    if (!isfinite(scat_0_nux)) ierr++;
    if (!isfinite(scat_0_anux)) ierr++;
    if (!isfinite(scat_1_nue)) ierr++;
    if (!isfinite(scat_1_anue)) ierr++;
    if (!isfinite(scat_1_nux)) ierr++;
    if (!isfinite(scat_1_anux)) ierr++;
    
    // convert back to code units
    R_nue = R_nue * (cgs2code_R / NORMFACT) * 1e21;
    R_anue = R_anue * (cgs2code_R / NORMFACT) * 1e21;
    R_nux = R_nux * (cgs2code_R / NORMFACT) * 1e21;
    R_anux = R_anux * (cgs2code_R / NORMFACT) * 1e21;
    Q_nue = Q_nue * cgs2code_Q * 1e21;
    Q_anue = Q_anue * cgs2code_Q * 1e21;
    Q_nux = Q_nux * cgs2code_Q * 1e21;
    Q_anux = Q_anux * cgs2code_Q * 1e21;
    sigma_0_nue = sigma_0_nue * cgs2code_kappa * 1e7;
    sigma_0_anue = sigma_0_anue * cgs2code_kappa * 1e7;
    sigma_0_nux = sigma_0_nux * cgs2code_kappa * 1e7;
    sigma_0_anux = sigma_0_anux * cgs2code_kappa * 1e7;
    sigma_1_nue = sigma_1_nue * cgs2code_kappa * 1e7;
    sigma_1_anue = sigma_1_anue * cgs2code_kappa * 1e7;
    sigma_1_nux = sigma_1_nux * cgs2code_kappa * 1e7;
    sigma_1_anux = sigma_1_anux * cgs2code_kappa * 1e7;
    scat_0_nue = scat_0_nue * cgs2code_kappa * 1e7;
    scat_0_anue = scat_0_anue * cgs2code_kappa * 1e7;
    scat_0_nux = scat_0_nux * cgs2code_kappa * 1e7;
    scat_0_anux = scat_0_anux * cgs2code_kappa * 1e7;
    scat_1_nue = scat_1_nue * cgs2code_kappa * 1e7;
    scat_1_anue = scat_1_anue * cgs2code_kappa * 1e7;
    scat_1_nux = scat_1_nux * cgs2code_kappa * 1e7;
    scat_1_anux = scat_1_anux * cgs2code_kappa * 1e7;
      
    return ierr;
  }

  
  //! \fn void NeutrinoDensity_ChemPot(Real nb, Real temp,
  //!                                  Real mu_n, Real mu_p, Real mu_e, 
  //!                                  Real &n_nue, Real &n_anue, Real &n_nux,
  //!                                  Real &en_nue, Real &en_anue, Real &en_nux, 
  //!                                  NuratesParams nurates_params)
  //
  //   \brief Computes the neutrino number and energy density, given chem potentials
  //
  //   \note  All input and output quantities are in code units
  //
  //   \param[in]  mu_n            neutron chemical potential
  //   \param[in]  mu_p            proton chemical potential
  //   \param[in]  mu_e            electron chemical potential
  //   \param[in]  nb              baryon number density
  //   \param[in]  temp            temperature
  //   \param[out] n_nue           number density electron neutrinos
  //   \param[out] n_anue          number density electron anti-neutrinos
  //   \param[out] n_nux           number density mu/tau neutrinos
  //   \param[out] en_nue          energy density electron neutrinos
  //   \param[out] en_anue         energy density electron anti-neutrinos
  //   \param[out] en_nux          energy density mu/tau neutrinos
  void BNSNuRates::NeutrinoDensity_ChemPot(Real nb, Real temp,
					   Real mu_n, Real mu_p, Real mu_e,
					   Real &n_nue, Real &n_anue, Real &n_nux,
					   Real &en_nue, Real &en_anue, Real &en_nux)
  {
    const Real nb_cgs = nb * code_units->NumberDensityConversion(*my_units); // [baryon/cm^-3]
    const Real temp_mev = temp * code_units->TemperatureConversion(*my_units);  // [MeV]
    
    if ( (nb_cgs < nurates_params.nb_min_cgs) ||
         (temp_mev < nurates_params.temp_min_mev) ) {
      n_nue = 0.;
      n_anue = 0.;
      n_nux = 0.;
      en_nue = 0.;
      en_anue = 0.;
      en_nux = 0.;
      return;
    }
    
    Real eta_nue = (mu_p + mu_e - mu_n) / temp; // ratio can use code units
    Real eta_anue = -eta_nue;
    Real eta_nux = 0.0;

    const Real hc_mevcm3 = hc_mevcm * hc_mevcm * hc_mevcm;
    const Real temp3 = temp_mev * temp_mev * temp_mev;
    const Real temp4 = temp3 * temp_mev;

    n_nue = 4.0 * M_PI / hc_mevcm3 * temp3 * Fermi::fermi2(eta_nue);    // [cm^-3]
    n_anue = 4.0 * M_PI / hc_mevcm3 * temp3 * Fermi::fermi2(eta_anue);  // [cm^-3]
    n_nux = 16.0 * M_PI / hc_mevcm3 * temp3 * Fermi::fermi2(eta_nux);   // [cm^-3]
    
    en_nue = 4.0 * M_PI / hc_mevcm3 * temp4 * Fermi::fermi3(eta_nue);    // [MeV cm^-3]
    en_anue = 4.0 * M_PI / hc_mevcm3 * temp4 * Fermi::fermi3(eta_anue);  // [MeV cm^-3]
    en_nux = 16.0 * M_PI / hc_mevcm3 * temp4 * Fermi::fermi3(eta_nux);   // [MeV cm^-3]

#if (ASSERT_NDEN_ISFINITE)
    assert(isfinite(n_nue));
    assert(isfinite(n_anue));
    assert(isfinite(n_nux));
    assert(isfinite(en_nue));
    assert(isfinite(en_anue));
    assert(isfinite(en_nux));
#endif
    
    // convert back to code units (adjusting for NORMFACT)
    const Real n_conv = my_units->NumberDensityConversion(*code_units) / NORMFACT;
    const Real e_conv = MEV_TO_ERG * my_units->EnergyDensityConversion(*code_units);

    n_nue = n_nue * n_conv;
    n_anue = n_anue * n_conv;
    n_nux = n_nux * n_conv;

    en_nue = en_nue * e_conv;
    en_anue = en_anue * e_conv;
    en_nux = en_nux * e_conv;    
  }

  
  //! \fn int WeakEquilibrium(Real rho, Real temp, Real ye,
  //!                         Real n_nue, Real n_nua, Real n_nux,
  //!                         Real e_nue, Real e_nua, Real e_nux,
  //!                         Real& temp_eq, Real& ye_eq,
  //!                         Real& n_nue_eq, Real& n_nua_eq, Real& n_nux_eq,
  //!                         Real& e_nue_eq, Real& e_nua_eq, Real& e_nux_eq)
  //
  //  \brief Calculates the equilibrium fluid temperature and electron fraction,
  //   and neutrino number and energy densities assuming energy and lepton number conservation.
  //
  //  \note  All input and output quantities are in code units
  int BNSNuRates::WeakEquilibrium(Real rho, Real temp, Real ye,
                                  Real n_nue, Real n_nua, Real n_nux,
                                  Real e_nue, Real e_nua, Real e_nux,
                                  Real& temp_eq, Real& ye_eq,
                                  Real& n_nue_eq, Real& n_nua_eq, Real& n_nux_eq,
                                  Real& e_nue_eq, Real& e_nua_eq, Real& e_nux_eq)
  {
    // conversion factors from code units to CGS + MeV
    Real n_conv = code_units->NumberDensityConversion(*my_units);  // code units to 1/cm^3
    Real e_conv = code_units->EnergyDensityConversion(*my_units);  // code units to erg/cm^3
    const Real MeV = code_units->TemperatureConversion(*my_units); // code units to MeV 
      
    int ierr = compute_weak_equilibrium(rho * code_units->MassDensityConversion(*my_units),
                                        temp * MeV,
                                        ye,
                                        n_nue * n_conv,
                                        n_nua * n_conv,
                                        n_nux * n_conv,
                                        e_nue * e_conv,
                                        e_nua * e_conv,
                                        e_nux * e_conv,
                                        temp_eq,
                                        ye_eq,
                                        n_nue_eq,
                                        n_nua_eq,
                                        n_nux_eq,
                                        e_nue_eq,
                                        e_nua_eq,
                                        e_nux_eq);

    // convert back from CGS+MeV units to code units
    temp_eq = temp_eq / MeV;

    n_conv = my_units->NumberDensityConversion(*code_units);
    e_conv = my_units->EnergyDensityConversion(*code_units);

    n_nue_eq = n_nue_eq * n_conv;
    n_nua_eq = n_nua_eq * n_conv; 
    n_nux_eq = n_nux_eq * n_conv; 
    e_nue_eq = e_nue_eq * e_conv; 
    e_nua_eq = e_nua_eq * e_conv; 
    e_nux_eq = e_nux_eq * e_conv;
    
    return ierr;
  }
  
  int BNSNuRates::compute_weak_equilibrium(Real rho,        // [g/cm^3]
                                           Real temp,       // [MeV]
                                           Real ye,         // [-]
                                           Real n_nue,      // [1/cm^3] 
                                           Real n_nua,      // [1/cm^3] 
                                           Real n_nux,      // [1/cm^3] 
                                           Real e_nue,      // [erg/cm^3] 
                                           Real e_nua,      // [erg/cm^3] 
                                           Real e_nux,      // [erg/cm^3]
                                           Real & temp_eq,  // [MeV]
                                           Real & ye_eq,    // [-] 
                                           Real & n_nue_eq, // [1/cm^3]
                                           Real & n_nua_eq, // [1/cm^3]
                                           Real & n_nux_eq, // [1/cm^3]
                                           Real & e_nue_eq, // [erg/cm^3]
                                           Real & e_nua_eq, // [erg/cm^3]
                                           Real & e_nux_eq  // [erg/cm^3]
                                           ) 
    {
      int iout = 0;

      // Everything is in CGS + MeV (my_units)
      
      // Do not do anything outside of this range
      if ( (rho < rho_min) || (temp < temp_min) ) {
        n_nue_eq  = 0.0;
        n_nua_eq  = 0.0;
        n_nux_eq  = 0.0;
        e_nue_eq = 0.0;
        e_nua_eq = 0.0;
        e_nux_eq = 0.0;
        return iout;
      } 

      Real nb = rho/atomic_mass;

      // Compute fractions
      Real y_in[4] = {0.0};
      y_in[0] = ye;
      y_in[1] = n_nue/nb;
      y_in[2] = n_nua/nb;
      y_in[3] = 0.25*n_nux/nb;
      
      // Compute energy
      // NB EOS calls require I/O unit conversion
      Real Y[1] = {ye};
      Real e_in[4] = {0.0};
      e_in[0] = pmy_block->peos->GetEOS().GetEnergy(nb
                                                    * my_units->NumberDensityConversion(*code_units),
                                                    temp
                                                    * my_units->TemperatureConversion(*code_units),
                                                    Y)
        * code_units->EnergyDensityConversion(*my_units); // [g/cm^3] 

      e_in[1] = e_nue;
      e_in[2] = e_nua;
      e_in[3] = e_nux;
      
      // Compute weak equilibrium
      Real y_eq[4] = {0.0};
      Real e_eq[4] = {0.0};
      int na = 0;
      int ierr = 0;
      weak_equil_wnu(rho, temp, y_in, e_in, temp_eq, y_eq, e_eq, na, ierr);
      ye_eq = y_eq[0];
      iout = (ierr != 0) ? -1 : 0 ;

      // Split output arrays from weak_equil_wnu
      n_nue_eq  = nb * y_eq[1];
      n_nua_eq  = nb * y_eq[2];
      n_nux_eq  = 4.0 * nb * y_eq[3];
      e_nue_eq = e_eq[1];
      e_nua_eq = e_eq[2];
      e_nux_eq = e_eq[3];
      
      return iout;
    } 
  
  
  // weak_equil_wnu
  //     input:
  //
  //     rho  ... fluid density [g/cm^3]
  //     T    ... fluid temperature [MeV]
  //     y_in ... incoming abundances
  //              y_in(1) ... initial electron fraction               [#/baryon]
  //              y_in(2) ... initial electron neutrino fraction      [#/baryon]
  //              y_in(3) ... initial electron antineutrino fraction  [#/baryon]
  //              y_in(4) ... initial heavy flavor neutrino fraction  [#/baryon]
  //                          The total one would be 0, so we are
  //                          assuming this to be each of the single ones.
  //                          Anyway, this value is useless for our
  //                          calculations. We could also assume it to be
  //                          the total and set it to 0
  //     e_eq ... incoming energies
  //              e_in(1) ... initial fluid energy, incl rest mass    [erg/cm^3]
  //              e_in(2) ... initial electron neutrino energy        [erg/cm^3]
  //              e_in(3) ... initial electron antineutrino energy    [erg/cm^3]
  //              e_in(4) ... total initial heavy flavor neutrino energy    [erg/cm^3]
  //                          This is assumed to be 4 times the energy of
  //                          each single heavy flavor neutrino species
  //
  //     output:
  //
  //     T_eq ... equilibrium temperature   [MeV]
  //     y_eq ... equilibrium abundances    [#/baryons]
  //              y_eq(1) ... equilibrium electron fraction              [#/baryon]
  //              y_eq(2) ... equilibrium electron neutrino fraction     [#/baryon]
  //              y_eq(3) ... equilibrium electron antineutrino fraction [#/baryon]
  //              y_eq(4) ... equilibrium heavy flavor neutrino fraction [#/baryon]
  //                          see explanation above and change if necessary
  //     e_eq ... equilibrium energies
  //              e_eq(1) ... equilibrium fluid energy                   [erg/cm^3]
  //              e_eq(2) ... equilibrium electron neutrino energy       [erg/cm^3]
  //              e_eq(3) ... equilibrium electron antineutrino energy   [erg/cm^3]
  //              e_eq(4) ... total equilibrium heavy flavor neutrino energy   [erg/cm^3]
  //                          see explanation above and change if necessary
  //     na   ... number of attempts in 2D Newton-Raphson
  //     ierr ... 0 success in Newton-Raphson
  //              1 failure in Newton-Raphson
  //
  void BNSNuRates::weak_equil_wnu(Real rho, Real T, Real y_in[4], Real e_in[4],
                                  Real& T_eq, Real y_eq[4], Real e_eq[4], int& na, int& ierr)
  {
    
    // Compute the total lepton fraction and internal energy
    Real yl = y_in[0] + y_in[1] - y_in[2];           // [#/baryon]
    Real u  = e_in[0] + e_in[1] + e_in[2] + e_in[3]; // [erg/cm^3]
    
    // vector with the coefficients for the different guesses............
    //     at the moment, to solve the 2D NR we assign guesses for the
    //     equilibrium ye and T close to the incoming ones. This array
    //     quantifies this closeness. Different guesses are used, one after
    //     the other, until a solution is found. Hopefully, the first one
    //     works already in most of the cases. The other ones are used as
    //     backups
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
    // ierr is the variable that check if equilibrium has been found:
    // ierr = 0   equilibrium found
    // ierr = 1   equilibrium not found
    */
    ierr = 1;
    
    // here we try different guesses x0 = [T,Ye]*vec_guess[i,:], one after 
    // the other, until success is obtained
    Real x0[2]; // Guess for T,Ye
    Real x1[2]; // Result for T,Ye
    
    while (ierr!=0 && na<n_at){
      
      // make an initial guess................................................
      x0[0] = vec_guess[na][0]*T;       // T guess  [MeV]
      x0[1] = vec_guess[na][1]*y_in[0]; // ye guess [#/baryon]
      
      // call the 2d Newton-Raphson...........................................
      new_raph_2dim(rho,u,yl,x0,x1,ierr);
      
      na += 1;
    } 
    
    // assign the output......................................................
    if (ierr==0) {
      // calculations worked
      T_eq = x1[0];
      y_eq[0] = x1[1];
    } else {
      // calculations did not work
      // as backup plan, we assign the initial values to all outputs..........
      T_eq = T;    // [MeV]
      for (int i=0;i<4;i++) {
        y_eq[i] = y_in[i]; // [#/baryon]
        e_eq[i] = e_in[i]; // [erg/cm^3]
      }
        return;
    } 
    
    // Here we want to compute the total energy and fractions in the
    // equilibrated state
    Real mu_n, mu_p, mu_e;
    ChemicalPotentials_npe_cgs(rho, T_eq, y_eq[0],  mu_n, mu_p, mu_e); 

    // in the original version of this function mus has size 3, whereas 
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
    
    // Compute the baryon number density
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
    
    // Check that the energy is positive
    // EOS calls requires code units
    // For tabulated eos we should check that the energy is above the minimum?
    Real Y[1] = {y_eq[0]};
    Real e_min = pmy_block->peos->GetEOS().GetEnergy(nb
                                                     * my_units->NumberDensityConversion(*code_units),
                                                     eos_temp_min
                                                     * my_units->TemperatureConversion(*code_units),
                                                     Y)
      * code_units->EnergyDensityConversion(*my_units); // [g/cm^3]                     
    
    if (e_eq[0] < e_min) {
      ierr = 1;
      T_eq = T;
      for (int i=0;i<4;i++) {
        y_eq[i] = y_in[i]; // [#/baryon]
        e_eq[i] = e_in[i]; // [erg/cm^3]
      }
      return;
    } 
    
    // check that Y_e is within the range
    // NB In WeakRates eos_ye_m* are table_ye_m*
    if (y_eq[0] < eos_ye_min || y_eq[0] > eos_ye_max) {
      ierr = 1;
      T_eq = T;
      for (int i=0;i<4;i++) {
        y_eq[i] = y_in[i]; // [#/baryon]
        e_eq[i] = e_in[i]; // [erg/cm^3]
      }
      return;
    } 
    
    return;
  }

  
  // Apply the EOS limit
  bool BNSNuRates::apply_eos_limits(Real& rho, Real& temp, Real& ye)
  {
    bool limits_applied = false;

    if (rho < eos_rho_min) {
      rho = eos_rho_min;
      limits_applied = true;
    } else if (rho > eos_rho_max) {
      rho = eos_rho_max;
      limits_applied = true;
    }
    
    if (temp < eos_temp_min) {
      temp = eos_temp_min;
      limits_applied = true;
    } else if (temp > eos_temp_max) {
      temp = eos_temp_max;
      limits_applied = true;
    }
    
    if (ye < eos_ye_min) {
      ye = eos_ye_min;
      limits_applied = true;
    } else if (ye > eos_ye_max) {
      ye = eos_ye_max;
      limits_applied = true;
    }
    
    return limits_applied;
  }
  

  // new_raph_2dim
  //
  //     input:
  //     rho ... density               [g/cm^3]
  //     u   ... total internal energy [erg/cm^3]
  //     yl  ... lepton number         [erg/cm^3]
  //     x0  ... T and ye guess
  //        x0(1) ... T               [MeV]
  //        x0(2) ... ye              [#/baryon]
  //
  //     output:
  //     x1 ... T and ye at equilibrium
  //        x1(1) ... T               [MeV]
  //        x1(2) ... ye              [#/baryon]
  //     ierr  ...
  //
  void BNSNuRates::new_raph_2dim(Real rho, Real u, Real yl, Real x0[2],
                                 Real x1[2], int& ierr) {
    
    // initialize the solution
    x1[0] = x0[0];
    x1[1] = x0[1];
    
    // KKT
    // If true then we satisfy the Karush-Kuhn-Tucker conditions.
    // This means that the equilibrium is out of the table and we have the best possible result.
    bool KKT = false;
    
    // compute the initial residuals
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
    while (err>eps_lim && n_iter<=n_max_iter && !KKT) {
      
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
      } 
      
      // invert the Jacobian
      inv_jacobi(det,J,invJ);
      
      // compute the next step
      dx1[0] = - (invJ[0][0]*y[0] + invJ[0][1]*y[1]);
      dx1[1] = - (invJ[1][0]*y[0] + invJ[1][1]*y[1]);
      
      // check if we are the boundary of the table
      if (x1[0] == eos_temp_min) {
          norm[0] = -1.0;
      } else if (x1[0] == eos_temp_max) {
        norm[0] = 1.0;
      } else { 
          norm[0] = 0.0;
      } 
      
      if (x1[1] == eos_ye_min) {
        norm[1] = -1.0;
      } else if (x1[1] == eos_ye_max) {
        norm[1] = 1.0;
      } else {
        norm[1] = 0.0;
      } 
      
      // Take the part of the gradient that is active (pointing within the eos domain)
      Real scal = norm[0]*norm[0] + norm[1]*norm[1];
      if (scal <= 0.5) { // this can only happen if norm = (0, 0)
        scal = 1.0;
      } 
      dxa[0] = dx1[0] - (dx1[0]*norm[0] + dx1[1]*norm[1])*norm[0]/scal;
      dxa[1] = dx1[1] - (dx1[0]*norm[0] + dx1[1]*norm[1])*norm[1]/scal;
        
      if ((dxa[0]*dxa[0] + dxa[1]*dxa[1]) < (eps_lim*eps_lim * (dx1[0]*dx1[0] + dx1[1]*dx1[1]))) {
        KKT = true;
        ierr = 2;
        return;
      }
      
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
        } 

        // Here we simple enforce limits with eos_*_min/max values.
        // This call is called ApplyTableLimits in weakrates/weak_eos
        // but it is with different options
        bool tabBoundsFlag = apply_eos_limits(rho, x1_tmp[0], x1_tmp[1]); 
        
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
      
    } // end do
    
    // if equilibrium has been found, set ierr=0 and return
    // if too many attempts have been performed, set ierr=1
    ierr =  (n_iter <= n_max_iter) ? 0 : 1;
    
    return;
  }

  
  // func_eq_weak
  //     Input:
  //
  //     rho ... density                                  [g/cm^3]
  //     u   ... total (fluid+radiation) internal energy  [erg/cm^3]
  //     yl  ... lepton number                            [#/baryon]
  //     x   ...  array with the temperature and ye
  //        x(1) ... T                                  [MeV]
  //        x(2) ... ye                                 [#/baryon]
  //
  //     Output:
  //
  //     y ... array with the function whose zeros we are searching for
  //
  void BNSNuRates::func_eq_weak(Real rho, Real u, Real yl, Real x[2], Real y[2]) {
    
    // Compute the baryon number density
    Real nb = rho / atomic_mass; // [#/cm^3]
    
    // Interpolate the chemical potentials (stored in MeV in the table)
    Real mu_n, mu_p, mu_e;
    ChemicalPotentials_npe_cgs(rho, x[0], x[1],  mu_n, mu_p, mu_e );
    Real mus[2] = {mu_e, mu_n - mu_p};
    
    // Call the EOS
    Real Y[1] = {x[1]};
    Real e = pmy_block->peos->GetEOS().GetEnergy(rho/atomic_mass
                                                 * my_units->NumberDensityConversion(*code_units),
                                                 x[0]
                                                 * my_units->TemperatureConversion(*code_units),
                                                 Y)
      * code_units->EnergyDensityConversion(*my_units); // [g/cm^3]
    
    // compute the neutrino degeneracy paramater at equilibrium..........
    Real eta_vec[2] = {0.0};
    nu_deg_param_trap(x[0],mus,eta_vec);
    Real eta = eta_vec[0]; // [-]
    Real eta2 = eta*eta;   // [-]
    
    // compute the function..............................................
    Real t3 = x[0]*x[0]*x[0];
    Real t4 = t3*x[0];
    y[0] = x[1] + pref1*t3*eta*(pi2 + eta2)/nb - yl;
    y[1] = (e+pref2*t4*((cnst5+0.5*eta2*(pi2+0.5*eta2))+cnst6))/u - 1.0;
    
    return;
  }

  
  // error_func_eq_weak
  //
  //     Input:
  //
  //     yl ... lepton number                            [#/baryon]
  //     u  ... total (fluid+radiation) internal energy  [erg/cm^3]
  //     y  ... array with residuals                     [-]
  //
  //     Output:
  //
  //     err ... error associated with the residuals     [-]
  //
  //
  // since the first equation is has yl as constant, we normalized the error to it.
  // since the second equation was normalized wrt u, we divide it by 1.
  // the modulus of the two contributions are then summed
  void BNSNuRates::error_func_eq_weak(Real yl, Real u, Real y[2], Real &err) {
    err = abs(y[0]/yl) + abs(y[1]/1.0);
    return;
  }


  // jacobi_eq_weak
  //
  //     Input:
  //
  //     rho ... density                   [g/cm^3]
  //     u   ... total energy density      [erg/cm^3]
  //     yl  ... lepton fraction           [#/baryon]
  //     x   ... array with T and ye
  //        x(1) ... T                     [MeV]
  //        x(2) ... ye                    [#/baryon]
  //
  //     Output:
  //
  //     J ... Jacobian for the 2D Newton-Raphson
  void BNSNuRates::jacobi_eq_weak(Real rho, Real u, Real yl, Real x[2], Real J[2][2], int &ierr)
  {
    // Interpolate the chemical potentials (stored in MeV in the table)
    Real t = x[0];
    Real ye = x[1];

    Real mu_n, mu_p, mu_e;
    ChemicalPotentials_npe_cgs(rho, t, ye,  mu_n, mu_p, mu_e);
    Real mus[2] = {0.0};
    mus[0] = mu_e;        // electron chemical potential (w rest mass) [MeV]
    mus[1] = mu_n - mu_p; // n minus p chemical potential (w rest mass) [MeV]
    
    Real eta_vec[3] = {0.0};
    
    // compute the degeneracy parameters
    nu_deg_param_trap(t,mus,eta_vec);
    Real eta = eta_vec[0]; // electron neutrinos degeneracy parameter
    Real eta2 = eta*eta;
    
    // compute the gradients of eta and of the internal energy
    Real detadt,detadye,dedt,dedye;
    eta_e_gradient(rho,t,ye,eta,detadt,detadye,dedt,dedye,ierr);
    if (ierr != 0) {
      return;
    } 
    
    // Compute the baryon number density
    Real nb = rho / atomic_mass; // [#/cm^3]
    
    // compute the Jacobian 
    //     J[0,0]: df1/dt
    //     J[0,1]: df1/dye
    //     J[1,0]: df2/dt
    //     J[1,1]: df2/dye
    
    Real t2 = t*t;
    Real t3 = t2*t;
    Real t4 = t3*t;
    J[0][0] = pref1/nb*t2*(3.e0*eta*(pi2+eta2)+t*(pi2+3.e0*eta2)*detadt);
    J[0][1] = 1.e0+pref1/nb*t3*(pi2+3.e0*eta2)*detadye;
    
    J[1][0] = (dedt+pref2*t3*(cnst3+cnst4+2.e0*eta2*(pi2+0.5*eta2)+eta*t*(pi2+eta2)*detadt))/u;
    J[1][1] = (dedye+pref2*t4*eta*(pi2+eta2)*detadye)/u;
    
    // check on the degeneracy parameters and temperature
    if (isnan(eta)) {
      ierr = 1;
      return;
    } 
    
    if (isnan(detadt)) {
      ierr = 1;
      return;
    } 
    
    if (isnan(t)) {
      ierr = 1;
      return;
    }
    
    ierr = 0;
    return;
  } 

  
  // eta_e_gradient
  //
  //     this subroutine computes the gradient of the degeneracy parameter
  //     and of the fluid internal energy with respect to temperature and ye
  //
  //     Input:
  //     rho ... density             [g/cm^3]
  //     t   ... temperature         [MeV]
  //     ye  ... electron fraction   [#/baryon]
  //     eta ... electron neutrino degeneracy parameter at equilibrium [-]
  //
  //     Output:
  //     detadt  ... derivative of eta wrt T (for ye and rho fixed)             [1/MeV]
  //     detadye ... derivative of eta wrt ye (for T and rho fixed)             [-]
  //     dedt  ... derivative of internal energy wrt T (for ye and rho fixed) [erg/cm^3/MeV]
  //     dedye ... derivative of internal energy wrt ye (for T and rho fixed) [erg/cm^3]
  //
  void BNSNuRates::eta_e_gradient(Real rho, Real t, Real ye, Real eta,
                                  Real& detadt, Real& detadye, Real& dedt, Real& dedye, int& ierr)
  {
    
    // gradients are computed numerically. To do it, we consider small
    // variations in ye and temperature, and we compute the detivative
    // using finite differencing. The real limitation is that this way
    // relies on the EOS table interpolation procedure
    //
    // the goal of this part is to obtain chemical potentials (mus1 and
    // mus2) in two points close to the point we are considering, first
    // varying wrt ye and then wrt T
    // vary the electron fraction............................................
    
    // these are the two calls to the EOS. The goal here is to get
    // the fluid internal energy and the chemical potential for
    // electrons and the difference between neutron and proton
    // chemical potential (usually called mu_hat) for two points with
    // slightly different ye

    Real nb = rho/atomic_mass;
    Real mu_n, mu_p, mu_e;
    
    //  first, for ye slightly smaller
    Real ye1 = std::max(ye - delta_ye, eos_ye_min);
    Real yev = ye1;
    
    ChemicalPotentials_npe_cgs(rho, t, yev,  mu_n, mu_p, mu_e);
    Real mus1[2] = {mu_e, mu_n - mu_p};

    Real Y[1] = {yev};
    Real e1 = pmy_block->peos->GetEOS().GetEnergy(nb
                                                  * my_units->NumberDensityConversion(*code_units),
                                                  t
                                                  * my_units->TemperatureConversion(*code_units),
                                                  Y)
      * code_units->EnergyDensityConversion(*my_units); // [g/cm^3] 
    
    // second, for ye slightly larger
    Real ye2 = std::min(ye + delta_ye, eos_ye_max);
    yev = ye2;
    
    ChemicalPotentials_npe_cgs(rho, t, yev,  mu_n, mu_p, mu_e);
    Real mus2[2] = {mu_e, mu_n - mu_p};

    Y[1] = {yev};
    Real e2 = pmy_block->peos->GetEOS().GetEnergy(nb
                                                  * my_units->NumberDensityConversion(*code_units),
                                                  t
                                                  * my_units->TemperatureConversion(*code_units),
                                                  Y)
      * code_units->EnergyDensityConversion(*my_units); // [g/cm^3] 
    
    // compute numerical derivaties.........................................
    Real dmuedye   = (mus2[0]-mus1[0])/(ye2 - ye1);
    Real dmuhatdye = (mus2[1]-mus1[1])/(ye2 - ye1);
    dedye          = (e2-e1)/(ye2 - ye1);
    
    // vary the temperature.................................................
    Real t1 = std::max(t - delta_t, eos_temp_min);
    Real t2 = std::min(t + delta_t, eos_temp_max);
    
    // these are the two other calls to the EOS. The goal here is to get
    // the fluid internal energy and the chemical potential for
    // electrons and the difference between neutron and proton
    // chemical potential (usually called mu_hat) for two points with
    // slightly different t
    
    // ye is the original one
    
    // first, for t slightly smaller
    Real tv = t1;    

    ChemicalPotentials_npe_cgs(rho, tv, ye,  mu_n, mu_p, mu_e );
    mus1[0] = mu_e;
    mus1[1] = mu_n - mu_p;

    Y[0] = ye;
    e1 = pmy_block->peos->GetEOS().GetEnergy(nb 
                                             * my_units->NumberDensityConversion(*code_units),
                                             t1
                                             * my_units->TemperatureConversion(*code_units),
                                             Y)
      * code_units->EnergyDensityConversion(*my_units); // [g/cm^3] 
    
    // second, for t slightly larger
    tv = t2;
    
    ChemicalPotentials_npe_cgs(rho, tv, ye,  mu_n, mu_p, mu_e);
    mus2[0] = mu_e;
    mus2[1] = mu_n - mu_p;
 
    Y[0] = ye;
    e2 = pmy_block->peos->GetEOS().GetEnergy(nb
                                             * my_units->NumberDensityConversion(*code_units),
                                             tv
                                             * my_units->TemperatureConversion(*code_units),
                                             Y)
      * code_units->EnergyDensityConversion(*my_units); // [g/cm^3] 
    
    // compute the derivatives wrt temperature..............................
    Real dmuedt   = (mus2[0] - mus1[0])/(t2 - t1);
    Real dmuhatdt = (mus2[1] - mus1[1])/(t2 - t1);
    dedt          = (e2   - e1  )/(t2 - t1);
    
    // combine the eta derivatives..........................................
    detadt  = (-eta + dmuedt - dmuhatdt)/t; // [1/MeV]
    detadye = (dmuedye - dmuhatdye)/t;      // [-]
    
    // check if the derivative has a problem................................
    ierr = isnan(detadt) ? 1 : 0 ;
    return;
  }
    
  // inv_jacobi
  //
  //     This subroutine inverts the Jacobian matrix, assuming it to be a
  //     2x2 matrix
  //
  //
  //     Input:
  //     det ... determinant of the Jacobian matrix
  //     J   ... Jacobian matrix
  //
  //     Output:
  //     invJ ... inverse of the Jacobian matrix
  void BNSNuRates::inv_jacobi(Real det, Real J[2][2], Real invJ[2][2]) {
    Real inv_det = 1.0/det;
    invJ[0][0] =  J[1][1]*inv_det;
    invJ[1][1] =  J[0][0]*inv_det;
    invJ[0][1] = -J[0][1]*inv_det;
    invJ[1][0] = -J[1][0]*inv_det;
  }

  
  // nu_deg_param_trap
  //
  //     In this subroutine, we compute the neutrino degeneracy parameters
  //     assuming weak and thermal equilibrium, i.e. using as input the
  //     local thermodynamical properties
  //
  //     Input:
  //     temp_m   ----> local matter temperature [MeV]
  //     chem_pot ----> matter chemical potential [MeV]
  //                    chem_pot(1): electron chemical potential (w rest mass)
  //                    chem_pot(2): n minus p chemical potential (w/o rest mass)
  //
  //     Output:
  //     eta      ----> neutrino degeneracy parameters [-]
  //                    eta(1): electron neutrino
  //                    eta(2): electron antineutrino
  //                    eta(3): mu and tau neutrinos
  //
  void BNSNuRates::nu_deg_param_trap(Real temp_m, Real chem_pot[2], Real eta[3])
  {
    if (temp_m>0.0) {
      eta[0] = (chem_pot[0] - chem_pot[1])/temp_m; // [-]
      eta[1] = - eta[0];                       // [-]
      eta[2] = 0.0;                            // [-]
    } else {
      eta[0] = 0.0; // [-]
      eta[1] = 0.0; // [-]
      eta[2] = 0.0; // [-]
    } 
  }

  
  // dens_nu_trap
  //     In this subroutine, we compute the neutrino densities in equilibrium
  //     conditions, using as input the local thermodynamical properties
  //
  //     Input:
  //     temp_m   ----> local matter temperature [MeV]
  //     eta_nu   ----> neutrino degeneracy parameter [-]
  //
  //     Output:
  //     nu_dens ----> neutrino density [particles/cm^3]
  //
  void BNSNuRates::dens_nu_trap(Real temp_m, Real eta_nu[3], Real nu_dens[3])
    {
      const Real pref=4.0*pi/(hc_mevcm*hc_mevcm*hc_mevcm); // [1/MeV^3/cm^3]
      Real temp_m3 = (temp_m*temp_m*temp_m);               // [MeV^3]
      
      for (int it=0; it<3; it++) {
#if (FERMI_ANALYTIC)
        Real f2 = Fermi::fermi2(eta_nu[it]);
#else
        Real f2 = 0.0;
        Fermi::fermiint(2.0,eta_nu[it],f2);
#endif
        nu_dens[it] = pref * temp_m3 * f2; // [#/cm^3]
      } 
    }
  
  
  // edens_nu_trap
  //     In this subroutine, we compute the neutrino densities in equilibrium
  //     conditions, using as input the local thermodynamical properties
  //
  //     Input:
  //     temp_m   ----> local matter temperature [MeV]
  //     eta_nu   ----> neutrino degeneracy parameter [-]
  //
  //     Output:
  //     enu_dens ----> neutrino density [MeV/cm^3]
  void BNSNuRates::edens_nu_trap(Real temp_m, Real eta_nu[3], Real enu_dens[3])
  {
    
    const Real pref = 4.0*pi/(hc_mevcm*hc_mevcm*hc_mevcm); // [1/MeV^3/cm^3]
    Real temp_m4 = temp_m*temp_m*temp_m*temp_m;          // [MeV^3]
    
    for (int it=0; it<3; it++) {
#if (FERMI_ANALYTIC)
      Real f3 = Fermi::fermi3(eta_nu[it]);
#else
      Real f3 = 0.0;
      Fermi::fermiint(3.0,eta_nu[it],f3);
#endif
      enu_dens[it] = pref * temp_m4 * f3;
    } 
  } 

  
} // namespace M1::Opacities::BNSNuRates

