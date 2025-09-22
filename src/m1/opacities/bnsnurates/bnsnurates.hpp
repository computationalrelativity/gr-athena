#ifndef M1_OPACITIES_BNSNURATES_HPP
#define M1_OPACITIES_BNSNURATES_HPP

#include <limits>

// Athena++ classes headers
#include "../../../athena.hpp"
#include "../../../eos/eos.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../m1.hpp"

#include "units.hpp"
#include "fermi.hpp"

// bns_nurates headers from:
// https://github.com/RelNucAs/bns_nurates
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"


namespace M1::Opacities::BNSNuRates {
  
  struct NuratesParams {
    Real rho_min_cgs;
    Real temp_min_mev;
    Real dm_eff;
    Real dU;
    
    bool use_abs_em;
    bool use_pair;
    bool use_brem;
    bool use_iso;
    bool use_inelastic_scatt;
    bool use_WM_ab;
    bool use_WM_sc;
    bool use_dU;
    bool use_dm_eff;
    bool use_equilibrium_distribution;
    bool use_NN_medium_corr;
    bool neglect_blocking;
    bool use_decay;
    bool use_BRT_brem;

    // no. of quadrature points in bns_nurates
    int quad_nx_1;  // beta_nucleon_scat
    int quad_nx_2;  // pair_bremsstrahlung_lepton_scat
    
    MyQuadrature quadrature_1;
    MyQuadrature quadrature_2;
   };
  
  class BNSNuRates {

    friend class ::M1::Opacities::Opacities; 
    
  public:
    
    BNSNuRates(MeshBlock *pmb, M1 * pm1, ParameterInput *pin) :
      pm1(pm1),
      pmy_mesh(pmb->pmy_mesh),
      pmy_block(pmb),
      pmy_coord(pmy_block->pcoord),
      N_GRPS(pm1->N_GRPS),
      N_SPCS(pm1->N_SPCS)
    {

#if !(USETM)
#pragma omp critical
      {
        std::cout << "Warning: ";
        std::cout << "M1::Opacities::BNSNuRates needs TEOS to work properly \n";
      }
#endif
    
#if !(NSCALARS>0)
#pragma omp critical
      {
        std::cout << "Warning: ";
        std::cout << "M1::Opacities::BNSNuRates needs NSCALARS>0 to work function \n";
      }
#endif

      // Set the units
      // my_units ......... CGS + MeV
      // nurates_units .... CGS + MeV + nm
      // code_units ....... Geometric + Solar masses
      my_units = &BNSNuRates_Units::WeakRatesUnits;
      nurates_units = &BNSNuRates_Units::NGS;
      code_units = &BNSNuRates_Units::GeometricSolar;
      
      // BNSNuRates only works for nu_e + nu_ae + nu_x
      assert(N_SPCS==3);
      
      // BNSNuRates only works for 1 group
      assert(N_GRPS==1);

      // Parameters for bns_nurates
      // Defaults should mimic WeakRates
      nurates_params.quad_nx_1 = pin->GetOrAddInteger("bns_nurates", "n_quad_points_beta_nucleon_scat", 6);
      nurates_params.quad_nx_2 = pin->GetOrAddInteger("bns_nurates", "n_quad_points_pair_bremsstrahlung_lepton_scat", -1);

      nurates_params.use_abs_em = pin->GetOrAddBoolean("bns_nurates", "use_abs_em", true); // semilept charge-current proc.
      nurates_params.use_pair = pin->GetOrAddBoolean("bns_nurates", "use_pair", true); // ep annihil.
      nurates_params.use_brem = pin->GetOrAddBoolean("bns_nurates", "use_brem", true); // 
      nurates_params.use_iso = pin->GetOrAddBoolean("bns_nurates", "use_iso", true); // isoenergetics scattering (nucleons)
      nurates_params.use_inelastic_scatt = pin->GetOrAddBoolean("bns_nurates", "use_inelastic_scatt", false); // e,p
      nurates_params.use_WM_ab = pin->GetOrAddBoolean("bns_nurates", "use_WM_ab", false); // weak magnetism charge currents
      nurates_params.use_WM_sc = pin->GetOrAddBoolean("bns_nurates", "use_WM_sc", false); // weak magnetism neutral currents
      nurates_params.use_NN_medium_corr = pin->GetOrAddBoolean("bns_nurates", "use_NN_medium_corr", false); // correction to bremstralhung
      nurates_params.neglect_blocking = pin->GetOrAddBoolean("bns_nurates", "neglect_blocking", false);
      nurates_params.use_decay = pin->GetOrAddBoolean("bns_nurates", "use_decay", false); // inverse beta
      nurates_params.use_BRT_brem = pin->GetOrAddBoolean("bns_nurates", "use_BRT_brem", false); // alternative bremstralung
     
      nurates_params.use_dU = pin->GetOrAddBoolean("bns_nurates", "use_dU", false); //TODO Not implemented!
      nurates_params.dU = pin->GetOrAddReal("bns_nurates", "dU", 0.0); //TODO Set effective potential difference (in MeV) from EOS

      nurates_params.use_dm_eff = pin->GetOrAddBoolean("bns_nurates", "use_dm_eff", false);
      nurates_params.dm_eff = pin->GetOrAddReal("bns_nurates", "effective_mass_diff", 0.0); //TODO Set effective mass difference (in MeV) from EOS 

      nurates_params.rho_min_cgs = pin->GetOrAddReal("bns_nurates", "rho_min_cgs", 0.); 
      nurates_params.temp_min_mev = pin->GetOrAddReal("bns_nurates", "temp_min_mev", 0.); 

      nurates_params.use_equilibrium_distribution = pin->GetOrAddBoolean("bns_nurates", "use_equilibrium_distribution", true); 

      // Set quadratures
      nurates_params.quadrature_1.nx = nurates_params.quad_nx_1;
      nurates_params.quadrature_1.dim = 1;
      nurates_params.quadrature_1.type = kGauleg;
      nurates_params.quadrature_1.x1 = 0.;
      nurates_params.quadrature_1.x2 = 1.;
      GaussLegendre(&nurates_params.quadrature_1);
     
      if (nurates_params.quad_nx_2 == -1) {
	nurates_params.quadrature_2 = nurates_params.quadrature_1;
      } else {
	nurates_params.quadrature_2.nx = nurates_params.quad_nx_2;
	nurates_params.quadrature_2.dim = 1;
	nurates_params.quadrature_2.type = kGauleg;
	nurates_params.quadrature_2.x1 = 0.;
	nurates_params.quadrature_2.x2 = 1.;
	GaussLegendre(&nurates_params.quadrature_2);
      }
      
      // Weak equilibrium parameters ----------------------------------------------

      verbose_warn_weak = pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true);

      // This parameter is actually used in the equilibration 
      use_kirchhoff_law = pin->GetOrAddBoolean("M1_opacities", "use_kirchoff_law", true);
      
      // These are the defaults in THC
      opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
      opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
      opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 3.0);

      // These min values are used at the begining of compute_weak_equilibrium()
      // Equilibration logic should do the job, may be not needed
      // Alternatively, could experiment with rho(CGS) ~ 1e11 in case.      
      // density below which nothing is done [g/cm^3]
      rho_min = pin->GetOrAddReal("M1_opacities", "equilibration_rho_min_cgs", 0.0); 
      // temperature below which nothing is done [MeV]
      temp_min = pin->GetOrAddReal("M1_opacities", "equilibration_temp_min_mev", 0.0);

      // EOS limits
      // NB These values need to be in my_units (CGS+MeV) 
      Real infty = std::numeric_limits<Real>::infinity();

      // Set maximal ranges (default)
      eos_rho_min = pin->GetOrAddReal("M1_opacities", "eos_rho_min_cgs", 0.0);
      eos_rho_max = pin->GetOrAddReal("M1_opacities", "eos_rho_max_cgs", infty);
      eos_temp_min = pin->GetOrAddReal("M1_opacities", "eos_temp_min_mev", 0.0);
      eos_temp_max = pin->GetOrAddReal("M1_opacities", "eos_temp_max_mev", infty);
      eos_ye_min = pin->GetOrAddReal("M1_opacities", "eos_ye_min", 0.0);
      eos_ye_max = pin->GetOrAddReal("M1_opacities", "eos_ye_max", 1.0);
      
      // Option to override the EOS limits with the table limits 
      bool enforce_table_limits = pin->GetOrAddBoolean("M1_opacities", "eos_limits_from_table", false);

      if (enforce_table_limits) {
        eos_rho_min = pmy_block->peos->GetEOS().GetMinimumDensity()
          * code_units->MassDensityConversion(*my_units);
        eos_rho_max = pmy_block->peos->GetEOS().GetMaximumDensity()
          * code_units->MassDensityConversion(*my_units);
        eos_temp_min = pmy_block->peos->GetEOS().GetMinimumTemperature()
          * code_units->TemperatureConversion(*my_units);
        eos_temp_max = pmy_block->peos->GetEOS().GetMaximumTemperature()
          * code_units->TemperatureConversion(*my_units);
        eos_ye_min = pmy_block->peos->GetEOS().GetMinimumSpeciesFraction(0);
        eos_ye_max = pmy_block->peos->GetEOS().GetMaximumSpeciesFraction(0);
      }

      // Options to override rho_min and eos_temp_min with floors
      bool enforce_rho_floor = pin->GetOrAddBoolean("M1_opacities", "eos_rho_min_usefloor", false);
      bool enforce_temp_floor = pin->GetOrAddBoolean("M1_opacities", "eos_temp_min_usefloor", false);

      if (enforce_rho_floor)
        eos_rho_min = pmy_block->peos->GetDensityFloor()
          * code_units->MassDensityConversion(*my_units);

      if (enforce_temp_floor)
        eos_temp_min = pmy_block->peos->GetTemperatureFloor()
          * code_units->TemperatureConversion(*my_units);
      
      // mb [g] 
      atomic_mass = pmy_block->peos->GetEOS().GetRawBaryonMass()
        * code_units->MassConversion(*my_units); 

    };
    
    ~BNSNuRates() {
    };

    // Main opacity computation needed by M1
    // N.B.
    // In general it will be faster to slice a fixed
    // choice of ix_g, ix_s and then loop over k,j,i
    inline int CalculateOpacityBNSNuRates(Real const dt, AA &u)
    {
      
      // Hydro * phydro = pmy_block->phydro;
      // PassiveScalars * pscalars = pmy_block->pscalars;
      AT_C_sca &sc_oo_sqrt_det_g = pm1->geom.sc_oo_sqrt_det_g;
      
      const int NUM_COEFF = 3;
      int ierr[NUM_COEFF];

      const Real mb_code = pmy_block->peos->GetEOS().GetRawBaryonMass();
      
      M1_FLOOP3(k, j, i)
        if (pm1->MaskGet(k, j, i))
          {
            Real rho = pm1->hydro.sc_w_rho(k, j, i);
            Real press = pm1->hydro.sc_w_p(k, j, i);
            Real nb = rho / mb_code; // baryon num dens 
            Real T = pm1->hydro.sc_T(k,j,i);
            
            Real Y[MAX_SPECIES] = {0.0};
            Y[0] = pm1->hydro.sc_w_Ye(k, j, i);
            Real Y_e = Y[0];
       
            // Chem potentials (code units)
            Real mu_n, mu_p, mu_e;
            ChemicalPotentials_npe(nb, T, Y_e,  mu_n, mu_p, mu_e);
            
            // Local undensitized neutrino quantities
            Real invsdetg = sc_oo_sqrt_det_g(k, j, i);
            Real nudens_0[4] = {0.}; // NB we have 3 species
            Real nudens_1[4] = {0.};
            Real chi_loc[4] = {0.};
            
            for (int nuidx = 0; nuidx < N_SPCS; ++nuidx) {
              nudens_0[nuidx] = pm1->rad.sc_n(0, nuidx)(k, j, i) * invsdetg;
              nudens_1[nuidx] = pm1->rad.sc_J(0, nuidx)(k, j, i) * invsdetg;
              chi_loc[nuidx] = pm1->lab_aux.sc_chi(0, nuidx)(k, j, i); 
            }
	    // Copy 4th species (assume anux = nux)
	    nudens_0[3] = nudens_0[2];
	    nudens_1[3] = nudens_1[2];
	    chi_loc[3] = chi_loc[2];

            // Get emissivities and opacities
            Real eta_0_loc[4]{}, eta_1_loc[4]{};
            Real abs_0_loc[4]{}, abs_1_loc[4]{};
            Real scat_0_loc[4]{}, scat_1_loc[4]{};
	    
            // Note: everything sent and received are in code units
		
            int opac_err =
              bns_nurates_wrapper(nb, T, Y_e, mu_n, mu_p, mu_e,
				  nudens_0[0], nudens_1[0], chi_loc[0], 
				  nudens_0[1], nudens_1[1], chi_loc[1],
				  nudens_0[2], nudens_1[2], chi_loc[2],
				  nudens_0[3], nudens_1[3], chi_loc[3], 
				  eta_0_loc[0], eta_0_loc[1], eta_0_loc[2], eta_0_loc[3],
				  eta_1_loc[0], eta_1_loc[1], eta_1_loc[2], eta_1_loc[3],
				  abs_0_loc[0], abs_0_loc[1], abs_0_loc[2], abs_0_loc[3],
				  abs_1_loc[0], abs_1_loc[1], abs_1_loc[2], abs_1_loc[3],
				  scat_0_loc[0], scat_0_loc[1], scat_0_loc[2], scat_0_loc[3],
				  scat_1_loc[0], scat_1_loc[1], scat_1_loc[2], scat_1_loc[3],
				  nurates_params); 
	    
            bool is_failing_opacity = (opac_err)? true : false;
	    
            // Dump some information when opacity calculation fails
	    if (is_failing_opacity) {
		std::ostringstream msg;
                msg << "CalculateOpacityBNSNuRates failure: " << opac_err << std::endl;		
		//msg << "T-conv-fact " << code_units->TemperatureConversion(*my_units) << std::endl;
		msg << "nb " << nb << std::endl;
		msg << "T " << T << std::endl;
		msg << "Ye " << Y_e << std::endl;
		msg << "mun " << mu_n << std::endl;
		msg << "mup " << mu_p << std::endl;
		msg << "mue " << mu_e << std::endl;
		pm1->StatePrintPoint(msg.str(), 0, 0, k, j, i, false);
                pm1->StatePrintPoint(msg.str(), 0, 1, k, j, i, false);
                pm1->StatePrintPoint(msg.str(), 0, 2, k, j, i, true); // assert(false)
              }
            
            // Equilibrium logic ----------------------------------------------------
            // Calculate equilibrium blackbody functions with trapped neutrinos	    

            Real tau{};
            Real nudens_0_trap[4]{}, nudens_0_thin[4]{};
            Real nudens_1_thin[4]{}, nudens_1_trap[4]{};

            // --------------------------------------------------------------------
	    if (use_kirchhoff_law ||
		nurates_params.use_equilibrium_distribution) {

	      tau = std::min(std::sqrt(abs_1_loc[0] * (abs_1_loc[0] + scat_1_loc[0])),
			     std::sqrt(abs_1_loc[1] * (abs_1_loc[1] + scat_1_loc[1]))
			     ) * dt;
	      
	      if (opacity_tau_trap >= 0.0 && tau > opacity_tau_trap) {
		
                Real T_trap;
                Real Y_e_trap;
		
                // Calculate equilibrated state
                ierr[0] = WeakEquilibrium(rho, T, Y_e,
                                          nudens_0[0],
                                          nudens_0[1],
                                          nudens_0[2]+nudens_0[3], // (assume anux = nux)
                                          nudens_1[0],
                                          nudens_1[1],
                                          nudens_1[2]+nudens_1[3], // (assume anux = nux)
                                          T_trap, Y_e_trap,
                                          nudens_0_trap[0],
                                          nudens_0_trap[1],
                                          nudens_0_trap[2],
                                          nudens_1_trap[0],
                                          nudens_1_trap[1],
                                          nudens_1_trap[2]);
                
                // If we can't get equilibrium, try again but ignore current neutrino
                // data
                if (ierr[0]) {
		  ierr[1] = WeakEquilibrium(rho, T, Y_e,
					    0.0, 0.0, 0.0,
					    0.0, 0.0, 0.0,
					    T_trap, Y_e_trap,
					    nudens_0_trap[0],
					    nudens_0_trap[1],
					    nudens_0_trap[2],
					    nudens_1_trap[0],
					    nudens_1_trap[1],
					    nudens_1_trap[2]);
		  
		  //assert(!ierr[1]); // THC treats this as a warning
		  if (verbose_warn_weak) {
		    std::printf("M1: can't get equilibrium @ (i,j,k)=(%d,%d,%d) (%.3e,%.3e,%.3e)\n",
				i, j, k,
				pmy_block->pcoord->x1v(i),
				pmy_block->pcoord->x2v(j),
				pmy_block->pcoord->x3v(k));
		  }
		}

		Real mu_n_eq;
		Real mu_p_eq;
		Real mu_e_eq;
		ChemicalPotentials_npe(nb, T_trap, Y_e_trap,
				       mu_n_eq, mu_p_eq, mu_e_eq);
		
		NeutrinoDensity_ChemPot(nb, T_trap,
					mu_n_eq, mu_p_eq, mu_e_eq,
					nudens_0_trap[0], nudens_0_trap[1], nudens_0_trap[2],
					nudens_1_trap[0], nudens_1_trap[1], nudens_1_trap[2]);
		
                assert(isfinite(nudens_0_trap[0]));
                assert(isfinite(nudens_0_trap[1]));
                assert(isfinite(nudens_0_trap[2]));
                assert(isfinite(nudens_1_trap[0]));
                assert(isfinite(nudens_1_trap[1]));
                assert(isfinite(nudens_1_trap[2]));

		// Copy 4th species (assume anux = nux) and half 3rd species
		nudens_0_trap[2] *= 0.5;
		nudens_1_trap[2] *= 0.5;
		nudens_0_trap[3] = nudens_0_trap[2];
		nudens_1_trap[3] = nudens_1_trap[2];
                
              } // if (opacity_tau_trap ...
            
	      // Calculate equilibrium blackbody functions with fixed T, Ye
	      NeutrinoDensity_ChemPot(nb, T,
				      mu_n, mu_p, mu_e,
				      nudens_0_thin[0], nudens_0_thin[1], nudens_0_thin[2],
				      nudens_1_thin[0], nudens_1_thin[1], nudens_1_thin[2]);

	      // Copy 4th species (assume anux = nux) and half 3rd species
	      nudens_0_thin[2] *= 0.5;
	      nudens_1_thin[2] *= 0.5;
	      nudens_0_thin[3] = nudens_0_thin[2];
	      nudens_1_thin[3] = nudens_1_thin[2];

	    } //  if (use_kirchhoff_law || nurates_params.use_equilibrium_distribution)

	    // Store opacities and emissivities	      
	    for (int nuidx= 0; nuidx < N_SPCS; ++nuidx) {	      
	      pm1->radmat.sc_eta_0(0, nuidx)(k, j, i) = eta_0_loc[nuidx];
	      pm1->radmat.sc_eta(0, nuidx)(k, j, i) = eta_1_loc[nuidx];	      
	      pm1->radmat.sc_kap_a_0(0, nuidx)(k, j, i) = abs_0_loc[nuidx];
	      pm1->radmat.sc_kap_a(0, nuidx)(k, j, i) = abs_1_loc[nuidx];
	      pm1->radmat.sc_kap_s(0, nuidx)(k, j, i) = scat_1_loc[nuidx];
	    }
	    // Fix for 4th species
	    pm1->radmat.sc_eta_0(0, 2)(k, j, i) += eta_0_loc[3]; // sum nux and anux emissivities 
	    pm1->radmat.sc_eta(0, 2)(k, j, i) += eta_1_loc[3]; // sum nux and anux emissivities 
	    pm1->radmat.sc_kap_a_0(0, 2)(k, j, i) += abs_0_loc[3]; // avg nux and anux absorbsivities
	    pm1->radmat.sc_kap_a_0(0, 2)(k, j, i) *= 0.5;
	    pm1->radmat.sc_kap_a(0, 2)(k, j, i) += abs_1_loc[3]; // avg nux and anux absorbsivities
	    pm1->radmat.sc_kap_a(0, 2)(k, j, i) *= 0.5;
	    pm1->radmat.sc_kap_s(0, 2)(k, j, i) += scat_1_loc[3]; // avg nux and anux abs scattering
	    pm1->radmat.sc_kap_s(0, 2)(k, j, i) *= 0.5;

	    for (int nuidx= 0; nuidx < N_SPCS; ++nuidx) {
	    
	      Real my_nudens_0{}, my_nudens_1{};
	      
	      if (use_kirchhoff_law || nurates_params.use_equilibrium_distribution) {
		// Combine optically thin and optically thick limits
		if (opacity_tau_trap < 0 ||
		    tau <= opacity_tau_trap) {
		  my_nudens_0 = nudens_0_thin[nuidx];
		  my_nudens_1 = nudens_1_thin[nuidx];
		} else if (tau > opacity_tau_trap +
			   opacity_tau_delta) {
		  my_nudens_0 = nudens_0_trap[nuidx];
		  my_nudens_1 = nudens_1_trap[nuidx];
		} else {
		  Real const lam = (tau - opacity_tau_trap) /
		    opacity_tau_delta;
		  my_nudens_0 = lam * nudens_0_trap[nuidx] +
		    (1 - lam) * nudens_0_thin[nuidx];
		  my_nudens_1 = lam * nudens_1_trap[nuidx] +
		    (1 - lam) * nudens_1_thin[nuidx];
		}		
	      }

	      // Correction factor for absorption opacities for non-LTE effects
	      // (kappa ~ E_nu^2)
	      Real corr_fac = 1.0;
	      if (nurates_params.use_equilibrium_distribution) {
		Real nu_e_avg = my_nudens_1 / my_nudens_0;
		pm1->radmat.sc_avg_nrg(0, nuidx)(k, j, i) = nu_e_avg;
		corr_fac = pm1->rad.sc_J(0, nuidx)(k, j, i) /
                  (pm1->rad.sc_n(0, nuidx)(k, j, i) * nu_e_avg);
		if (!std::isfinite(corr_fac)) {
                  corr_fac = 1.0;
                }
		corr_fac *= corr_fac;
		corr_fac = std::max(1.0 / opacity_corr_fac_max,
				    std::min(corr_fac, opacity_corr_fac_max));
	      }
	      
	      pm1->radmat.sc_kap_s(0, nuidx)(k, j, i) *= corr_fac;

	      if (use_kirchhoff_law) {
		// Enforce Kirchhoff's law
		// For electron lepton neutrinos we change the emissivity
		// For heavy lepton neutrinos we change the opacity
		// NB bns_nurates does not need to impose Kirchoff law
		if (nuidx == 0 || nuidx == 1) {
		  pm1->radmat.sc_kap_a_0(0, nuidx)(k, j, i) *= corr_fac;
		  pm1->radmat.sc_kap_a(0, nuidx)(k, j, i) *= corr_fac;
		  pm1->radmat.sc_eta_0(0, nuidx)(k, j, i) =
		    pm1->radmat.sc_kap_a_0(0, nuidx)(k, j, i) * my_nudens_0;
		  pm1->radmat.sc_eta(0, nuidx)(k, j, i) =
		    pm1->radmat.sc_kap_a(0, nuidx)(k, j, i) * my_nudens_1;
		} else {
		  pm1->radmat.sc_kap_a_0(0, nuidx)(k, j, i) =
                    (my_nudens_0 > pm1->opt.fl_nG //TODO Is this the floor for N?
                         ? pm1->radmat.sc_eta_0(0, nuidx)(k, j, i) / my_nudens_0
                         : 0);
		  pm1->radmat.sc_kap_a(0, nuidx)(k, j, i) =
                    (my_nudens_1 >  pm1->opt.fl_E
                         ? pm1->radmat.sc_eta(0, nuidx)(k, j, i) / my_nudens_1
                         : 0);
		}	
	      } else {
		pm1->radmat.sc_kap_a_0(0, nuidx)(k, j, i) *= corr_fac;
		pm1->radmat.sc_kap_a(0, nuidx)(k, j, i) *= corr_fac;
		pm1->radmat.sc_eta_0(0, nuidx)(k, j, i) *= corr_fac;
		pm1->radmat.sc_eta(0, nuidx)(k, j, i) *= corr_fac;
	      }
	      
	    } // nuidx
	  } // Mask      
      return 0;
    };
    
  private:
    
    M1 *pm1;
    Mesh *pmy_mesh;
    MeshBlock *pmy_block;
    Coordinates *pmy_coord;
    
    const int N_GRPS;
    const int N_SPCS;

    BNSNuRates_Units::UnitSystem* my_units;
    BNSNuRates_Units::UnitSystem* nurates_units;
    BNSNuRates_Units::UnitSystem* code_units;
    
    // pars for nurates (choice of reactions, quadratures)
    NuratesParams nurates_params;  
        
    // Main driver for bns_nurates
    // (input & output in code units)
    int bns_nurates_wrapper(Real &nb, Real &temp, Real &ye,
			    Real &mu_n, Real &mu_p, Real &mu_e,
			    Real &n_nue, Real &j_nue, Real &chi_nue,
			    Real &n_anue, Real &j_anue, Real &chi_anue,
			    Real &n_nux, Real &j_nux, Real &chi_nux,
			    Real &n_anux, Real &j_anux, Real &chi_anux,
			    Real &R_nue, Real &R_anue, Real &R_nux, Real &R_anux,
			    Real &Q_nue, Real &Q_anue, Real &Q_nux, Real &Q_anux,
			    Real &sigma_0_nue, Real &sigma_0_anue, Real &sigma_0_nux,
			    Real &sigma_0_anux, Real &sigma_1_nue, Real &sigma_1_anue,
			    Real &sigma_1_nux, Real &sigma_1_anux, Real &scat_0_nue,
			    Real &scat_0_anue, Real &scat_0_nux, Real &scat_0_anux,
			    Real &scat_1_nue, Real &scat_1_anue, Real &scat_1_nux, Real &scat_1_anux,
			    const NuratesParams nurates_params);
    
    // Weak equilibrium stuff -----------------------------------------------------------------

    bool use_kirchhoff_law;
    
    // Options for controlling weakrates opacities
    Real opacity_tau_trap;
    Real opacity_tau_delta;
    Real opacity_corr_fac_max;
   
    bool verbose_warn_weak;
    
    Real rho_min;                // density below which nothing is done [g/cm^3]
    Real temp_min;               // temperature below which nothing is done [MeV]
    Real atomic_mass;            // atomic mass [g] (to convert mass density to number density)

    // EOS limits
    Real eos_rho_min; 
    Real eos_rho_max;
    Real eos_temp_min;
    Real eos_temp_max;
    Real eos_ye_min;
    Real eos_ye_max;

    // Some parameters later used in the calculations.
    const Real eps_lim = 1.e-7;  // standard tollerance in 2D NR
    const int n_cut_max = 8;     // number of bisections of dx
    const int n_max_iter = 100;  // Newton-Raphson max number of iterations
    const int n_at = 16;  // number of independent initial guesses

    // deltas to compute numerical derivatives in the EOS tables
    const Real delta_ye = 0.005;
    const Real delta_t  = 0.01;

    // Some constants
    const Real pi = M_PI;                      // 3.14159265358979323846;
    const Real pi2 = SQR(pi);                          
    const Real pi4 = POW4(pi);                 
    const Real mev_to_erg = 1.60217733e-6;     // conversion from MeV to erg
    const Real hc_mevcm = 1.23984172e-10;      // hc in units of MeV*cm
    const Real oo_hc_mevcm3 = 1.0/POW3(hc_mevcm);

    const Real pref1 = 4.0/3.0 * pi * oo_hc_mevcm3;        // 4/3 *pi/(hc)**3 [MeV^3/cm^3]
    const Real pref2 = 4.0 * pi*mev_to_erg * oo_hc_mevcm3; // 4*pi/(hc)**3 [erg/MeV^4/cm^3]
    const Real cnst1 = 7.0 * pi4 /20.0;                // 7*pi**4/20 [-]
    const Real cnst5 = 7.0 * pi4 /60.0;                // 7*pi**4/60 [-]
    const Real cnst6 = 7.0 * pi4 /30.0;                // 7*pi**4/30 [-]
    const Real cnst2 = 7.0 * pi4 /5.0;                 // 7*pi**4/5 [-]
    const Real cnst3 = 7.0 * pi4 /15.0;                // 7*pi**4/15 [-]
    const Real cnst4 = 14.0 *pi4 /15.0;                // 14*pi**4/15 [-]


    // Chem potentials calculation (input & output in code units)
    void ChemicalPotentials_npe(Real nb, Real T, Real Ye,
                                Real &mu_n, Real &mu_p, Real &mu_e);

    // Wrapper for chem potential calculation in CGS + MeV
    void ChemicalPotentials_npe_cgs(Real rho, Real temp, Real Ye,
				    Real &mu_n, Real &mu_p, Real &mu_e);
    
    // Computes the neutrino number and energy density at equilibrium
    // Implements the computation, given the chemical potentials
    // (input & output in code units)
    void NeutrinoDensity_ChemPot(Real nb, Real temp,
				 Real mu_n, Real mu_p, Real mu_e,
				 Real &n_nue, Real &n_anue, Real &n_nux,
				 Real &en_nue, Real &en_anue, Real &en_nux);
    
    // Computes the neutrino number and energy density at equilibrium
    // This is needed by M1 in this form, all I/O in code units
    int NeutrinoDensity(Real rho, Real T, Real Y_e, 
			Real &n_nue, Real &n_anue, Real &n_nux,
			Real &e_nue, Real &e_anue, Real &e_nux);
    
    // Wrapper for weak equilibrium computation
    int WeakEquilibrium(Real rho, Real temp, Real ye,
                        Real n_nue, Real n_nua, Real n_nux,
                        Real e_nue, Real e_nua, Real e_nux,
                        Real& temp_eq, Real& ye_eq,
                        Real& n_nue_eq, Real& n_nua_eq, Real& n_nux_eq,
                        Real& e_nue_eq, Real& e_nua_eq, Real& e_nux_eq);
    
    int compute_weak_equilibrium(Real rho,        // [g/cm^3]
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
                                 );

    void weak_equil_wnu(Real rho, Real T, Real y_in[4], Real e_in[4],
                        Real& T_eq, Real y_eq[4], Real e_eq[4], int& na, int& ierr);
    bool apply_eos_limits(Real& rho, Real& temp, Real& ye);
    void new_raph_2dim(Real rho, Real u, Real yl, Real x0[2],
                       Real x1[2], int& ierr);
    void func_eq_weak(Real rho, Real u, Real yl, Real x[2], Real y[2]);
    void error_func_eq_weak(Real yl, Real u, Real y[2], Real &err);
    void jacobi_eq_weak(Real rho, Real u, Real yl, Real x[2], Real J[2][2], int &ierr);
    void eta_e_gradient(Real rho, Real t, Real ye, Real eta,
                        Real& detadt, Real& detadye, Real& dedt, Real& dedye, int& ierr);
    void inv_jacobi(Real det, Real J[2][2], Real invJ[2][2]);
    void nu_deg_param_trap(Real temp_m, Real chem_pot[2], Real eta[3]);
    void dens_nu_trap(Real temp_m, Real eta_nu[3], Real nu_dens[3]);
    void edens_nu_trap(Real temp_m, Real eta_nu[3], Real enu_dens[3]);
    
  };

} // namespace M1::Opacities::BNSNuRates

#endif //M1_OPACITIES_BNSNURATES_HPP
