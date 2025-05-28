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

// bns_nurates headers
// https://github.com/RelNucAs/bns_nurates
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"

namespace M1::Opacities::BNSNuRates {

  struct NuratesParams {
    int nurates_quad_nx;     // no. of quadrature points for 1d integration (bns_nurates)
    int nurates_quad_ny;     // no. of quadrature points for 2d integration (bns_nurates)
    Real opacity_tau_trap;   // incl. effects of neutrino trapping above this optical depth
    Real opacity_tau_delta;  // range of optical depths over which trapping is introduced
    Real opacity_corr_fac_max;  // maximum correction factor for optically thin regime
    Real rho_min_cgs;
    Real temp_min_mev;
    
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
    bool use_kirchhoff_law;
    
    MyQuadrature my_quadrature_1d;
    MyQuadrature my_quadrature_2d;
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
      // my_units ...... CGS + MeV                 -> used by BNSNuRates
      // code_units .... Geometric + SOlar masses  -> used by Z4c, GRMHD 
      my_units   = &BNSNuRates_Units::WeakRatesUnits;
      code_units = &BNSNuRates_Units::GeometricSolar;
      
      // BNSNuRates only works for nu_e + nu_ae + nu_x
      assert(N_SPCS==3);
      
      // BNSNuRates only works for 1 group
      assert(N_GRPS==1);
      
      // Parameters for bns_nurates
      nurates_params.nurates_quad_nx = pin->GetOrAddInteger("bns_nurates", "nurates_quad_nx", 10);
      nurates_params.nurates_quad_ny = pin->GetOrAddInteger("bns_nurates", "nurates_quad_ny", 10);
      nurates_params.opacity_tau_trap = pin->GetOrAddReal("bns_nurates", "opacity_tau_trap", 1.0);
      nurates_params.opacity_tau_delta = pin->GetOrAddReal("bns_nurate", "opacity_tau_delta", 1.0);
      nurates_params.opacity_corr_fac_max = pin->GetOrAddReal("bns_nurates", "opacity_corr_fac_max", 3.0);
      nurates_params.nb_min_cgs = pin->GetOrAddReal("bns_nurates", "nb_min_cgs", 0.);
      nurates_params.temp_min_mev = pin->GetOrAddReal("bns_nurates", "temp_min_mev", 0.);
      
      nurates_params.use_abs_em = pin->GetOrAddBoolean("bns_nurates", "use_abs_em", true);
      nurates_params.use_pair = pin->GetOrAddBoolean("bns_nurates", "use_pair", true);
      nurates_params.use_brem = pin->GetOrAddBoolean("bns_nurates", "use_brem", true);
      nurates_params.use_iso = pin->GetOrAddBoolean("bns_nurates", "use_iso", true);
      nurates_params.use_inelastic_scatt = pin->GetOrAddBoolean("bns_nurates", "use_inelastic_scatt", true);
      nurates_params.use_WM_ab = pin->GetOrAddBoolean("bns_nurates", "use_WM_ab", true);
      nurates_params.use_WM_sc = pin->GetOrAddBoolean("bns_nurates", "use_WM_sc", true);
      nurates_params.use_dU = pin->GetOrAddBoolean("bns_nurates", "use_dU", false);
      nurates_params.use_dm_eff = pin->GetOrAddBoolean("bns_nurates", "use_dm_eff", true);
      nurates_params.use_equilibrium_distribution = pin->GetOrAddBoolean("bns_nurates", "use_equilibrium_distribution", false);
      nurates_params.use_kirchhoff_law = pin->GetOrAddBoolean("bns_nurates", "use_kirchoff_law", false);
      
      nurates_params.my_quadrature_1d.nx = nurates_params.nurates_quad_nx;
      nurates_params.my_quadrature_1d.dim = 1;
      nurates_params.my_quadrature_1d.type = kGauleg;
      nurates_params.my_quadrature_1d.x1 = 0.;
      nurates_params.my_quadrature_1d.x2 = 1.;
      nurates_params.my_quadrature_2d.nx = nurates_params.nurates_quad_nx;
      nurates_params.my_quadrature_2d.ny = nurates_params.nurates_quad_ny;
      nurates_params.my_quadrature_2d.dim = 2;
      nurates_params.my_quadrature_2d.type = kGauleg;
      nurates_params.my_quadrature_2d.x1 = 0.;
      nurates_params.my_quadrature_2d.x2 = 1.;
      nurates_params.my_quadrature_2d.y1 = 0.;
      nurates_params.my_quadrature_2d.y2 = 1.;
      //GaussLegendreMultiD(&nurates_params.my_quadrature_1d);

      // Weak equilibrium parameters

      verbose_warn_weak = pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true);
      
      // These are the defaults in THC
      opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
      opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
      opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 3.0);

      // density below which nothing is done [g/cm^3]
      rho_min = pin->GetOrAddReal("M1_opacities", "equilibration_rho_min_cgs", -1.0);
      // temperature below which nothing is done [MeV]
      temp_min = pin->GetOrAddReal("M1_opacities", "equilibration_temp_min_mev", -1.0);

      // EOS limits
      Real infty = std::numeric_limits<Real>::infinity();
      eos_rho_min = pin->GetOrAddReal("M1_opacities", "eos_rho_min_cgs", 0.0);
      eos_rho_max = pin->GetOrAddReal("M1_opacities", "eos_rho_max_cgs", infty);
      eos_temp_min = pin->GetOrAddReal("M1_opacities", "eos_temp_min_mev", 0.0);
      eos_temp_max = pin->GetOrAddReal("M1_opacities", "eos_temp_max_mev", infty);
      eos_ye_min = pin->GetOrAddReal("M1_opacities", "eos_ye_min", 0.0);
      eos_ye_max = pin->GetOrAddReal("M1_opacities", "eos_ye_max", 1.0);

      // Option to override the EOS limits with the table limits
      bool set_table_limits = pin->GetOrAddReal("M1_opacities", "eos_limits_from_table", true);

      // Options to override rho_min and eos_temp_min with floors
      bool enforce_rho_floor = pin->GetOrAddReal("M1_opacities", "eos_rho_min_usefloor", false);
      bool enforce_temp_floor = pin->GetOrAddReal("M1_opacities", "eos_temp_min_usefloor", false);

      // Set some stuff

      // Override EOS limits?
      if (set_table_limits) {
        // NB These values need to be in my_units (CGS+MeV) 
        table_rho_min = pmy_block->peos->GetMinimumDensity()
          * code_units->MassDensityConversion(*my_units);
        table_rho_max = pmy_block->peos->GetMaximumDensity()
          * code_units->MassDensityConversion(*my_units);
        table_temp_min = pmy_block->peos->GetMinimumTemperature()
          * code_units->TemperatureConversion(*my_units);
        table_temp_max = pmy_block->peos->GetMaximumTemperature()
          * code_units->TemperatureConversion(*my_units);
        table_ye_min = pmy_block->peos->GetMinimumSpeciesFraction(0);
        table_ye_max = pmy_block->peos->GetMaximumSpeciesFraction(0);
      }
      if (enforce_rho_floor)
        table_rho_min = pmy_block->peos->GetDensityFloor()
          * code_units->MassDensityConversion(*my_units);
      if (enforce_temp_floor)
        table_temp_min = pmy_block->peos->GetTemperatureFloor()
          * code_units->TemperatureConversion(*my_units);
      
      // mb [g] 
      atomic_mass = pmy_block->peos->GetRawBaryonMass()
        * code_units->MassConversion(*my_units); 

    };
    
    ~BNSNuRates() {
    };

    // N.B
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
            Real const nb = rho / mb_code; // baryon num dens 
            Real T = pm1->hydro.sc_T(k,j,i);
            
            Real Y[MAX_SPECIES] = {0.0};
            Y[0] = pm1->hydro.sc_w_Ye(k, j, i);
            Real Y_e = Y[0];
       
            // Chem potentials (code units)
            Real mu_n, mu_p, mu_e;
            ChemicalPotentials_npe(nb, T, Y_e,  mu_n, mu_p, mu_e);
            
            // local undensitized neutrino quantities
            Real invsdetg = sc_oo_sqrt_det_g(k, j, i);
            Real dens_n[4] = {0.}; // NB we have 3 species
            Real dens_e[4] = {0.};
            Real chi_loc[4] = {0.};
            
            for (int nuidx = 0; nuidx < N_SPCS; ++nuidx) {
              dens_n[nuidx] = pm1->rad.sc_n(0, nuidx)(k, j, i) * invsdetg;
              dens_e[nuidx] = pm1->rad.sc_J(0, nuidx)(k, j, i) * invsdetg;
              chi_loc[nuidx] = pm1->rad.sc_J(0, nuidx)(k, j, i); 
            }

            // get emissivities and opacities
            Real eta_0_loc[4]{}, eta_1_loc[4]{};
            Real abs_0_loc[4]{}, abs_1_loc[4]{};
            Real scat_0_loc[4]{}, scat_1_loc[4]{};

            // Note: everything sent and received are in code units

            int opac_err =
              bns_nurates(nb, T, Y, mu_n, mu_p, mu_e,
                          dens_n[0], dens_e[0], chi_loc[0], 
                          dens_n[1], dens_e[1], chi_loc[1],
                          dens_n[2], dens_e[2], chi_loc[2],
                          dens_n[3], dens_e[3], chi_loc[3],
                          eta_0_loc[0], eta_0_loc[1], eta_0_loc[2], eta_0_loc[3],
                          eta_1_loc[0], eta_1_loc[1], eta_1_loc[2], eta_1_loc[3],
                          abs_0_loc[0], abs_0_loc[1], abs_0_loc[2], abs_0_loc[3],
                          abs_1_loc[0], abs_1_loc[1], abs_1_loc[2], abs_1_loc[3],
                          scat_0_loc[0], scat_0_loc[1], scat_0_loc[2], scat_0_loc[3],
                          scat_1_loc[0], scat_1_loc[1], scat_1_loc[2], scat_1_loc[3],
                          nurates_params); 

            bool is_failing_opacity = (opac_err)? true : false;
            
            // Dump some information when opacity calculation fails
            if (is_failing_opacity)
              {
                std::ostringstream msg;
                msg << "CalculateOpacityBNSNuRates failure: " << ierr;
                pm1->StatePrintPoint(msg.str(), 0, 0, k, j, i, false);
                pm1->StatePrintPoint(msg.str(), 0, 1, k, j, i, false);
                pm1->StatePrintPoint(msg.str(), 0, 2, k, j, i, true);
              }

            // unpack to variables used below.
            // this is just to avoid to touch the code below.
            //TODO WeakRate code has been updated since this branch started
            //     some varnames have changed, fix this code as well.
            //     temporarily we copy to old varnames:
            Real eta_n_nue = eta_0_loc[0];
            Real eta_n_nua = eta_0_loc[1];
            Real eta_n_nux = eta_0_loc[2] + eta_0_loc[3]; // sum up anux contrib.
            Real eta_e_nue = eta_1_loc[0];
            Real eta_e_nua = eta_1_loc[1];
            Real eta_e_nux = eta_1_loc[2] + eta_0_loc[3];

            // nu absorption opacities
            Real kap_a_n_nue = abs_0_loc[0];
            Real kap_a_n_nua = abs_0_loc[1];
            Real kap_a_n_nux = 0.5*(abs_0_loc[2] + abs_0_loc[3]); // avg nux and anux contrib.
            Real kap_a_e_nue = abs_1_loc[0];
            Real kap_a_e_nua = abs_1_loc[2];
            Real kap_a_e_nux = 0.5*(abs_1_loc[2] + abs_1_loc[3]);

            // nu scattering opacities
            Real kap_s_n_nue = scat_0_loc[0];
            Real kap_s_n_nua = scat_0_loc[1];
            Real kap_s_n_nux = 0.5*(scat_0_loc[2] + scat_0_loc[3]); // avg nux and anux contrib.
            Real kap_s_e_nue = scat_1_loc[0];
            Real kap_s_e_nua = scat_1_loc[1];
            Real kap_s_e_nux = 0.5*(scat_1_loc[2] + scat_1_loc[3]);
            
            
            // Equilibrium logic ----------------------------------------------------
            // Following is identical to WeakRates, we are duplicating code.
            
            Real tau = std::min(
                                std::sqrt(kap_a_e_nue * (kap_a_e_nue + kap_s_e_nue)),
                                std::sqrt(kap_a_e_nua * (kap_a_e_nua + kap_s_e_nua))
                                ) * dt;
            
            // Calculate equilibrium blackbody functions with trapped neutrinos
            Real dens_n_trap[3];
            Real dens_e_trap[3];

            // --------------------------------------------------------------------
            if (opacity_tau_trap >= 0.0 && tau > opacity_tau_trap)
              {
                // Ensure evolution method delegated based on here detected equilibrium
                const static int ix_g = 0;
                typedef M1::evolution_strategy::opt_solution_regime osr_r;
                typedef M1::evolution_strategy::opt_source_treatment ost_r;
                AthenaArray<osr_r> & sol_r = pm1->ev_strat.masks.solution_regime;
                AthenaArray<ost_r> & src_r = pm1->ev_strat.masks.source_treatment;
                
                for (int ix_s=0; ix_s<3; ++ix_s)
                  {
                    sol_r(ix_g,ix_s,k,j,i) = osr_r::equilibrium;
                    src_r(ix_g,ix_s,k,j,i) = ost_r::set_zero;
                  }
                
                Real T_star;
                Real Y_e_star;

                /*
                Real dens_n[3];
                Real dens_e[3];
                Real invsdetg = sc_oo_sqrt_det_g(k, j, i);
                
                // FF number density
                dens_n[0] = pm1->rad.sc_n(0, 0)(k, j, i) * invsdetg;
                dens_n[1] = pm1->rad.sc_n(0, 1)(k, j, i) * invsdetg;
                dens_n[2] = pm1->rad.sc_n(0, 2)(k, j, i) * invsdetg;
                
                // FF energy density
                dens_e[0] = pm1->rad.sc_J(0, 0)(k, j, i) * invsdetg;
                dens_e[1] = pm1->rad.sc_J(0, 1)(k, j, i) * invsdetg;
                dens_e[2] = pm1->rad.sc_J(0, 2)(k, j, i) * invsdetg;
                */
                
                // Calculate equilibrated state
                ierr[0] = WeakEquilibrium(rho, T, Y_e,
                                          dens_n[0],
                                          dens_n[1],
                                          dens_n[2],
                                          dens_e[0],
                                          dens_e[1],
                                          dens_e[2],
                                          T_star, Y_e_star,
                                          dens_n_trap[0],
                                          dens_n_trap[1],
                                          dens_n_trap[2],
                                          dens_e_trap[0],
                                          dens_e_trap[1],
                                          dens_e_trap[2]);
                
                // If we can't get equilibrium, try again but ignore current neutrino
                // data
                if (ierr[0])
                  {
                    ierr[1] = WeakEquilibrium(rho, T, Y_e,
                                              0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0,
                                              T_star, Y_e_star,
                                              dens_n_trap[0],
                                              dens_n_trap[1],
                                              dens_n_trap[2],
                                              dens_e_trap[0],
                                              dens_e_trap[1],
                                              dens_e_trap[2]);
                    
                    //assert(!ierr[1]); // THC treats this as a warning

                    if (verbose_warn_weak)
                      {
                        std::printf("M1: can't get equilibrium @ (i,j,k)=(%d,%d,%d) (%.3e,%.3e,%.3e)\n",
                                    i, j, k,
                                    pmy_block->pcoord->x1v(i),
                                    pmy_block->pcoord->x2v(j),
                                    pmy_block->pcoord->x3v(k));
                      }
                  }
                
                assert(isfinite(dens_n_trap[0]));
                assert(isfinite(dens_n_trap[1]));
                assert(isfinite(dens_n_trap[2]));
                assert(isfinite(dens_e_trap[0]));
                assert(isfinite(dens_e_trap[1]));
                assert(isfinite(dens_e_trap[2]));
                
              } // if (opacity_tau_trap ...
            // --------------------------------------------------------------------
            
            // Calculate equilibrium blackbody functions with fixed T, Ye
            Real dens_n_thin[3];
            Real dens_e_thin[3];
            ierr[0] = NeutrinoDensity(nb, T,
                                      mu_n, mu_p, mu_e,
                                      dens_n_thin[0], dens_n_thin[1], dens_n_thin[2],
                                      dens_e_thin[0], dens_e_thin[1], dens_e_thin[2]);
            assert(!ierr[0]);
            
            // Set the black body function
            Real dens_n[3];
            Real dens_e[3];
            if (opacity_tau_trap < 0 || tau <= opacity_tau_trap)
              {
                dens_n[0] = dens_n_thin[0];
                dens_n[1] = dens_n_thin[1];
                dens_n[2] = dens_n_thin[2];
                
                dens_e[0] = dens_e_thin[0];
                dens_e[1] = dens_e_thin[1];
                dens_e[2] = dens_e_thin[2];
              }
            else if (tau > opacity_tau_trap + opacity_tau_delta)
              {
                dens_n[0] = dens_n_trap[0];
                dens_n[1] = dens_n_trap[1];
                dens_n[2] = dens_n_trap[2];
                
                dens_e[0] = dens_e_trap[0];
                dens_e[1] = dens_e_trap[1];
                dens_e[2] = dens_e_trap[2];
              }
            else
              {
                Real lam = (tau - opacity_tau_trap) / opacity_tau_delta;
                
                dens_n[0] = lam * dens_n_trap[0] + (1 - lam) * dens_n_thin[0];
                dens_n[1] = lam * dens_n_trap[1] + (1 - lam) * dens_n_thin[1];
                dens_n[2] = lam * dens_n_trap[2] + (1 - lam) * dens_n_thin[2];
                
                dens_e[0] = lam * dens_e_trap[0] + (1 - lam) * dens_e_thin[0];
                dens_e[1] = lam * dens_e_trap[1] + (1 - lam) * dens_e_thin[1];
                dens_e[2] = lam * dens_e_trap[2] + (1 - lam) * dens_e_thin[2];
              }
            
            // Calculate correction factors
            Real corr_fac[3];
            for (int s_idx = 0; s_idx < N_SPCS; ++s_idx)
              {
                pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i) =
                  dens_e[s_idx] / dens_n[s_idx];
                corr_fac[s_idx] = pm1->rad.sc_J(0, s_idx)(k, j, i) /
                  (pm1->rad.sc_n(0, s_idx)(k, j, i) *
                   pm1->radmat.sc_avg_nrg(0, s_idx)(k, j, i));
                
                if (!std::isfinite(corr_fac[s_idx])) {
                  corr_fac[s_idx] = 1.0;
                  // should never land here (due to flooring prior to call of opac.)
                  // assert(false);
                }
                
                corr_fac[s_idx] *= corr_fac[s_idx];
                corr_fac[s_idx] =
                  std::max(1.0 / opacity_corr_fac_max,
                           std::min(corr_fac[s_idx], opacity_corr_fac_max));
              }
            
            // Energy scattering
            pm1->radmat.sc_kap_s(0, 0)(k, j, i) = corr_fac[0] * kap_s_e_nue;
            pm1->radmat.sc_kap_s(0, 1)(k, j, i) = corr_fac[1] * kap_s_e_nua;
            pm1->radmat.sc_kap_s(0, 2)(k, j, i) = corr_fac[2] * kap_s_e_nux;
            
            // Enforce Kirchhoff's law
            // For electron lepton neutrinos we change the opacity
            // For heavy lepton neutrinos we change the emissivity
            
            // Electron neutrinos
            pm1->radmat.sc_kap_a_0(0, 0)(k, j, i) = corr_fac[0] * kap_a_n_nue;
            pm1->radmat.sc_kap_a(0, 0)(k, j, i) = corr_fac[0] * kap_a_e_nue;
            
            pm1->radmat.sc_eta_0(0, 0)(k, j, i) =
              pm1->radmat.sc_kap_a_0(0, 0)(k, j, i) * dens_n[0];
            pm1->radmat.sc_eta(0, 0)(k, j, i) =
              pm1->radmat.sc_kap_a(0, 0)(k, j, i) * dens_e[0];
            
            // Electron anti-neutrinos
            pm1->radmat.sc_kap_a_0(0, 1)(k, j, i) = corr_fac[1] * kap_a_n_nua;
            pm1->radmat.sc_kap_a(0, 1)(k, j, i) = corr_fac[1] * kap_a_e_nua;
            
            pm1->radmat.sc_eta_0(0, 1)(k, j, i) =
              pm1->radmat.sc_kap_a_0(0, 1)(k, j, i) * dens_n[1];
            pm1->radmat.sc_eta(0, 1)(k, j, i) =
              pm1->radmat.sc_kap_a(0, 1)(k, j, i) * dens_e[1];
            
            // Heavy lepton neutrinos
            pm1->radmat.sc_eta_0(0, 2)(k, j, i) = corr_fac[2] * eta_n_nux;
            pm1->radmat.sc_eta(0, 2)(k, j, i) = corr_fac[2] * eta_e_nux;
            
            pm1->radmat.sc_kap_a_0(0, 2)(k, j, i) =
              (dens_n[2] > 1e-20 ? pm1->radmat.sc_eta_0(0, 2)(k, j, i) / dens_n[2]
               : 0.0);
            pm1->radmat.sc_kap_a(0, 2)(k, j, i) =
              (dens_e[2] > 1e-20 ? pm1->radmat.sc_eta(0, 2)(k, j, i) / dens_e[2] : 0.0);
          }
      
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
    BNSNuRates_Units::UnitSystem* code_units;
    
    // pars for nurates (choice of reactions, quadratures)
    NuratesParams nurates_params;  
    
    // Chem potentials (input & output in code units)
    void ChemicalPotentials_npe(Real nb, Real T, Real Ye,
                                Real &mu_n, Real &mu_p, Real mu_n) {
      Real mu_b = pmy_block->peos->GetEOS().GetBaryonChemicalPotential(nb, T, Y_e);
      Real mu_q = pmy_block->peos->GetEOS().GetChargeChemicalPotential(nb, T, Y_e);
      Real mu_l = pmy_block->peos->GetEOS().GetElectronLeptonChemicalPotential(nb, T, Y_e);
      mu_n = mu_b;
      mu_p = mu_b + mu_q;
      mu_e = mu_l - mu_q;
    }

    // Main wrapper to bns_nurates
    bool bns_nurates(Real &nb, Real &temp, Real &ye, Real &mu_n, Real &mu_p, Real &mu_e,
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

    // Computes the neutrino number and energy density
    void NeutrinoDensity(Real nb, Real temp,
                         Real mu_n, Real mu_p, Real mu_e,
                         Real &n_nue, Real &n_anue, Real &n_nux,
                         Real &e_nue, Real &e_anue, Real &e_nux);

    // Weak equilibrium stuff -----------------------------------------------------------------

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
    const int n_max = 100;       // Newton-Raphson max number of iterations
    static const int n_at = 16;  // number of independent initial guesses

    // deltas to compute numerical derivatives in the EOS tables
    const Real delta_ye = 0.005;
    const Real delta_t  = 0.01;

    // Some constants
    const Real pi = M_PI;                      // 3.14159265358979323846; 
    const Real mev_to_erg = 1.60217733e-6;     // conversion from MeV to erg
    const Real hc_mevcm = 1.23984172e-10;      // hc in units of MeV*cm

#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define QUAD(x) ((x)*(x)*(x)*(x))

    const Real pi2   = SQR(pi);                          // pi**2 [-]
    const Real pref1 = 4.0/3.0*pi/CUBE(hc_mevcm);        // 4/3 *pi/(hc)**3 [MeV^3/cm^3]
    const Real pref2 = 4.0*pi*mev_to_erg/CUBE(hc_mevcm); // 4*pi/(hc)**3 [erg/MeV^4/cm^3]
    const Real cnst1 = 7.0*QUAD(pi)/20.0;                // 7*pi**4/20 [-]
    const Real cnst5 = 7.0*QUAD(pi)/60.0;                // 7*pi**4/60 [-]
    const Real cnst6 = 7.0*QUAD(pi)/30.0;                // 7*pi**4/30 [-]
    const Real cnst2 = 7.0*QUAD(pi)/5.0;                 // 7*pi**4/5 [-]
    const Real cnst3 = 7.0*QUAD(pi)/15.0;                // 7*pi**4/15 [-]
    const Real cnst4 = 14.0*QUAD(pi)/15.0;               // 14*pi**4/15 [-]

#undef SQR
#undef CUBE
#undef QUAD

    // Factors needed for some unit conversion
    const Real NORMFACT = 1e50;
    const Real MEV_TO_ERG = 1.6021766339999e-6;
    
    // Main wrapper
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

    // Wrapper working with CGS + MeV units in input/output
    // (several calls in equilibration code requires this)
    void ChemicalPotentials_npe_cgs(Real rho, Real temp, Real Ye,
                                    Real &mu_n, Real &mu_p, Real mu_n) {
      const Real MeV = code_units->TemperatureConversion(*my_units); 
      ChemicalPotentials_npe( rho / atomic_mass * my_units->NumberDensityConversion(*code_units), 
                              temp / MeV,
                              Ye,
                              mu_n, mu_p, mu_n);
      mu_n *= Mev:
      mu_p *= MeV;
      mu_e *= MeV;
      return
    }

    void weak_equil_wnu(Real rho, Real T, Real y_in[4], Real e_in[4],
                        Real& T_eq, Real y_eq[4], Real e_eq[4], int& na, int& ierr);
    bool apply_table_limits(Real& rho, Real& temp, Real& ye);
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
