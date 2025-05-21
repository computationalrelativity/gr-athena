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

// https://github.com/RelNucAs/bns_nurates
// bns_nurates headers
#include "bns_nurates/include/bns_nurates.hpp"
#include "bns_nurates/include/constants.hpp"
#include "bns_nurates/include/distribution.hpp"
#include "bns_nurates/include/functions.hpp"
#include "bns_nurates/include/integration.hpp"
#include "bns_nurates/include/m1_opacities.hpp"

//TODO reference K code (note also there it is under construction...)
// https://github.com/IAS-Astrophysics/athenak/blob/radiation-m1-nurates/src/radiation_m1/radiation_m1_nurates.hpp
// https://github.com/IAS-Astrophysics/athenak/blob/radiation-m1-nurates/src/radiation_m1/radiation_m1_calc_opacities_nurates.cpp

namespace M1::Opacities::BNSNuRates {

#define NORMFACT (1e50)
  
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

      //TODO there is code in WeakRates units.hpp ... scheiBe
      // my_units (of BNSNuRates) vs code_units (of GRMHD)
      //my_units   = &WeakRates_Units::WeakRatesUnits;
      //code_units = &WeakRates_Units::GeometricSolar;

      
      // BNSNuRates only works for nu_e + nu_ae + nu_x
      assert(N_SPCS==3);
      
      // BNSNuRates only works for 1 group
      assert(N_GRPS==1);
      
      // These are the defaults in THC
      opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
      opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
      opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 3.0);
      
      verbose_warn_weak = pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true);

      // Parameters for bns_nurates
      nurates_params.nurates_quad_nx = pin->GetOrAddInteger("bns_nurates", "nurates_quad_nx", 10);
      nurates_params.nurates_quad_ny = pin->GetOrAddInteger("bns_nurates", "nurates_quad_ny", 10);
      nurates_params.opacity_tau_trap = pin->GetOrAddReal("bns_nurates", "opacity_tau_trap", 1.0);
      nurates_params.opacity_tau_delta = pin->GetOrAddReal("bns_nurate", "opacity_tau_delta", 1.0);
      nurates_params.opacity_corr_fac_max = pin->GetOrAddReal("bns_nurates", "opacity_corr_fac_max", 3.0);
      nurates_params.nb_min_cgs = pin->GetOrAddReal("bns_nurates", "rho_min_cgs", 0.);
      nurates_params.temp_min_mev = pin->GetOrAddReal("bns_nurates", "temp_min_mev", 0.);
      
      nurates_params.use_abs_em = pin->GetOrAddBoolean("bns_nurates", "use_abs_em", true);
      nurates_params.use_pair = pin->GetOrAddBoolean("bns_nurates", "use_pair", true);
      nurates_params.use_brem = pin->GetOrAddBoolean("bns_nurates", "use_brem", true);
      nurates_params.use_iso = pin->GetOrAddBoolean("bns_nurates", "use_iso", true);
      nurates_params.use_inelastic_scatt = pin->GetOrAddBoolean("bns_nurates", "use_inelastic_scatt", true);
      nurates_params.use_WM_ab = pin->GetOrAddBoolean("bns_nurates", "use_WM_ab", true);
      nurates_params.use_WM_sc = pin->GetOrAddBoolean("bns_nurates", "use_WM_sc", true);
      nurates_params.use_dU = pin->GetOrAddBoolean("bns_nurates", "use_dU", true);
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
      
      // These are the defaults in THC
      opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
      opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
      opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 3.0);
      
      verbose_warn_weak = pin->GetOrAddBoolean("M1_opacities", "verbopmy_block->peos->GetEOS());
      
      // These are the defaults in THC
      opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
      opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
      opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 3.0);
      
      verbose_warn_weak = pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true);

      // Parameters for bns_nurates
      nurates_params.nurates_quad_nx = pin->GetOrAddInteger("bns_nurates", "nurates_quad_nx", 10);
      nurates_params.nurates_quad_ny = pin->GetOrAddInteger("bns_nurates", "nurates_quad_ny", 10);
      nurates_params.opacity_tau_trap = pin->GetOrAddReal("bns_nurates", "opacity_tau_trap", 1.0);
      nurates_params.opacity_tau_delta = pin->GetOrAddReal("bns_nurate", "opacity_tau_delta", 1.0);
      nurates_params.opacity_corr_fac_max = pin->GetOrAddReal("bns_nurates", "opacity_corr_fac_max", 3.0);
      nurates_params.nb_min_cgs = pin->GetOrAddReal("bns_nurates", "rho_min_cgs", 0.);
      nurates_params.temp_min_mev = pin->GetOrAddReal("bns_nurates", "temp_min_mev", 0.);
      
      nurates_params.use_abs_em = pin->GetOrAddBoolean("bns_nurates", "use_abs_em", true);
      nurates_params.use_pair = pin->GetOrAddBoolean("bns_nurates", "use_pair", true);
      nurates_params.use_brem = pin->GetOrAddBoolean("bns_nurates", "use_brem", true);
      nurates_params.use_iso = pin->GetOrAddBoolean("bns_nurates", "use_iso", true);
      nurates_params.use_inelastic_scatt = pin->GetOrAddBoolean("bns_nurates", "use_inelastic_scatt", true);
      nurates_params.use_WM_ab = pin->GetOrAddBoolean("bns_nurates", "use_WM_ab", true);
      nurates_params.use_WM_sc = pin->GetOrAddBoolean("bns_nurates", "use_WM_sc", true);
      nurates_params.use_dU = pin->GetOrAddBoolean("bns_nurates", "use_dU", true);
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
se_warn_weak", true);

      // Parameters for bns_nurates
      nurates_params.nurates_quad_nx = pin->GetOrAddInteger("bns_nurates", "nurates_quad_nx", 10);
      nurates_params.nurates_quad_ny = pin->GetOrAddInteger("bns_nurates", "nurates_quad_ny", 10);
      nurates_params.opacity_tau_trap = pin->GetOrAddReal("bns_nurates", "opacity_tau_trap", 1.0);
      nurates_params.opacity_tau_delta = pin->GetOrAddReal("bns_nurate", "opacity_tau_delta", 1.0);
      nurates_params.opacity_corr_fac_max = pin->GetOrAddReal("bns_nurates", "opacity_corr_fac_max", 3.0);
      nurates_params.nb_min_cgs = pin->GetOrAddReal("bns_nurates", "rho_min_cgs", 0.);
      nurates_params.temp_min_mev = pin->GetOrAddReal("bns_nurates", "temp_min_mev", 0.);
      
      nurates_params.use_abs_em = pin->GetOrAddBoolean("bns_nurates", "use_abs_em", true);
      nurates_params.use_pair = pin->GetOrAddBoolean("bns_nurates", "use_pair", true);
      nurates_params.use_brem = pin->GetOrAddBoolean("bns_nurates", "use_brem", true);
      nurates_params.use_iso = pin->GetOrAddBoolean("bns_nurates", "use_iso", true);
      nurates_params.use_inelastic_scatt = pin->GetOrAddBoolean("bns_nurates", "use_inelastic_scatt", true);
      nurates_params.use_WM_ab = pin->GetOrAddBoolean("bns_nurates", "use_WM_ab", true);
      nurates_params.use_WM_sc = pin->GetOrAddBoolean("bns_nurates", "use_WM_sc", true);
      nurates_params.use_dU = pin->GetOrAddBoolean("bns_nurates", "use_dU", true);
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
      //GaussLegendreMultiD(&nurates_params.my_quadrature_2d);
            
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

      //TODO this is what is done in the K to pass unit conversions factors to bnsnurates()
      // conversion factors from cgs to code units
      // auto cgs_units = Primitive::MakeCGS();
      // auto code_units = eos.GetCodeUnitSystem();
      // const RadiationM1Units cgs2codeunits = {
      //   .cgs2code_length = 1. / code_units.LengthConversion(cgs_units),
      //   .cgs2code_time = 1. / code_units.TimeConversion(cgs_units),
      //   .cgs2code_rho = 1. / code_units.DensityConversion(cgs_units),
      //   .cgs2code_energy = 1. / code_units.EnergyConversion(cgs_units),
      // };

      const Real mb_code = pmy_block->peos->GetEOS().GetBaryonMass();
      
      M1_FLOOP3(k, j, i)
        if (pm1->MaskGet(k, j, i))
          {
            Real rho = pm1->hydro.sc_w_rho(k, j, i);
            Real press = pm1->hydro.sc_w_p(k, j, i);
            Real const nb = rho / mb_code; // baryon num dens (code units)
            
            Real Y[MAX_SPECIES] = {0.0};
            Y[0] = pm1->hydro.sc_w_Ye(k, j, i);
            Real Y_e = Y[0];
            
            Real T = pm1->hydro.sc_T(k,j,i); // this should be in MeV
            Real T_code = T * my_units->TemperatureConversion(*code_units);
       
            // Chem potentials (code units)
            //CHECK T (Mev) or T_code for these calls?
            /*
            Real mu_b = pmy_block->peos->GetEOS().GetBaryonChemicalPotential(nb, T_code, Y_e);
            Real mu_q = pmy_block->peos->GetEOS().GetChargeChemicalPotential(nb, T_code, Y_e);
            Real mu_l = pmy_block->peos->GetEOS().GetElectronLeptonChemicalPotential(nb, T_code, Y_e);
            
            Real mu_n = mu_b;
            Real mu_p = mu_b + mu_q;
            Real mu_e = mu_l - mu_q;
            */
            Real mu_n, mu_p, mu_e;
            ChemicalPotentials_npe(nb, T_code, Y_e,  mu_n, mu_p, mu_e);

            Real eta_n_nue;
            Real eta_n_nua;
            Real eta_n_nux;
            Real eta_e_nue;
            Real eta_e_nua;
            Real eta_e_nux;
            
            Real kap_a_n_nue;
            Real kap_a_n_nua;
            Real kap_a_n_nux;
            Real kap_a_e_nue;
            Real kap_a_e_nua;
            Real kap_a_e_nux;
            
            Real kap_s_n_nue;
            Real kap_s_n_nua;
            Real kap_s_n_nux;
            Real kap_s_e_nue;
            Real kap_s_e_nua;
            Real kap_s_e_nux;
            
            // Note: everything sent and received are in code units
            //TODO change below var names with those above 
            bool is_failing_opacity =
              bns_nurates(nb, T, Y, mu_n, mu_p, mu_e,
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
    
            // Dump some information when opacity calculation fails
            if (is_failing_opacity)
              {
                std::ostringstream msg;
                msg << "CalculateOpacityBNSNuRates failure: ";
                pm1->StatePrintPoint(msg.str(), 0, 0, k, j, i, false);
                pm1->StatePrintPoint(msg.str(), 0, 1, k, j, i, false);
                pm1->StatePrintPoint(msg.str(), 0, 2, k, j, i, true);  // assert(false)
              }
            
            // Equilibrium logic ----------------------------------------------------

            //TODO Following is identical to WeakRates...
            // ... there should be a single place where this is done at level
            //   src/m1/opacity/
            // then we can vary the rates.
            
            Real tau = std::min(
                                std::sqrt(kap_a_e_nue * (kap_a_e_nue + kap_s_e_nue)),
                                std::sqrt(kap_a_e_nua * (kap_a_e_nua + kap_s_e_nua))
                                ) * dt;
            
            // Calculate equilibrium blackbody functions with trapped neutrinos
            Real dens_n_trap[3];
            Real dens_e_trap[3];
            
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
                // --------------------------------------------------------------------
                
                Real T_star;
                Real Y_e_star;
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
                    
                    // THC treats this as a warning
                    //assert(!ierr[1]);
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
              }
            
            // Calculate equilibrium blackbody functions with fixed T, Ye
            Real dens_n_thin[3];
            Real dens_e_thin[3];
            ierr[0] = NeutrinoDensity(rho, T, Y_e,
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
    
    int NucleiAbar(Real rho, Real temp, Real ye, Real& abar) {
      //NB apparently also not implemented in WeakRates
      std::cout<<"NucleiAbar not implemented"<<std::endl;
      return -1;
    }

    Real AverageBaryonMass() {
      Real mb_code = pmy_block->peos->GetRawBaryonMass();
      Real mb_cgs  = mb_code * code_units->MassConversion(*my_units);
      return mb_cgs;
    }
    
  private:
    M1 *pm1;
    Mesh *pmy_mesh;
    MeshBlock *pmy_block;
    Coordinates *pmy_coord;
    
    const int N_GRPS;
    const int N_SPCS;
    
    // Options for controlling weakrates opacities
    Real opacity_tau_trap;
    Real opacity_tau_delta;
    Real opacity_corr_fac_max;
   
    bool verbose_warn_weak;

    // pars for nurates (choice of reactions, quadratures)
    NuratesParams nurates_params;  


    // Chem potentials (everything in code units)
    //CHECK T (Mev) or T_code for these calls?
    void ChemicalPotentials_npe(Real nb, Real T, Real Ye,
                                Real &mu_n, Real &mu_p, Real mu_n) {
      Real mu_b = pmy_block->peos->GetEOS().GetBaryonChemicalPotential(nb, T_code, Y_e);
      Real mu_q = pmy_block->peos->GetEOS().GetChargeChemicalPotential(nb, T_code, Y_e);
      Real mu_l = pmy_block->peos->GetEOS().GetElectronLeptonChemicalPotential(nb, T_code, Y_e);
      mu_n = mu_b;
      mu_p = mu_b + mu_q;
      mu_e = mu_l - mu_q;
    }

    
    // Main wrapper to bns_nurates

    //! \fn void bns_nurates(Real &nb, Real &temp, Real &ye, Real &mu_n, Real &mu_p,
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
    //   \note  All input and output quantities are in code units, except temperature (MeV)
    //
    //   \param[in] nb              baryon number density
    //   \param[in] temp            temperature (MeV)
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
    
    bool bns_nurates(Real &nb, Real &temp, Real &ye, Real &mu_n, Real &mu_p, Real &mu_e,
                     Real &n_nue, Real &j_nue, Real &chi_nue, Real &n_anue, Real &j_anue,
                     Real &chi_anue, Real &n_nux, Real &j_nux, Real &chi_nux, Real &n_anux,
                     Real &j_anux, Real &chi_anux, Real &R_nue, Real &R_anue, Real &R_nux,
                     Real &R_anux, Real &Q_nue, Real &Q_anue, Real &Q_nux, Real &Q_anux,
                     Real &sigma_0_nue, Real &sigma_0_anue, Real &sigma_0_nux,
                     Real &sigma_0_anux, Real &sigma_1_nue, Real &sigma_1_anue,
                     Real &sigma_1_nux, Real &sigma_1_anux, Real &scat_0_nue,
                     Real &scat_0_anue, Real &scat_0_nux, Real &scat_0_anux, Real &scat_1_nue,
                     Real &scat_1_anue, Real &scat_1_nux, Real &scat_1_anux,
                     const NuratesParams nurates_params)

    {
      
      bool success = true;
      
      //TODO unit conversion with GRA structure!
      
      // unit conversion factors
      // from cm^-3 to code
      const Real n_cgs2code = 1. / std::pow(units.cgs2code_length, 3);
      // from MeV cm^-3 to code
      const Real j_cgs2code = units.cgs2code_energy / std::pow(units.cgs2code_length, 3);
      // from cm^-3 s^-1 to code
      const Real r_cgs2code =
        1. / (units.cgs2code_time * std::pow(units.cgs2code_length, 3));
      // from MeV cm^-3 s^-1 to code
      const Real q_cgs2code = MEV_TO_ERG * units.cgs2code_energy /
        (units.cgs2code_time * std::pow(units.cgs2code_length, 3));
      // from cm^-1 to code
      const Real kappa_cgs2code = 1. / (units.cgs2code_length);

      // const Real ten_to_7 = 1e7;
      // const Real ten_to_21 = 1e21;
      // const Real ten_to_minus_21 = 1e-21;
      
      if (nb * (1.0/units.cgs2code_energy) < nurates_params.rho_min_cgs ||
          temp < nurates_params.temp_min_mev) {
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
        return success;
      }
      
      // convert neutrino quantities to cgs
      const Real n_nue_cgs = n_nue / (cgs2code_n / NORMFACT) * 1e-21;
      const Real n_anue_cgs = n_anue / (cgs2code_n / NORMFACT) * 1e-21;
      const Real n_nux_cgs = n_nux / (cgs2code_n / NORMFACT) * 1e-21;
      const Real n_anux_cgs = n_anux / (cgs2code_n / NORMFACT) * 1e-21;
      const Real j_nue_cgs = j_nue / cgs2code_j;
      const Real j_anue_cgs = j_anue / cgs2code_j;
      const Real j_nux_cgs = j_nux / cgs2code_j;
      const Real j_anux_cgs = j_anux / cgs2code_j;
      const Real nb_cgs = nb / units.cgs2code_rho * 1e-21;
        
      // opacity params structure
      GreyOpacityParams my_grey_opacity_params{};
      
      // reaction flags
      my_grey_opacity_params.opacity_flags = opacity_flags_default_none;
      my_grey_opacity_params.opacity_flags.use_abs_em = nurates_params.use_abs_em;
      my_grey_opacity_params.opacity_flags.use_brem = nurates_params.use_brem;
      my_grey_opacity_params.opacity_flags.use_pair = nurates_params.use_pair;
      my_grey_opacity_params.opacity_flags.use_iso = nurates_params.use_iso;
      my_grey_opacity_params.opacity_flags.use_inelastic_scatt =
        nurates_params.use_inelastic_scatt;

      // other flags
      my_grey_opacity_params.opacity_pars = opacity_params_default_none;
      my_grey_opacity_params.opacity_pars.use_WM_ab = nurates_params.use_WM_ab;
      my_grey_opacity_params.opacity_pars.use_WM_sc = nurates_params.use_WM_sc;
      my_grey_opacity_params.opacity_pars.use_dU = nurates_params.use_dU;
      my_grey_opacity_params.opacity_pars.use_dm_eff = nurates_params.use_dm_eff;

      //TODO logic for nurates_params.use_dU not implemented from bns_nurates_wrap.cpp

      // populate EOS quantities
      my_grey_opacity_params.eos_pars.mu_e = mu_e;
      my_grey_opacity_params.eos_pars.mu_p = mu_p;
      my_grey_opacity_params.eos_pars.mu_n = mu_n;
      my_grey_opacity_params.eos_pars.temp = temp;
      my_grey_opacity_params.eos_pars.yp = ye;
      my_grey_opacity_params.eos_pars.yn = 1 - ye;
      my_grey_opacity_params.eos_pars.nb = nb_cgs;
      
      // populate M1 quantities
      // The factors of 1/2 come from the fact that bns_nurates and THC weight the
      // heavy neutrinos differently. THC weights them with a factor of 2 (because
      // "nux" means "mu AND tau"), bns_nurates with a factor of 1 (because "nux"
      // means "mu OR tau").
      // GR-Athena++ uses same treatment as THC.
      my_grey_opacity_params.m1_pars.n[id_nue] = n_nue_cgs;
      my_grey_opacity_params.m1_pars.J[id_nue] = j_nue_cgs;
      my_grey_opacity_params.m1_pars.chi[id_nue] = chi_nue;
      my_grey_opacity_params.m1_pars.n[id_anue] = n_anue_cgs;
      my_grey_opacity_params.m1_pars.J[id_anue] = j_anue_cgs;
      my_grey_opacity_params.m1_pars.chi[id_anue] = chi_anue;
      my_grey_opacity_params.m1_pars.n[id_nux] = n_nux_cgs * 0.5;
      my_grey_opacity_params.m1_pars.J[id_nux] = j_nux_cgs * 0.5;
      my_grey_opacity_params.m1_pars.chi[id_nux] = chi_nux;
      my_grey_opacity_params.m1_pars.n[id_anux] = n_anux_cgs * 0.5;
      my_grey_opacity_params.m1_pars.J[id_anux] = j_anux_cgs * 0.5;
      my_grey_opacity_params.m1_pars.chi[id_anux] = chi_anux;
      
      // reconstruct distribution function
      if (!nurates_params.use_equilibrium_distribution) {
        my_grey_opacity_params.distr_pars =
          CalculateDistrParamsFromM1(
                                     &my_grey_opacity_params.m1_pars,
                                     &my_grey_opacity_params.eos_pars);
      } else {
        my_grey_opacity_params.distr_pars =
          NuEquilibriumParams(&my_grey_opacity_params.eos_pars);
        
        // compute neutrino number and energy densities
        ComputeM1DensitiesEq(&my_grey_opacity_params.eos_pars,
                             &my_grey_opacity_params.distr_pars,
                             &my_grey_opacity_params.m1_pars);

        // populate M1 quantities
        my_grey_opacity_params.m1_pars.chi[id_nue] = 0.333333333333333333333333333;
        my_grey_opacity_params.m1_pars.chi[id_anue] = 0.333333333333333333333333333;
        my_grey_opacity_params.m1_pars.chi[id_nux] = 0.333333333333333333333333333;
        my_grey_opacity_params.m1_pars.chi[id_anux] = 0.333333333333333333333333333;
        
        // convert neutrino energy density to mixed MeV and cgs as requested by bns_nurates
        my_grey_opacity_params.m1_pars.J[id_nue] *= kBS_MeV;
        my_grey_opacity_params.m1_pars.J[id_anue] *= kBS_MeV;
        my_grey_opacity_params.m1_pars.J[id_nux] *= kBS_MeV;
        my_grey_opacity_params.m1_pars.J[id_anux] *= kBS_MeV;
      }
      
      // compute opacities
      M1Opacities opacities =
        ComputeM1Opacities(&nurates_params.my_quadrature_1d,
                           &nurates_params.my_quadrature_1d, &my_grey_opacity_params);
      
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
      //TODO catch these and return "not success"
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
      
      // convert to code units
      R_nue = R_nue * (r_cgs2code / NORMFACT) * 1e21;
      R_anue = R_anue * (r_cgs2code / NORMFACT) * 1e21;
      R_nux = R_nux * (r_cgs2code / NORMFACT) * 1e21;
      R_anux = R_anux * (r_cgs2code / NORMFACT) * 1e21;
      Q_nue = Q_nue * q_cgs2code * 1e21;
      Q_anue = Q_anue * q_cgs2code * 1e21;
      Q_nux = Q_nux * q_cgs2code * 1e21;
      Q_anux = Q_anux * q_cgs2code * 1e21;
      sigma_0_nue = sigma_0_nue * kappa_cgs2code * 1e7;
      sigma_0_anue = sigma_0_anue * kappa_cgs2code * 1e7;
      sigma_0_nux = sigma_0_nux * kappa_cgs2code * 1e7;
      sigma_0_anux = sigma_0_anux * kappa_cgs2code * 1e7;
      sigma_1_nue = sigma_1_nue * kappa_cgs2code * 1e7;
      sigma_1_anue = sigma_1_anue * kappa_cgs2code * 1e7;
      sigma_1_nux = sigma_1_nux * kappa_cgs2code * 1e7;
      sigma_1_anux = sigma_1_anux * kappa_cgs2code * 1e7;
      scat_0_nue = scat_0_nue * kappa_cgs2code * 1e7;
      scat_0_anue = scat_0_anue * kappa_cgs2code * 1e7;
      scat_0_nux = scat_0_nux * kappa_cgs2code * 1e7;
      scat_0_anux = scat_0_anux * kappa_cgs2code * 1e7;
      scat_1_nue = scat_1_nue * kappa_cgs2code * 1e7;
      scat_1_anue = scat_1_anue * kappa_cgs2code * 1e7;
      scat_1_nux = scat_1_nux * kappa_cgs2code * 1e7;
      scat_1_anux = scat_1_anux * kappa_cgs2code * 1e7;

      return success;
    }

    
    //! \fn void NeutrinoDensity(Real mu_n, Real mu_p, Real mu_e, Real nb, Real temp,
    //!                       Real &n_nue, Real &n_anue, Real &n_nux, Real &en_nue,
    //!                       Real &en_anue, Real &en_nux, NuratesParams nurates_params)
    //
    //   \brief Computes the neutrino number and energy density
    //
    //   \note  All input and output quantities are in code units, except temperature (MeV)
    //
    //   \param[in]  mu_n            neutron chemical potential
    //   \param[in]  mu_p            proton chemical potential
    //   \param[in]  mu_e            electron chemical potential
    //   \param[in]  nb              baryon number density
    //   \param[in]  temp            temperature (MeV).
    //   \param[out] n_nue           number density electron neutrinos
    //   \param[out] n_anue          number density electron anti-neutrinos
    //   \param[out] n_nux           number density mu/tau neutrinos
    //   \param[out] en_nue          energy density electron neutrinos
    //   \param[out] en_anue         energy density electron anti-neutrinos
    //   \param[out] en_nux          energy density mu/tau neutrinos
    
    void NeutrinoDensity(Real mu_n, Real mu_p, Real mu_e, Real nb, Real temp, Real &n_nue,
                         Real &n_anue, Real &n_nux, Real &en_nue, Real &en_anue, Real &en_nux)
    {
      if ((nb < nurates_params.rho_min_cgs) || (temp < nurates_params.temp_min_mev)) {
        n_nue = 0.;
        n_anue = 0.;
        n_nux = 0.;
        en_nue = 0.;
        en_anue = 0.;
        en_nux = 0.;
        return;
      }
      
      Real eta_nue = (mu_p + mu_e - mu_n) / temp;
      Real eta_anue = -eta_nue;
      Real eta_nux = 0.0;
      
      n_nue = 4.0 * M_PI / std::pow(HC_MEVCM, 3) * std::pow(temp, 3) *
        Fermi::fermi2(eta_nue);
      n_anue = 4.0 * M_PI / std::pow(HC_MEVCM, 3) * std::pow(temp, 3) *
        Fermi::fermi2(eta_anue);
      n_nux = 16.0 * M_PI / std::pow(HC_MEVCM, 3) * std::pow(temp, 3) *
          Fermi::fermi2(eta_nux);
      
      en_nue = 4.0 * M_PI / std::pow(HC_MEVCM, 3) * std::pow(temp, 4) *
        Fermi::fermi3(eta_nue);
      en_anue = 4.0 * M_PI / std::pow(HC_MEVCM, 3) * std::pow(temp, 4) *
        Fermi::fermi3(eta_anue);
      en_nux = 16.0 * M_PI / std::pow(HC_MEVCM, 3) * std::pow(temp, 4) *
        Fermi::fermi3(eta_nux);
      
      assert(isfinite(n_nue));
      assert(isfinite(n_anue));
      assert(isfinite(n_nux));
      assert(isfinite(en_nue));
      assert(isfinite(en_anue));
      assert(isfinite(en_nux));
      
      const Real fact1 = std::pow(units.cgs2code_length, 3) * NORMFACT;
      n_nue = n_nue / fact1;
      n_anue = n_anue / fact1;
      n_nux = n_nux / fact1;
      
      const Real fact2 =
        MEV_TO_ERG * units.cgs2code_energy / std::pow(units.cgs2code_length, 3);
      en_nue = en_nue / fact2;
      en_anue = en_anue / fact2;
      en_nux = en_nux / fact2;
    }


    //! \fn 
    //!      
    //!      
    //
    //  \brief Calculates the equilibrium fluid temperature and electron fraction,
    //   and neutrino number and energy densities assuming energy and lepton number conservation.
    //
    //  \note  All input and output quantities are in code units, except temperature (MeV)

    //TODO double check the units of temperature!
    
    int WeakEquilibrium(Real rho, Real temp, Real ye,
                        Real n_nue, Real n_nua, Real n_nux,
                        Real e_nue, Real e_nua, Real e_nux,
                        Real& temp_eq, Real& ye_eq,
                        Real& n_nue_eq, Real& n_nua_eq, Real& n_nux_eq,
                        Real& e_nue_eq, Real& e_nua_eq, Real& e_nux_eq) {
      Real n_conv = code_units->NumberDensityConversion(*my_units);
      Real e_conv = code_units->EnergyDensityConversion(*my_units);

      int ierr = weak_equilibrium(
        rho * code_units->MassDensityConversion(*my_units),
        temp * code_units->TemperatureConversion(*my_units),
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
        
      temp_eq = temp_eq * my_units->TemperatureConversion(*code_units);
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


    // Weak equilibrium stuff ------------------------------------------------------------

    Real rho_min;                  // density below which nothing is done in g cm-3
    Real temp_min;                 // temperature below which nothing is done in g cm-3
    Real atomic_mass;              // atomic mass in g (to convert mass density to number density)

    // Some parameters later used in the calculations.
    const Real eps_lim  = 1.e-7; // standard tollerance in 2D NR
    const int n_cut_max = 8;     // number of bisections of dx
    const int n_max = 100;       // Newton-Raphson max number of iterations
    static const int n_at = 16;  // number of independent initial guesses

    // deltas to compute numerical derivatives in the EOS tables
    const Real delta_ye = 0.005;
    const Real delta_t  = 0.01;

    //TODO: these should come from the EoS  
    const Real mass_fact = 9.223158894119980e2; //TODO: mass_fact is set in the table read
    Real eos_rhomin;
    Real eos_rhomax;
    Real eos_tempmin;
    Real eos_tempmax;
    Real eos_yemin;
    Real eos_yemax;

#define WR_SQR(x) ((x)*(x))
#define WR_CUBE(x) ((x)*(x)*(x))
#define WR_QUAD(x) ((x)*(x)*(x)*(x))

    // Some constants
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

    const Real clight = 2.99792458e10;
    const Real mev_to_erg = 1.60217733e-6;     // conversion from MeV to erg
    const Real hc_mevcm = 1.23984172e-10;      // hc in units of MeV*cm
    const Real pi    = 3.14159265358979323846; // pi
    
    // Wariable to switch between analytic and numerical solutions of Fermi integrals
    // const bool fermi_analytics = true;
#define WR_FERMI_ANALYTIC (1)
    
    int weak_equilibrium(Real rho,        // [g/cm^3]
                         Real temp,       // [MeV]
                         Real ye,         // [-]
                         Real n_nue,      // [1/cm^3] 
                         Real n_nua,      // [1/cm^3] 
                         Real n_nux,      // [1/cm^3] 
                         Real e_nue,     // [erg/cm^3] 
                         Real e_nua,     // [erg/cm^3] 
                         Real e_nux,     // [erg/cm^3]
                         Real & temp_eq,   // [MeV]
                         Real & ye_eq,     // [-] 
                         Real & n_nue_eq,  // [1/cm^3]
                         Real & n_nua_eq,  // [1/cm^3]
                         Real & n_nux_eq,  // [1/cm^3]
                         Real & e_nue_eq, // [erg/cm^3]
                         Real & e_nua_eq, // [erg/cm^3]
                         Real & e_nux_eq  // [erg/cm^3]
                         ) 
    {
      int iout = 0;
      
      // We will already be in cgs units here
  
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
        return iout;
      } 

      // Enforce table bounds
      //TODO something something EoS, return?
      // boundsErr = enforceTableBounds(rho0, temp0, ye0)
           
      // Compute baryon number density in cgs.
      Real nb = rho0/atomic_mass;

      // Compute fractions
      Real y_in[4] = {0.0};
      y_in[0] = ye0;
      y_in[1] = n_nue/nb;
      y_in[2] = n_nua/nb;
      y_in[3] = 0.25*n_nux/nb;
      
      // Compute energy (note that tab3d_eps works in Cactus units)
      Real e_in[4] = {0.0};
      e_in[0] = EoS->GetEnergyDensity(rho0, temp0, ye0);//TODO
      e_in[1] = e_nue;
      e_in[2] = e_nua;
      e_in[3] = e_nux;
      
      // Compute weak equilibrium
      Real y_eq[4] = {0.0};
      Real e_eq[4] = {0.0};
      int na = 0;
      int ierr = 0;
      weak_equil_wnu(rho0, temp0, y_in, e_in, temp_eq, y_eq, e_eq, na, ierr);
      ye_eq = y_eq[0];
      iout = (ierr != 0) -1 : 0 ;

      // Convert results to Cactus units
      // Conversion no longer here, just split output arrays from weak_equil_wnu
      n_nue_eq  = nb*y_eq[1];
      n_nua_eq  = nb*y_eq[2];
      n_nux_eq  = 4.0*nb*y_eq[3];
      e_nue_eq = e_eq[1];
      e_nua_eq = e_eq[2];
      e_nux_eq = e_eq[3];
      
      return iout;

    } 

    void weak_equil_wnu(Real rho, Real T, Real y_in[4], Real e_in[4],
                        Real& T_eq, Real y_eq[4], Real e_eq[4], int& na, int& ierr) {
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
      
      // Compute the total lepton fraction and internal energy
      Real yl = y_in[0] + y_in[1] - y_in[2];           // [#/baryon]
      Real u  = e_in[0] + e_in[1] + e_in[2] + e_in[3]; // [erg/cm^3]

      /*
        ! vector with the coefficients for the different guesses............
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
        
        // make an initial guess................................................
        x0[0] = vec_guess[na][0]*T;       // T guess  [MeV]
        x0[1] = vec_guess[na][1]*y_in[0]; // ye guess [#/baryon]
        
        // call the 2d Newton-Raphson...........................................
        new_raph_2dim(rho,u,yl,x0,x1,ierr);
        
        na += 1;
      } // end while
      
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

      //TODO ChemPot for weak equilibrium parts might require different UNITS than BNSNuRates!
      
      // Interpolate the chemical potentials (stored in MeV in the table)
      // Real mu_n = EoS->GetNeutronChemicalPotential(rho, T_eq, y_eq[0]);
      // Real mu_p = EoS->GetProtonChemicalPotential(rho, T_eq, y_eq[0]);
      // Real mu_e = EoS->GetElectronChemicalPotential(rho, T_eq, y_eq[0]);
      Real mu_n, mu_p, mu_e;
      ChemicalPotentials_npe(rho, T_eq, y_eq[0],  mu_n, mu_p, mu_e);

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
      
      // check that the energy is positive
      // For tabulated eos we should check that the energy is above the minimum?

      Real e_min = EoS->GetMinimumEnergyDensity(rho, y_eq[0]);//TODO
      if (e_eq[0]<e_min) {
        ierr = 1;
        T_eq = T;
        for (int i=0;i<4;i++) {
          y_eq[i] = y_in[i]; // [#/baryon]
          e_eq[i] = e_in[i]; // [erg/cm^3]
        }
        return;
      } 
  
      // check that Y_e is within the range
      Real table_ye_min, table_ye_max;
      EoS->GetTableLimitsYe(table_ye_min, table_ye_max);//TODO
      if (y_eq[0]<table_ye_min || y_eq[0]>table_ye_max) {
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

    void new_raph_2dim(Real rho, Real u, Real yl, Real x0[2],
                       Real x1[2], int& ierr) {
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
        } 

        // invert the Jacobian
        inv_jacobi(det,J,invJ);
        
        // compute the next step
        dx1[0] = - (invJ[0][0]*y[0] + invJ[0][1]*y[1]);
        dx1[1] = - (invJ[1][0]*y[0] + invJ[1][1]*y[1]);
        
        // check if we are the boundary of the table
        //TODO: Replace with EoS calls
        if (x1[0] == eos_tempmin) {
          norm[0] = -1.0;
        } else if (x1[0] == eos_tempmax) {
          norm[0] = 1.0;
        } else { 
          norm[0] = 0.0;
        } 

        if (x1[1] == eos_yemin) {
          norm[1] = -1.0;
        } else if (x1[1] == eos_yemax) {
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

          bool tabBoundsFlag = EoS->ApplyTableLimits(rho, x1_tmp[0], x1_tmp[1]);//TODO

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
      ierr =  (n_iter <= n_max) ? 0 : 1;
      
      return;
    }
    
    void func_eq_weak(Real rho, Real u, Real yl, Real x[2], Real y[2]) {
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

      // Compute the baryon number density (mass_fact is given in MeV)
      // Real mass_fact_cgs = mass_fact * mev_to_erg / (clight*clight);
      // Real nb = rho / mass_fact_cgs; // [#/cm^3]
      Real nb = rho / atomic_mass; // [#/cm^3]
      
      // Interpolate the chemical potentials (stored in MeV in the table)
      // Real mu_n = EoS->GetNeutronChemicalPotential(rho, x[0], x[1]);
      // Real mu_p = EoS->GetProtonChemicalPotential(rho, x[0], x[1]);
      // Real mu_e = EoS->GetElectronChemicalPotential(rho, x[0], x[1]);
      Real mu_n, mu_p, mu_e;
      ChemicalPotentials_npe(rho, x[0], x[1],  mu_n, mu_p, mu_e );
      
      Real mus[2] = {mu_e, mu_n - mu_p};
      
      // Call the EOS
      Real e = EoS->GetEnergyDensity(rho, x[0], x[1]);//TODO
      
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
    
    void error_func_eq_weak(Real yl, Real u, Real y[2], Real &err) {
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
      // since the first equation is has yl as constant, we normalized the error to it.
      // since the second equation was normalized wrt u, we divide it by 1.
      // the modulus of the two contributions are then summed
      err = abs(y[0]/yl) + abs(y[1]/1.0);
      return;
    }
    
    void jacobi_eq_weak(Real rho, Real u, Real yl, Real x[2], Real J[2][2], int &ierr) {
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
      
      // Interpolate the chemical potentials (stored in MeV in the table)
      Real t = x[0];
      Real ye = x[1];

       //Real mu_n = EoS->GetNeutronChemicalPotential(rho, t, ye);
      //Real mu_p = EoS->GetProtonChemicalPotential(rho, t, ye);
      //Real mu_e = EoS->GetElectronChemicalPotential(rho, t, ye);
      //Real mu_n, mu_p, mu_e;
      ChemicalPotentials_npe(rho, t, ye,  mu_n, mu_p, mu_e);

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

      // compute the Jacobian 
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

    void eta_e_gradient(Real rho, Real t, Real ye, Real eta,
                        Real& detadt, Real& detadye, Real& dedt, Real& dedye, int& ierr) {
      /*----------------------------------------------------------------------
        !
        !     subroutine: eta_e_gradient
        !
        !     this subroutine computes the gradient of the degeneracy parameter
        !     and of the fluid internal energy with respect to temperature and ye
        !
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
        ! gradients are computed numerically. To do it, we consider small
        ! variations in ye and temperature, and we compute the detivative
        ! using finite differencing. The real limitation is that this way
        ! relies on the EOS table interpolation procedure
        !
        ! the goal of this part is to obtain chemical potentials (mus1 and
        ! mus2) in two points close to the point we are considering, first
        ! varying wrt ye and then wrt T
      */
      
      // vary the electron fraction............................................

      /*
        ! these are the two calls to the EOS. The goal here is to get
        ! the fluid internal energy and the chemical potential for
        ! electrons and the difference between neutron and proton
        ! chemical potential (usually called mu_hat) for two points with
        ! slightly different ye
      */

      //  first, for ye slightly smaller
      Real ye1 = max(ye - delta_ye, eos_yemin);
      Real yev = ye1;

      //Real mu_n = EoS->GetNeutronChemicalPotential(rho, t, yev);
      //Real mu_p = EoS->GetProtonChemicalPotential(rho, t, yev);
      //Real mu_e = EoS->GetElectronChemicalPotential(rho, t, yev);
      Real mu_n, mu_p, mu_e;
      ChemicalPotentials_npe(rho, t, yev,  mu_n, mu_p, mu_e);
      
      Real e1 = EoS->GetEnergyDensity(rho, t, yev);

      Real mus1[2] = {mu_e, mu_n - mu_p};

      // second, for ye slightly larger
      Real ye2 = min(ye + delta_ye, eos_yemax);
      yev = ye2;

      //mu_n = EoS->GetNeutronChemicalPotential(rho, t, yev);
      //mu_p = EoS->GetProtonChemicalPotential(rho, t, yev);
      //mu_e = EoS->GetElectronChemicalPotential(rho, t, yev);
      ChemicalPotentials_npe(rho, t, yev,  mu_n, mu_p, mu_e);
      Real mus2[2] = {mu_e, mu_n - mu_p};

      Real e2 = EoS->GetEnergyDensity(rho, t, yev);
      
      // compute numerical derivaties.........................................
      Real dmuedye   = (mus2[0]-mus1[0])/(ye2 - ye1);
      Real dmuhatdye = (mus2[1]-mus1[1])/(ye2 - ye1);
      dedye          = (e2-e1)/(ye2 - ye1);

      // vary the temperature.................................................
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

      //mu_n = EoS->GetNeutronChemicalPotential(rho, tv, ye);
      //mu_p = EoS->GetProtonChemicalPotential(rho, tv, ye);
      //mu_e = EoS->GetElectronChemicalPotential(rho, tv, ye);
      Real mu_n, mu_p, mu_e;
      ChemicalPotentials_npe(rho, t1, ye,  mu_n, mu_p, mu_e );
      Real mus1[2] = {mu_e, mu_n - mu_p};
      
      e1 = EoS->GetEnergyDensity(rho, t1, ye);
      
      // second, for t slightly larger
      tv = t2;

      //mu_n = EoS->GetNeutronChemicalPotential(rho, tv, ye);
      //mu_p = EoS->GetProtonChemicalPotential(rho, tv, ye);
      //mu_e = EoS->GetElectronChemicalPotential(rho, tv, ye);
      ChemicalPotentials_npe(rho, tv, ye,  mu_n, mu_p, mu_e);
      Real mus1[2] = {mu_e, mu_n - mu_p};

      e2 = EoS->GetEnergyDensity(rho, tv, ye);

      // compute the derivatives wrt temperature..............................
      Real dmuedt   = (mus2[0] - mus1[0])/(t2 - t1);
      Real dmuhatdt = (mus2[1] - mus1[1])/(t2 - t1);
      dedt          = (e2   - e1  )/(t2 - t1);
      
      // combine the eta derivatives..........................................
      detadt  = (-eta + dmuedt - dmuhatdt)/t; // [1/MeV]
      detadye = (dmuedye - dmuhatdye)/t;      // [-]
      
      // check if the derivative has a problem................................
      if (isnan(detadt)) {
        ierr = 1;
        return; 
      }

      ierr = 0;
      return;
    }

    void inv_jacobi(Real det, Real J[2][2], Real invJ[2][2]) {
      /*======================================================================
        !
        !     subroutine: inv_jacobi
        !
        !     This subroutine inverts the Jacobian matrix, assuming it to be a
        !     2x2 matrix
        !
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
    }
    

    void nu_deg_param_trap(Real temp_m, Real chem_pot[2], Real eta[3]) {
      /*======================================================================
        !
        !     subroutine: nu_deg_param_trap
        !
        !     In this subroutine, we compute the neutrino degeneracy parameters
        !     assuming weak and thermal equilibrium, i.e. using as input the
        !     local thermodynamical properties
        !
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
        eta[0] = 0.0; // [-]
        eta[1] = 0.0; // [-]
        eta[2] = 0.0; // [-]
      } 
    }
    
    void dens_nu_trap(Real temp_m, Real eta_nu[3], Real nu_dens[3]) {
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
#else
        Real f2 = 0.0;
        fermiint(2.0,eta_nu[it],f2);
#endif
        nu_dens[it] = pref * temp_m3 * f2; // [#/cm^3]
      } 
    }
    
    void edens_nu_trap(Real temp_m, Real eta_nu[3], Real enu_dens[3]) {
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
#else
        Real f3 = 0.0;
        fermiint(3.0,eta_nu[it],f3);
#endif
        enu_dens[it] = pref * temp_m4 * f3;
      } 
    } 
    
    // Done with weak equilibrium stuff 
    
  };

} // namespace M1::Opacities::BNSNuRates

#endif //M1_OPACITIES_BNSNURATES_HPP
