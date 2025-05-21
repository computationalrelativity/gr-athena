#ifndef M1_OPACITIES_BNSNURATES_HPP
#define M1_OPACITIES_BNSNURATES_HPP

#include <limits>

// Athena++ classes headers
#include "../../../athena.hpp"
#include "../../m1.hpp"
#include "../../../hydro/hydro.hpp"

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

#define NORMFACT 1e50
  
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
      
      // BNSNuRates only works for nu_e + nu_ae + nu_x
      assert(N_SPCS==3);
      
      // BNSNuRates only works for 1 group
      assert(N_GRPS==1);

      // Create instance of BNSNuRatesNeutrinos::BNSNuRates
      // Set EoS from PS
      pmy_weakrates = new BNSNuRatesNeutrinos::BNSNuRates(pin,
                                                          &pmy_block->peos->GetEOS());
      
      // These are the defaults in THC
      opacity_tau_trap = pin->GetOrAddReal("M1_opacities", "tau_trap", 1.0);
      opacity_tau_delta = pin->GetOrAddReal("M1_opacities", "tau_delta", 1.0);
      opacity_corr_fac_max = pin->GetOrAddReal("M1_opacities", "max_correction_factor", 3.0);
      
      verbose_warn_weak = pin->GetOrAddBoolean("M1_opacities", "verbose_warn_weak", true);

      // nurates

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
      if (pmy_weakrates!= nullptr) {
        delete pmy_weakrates;
      }
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
      
      M1_FLOOP3(k, j, i)
        if (pm1->MaskGet(k, j, i))
          {
            Real rho = pm1->hydro.sc_w_rho(k, j, i);
            Real press = pm1->hydro.sc_w_p(k, j, i);
            Real const nb = rho / (pmy_block->peos->GetEOS().GetBaryonMass());
            
            Real Y[MAX_SPECIES] = {0.0};
            Y[0] = pm1->hydro.sc_w_Ye(k, j, i);
            
            Real T = pm1->hydro.sc_T(k,j,i);
            Real Y_e = Y[0];
            
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
            //TODO change var names
            bool is_failing_opacity =
              bns_nurates(nb, T, Y, mu_n, mu_p, mu_e, nudens_0[0], nudens_1[0],
                          chi_loc[0], 
                          nudens_0[1], nudens_1[1], chi_loc[1], nudens_0[2], nudens_1[2], chi_loc[2],
                          nudens_0[3], nudens_1[3], chi_loc[3], eta_0_loc[0], eta_0_loc[1],
                          eta_0_loc[2], eta_0_loc[3], eta_1_loc[0], eta_1_loc[1], eta_1_loc[2],
                          eta_1_loc[3], abs_0_loc[0], abs_0_loc[1], abs_0_loc[2], abs_0_loc[3],
                          abs_1_loc[0], abs_1_loc[1], abs_1_loc[2], abs_1_loc[3], scat_0_loc[0],
                          scat_0_loc[1], scat_0_loc[2], scat_0_loc[3], scat_1_loc[0], scat_1_loc[1],
                          scat_1_loc[2], scat_1_loc[3],
                          nurates_params_); // , cgs2codeunits);
    
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
            
            //TODO following is identical to weakrates, the Equilibrium method need to be imported here ...
            
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
                
                // Calculate equilibriated state
                //TODO we need a WeakEquilibrium method,
                //     in principle this should be independent on the specific rate used ...
                //     ... but needs checking and import!
                ierr[0] = pmy_weakrates->WeakEquilibrium(rho, T, Y_e,
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
                    ierr[1] = pmy_weakrates->WeakEquilibrium(rho, T, Y_e,
                                                             0.0, 0.0, 0.0,
                                                             0.0, 0.0, 0.0,
                                                             T_star, Y_e_star,
                                                             dens_n_trap[0],
                                                             dens_n_trap[1],
                                                             dens_n_trap[2],
                                                             dens_e_trap[0],
                                                             dens_e_trap[1],
                                                             dens_e_trap[2]);
                    
                    // TODO THC BNSNuRates treats this as a warning
                    // assert(!ierr[1]);
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
            ierr[0] = pmy_weakrates->NeutrinoDensity(
                                                     rho, T, Y_e, dens_n_thin[0], dens_n_thin[1], dens_n_thin[2],
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


    // Calculate the equilibrium fluid temperature and electron fraction,
    // and neutrino number and energy densities assuming energy and lepton number conservation.
    
    //TODO this is a copy from weak rates, the implementation can be found in
    //     weak_equilibrium.hpp
    //     and it is not yet ported here

    int WeakEquilibrium(Real rho, Real temp, Real ye,
                        Real n_nue, Real n_nua, Real n_nux,
                        Real e_nue, Real e_nua, Real e_nux,
                        Real& temp_eq, Real& ye_eq,
                        Real& n_nue_eq, Real& n_nua_eq, Real& n_nux_eq,
                        Real& e_nue_eq, Real& e_nua_eq, Real& e_nux_eq) {
      Real n_conv = code_units->NumberDensityConversion(*my_units);
      Real e_conv = code_units->EnergyDensityConversion(*my_units);

      int ierr = WR_Equilibrium.WeakEquilibriumImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
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
        
      temp_eq = temp_eq*my_units->TemperatureConversion(*code_units);
      // ye_eq = ye_eq;
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
    
    int NucleiAbar(Real rho, Real temp, Real ye, Real& abar) {
      int ierr = WR_EoS.NucleiAbarImpl(
        rho*code_units->MassDensityConversion(*my_units),
        temp*code_units->TemperatureConversion(*my_units),
        ye,
        abar);
      return ierr;
    }

    Real AverageBaryonMass() {
      Real atomic_mass = WR_EoS.AtomicMassImpl();
      atomic_mass = atomic_mass * my_units->MassConversion(*code_units);
      return atomic_mass;
    }
    
  private:
    M1 *pm1;
    Mesh *pmy_mesh;
    MeshBlock *pmy_block;
    Coordinates *pmy_coord;
    BNSNuRatesNeutrinos::BNSNuRates *pmy_weakrates = nullptr;

    const int N_GRPS;
    const int N_SPCS;
    
    // Options for controlling weakrates opacities
    Real opacity_tau_trap;
    Real opacity_tau_delta;
    Real opacity_corr_fac_max;
    
    bool verbose_warn_weak;

    // pars for nurates (choice of reactions, quadratures)
    NuratesParams nurates_params;  
    
    // Main wrappers to bns_nurates
    //TODO inline ?

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

      //TODO: logic for nurates_params.use_dU not implemented from bns_nurates_wrap.cpp

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
        //TODO check these settings ?!
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
    //   \param[in]  nurates_params  struct for nurates parameters
    
    void NeutrinoDensity(Real mu_n, Real mu_p, Real mu_e, Real nb, Real temp, Real &n_nue,
                         Real &n_anue, Real &n_nux, Real &en_nue, Real &en_anue, Real &en_nux)
    {
      
      // NuratesParams nurates_params, RadiationM1Units units) {
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
    
  };

} // namespace M1::Opacities::BNSNuRates

#endif //M1_OPACITIES_BNSNURATES_HPP
