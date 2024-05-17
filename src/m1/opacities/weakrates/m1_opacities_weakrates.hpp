#ifndef M1_OPACITIES_WEAKRATES_HPP
#define M1_OPACITIES_WEAKRATES_HPP

// Athena++ classes headers
#include "../../../athena.hpp"
#include "../../m1.hpp"
#include "../../../hydro/hydro.hpp"

// Weakrates header
#include "weak_rates.hpp"

namespace M1::Opacities::WeakRates {

class WeakRates {

public:
  WeakRates(MeshBlock *pmb, M1 * pm1, ParameterInput *pin) :
    pm1(pm1),
    pmy_mesh(pmb->pmy_mesh),
    pmy_block(pmb),
    pmy_coord(pmy_block->pcoord),
    N_GRPS(pm1->N_GRPS),
    N_SPCS(pm1->N_SPCS)
  {

#if !USETM
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "M1::Opacities::WeakRates needs TEOS to work properly \n";
    }
#endif

#if !(NSCALARS>0)
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "M1::Opacities::WeakRates needs NSCALARS>0 to work function \n";
    }
#endif

    // Weakrates only works for nu_e + nu_ae + nu_x
    assert(N_SPCS==3);

    // Weakrates only works for 1 group
    assert(N_GRPS==1);

    // Create instance of WeakRatesNeutrinos::WeakRates
    // Set EoS from PS
    pmy_weakrates = new WeakRatesNeutrinos::WeakRates();
    pmy_weakrates->SetEoS(&pmy_block->peos->GetEOS());

    // These are the defaults in THC, TODO convert to pin parameters
    opacity_tau_trap = 1.0;
    opacity_tau_delta = 1.0;
    opacity_corr_fac_max = 3.0;
  };

  ~WeakRates() {
    if (pmy_weakrates!= nullptr) {
      delete pmy_weakrates;
    }
  };
  // N.B
  // In general it will be faster to slice a fixed
  // choice of ix_g, ix_s and then loop over k,j,i
  inline int CalculateOpacityWeakRates(Real const dt, AA & u)
  {

    Hydro * phydro = pmy_block->phydro;
    PassiveScalars * pscalars = pmy_block->pscalars;
    AT_C_sca & sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g; // TODO is this safe to use?

    const int NUM_COEFF = 3;
    int ierr[NUM_COEFF];

    M1_FLOOP3(k,j,i)
    if (pm1->MaskGet(k,j,i))
    {
      Real rho = phydro->w(IDN, k, j, i);
      Real press = phydro->w(IPR, k, j, i);
      Real Y[MAX_SPECIES] = {0.0};
      for (int n=0; n<NSCALARS; n++) {
        Y[n] = pscalars->r(n, k, j, i);
      }
      Real T = pmy_block->peos->GetEOS().GetTemperatureFromP(rho, press, Y);
      Real Y_e = Y[0];

      Real eta_n_nue;
      Real eta_n_nua;
      Real eta_n_nux;
      Real eta_e_nue;
      Real eta_e_nua;
      Real eta_e_nux;

      ierr[0] = pmy_weakrates->NeutrinoEmission(rho, T, Y_e, eta_n_nue, eta_n_nua, eta_n_nux, eta_e_nue, eta_e_nua, eta_n_nux);

      Real kap_a_n_nue;
      Real kap_a_n_nua;
      Real kap_a_n_nux;
      Real kap_a_e_nue;
      Real kap_a_e_nua;
      Real kap_a_e_nux;

      ierr[1] = pmy_weakrates->NeutrinoAbsorptionOpacity(rho, T, Y_e, kap_a_n_nue, kap_a_n_nua, kap_a_n_nux, kap_a_e_nue, kap_a_e_nua, kap_a_e_nux);

      Real kap_s_n_nue;
      Real kap_s_n_nua;
      Real kap_s_n_nux;
      Real kap_s_e_nue;
      Real kap_s_e_nua;
      Real kap_s_e_nux;

      ierr[2] = pmy_weakrates->NeutrinoScatteringOpacity(rho, T, Y_e, kap_s_n_nue, kap_s_n_nua, kap_s_n_nux, kap_s_e_nue, kap_s_e_nua, kap_s_e_nux);

      for (int r=0; r<NUM_COEFF; ++r)
      {
        assert(!ierr[r]);
      }

      Real tau = std::min(std::sqrt(kap_a_e_nue*(kap_a_e_nue + kap_s_e_nue)), std::sqrt(kap_a_e_nua*(kap_a_e_nua + kap_s_e_nua)))*dt;
      
      // Calculate equilibrium blackbody functions with trapped neutrinos
      Real dens_n_trap[3];
      Real dens_e_trap[3];
      if (opacity_tau_trap >= 0.0 && tau > opacity_tau_trap) {
        Real T_star;
        Real Y_e_star;
        Real dens_n[3];
        Real dens_e[3];

        // TODO undensitise
        Real invsdetg = 1.0/sc_sqrt_det_g(k, j, i);
        // FF number density
        dens_n[0] = pm1->rad.sc_nnu(0,0)(k, j, i)*invsdetg;
        dens_n[1] = pm1->rad.sc_nnu(0,1)(k, j, i)*invsdetg;
        dens_n[2] = pm1->rad.sc_nnu(0,2)(k, j, i)*invsdetg;
        
        // FF energy density
        dens_e[0] = pm1->rad.sc_J(0,0)(k, j, i)*invsdetg;
        dens_e[1] = pm1->rad.sc_J(0,1)(k, j, i)*invsdetg;
        dens_e[2] = pm1->rad.sc_J(0,2)(k, j, i)*invsdetg;
        
        // Calculate equilibriated state
        ierr[0] = pmy_weakrates->WeakEquilibrium(rho, T, Y_e, dens_n[0], dens_n[1], dens_n[2], dens_e[0], dens_e[1], dens_e[2], T_star, Y_e_star, dens_n_trap[0], dens_n_trap[1], dens_n_trap[2], dens_e_trap[0], dens_e_trap[1], dens_e_trap[2]);

        // If we can't get equilibrium, try again but ignore current neutrino data
        if (ierr[0]) {
          ierr[1] = pmy_weakrates->WeakEquilibrium(rho, T, Y_e, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, T_star, Y_e_star, dens_n_trap[0], dens_n_trap[1], dens_n_trap[2], dens_e_trap[0], dens_e_trap[1], dens_e_trap[2]);

          // TODO THC WeakRates treats this as a warning
          assert(!ierr[1]);
        }
      }

      // Calculate equilibrium blackbody functions with fixed T, Ye
      Real dens_n_thin[3];
      Real dens_e_thin[3];
      ierr[0] = pmy_weakrates->NeutrinoDensity(rho, T, Y_e, dens_n_thin[0], dens_n_thin[1], dens_n_thin[2], dens_e_thin[0], dens_e_thin[1], dens_e_thin[2]);
      assert(!ierr[0]);
      
      // Set the black body function
      Real dens_n[3];
      Real dens_e[3];
      if (opacity_tau_trap < 0 || tau <= opacity_tau_trap) {
        dens_n[0] = dens_n_thin[0];
        dens_n[1] = dens_n_thin[1];
        dens_n[2] = dens_n_thin[2];

        dens_e[0] = dens_e_thin[0];
        dens_e[1] = dens_e_thin[1];
        dens_e[2] = dens_e_thin[2];
      } else if (tau > opacity_tau_trap + opacity_tau_delta) {
        dens_n[0] = dens_n_trap[0];
        dens_n[1] = dens_n_trap[1];
        dens_n[2] = dens_n_trap[2];

        dens_e[0] = dens_e_trap[0];
        dens_e[1] = dens_e_trap[1];
        dens_e[2] = dens_e_trap[2];
      } else {
        Real lam = (tau - opacity_tau_trap)/opacity_tau_delta;

        dens_n[0] = lam*dens_n_trap[0] + (1-lam)*dens_n_thin[0];
        dens_n[1] = lam*dens_n_trap[1] + (1-lam)*dens_n_thin[1];
        dens_n[2] = lam*dens_n_trap[2] + (1-lam)*dens_n_thin[2];

        dens_e[0] = lam*dens_e_trap[0] + (1-lam)*dens_e_thin[0];
        dens_e[1] = lam*dens_e_trap[1] + (1-lam)*dens_e_thin[1];
        dens_e[2] = lam*dens_e_trap[2] + (1-lam)*dens_e_thin[2];
      }

      // Calculate correction factors
      Real e_ave[3];
      Real corr_fac[3];
      for (int s_idx=0; s_idx<N_SPCS; ++s_idx) {
        pm1->radmat.sc_avg_nrg(0,s_idx)(k, j, i) = dens_e[s_idx]/dens_n[s_idx];
        corr_fac[s_idx] = pm1->rad.sc_J(0,s_idx)(k, j, i) / (pm1->rad.sc_nnu(0,s_idx)(k, j, i)*pm1->radmat.sc_avg_nrg(0,s_idx)(k, j, i));

        if (!std::isfinite(corr_fac[s_idx])) {
          corr_fac[s_idx] = 1.0;
        }

        corr_fac[s_idx] *= corr_fac[s_idx];
        corr_fac[s_idx] = std::max(1.0/opacity_corr_fac_max, std::min(corr_fac[s_idx], opacity_corr_fac_max));
      }

      // Energy scattering
      pm1->radmat.sc_kap_s(0,0)(k,j,i) = corr_fac[0]*kap_s_e_nue;
      pm1->radmat.sc_kap_s(0,1)(k,j,i) = corr_fac[1]*kap_s_e_nua;
      pm1->radmat.sc_kap_s(0,2)(k,j,i) = corr_fac[2]*kap_s_e_nux;

      // Enforce Kirchhoff's law
      // For electron lepton neutrinos we change the opacity
      // For heavy lepton neutrinos we change the emissivity
      // TODO Floors for dens_n and dens_e?

      // Electron neutrinos
      pm1->radmat.sc_eta_0(0,0)(k,j,i)   = corr_fac[0]*eta_n_nue;
      pm1->radmat.sc_eta(0,0)(k,j,i)     = corr_fac[0]*eta_e_nue;
      
      pm1->radmat.sc_kap_a_0(0,0)(k,j,i) = (dens_n[0] > 0.0 ? pm1->radmat.sc_eta_0(0,0)(k,j,i)/dens_n[0] : 0.0);
      pm1->radmat.sc_kap_a(0,0)(k,j,i)   = (dens_e[0] > 0.0 ? pm1->radmat.sc_eta(0,0)(k,j,i)/dens_e[0]   : 0.0);
      
      // Electron anti-neutrinos
      pm1->radmat.sc_eta_0(0,1)(k,j,i)   = corr_fac[1]*eta_n_nua;
      pm1->radmat.sc_eta(0,1)(k,j,i)     = corr_fac[1]*eta_e_nua;
      
      pm1->radmat.sc_kap_a_0(0,1)(k,j,i) = (dens_n[1] > 0.0 ? pm1->radmat.sc_eta_0(0,1)(k,j,i)/dens_n[1] : 0.0);
      pm1->radmat.sc_kap_a(0,1)(k,j,i)   = (dens_e[1] > 0.0 ? pm1->radmat.sc_eta(0,1)(k,j,i)/dens_e[1]   : 0.0);
      
      // Heavy lepton neutrinos
      pm1->radmat.sc_kap_a_0(0,1)(k,j,i) = corr_fac[2]*kap_a_n_nux;
      pm1->radmat.sc_kap_a(0,1)(k,j,i)   = corr_fac[2]*kap_a_e_nux;
      
      pm1->radmat.sc_eta_0(0,2)(k,j,i)   = pm1->radmat.sc_kap_a_0(0,1)(k,j,i)*dens_n[2];
      pm1->radmat.sc_eta(0,2)(k,j,i)     = pm1->radmat.sc_kap_a(0,1)(k,j,i)*dens_e[2];
      
    }

    return 0;
  };

private:
  M1 *pm1;
  Mesh *pmy_mesh;
  MeshBlock *pmy_block;
  Coordinates *pmy_coord;
  WeakRatesNeutrinos::WeakRates *pmy_weakrates = nullptr;

  const int N_GRPS;
  const int N_SPCS;

  // Options for controlling weakrates opacities
  Real opacity_tau_trap;
  Real opacity_tau_delta;
  Real opacity_corr_fac_max;

};

} // namespace M1::Opacities::WeakRates

#endif //M1_OPACITIES_WEAKRATES_HPP