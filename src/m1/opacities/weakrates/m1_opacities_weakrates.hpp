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

      // Corrections due to Kirchhoff's will replace this
      // Number emissivity
      pm1->radmat.sc_eta_0(0,0)(k,j,i) = eta_n_nue;
      pm1->radmat.sc_eta_0(0,1)(k,j,i) = eta_n_nua;
      pm1->radmat.sc_eta_0(0,2)(k,j,i) = eta_n_nux;

      // Number absorption
      pm1->radmat.sc_kap_a_0(0,0)(k,j,i) = kap_a_n_nue;
      pm1->radmat.sc_kap_a_0(0,1)(k,j,i) = kap_a_n_nua;
      pm1->radmat.sc_kap_a_0(0,2)(k,j,i) = kap_a_n_nux;

      // Energy emissivity
      pm1->radmat.sc_eta(0,0)(k,j,i) = eta_e_nue;
      pm1->radmat.sc_eta(0,1)(k,j,i) = eta_e_nua;
      pm1->radmat.sc_eta(0,2)(k,j,i) = eta_e_nux;

      // Energy absorption
      pm1->radmat.sc_kap_a(0,0)(k,j,i) = kap_a_e_nue;
      pm1->radmat.sc_kap_a(0,1)(k,j,i) = kap_a_e_nua;
      pm1->radmat.sc_kap_a(0,2)(k,j,i) = kap_a_e_nux;

      // Energy scattering
      pm1->radmat.sc_kap_s(0,0)(k,j,i) = kap_s_e_nue;
      pm1->radmat.sc_kap_s(0,1)(k,j,i) = kap_s_e_nua;
      pm1->radmat.sc_kap_s(0,2)(k,j,i) = kap_s_e_nux;
      
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

};

} // namespace M1::Opacities::WeakRates

#endif //M1_OPACITIES_WEAKRATES_HPP