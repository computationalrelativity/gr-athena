#ifndef M1_OPACITIES_WEAKRATES_HPP
#define M1_OPACITIES_WEAKRATES_HPP

#include "../common/rates_pipeline.hpp"
#include "../common/utils.hpp"
#include "weak_rates.hpp"

namespace M1::Opacities::WeakRates
{

class WeakRates
{
  friend class ::M1::Opacities::Opacities;

  private:
  // Type aliases from OpacityUtils
  using cstate           = Common::OpacityUtils::cstate;
  using cmp_eql_dens_ini = Common::OpacityUtils::cmp_eql_dens_ini;

  public:
  // Shared utility object (public so pipeline code can access opu members)
  Common::OpacityUtils opu;

  WeakRates(MeshBlock* pmb, M1* pm1, ParameterInput* pin)
      : pm1(pm1),
        pmy_mesh(pmb->pmy_mesh),
        pmy_block(pmb),
        pmy_coord(pmy_block->pcoord),
        // opu is initialized here
        opu(pm1, pmb, pin, "(WeakRates)")
  {
#if !FLUID_ENABLED
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "M1::Opacities::WeakRates needs TEOS to work properly \n";
    }
#endif

#if !(NSCALARS > 0)
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout
        << "M1::Opacities::WeakRates needs NSCALARS>0 to work function \n";
    }
#endif

    // Weakrates only works for nu_e + nu_ae + nu_x
    assert(opu.N_SPCS == 3);

    // Weakrates only works for 1 group
    assert(opu.N_GRPS == 1);

    // logic simplified with retained eql; needs a patch otherwise:
    // TODO: BD- could be fixed to not need this (minor)
    if (!pm1->opt.retain_equilibrium)
    {
      assert(false);
    }

    // Create instance of WeakRatesNeutrinos::WeakRates
    // Set EoS from PS
    pmy_weakrates =
      new WeakRatesNeutrinos::WeakRates(opu.opt, &pmy_block->peos->GetEOS());
  };

  ~WeakRates()
  {
    if (pmy_weakrates != nullptr)
    {
      delete pmy_weakrates;
    }
  };

  // =========================================================================
  // CalculateOpacityCoefficients
  //   Calls WeakRatesNeutrinos emission/absorption/scattering routines,
  //   stores 3-species results directly in pm1->radmat.
  //
  // Input:
  //   k, j, i         grid indices
  //   rho             [code_units mass density]   (GeometricSolar)
  //   T               [MeV]
  //   Y_e             [-]                         (dimensionless)
  //
  // Output (stored in pm1->radmat, code_units):
  //   eta_0            [code_units number rate density]  (1/time/volume)
  //   eta              [code_units energy rate density]  (energy/time/volume)
  //   kap_a_0          [1/code_units length]             (number absorption)
  //   kap_a            [1/code_units length]             (energy absorption)
  //   kap_s            [1/code_units length]             (energy scattering)
  //
  //   Note: NeutrinoScatteringOpacity also returns number scattering
  //   opacities (kap_s_n), but these are discarded (not stored in radmat).
  //
  // Species / NUX convention:
  //   NUE, NUA: single species, stored directly.
  //   NUX emissivities (eta_0, eta): TOTAL for all 4 heavy-lepton species
  //     (nu_mu, nu_tau, anti-nu_mu, anti-nu_tau).  The x4 factor is applied
  //     inside the physics routines (Emissions_cgs): bremsstrahlung uses an
  //     explicit 4x factor; pair and plasmon use NUX-specific couplings that
  //     yield the 4-species total.
  //   NUX opacities (kap_a_0, kap_a, kap_s): PER-SINGLE-SPECIES.
  //     kap_a_nux = 0 always (no charged-current absorption for heavy
  //     flavors).  kap_s_nux uses flavor-blind nucleon/nucleus cross-sections.
  //
  //   Unlike BNS_NuRates, there is no 3<->4 species split/collapse here;
  //   the weak_rates.hpp physics layer handles the NUX factors internally.
  // =========================================================================
  inline int CalculateOpacityCoefficients(int k,
                                          int j,
                                          int i,
                                          Real rho,
                                          Real T,
                                          Real Y_e)
  {
    int iem, iab, isc;

    typedef M1::vars_RadMat RM;
    RM& rm = pm1->radmat;

    // Order in the arrays is: e, a, x
    const int ix_g = 0;
    iem            = pmy_weakrates->NeutrinoEmission(
      rho,
      T,
      Y_e,
      rm.sc_eta_0(ix_g, NUE)(k, j, i),  // eta_n[NUE]
      rm.sc_eta_0(ix_g, NUA)(k, j, i),  // eta_n[NUA]
      rm.sc_eta_0(ix_g, NUX)(k, j, i),  // eta_n[NUX]
      rm.sc_eta(ix_g, NUE)(k, j, i),    // eta_e[NUE]
      rm.sc_eta(ix_g, NUA)(k, j, i),    // eta_e[NUA]
      rm.sc_eta(ix_g, NUX)(k, j, i)     // eta_e[NUX]
    );

    iab = pmy_weakrates->NeutrinoAbsorptionOpacity(
      rho,
      T,
      Y_e,
      rm.sc_kap_a_0(ix_g, NUE)(k, j, i),  // kap_a_n[NUE]
      rm.sc_kap_a_0(ix_g, NUA)(k, j, i),  // kap_a_n[NUA]
      rm.sc_kap_a_0(ix_g, NUX)(k, j, i),  // kap_a_n[NUX]
      rm.sc_kap_a(ix_g, NUE)(k, j, i),    // kap_a_e[NUE]
      rm.sc_kap_a(ix_g, NUA)(k, j, i),    // kap_a_e[NUA]
      rm.sc_kap_a(ix_g, NUX)(k, j, i)     // kap_a_e[NUX]
    );

    Real kap_s_n[opu.N_SPCS];
    isc = pmy_weakrates->NeutrinoScatteringOpacity(
      rho,
      T,
      Y_e,
      kap_s_n[NUE],
      kap_s_n[NUA],
      kap_s_n[NUX],
      rm.sc_kap_s(ix_g, NUE)(k, j, i),  // kap_s_e[NUE]
      rm.sc_kap_s(ix_g, NUA)(k, j, i),  // kap_s_e[NUA]
      rm.sc_kap_s(ix_g, NUX)(k, j, i)   // kap_s_e[NUX]
    );

    int res = iem || iab || isc;
    return res;
  }

  // =========================================================================
  // ComputeEquilibriumDensities
  //   Computes thin + trapped equilibrium densities, interpolates based on
  //   tau, stores in pm1->eql.  Delegates to opu.ComputeEquilibriumDensities.
  //
  // Input:
  //   k, j, i                   grid indices
  //   dt                        [code_units time]
  //   rho                       [code_units mass density]
  //   T                         [MeV]
  //   Y_e                       [-]
  //   tau                       [code_units time]  (equilibration timescale)
  //   initial_guess             enum selecting initial state for NR solver
  //   using_averaging_fix       whether hydro averaging fallback is active
  //
  // Output (stored in pm1->eql, densitized, code_units):
  //   eql.sc_n(0, ix_s)(k,j,i)  [code_units number density x sqrt(det_g)]
  //   eql.sc_J(0, ix_s)(k,j,i)  [code_units energy density x sqrt(det_g)]
  //
  // Species / NUX convention:
  //   NUX densities are TOTAL (all 4 heavy-lepton species).
  // =========================================================================
  inline int ComputeEquilibriumDensities(
    const int k,
    const int j,
    const int i,
    const Real dt,
    const Real rho,
    const Real T,
    const Real Y_e,
    const Real tau,
    const cmp_eql_dens_ini initial_guess,
    const bool using_averaging_fix  // call with true to prevent inf. rec.
  )
  {
    auto recompute = [this](int k, int j, int i, Real rho, Real T, Real Y_e)
    { return this->CalculateOpacityCoefficients(k, j, i, rho, T, Y_e); };
    return opu.ComputeEquilibriumDensities(pmy_weakrates,
                                           k,
                                           j,
                                           i,
                                           dt,
                                           rho,
                                           T,
                                           Y_e,
                                           tau,
                                           initial_guess,
                                           using_averaging_fix,
                                           recompute);
  }

  // =========================================================================
  // CalculateOpacity - main pipeline orchestrator
  //
  // Input:
  //   dt   [code_units time]
  //   u    conserved state array (hydro + radiation)
  //
  // Delegates to the shared RatesPipeline which orchestrates:
  //   1. CalculateOpacityCoefficients (raw opacities in code_units)
  //   2. ComputeEquilibriumDensities  (eql densities in code_units)
  //   3. Kirchhoff correction
  //   4. NN fixes, equilibrium flagging, validation
  //
  // All species use the 3-species M1 convention:
  //   NUE=0, NUA=1, NUX=2.
  //   NUX emissivities/densities = TOTAL for all 4 heavy-lepton species.
  //   NUX opacities = PER-SINGLE-SPECIES.
  // =========================================================================
  inline int CalculateOpacity(Real const dt, AA& u)
  {
    return Common::RatesPipeline(dt, u, *this);
  };

  // Public so the shared pipeline (RatesPipeline) can access pm1.
  M1* pm1;

  private:
  Mesh* pmy_mesh;
  MeshBlock* pmy_block;
  Coordinates* pmy_coord;
  WeakRatesNeutrinos::WeakRates* pmy_weakrates = nullptr;

  // Species index aliases (delegate to OpacityUtils constants)
  static constexpr int NUE = Common::OpacityUtils::NUE;
  static constexpr int NUA = Common::OpacityUtils::NUA;
  static constexpr int NUX = Common::OpacityUtils::NUX;
};

}  // namespace M1::Opacities::WeakRates

#endif  // M1_OPACITIES_WEAKRATES_HPP
