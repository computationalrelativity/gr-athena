#ifndef M1_OPACITIES_BNS_NURATES_HPP_
#define M1_OPACITIES_BNS_NURATES_HPP_

// BNSNuRates refactored headers
#include "../common/eos.hpp"
#include "../common/rates_pipeline.hpp"
#include "../common/utils.hpp"
#include "../common/weak_equilibrium.hpp"
#include "bnsnu_wrapper.hpp"

namespace M1::Opacities::BNS_NuRates
{

namespace Units = ::M1::Opacities::Common::Units;

class BNSNuRates
{
  friend class ::M1::Opacities::Opacities;

  private:
  // Type aliases from OpacityUtils
  using cstate           = Common::OpacityUtils::cstate;
  using cmp_eql_dens_ini = Common::OpacityUtils::cmp_eql_dens_ini;

  public:
  // Shared utility object (public so pipeline code can access opu members)
  Common::OpacityUtils opu;

  BNSNuRates(MeshBlock* pmb, M1* pm1, ParameterInput* pin)
      : pm1(pm1),
        pmy_mesh(pmb->pmy_mesh),
        pmy_block(pmb),
        pmy_coord(pmy_block->pcoord),
        // opu is initialized here
        opu(pm1, pmb, pin, "(BNSNuRates)")
  {
#if !USETM
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "M1::Opacities::BNS_NuRates needs TEOS to work properly \n";
    }
#endif

#if !(NSCALARS > 0)
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout
        << "M1::Opacities::BNS_NuRates needs NSCALARS>0 to work function \n";
    }
#endif

    // BNSNuRates only works for nu_e + nu_ae + nu_x
    assert(opu.N_SPCS == 3);

    // BNSNuRates only works for 1 group
    assert(opu.N_GRPS == 1);

    // logic simplified with retained eql; needs a patch otherwise
    if (!pm1->opt.retain_equilibrium)
    {
      assert(false);
    }

    // Create sub-objects
    pmy_eos     = new Common::EoS::EoSWrapper(opu.opt, &(pmb->peos->GetEOS()));
    pmy_nurates = new BNSNu_Wrapper::BNSNuRatesWrapper(pmy_eos, pin);

    // Cache unit pointers
    code_units = pmy_eos->GetCodeUnits();
    wr_units   = pmy_eos->GetWrUnits();
    eos_units  = pmy_eos->GetEosUnits();

    // Initialize common equilibrium solver (replaces BNSNuEquilibrium)
    solver_.Initialize(pmy_eos);
  };

  ~BNSNuRates()
  {
    if (pmy_nurates != nullptr)
      delete pmy_nurates;
    if (pmy_eos != nullptr)
      delete pmy_eos;
  };

  // =========================================================================
  // NeutrinoDensity - public interface needed by m1_opacities.hpp
  //   Converts code->CGS, calls solver, converts CGS->code.
  //
  // Input:
  //   rho   [code_units mass density]   (GeometricSolar)
  //   T     [MeV]
  //   Y_e   [-]                         (dimensionless)
  //
  // Output:
  //   n_nue, n_anue, n_nux  [code_units number density]
  //   e_nue, e_anue, e_nux  [code_units energy density]
  //
  // Species / NUX convention:
  //   NUE, NUA are single species.
  //   NUX outputs are TOTAL for all 4 heavy-lepton species (nu_mu, nu_tau,
  //   anti-nu_mu, anti-nu_tau).  The x4 factor is applied inside the
  //   common solver (NeutrinoDensity_cgs_erg).
  // =========================================================================
  int NeutrinoDensity(Real rho,
                      Real T,
                      Real Y_e,
                      Real& n_nue,
                      Real& n_anue,
                      Real& n_nux,
                      Real& e_nue,
                      Real& e_anue,
                      Real& e_nux)
  {
    // Convert mass density from code units to CGS [g/cm^3]
    const Real rho_cgs = rho * code_units->MassDensityConversion(*wr_units);
    // Temperature is already in MeV in both unit systems

    // Delegate to common solver (returns n in cm^-3, e in erg/cm^3)
    int ierr = solver_.NeutrinoDensity_cgs_erg(
      rho_cgs, T, Y_e, n_nue, n_anue, n_nux, e_nue, e_anue, e_nux);

    // Convert outputs from CGS to code units
    //   NeutrinoDensity_cgs_erg already applied mev_to_erg for energy
    const Real n_conv = wr_units->NumberDensityConversion(*code_units);
    const Real e_conv = wr_units->EnergyDensityConversion(*code_units);

    n_nue *= n_conv;
    n_anue *= n_conv;
    n_nux *= n_conv;
    e_nue *= e_conv;
    e_anue *= e_conv;
    e_nux *= e_conv;

    return ierr;
  }

  // =========================================================================
  // WeakEquilibrium - public interface needed by ComputeEquilibriumDensities
  //   Converts code->CGS, calls solver, converts CGS->code.
  //
  // Input:
  //   rho             [code_units mass density]   (GeometricSolar)
  //   temp            [MeV]
  //   ye              [-]
  //   n_nue, n_nua    [code_units number density]  (single species each)
  //   n_nux           [code_units number density]  TOTAL (all 4 heavy-lepton)
  //   e_nue, e_nua    [code_units energy density]  (single species each)
  //   e_nux           [code_units energy density]  TOTAL (all 4 heavy-lepton)
  //
  // Output:
  //   temp_eq         [MeV]
  //   ye_eq           [-]
  //   n_*_eq          [code_units number density]
  //   e_*_eq          [code_units energy density]
  //
  // Species / NUX convention:
  //   NUX inputs and outputs are TOTAL for all 4 heavy-lepton species.
  //   The solver internally divides by 4 for its single-species treatment,
  //   then multiplies back by 4 on output.
  // =========================================================================
  int WeakEquilibrium(Real rho,
                      Real temp,
                      Real ye,
                      Real n_nue,
                      Real n_nua,
                      Real n_nux,
                      Real e_nue,
                      Real e_nua,
                      Real e_nux,
                      Real& temp_eq,
                      Real& ye_eq,
                      Real& n_nue_eq,
                      Real& n_nua_eq,
                      Real& n_nux_eq,
                      Real& e_nue_eq,
                      Real& e_nua_eq,
                      Real& e_nux_eq)
  {
    const Real n_conv = code_units->NumberDensityConversion(*wr_units);
    const Real e_conv = code_units->EnergyDensityConversion(*wr_units);
    const Real T_conv = code_units->TemperatureConversion(*wr_units);

    int ierr = solver_.WeakEquilibrium_cgs(
      rho * code_units->MassDensityConversion(*wr_units),
      temp * T_conv,
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

    // Convert back to code units
    temp_eq *= wr_units->TemperatureConversion(*code_units);

    const Real n_conv_back = wr_units->NumberDensityConversion(*code_units);
    const Real e_conv_back = wr_units->EnergyDensityConversion(*code_units);
    n_nue_eq *= n_conv_back;
    n_nua_eq *= n_conv_back;
    n_nux_eq *= n_conv_back;
    e_nue_eq *= e_conv_back;
    e_nua_eq *= e_conv_back;
    e_nux_eq *= e_conv_back;

    return ierr;
  }

  // =========================================================================
  // CalculateOpacityCoefficients
  //   Calls external library via BNSNuRatesWrapper, stores 3-species results.
  //
  // Input:
  //   k, j, i         grid indices
  //   rho             [code_units mass density]   (GeometricSolar)
  //   T               [MeV]
  //   Y_e             [-]
  //
  // Internal workflow:
  //   1. Reads densitized neutrino fields from pm1->rad, undensitizes.
  //   2. Performs 3->4 species split for NUX:
  //        nudens[2] *= 0.25  (total -> per single heavy-lepton species)
  //        nudens[3]  = nudens[2]  (duplicate: nux = anux)
  //   3. Calls external library (4-species, per-single-species convention).
  //   4. Collapses 4->3 species for NUX:
  //        Emissivities (eta_0, eta):
  //          eta_nux = 2*eta[2] + 2*eta[3]  -> TOTAL for all 4 heavy-lepton
  //        Opacities (kap_a_0, kap_a, kap_s):
  //          kap_nux = (kap[2] + kap[3]) / 2 -> AVERAGED over nux & anux
  //
  // Output (stored in pm1->radmat, code_units):
  //   eta_0            [code_units number rate density]  (1/time/volume)
  //   eta              [code_units energy rate density]  (energy/time/volume)
  //   kap_a_0          [1/code_units length]             (number absorption)
  //   kap_a            [1/code_units length]             (energy absorption)
  //   kap_s            [1/code_units length]             (scattering)
  //
  // Species / NUX convention:
  //   NUE, NUA: single species, stored directly.
  //   NUX emissivities: TOTAL (all 4 heavy-lepton species summed).
  //   NUX opacities: AVERAGED over nux and anux (representative value for
  //     the combined NUX radiation field).
  // =========================================================================
  inline int CalculateOpacityCoefficients(int k,
                                          int j,
                                          int i,
                                          Real rho,
                                          Real T,
                                          Real Y_e)
  {
    typedef M1::vars_RadMat RM;
    RM& rm = pm1->radmat;

    const Real mb = pmy_eos->GetRawBaryonMass();  // baryon mass in code units
    const Real nb = rho / mb;

    // Chemical potentials (code units input, MeV output)
    Real nb_eos = nb * code_units->NumberDensityConversion(*eos_units);
    Real mu_n, mu_p, mu_e;
    pmy_eos->ChemicalPotentials_npe(nb_eos, T, Y_e, mu_n, mu_p, mu_e);

    // Undensitized neutrino quantities
    Real invsdetg    = pm1->geom.sc_oo_sqrt_det_g(k, j, i);
    Real nudens_0[4] = { 0. };
    Real nudens_1[4] = { 0. };
    Real chi_loc[4]  = { 0. };

    for (int nuidx = 0; nuidx < opu.N_SPCS; ++nuidx)
    {
      nudens_0[nuidx] = pm1->rad.sc_n(0, nuidx)(k, j, i) * invsdetg;
      nudens_1[nuidx] = pm1->rad.sc_J(0, nuidx)(k, j, i) * invsdetg;
      chi_loc[nuidx]  = pm1->lab_aux.sc_chi(0, nuidx)(k, j, i);
    }

    // Split 3->4 species: nux -> single heavy species
    // M1 transport carries NUX as total (all 4 heavy-lepton); the external
    // library expects per-single-species, so divide by 4 and duplicate.
    nudens_0[2] = 0.25 * nudens_0[2];  // total -> per single species
    nudens_1[2] = 0.25 * nudens_1[2];  // total -> per single species
    nudens_0[3] = nudens_0[2];         // anux = nux (symmetric)
    nudens_1[3] = nudens_1[2];         // anux = nux (symmetric)
    chi_loc[3]  = chi_loc[2];

    // Call external library (4-species output)
    Real eta_0[4]{}, eta_1[4]{};
    Real abs_0[4]{}, abs_1[4]{};
    Real scat[4]{};

    int opac_err = pmy_nurates->ComputeOpacities(nb,
                                                 T,
                                                 Y_e,
                                                 mu_n,
                                                 mu_p,
                                                 mu_e,
                                                 nudens_0[0],
                                                 nudens_1[0],
                                                 chi_loc[0],
                                                 nudens_0[1],
                                                 nudens_1[1],
                                                 chi_loc[1],
                                                 nudens_0[2],
                                                 nudens_1[2],
                                                 chi_loc[2],
                                                 nudens_0[3],
                                                 nudens_1[3],
                                                 chi_loc[3],
                                                 eta_0,
                                                 eta_1,
                                                 abs_0,
                                                 abs_1,
                                                 scat);

    if (opac_err)
    {
      return OPAC_NONFINITE;
    }

    // Store 3-species in radmat (before collapse for nux)
    const int ix_g = 0;
    for (int nuidx = 0; nuidx < opu.N_SPCS; ++nuidx)
    {
      rm.sc_eta_0(ix_g, nuidx)(k, j, i)   = eta_0[nuidx];
      rm.sc_eta(ix_g, nuidx)(k, j, i)     = eta_1[nuidx];
      rm.sc_kap_a_0(ix_g, nuidx)(k, j, i) = abs_0[nuidx];
      rm.sc_kap_a(ix_g, nuidx)(k, j, i)   = abs_1[nuidx];
      rm.sc_kap_s(ix_g, nuidx)(k, j, i)   = scat[nuidx];
    }

    // Collapse 4->3 species for heavy neutrinos
    //
    // Emissivities: sum all 4 heavy flavors to get TOTAL NUX emissivity.
    //   eta_nux = 2 * eta[2](nux) + 2 * eta[3](anux)
    //   The factor 2 accounts for mu + tau generations per
    //   particle/antiparticle.
    rm.sc_eta_0(ix_g, NUX)(k, j, i) *= 2.;
    rm.sc_eta_0(ix_g, NUX)(k, j, i) += 2. * eta_0[3];
    rm.sc_eta(ix_g, NUX)(k, j, i) *= 2.;
    rm.sc_eta(ix_g, NUX)(k, j, i) += 2. * eta_1[3];

    // Opacities: average nux and anux to get representative AVERAGED opacity.
    //   kap_nux = (kap[2](nux) + kap[3](anux)) / 2
    //   Averaging is appropriate because the opacity acts on the total NUX
    //   field.
    rm.sc_kap_a_0(ix_g, NUX)(k, j, i) += abs_0[3];
    rm.sc_kap_a_0(ix_g, NUX)(k, j, i) *= 0.5;
    rm.sc_kap_a(ix_g, NUX)(k, j, i) += abs_1[3];
    rm.sc_kap_a(ix_g, NUX)(k, j, i) *= 0.5;
    rm.sc_kap_s(ix_g, NUX)(k, j, i) += scat[3];
    rm.sc_kap_s(ix_g, NUX)(k, j, i) *= 0.5;

    return 0;
  }

  // =========================================================================
  // ComputeEquilibriumDensities
  //   Mirrors weakrates: computes thin + trapped equilibrium densities,
  //   interpolates based on tau, stores in pm1->eql.
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
  //   NUX densities throughout are TOTAL (all 4 heavy-lepton species).
  //   Passed to WeakEquilibrium / NeutrinoDensity which also use the TOTAL
  //   convention.
  // =========================================================================
  inline int ComputeEquilibriumDensities(const int k,
                                         const int j,
                                         const int i,
                                         const Real dt,
                                         const Real rho,
                                         const Real T,
                                         const Real Y_e,
                                         const Real tau,
                                         const cmp_eql_dens_ini initial_guess,
                                         const bool using_averaging_fix)
  {
    auto recompute = [this](int k, int j, int i, Real rho, Real T, Real Y_e)
    {
      return this->CalculateOpacityCoefficients(k, j, i, rho, T, Y_e);
    };

    return opu.ComputeEquilibriumDensities(this,
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
  //   NUE=0, NUA=1, NUX=2 (NUX = total for all 4 heavy-lepton species).
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

  // Sub-objects
  Common::EoS::EoSWrapper* pmy_eos              = nullptr;
  BNSNu_Wrapper::BNSNuRatesWrapper* pmy_nurates = nullptr;

  // Equilibrium solver
  Common::WeakEquilibrium::WeakEquilibriumSolver<Common::EoS::EoSWrapper>
    solver_;

  // Cached unit pointers
  Units::UnitSystem* code_units;
  Units::UnitSystem* wr_units;
  Units::UnitSystem* eos_units;

  // Convenience aliases (from opu)
  static constexpr int NUE = Common::OpacityUtils::NUE;
  static constexpr int NUA = Common::OpacityUtils::NUA;
  static constexpr int NUX = Common::OpacityUtils::NUX;
};

}  // namespace M1::Opacities::BNS_NuRates

#endif  // M1_OPACITIES_BNS_NURATES_HPP_
