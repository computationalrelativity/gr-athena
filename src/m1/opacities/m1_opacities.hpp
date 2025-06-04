#ifndef M1_OPACITIES_HPP
#define M1_OPACITIES_HPP

// c++
// ...

// Athena++ classes headers
#include "../m1.hpp"
#include "../m1_macro.hpp"
#include "fake/m1_opacities_fake.hpp"
#include "photon/m1_opacities_photon.hpp"

#if (M1_WEAKRATES)
#include "weakrates/m1_opacities_weakrates.hpp"
#elif (M1_BNSNURATES)
#include "bnsnurates/bnsnurates.hpp"
#endif

// ============================================================================
namespace M1::Opacities {
// ============================================================================

class Opacities
{
// methods ====================================================================
public:
  Opacities(MeshBlock *pmb, M1 * pm1, ParameterInput *pin) :
    popac(this),
    pm1(pm1),
    pmy_mesh(pmb->pmy_mesh),
    pmy_block(pmb),
    pmy_coord(pmy_block->pcoord)
  {
    PopulateOptionsOpacities(pin);
  };
  ~Opacities()
  {
    if (popac_fake != nullptr)
      delete popac_fake;

    if (popac_photon != nullptr)
      delete popac_photon;

#if (M1_WEAKRATES)
    if (popac_weakrates != nullptr)
      delete popac_weakrates;
#elif (M1_BNSNURATES)
    if (popac_bnsnurates != nullptr)
      delete popac_bnsnurates;
#endif
  };

  // handler
  void CalcOpacity(Real const dt, AA & u)
  {
    switch (opt.opacity_variety)
    {
      case (opt_opacity_variety::none):
      {
        return;
      }
      case (opt_opacity_variety::zero):
      {
        CalculateOpacityZero(dt, u);
        break;
      }
      case (opt_opacity_variety::fake):
      {
        popac_fake->CalculateOpacityFake(dt, u);
        break;
      }
      case (opt_opacity_variety::photon):
      {
        popac_photon->CalculateOpacityPhoton(dt, u);
        break;
      }
#if (M1_WEAKRATES)
      case (opt_opacity_variety::weakrates):
      {
        popac_weakrates->CalculateOpacityWeakRates(dt, u);
        break;
      }
#elif (M1_BNSNURATES)
    case (opt_opacity_variety::bnsnurates):
      {
        popac_bnsnurates->CalculateOpacityBNSNuRates(dt, u);
        break;
      }
#endif
      default:
      {
        assert(0);
        std::exit(0);
      }
    }
  }

  // Photon:
  // [[n_nue, n_nua, n_nux, n_nux]
  //  [e_nue, e_nua, e_nux, e_nux]]
  //
  // WeakRates:
  // [[n_nue, n_nua, n_nux, n_nux]
  //  [e_nue, e_nua, e_nux, e_nux]]
  void CalculateEquilibriumDensity(
    const Real w_rho,
    const Real w_T,
    const Real w_Y_e,
    AA & nudens)
  {
    switch (opt.opacity_variety)
    {
      case opt_opacity_variety::photon:
      {
        nudens(1, 0) = popac_photon->black_body(w_T);
        break;
      }
#if (M1_WEAKRATES)
      case opt_opacity_variety::weakrates:
      {
        // For 4 species neutrino transport we need to divide the nux luminosity
        // by 2
        const Real nux_weight = (pm1->N_SPCS == 3 ? 1.0 : 0.5);

        const int ierr = popac_weakrates->pmy_weakrates->NeutrinoDensity( 
          w_rho,        // Real rho,
          w_T,          // Real temp,
          w_Y_e,        // Real ye,
          nudens(0, 0), // Real &n_nue,
          nudens(0, 1), // Real &n_nua,
          nudens(0, 2), // Real &n_nux,
          nudens(1, 0), // Real &e_nue,
          nudens(1, 1), // Real &e_nua,
          nudens(1, 2)  // Real &e_nux
        );

        assert(!ierr);

        nudens(0,2) *= nux_weight;
        nudens(1,2) *= nux_weight;

        nudens(0,3) = nudens(0,2);
        nudens(1,3) = nudens(1,2);

        break;
      }
#elif (M1_BNSNURATES)
    case opt_opacity_variety::bnsnurates:
      {
        // For 4 species neutrino transport we need to divide the nux luminosity
        // by 2
        const Real nux_weight = (pm1->N_SPCS == 3 ? 1.0 : 0.5);

        const int ierr = popac_bnsnurates->NeutrinoDensity( //TODO fix
          w_rho,        // Real rho,
          w_T,          // Real temp,
          w_Y_e,        // Real ye,
          nudens(0, 0), // Real &n_nue,
          nudens(0, 1), // Real &n_nua,
          nudens(0, 2), // Real &n_nux,
          nudens(1, 0), // Real &e_nue,
          nudens(1, 1), // Real &e_nua,
          nudens(1, 2)  // Real &e_nux
        );

        assert(!ierr);

        nudens(0,2) *= nux_weight;
        nudens(1,2) *= nux_weight;

        nudens(0,3) = nudens(0,2);
        nudens(1,3) = nudens(1,2);

        break;
      }

#endif  // (M1_WEAKRATES)
      default:
      {
        assert(false);
      }
    }
  }

// internal methods / data ====================================================
private:
  Opacities *popac;
  M1 *pm1;
  Mesh *pmy_mesh;
  MeshBlock *pmy_block;
  Coordinates *pmy_coord;

  // some varieties have their own classes ------------------------------------
  Fake::Fake           * popac_fake   = nullptr;
  Photon::Photon       * popac_photon = nullptr;
#if (M1_WEAKRATES)
  WeakRates::WeakRates * popac_weakrates = nullptr;
#elif (M1_BNSNURATES)
  BNSNuRates::BNSNuRates * popac_bnsnurates = nullptr;
#endif
  // --------------------------------------------------------------------------

  void PopulateOptionsOpacities(ParameterInput *pin)
  {
    std::string tmp;
    std::ostringstream msg;

    { // opacities
      tmp = pin->GetOrAddString("M1_opacities", "variety", "none");
      if (tmp == "none")
      {
        opt.opacity_variety = opt_opacity_variety::none;
      }
      else if (tmp == "zero")
      {
        opt.opacity_variety = opt_opacity_variety::zero;
      }
      else if (tmp == "fake")
      {
        opt.opacity_variety = opt_opacity_variety::fake;
        popac_fake = new Fake::Fake(pmy_block, pm1, pin);
      }
      else if (tmp == "photon")
      {
        opt.opacity_variety = opt_opacity_variety::photon;
        popac_photon = new Photon::Photon(pmy_block, pm1, pin);
      }
#if (M1_WEAKRATES)
      else if (tmp == "weakrates")
      {
        opt.opacity_variety = opt_opacity_variety::weakrates;
        popac_weakrates = new WeakRates::WeakRates(pmy_block, pm1, pin);
      }
#elif (M1_BNSNURATES)
      else if (tmp == "bnsnurates")
      {
        opt.opacity_variety = opt_opacity_variety::bnsnurates;
        popac_bnsnurates = new BNSNuRates::BNSNuRates(pmy_block, pm1, pin);
      }
#endif
      else
      {
        msg << "M1_opacities/variety unknown " << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }

  inline int CalculateOpacityZero(Real const dt, AA & u)
  {
    for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
    {
      pm1->radmat.sc_eta_0(  ix_g,ix_s).ZeroClear();
      pm1->radmat.sc_kap_a_0(ix_g,ix_s).ZeroClear();

      pm1->radmat.sc_eta(  ix_g,ix_s).ZeroClear();
      pm1->radmat.sc_kap_a(ix_g,ix_s).ZeroClear();
      pm1->radmat.sc_kap_s(ix_g,ix_s).ZeroClear();
    }

    return 0;
  }

// configuration ==============================================================
public:
  enum class opt_opacity_variety { none, zero, fake, photon, weakrates, bnsnurates };

  struct
  {
    opt_opacity_variety opacity_variety;
  } opt;

};

// ============================================================================
}  // M1::Opacities
// ============================================================================

#endif // M1_OPACITIES_HPP

