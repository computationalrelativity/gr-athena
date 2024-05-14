#ifndef M1_OPACITIES_HPP
#define M1_OPACITIES_HPP

// c++
// ...

// Athena++ classes headers
#include "../m1.hpp"
#include "../m1_macro.hpp"
#include "fake/m1_opacities_fake.hpp"
#include "photon/m1_opacities_photon.hpp"
#include "weakrates/m1_opacities_weakrates.hpp"

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

    if (popac_weakrates != nullptr)
      delete popac_weakrates;
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
      case (opt_opacity_variety::weakrates):
      {
        popac_weakrates->CalculateOpacityWeakRates(dt, u);
        break;
      }
      default:
      {
        assert(0);
        std::exit(0);
      }
    }
  };

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
  WeakRates::WeakRates * popac_weakrates = nullptr;
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
      else if (tmp == "weakrates")
      {
        opt.opacity_variety = opt_opacity_variety::weakrates;
        popac_weakrates = new WeakRates::WeakRates(pmy_block, pm1, pin);
      }
      else
      {
        msg << "M1_opacities/variety unknown" << std::endl;
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
private:
  enum class opt_opacity_variety { none, zero, fake, photon, weakrates };

  struct
  {
    opt_opacity_variety opacity_variety;
  } opt;

};

// ============================================================================
}  // M1::Opacities
// ============================================================================

#endif // M1_OPACITIES_HPP

