#ifndef M1_OPACITIES_FAKE_HPP
#define M1_OPACITIES_FAKE_HPP

// c++
// ...

// Athena++ classes headers
#include "../../m1.hpp"

// ============================================================================
namespace M1::Opacities::Fake {
// ============================================================================

class Fake {

public:
  Fake(MeshBlock *pmb, M1 * pm1, ParameterInput *pin) :
    pm1(pm1),
    pmy_mesh(pmb->pmy_mesh),
    pmy_block(pmb),
    pmy_coord(pmy_block->pcoord),
    // storage for opacities
    N_GRPS(pm1->N_GRPS),
    N_SPCS(pm1->N_SPCS),
    eta   {N_GRPS, N_SPCS},
    kap_a {N_GRPS, N_SPCS},
    kap_s {N_GRPS, N_SPCS}
  {

    const std::string option_block {"M1_opacities"};

    auto GoA_Real = [&](const std::string & name, const Real default_value)
    {
      return pin->GetOrAddReal(option_block, name, default_value);
    };

    auto GoA_Real_gs = [&](const std::string & name,
                           const int ix_g, const int ix_s,
                           const Real default_value)
    {
      std::string name_gs = name + "_" + std::to_string(ix_g) +
                                   "_" + std::to_string(ix_s);
      return pin->GetOrAddReal(option_block, name_gs, default_value);
    };


    avg_atomic_mass = GoA_Real("fake_avg_atomic_mass", 1.0);
    avg_baryon_mass = GoA_Real("fake_avg_baryon_mass", 1.0);

    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      eta(ix_g,ix_s)   = GoA_Real_gs("fake_eta",   ix_g, ix_s, 0.0);
      kap_a(ix_g,ix_s) = GoA_Real_gs("fake_kap_a", ix_g, ix_s, 0.0);
      kap_s(ix_g,ix_s) = GoA_Real_gs("fake_kap_s", ix_g, ix_s, 0.0);
    }

  };
  ~Fake() { };

  // N.B
  // Here it doesn't matter, but in general it will be faster to slice a fixed
  // choice of ix_g, ix_s and then loop over k,j,i
  inline int CalculateOpacityFake(Real const dt, AA & u)
  {
    const int NUM_COEFF = 5;
    int ierr[NUM_COEFF];

    M1_FLOOP3(k,j,i)
    if (pm1->MaskGet(k,j,i))
    {
      ierr[0] = calc_eta(pm1->radmat.sc_eta_0, k, j, i, rho, T, Y_e);
      ierr[1] = calc_eta(pm1->radmat.sc_eta,   k, j, i, rho, T, Y_e);

      ierr[2] = calc_kap_a(pm1->radmat.sc_kap_a_0, k, j, i, rho, T, Y_e);
      ierr[3] = calc_kap_a(pm1->radmat.sc_kap_a,   k, j, i, rho, T, Y_e);

      ierr[4] = calc_kap_s(pm1->radmat.sc_kap_s,   k, j, i, rho, T, Y_e);

      for (int r=0; r<NUM_COEFF; ++r)
      {
        assert(!ierr[r]);
      }
    }

    return 0;
  }

// internal methods / data ====================================================
private:
  M1 *pm1;
  Mesh *pmy_mesh;
  MeshBlock *pmy_block;
  Coordinates *pmy_coord;

  const int N_GRPS;
  const int N_SPCS;

  // Dummy variables
  const Real rho = 1.0;
  const Real T   = 0.0;
  const Real Y_e = 0.0;

  // Input populated
  AA eta;
  AA kap_a;
  AA kap_s;

  Real avg_atomic_mass;
  Real avg_baryon_mass;

  inline int calc_eta(GroupSpeciesContainer<AT_C_sca> & eta,
                      const int k, const int j, const int i,
                      const Real rho,
                      const Real T,
                      const Real Y_e)
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      eta(ix_g,ix_s)(k,j,i) = rho * this->eta(ix_g,ix_s);
    }

    return 0;
  }

  inline int calc_kap_a(GroupSpeciesContainer<AT_C_sca> & kap_a,
                        const int k, const int j, const int i,
                        const Real rho,
                        const Real T,
                        const Real Y_e)
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      kap_a(ix_g,ix_s)(k,j,i) = rho * this->kap_a(ix_g,ix_s);
    }

    return 0;
  }

  inline int calc_kap_s(GroupSpeciesContainer<AT_C_sca> & kap_s,
                        const int k, const int j, const int i,
                        const Real rho,
                        const Real T,
                        const Real Y_e)
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      kap_s(ix_g,ix_s)(k,j,i) = rho * this->kap_s(ix_g,ix_s);
    }

    return 0;
  }

};

// ============================================================================
}  // M1::Opacities
// ============================================================================

#endif // M1_OPACITIES_FAKE_HPP

