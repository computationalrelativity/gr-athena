#ifndef M1_OPACITIES_PHOTON_HPP
#define M1_OPACITIES_PHOTON_HPP

// c++
// ...

// Athena++ classes headers
#include "../../m1.hpp"
#include "../../../hydro/hydro.hpp"
#include "../../../eos/eos.hpp"

// ============================================================================
namespace M1::Opacities::Photon {
// ============================================================================

class Photon {

friend class ::M1::Opacities::Opacities;

public:
  Photon(MeshBlock *pmb, M1 * pm1, ParameterInput *pin) :
    pm1(pm1),
    pmy_mesh(pmb->pmy_mesh),
    pmy_block(pmb),
    pmy_coord(pmy_block->pcoord),
    // storage for opacities
    N_GRPS(pm1->N_GRPS),
    N_SPCS(pm1->N_SPCS)
  {

    // Implementation is restricted
    assert(N_GRPS == 1);
    assert(N_SPCS == 1);

#if !USETM
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "Opacities::Photon needs TEOS to work properly \n";
    }
#endif

#if !(NSCALARS>0)
    #pragma omp critical
    {
      std::cout << "Warning: ";
      std::cout << "Opacities::Photon needs NSCALARS>0 to work function \n";
    }
#endif

    const std::string option_block {"M1_opacities"};

    auto GoA_Real = [&](const std::string & name, const Real default_value)
    {
      return pin->GetOrAddReal(option_block, name, default_value);
    };

    rad_constant = GoA_Real("photon_rad_constant", 2.471313401078565e-13);

    kap_a = GoA_Real("photon_kap_a", 1.0);
    kap_s = GoA_Real("photon_kap_s", 1.0);
  };
  ~Photon() { };

  inline int CalculateOpacityPhoton(Real const dt, AA & u)
  {
    const int NUM_COEFF = 2;
    int ierr[NUM_COEFF];

    const int ix_g = 0;
    const int ix_s = 0;

    AT_C_sca & sc_eta   = pm1->radmat.sc_eta(  ix_g,ix_s);
    AT_C_sca & sc_kap_a = pm1->radmat.sc_kap_a(ix_g,ix_s);
    AT_C_sca & sc_kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s);

    M1_FLOOP3(k,j,i)
    if (pm1->MaskGet(k,j,i))
    {

// TODO: check
#if USETM
      const Real rho = 1.0;
      const Real T   = 0.0;
      const Real Y_e = 0.0;
#else
      const Real rho = 1.0;
      const Real T   = 0.0;
      const Real Y_e = 0.0;
#endif

      ierr[0] = calc_kap_a(sc_kap_a, k, j, i, rho, T, Y_e);
      ierr[1] = calc_kap_s(sc_kap_s, k, j, i, rho, T, Y_e);

      for (int r=0; r<NUM_COEFF; ++r)
      {
        assert(!ierr[r]);
      }

      sc_eta(k,j,i) = sc_kap_a(k,j,i) * black_body(T);
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

  // Input populated
  Real rad_constant;
  Real kap_a;
  Real kap_s;

  inline Real black_body(const Real T)
  {
    return rad_constant * POW4(T);
  }

  inline int calc_kap_a(AT_C_sca & kap_a,
                        const int k, const int j, const int i,
                        const Real rho,
                        const Real T,
                        const Real Y_e)
  {
    kap_a(k,j,i) = rho * this->kap_a;
    return 0;
  }

  inline int calc_kap_s(AT_C_sca & kap_s,
                        const int k, const int j, const int i,
                        const Real rho,
                        const Real T,
                        const Real Y_e)
  {
    kap_s(k,j,i) = rho * this->kap_s;
    return 0;
  }

};

// ============================================================================
}  // M1::Opacities
// ============================================================================

#endif // M1_OPACITIES_PHOTON_HPP

