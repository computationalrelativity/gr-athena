#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../utils/tensor.hpp"

using namespace utils::tensor;
//----------------------------------------------------------------------------------------
// \!fn void Z4c::KerrSchild(AthenaArray<Real> & u)
// \brief Initialize vars to KerSchild with spin 0

void Z4c::ADMKerrSchild(Real const x, Real const y, Real const z,
                        TensorPointwise<Real, Symmetries::NONE, NDIM, 0> & alpha,
                        TensorPointwise<Real, Symmetries::NONE, NDIM, 1> & beta_u,
                        TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> & gamma_dd,
                        TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> & K_dd)
{
  Real const r = std::sqrt(SQR(x)+SQR(y)+SQR(z));
  Real const l[3] = {x/r, y/r, z/r};

  alpha() = std::pow(1.0 + 2.0/r, -0.5);
  for (int a = 0; a < NDIM; ++a) {
    beta_u(a) = 2.0/r * SQR(alpha()) * l[a];
    for(int b = 0; b < NDIM; ++b) {
      Real const eta_ab = (a == b ? 1.0 : 0.0);
      gamma_dd(a,b) = eta_ab + 2/r*l[a]*l[b];
      K_dd(a,b) = 2*alpha()/SQR(r) * (eta_ab - (2.0+1./r)*l[a]*l[b]);
    }
  }
}

void Z4c::SetKerrSchild(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_z4c) {
  ADM_vars adm;
  Z4c_vars z4c;
  SetADMAliases(u_adm, adm);
  SetZ4cAliases(u_z4c, z4c);
  adm.psi4.Fill(1.);
  adm.K_dd.ZeroClear();
  z4c.alpha.Fill(1.);
  z4c.beta_u.Fill(0.);

  TensorPointwise<Real, Symmetries::NONE, NDIM, 0> alpha;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gamma_dd;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> K_dd;

  alpha.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  gamma_dd.NewTensorPointwise();
  K_dd.NewTensorPointwise();

  GLOOP3(k,j,i) {
    Real const z = pmy_block->pcoord->x3f(k);
    Real const y = pmy_block->pcoord->x2f(j);
    Real const x = pmy_block->pcoord->x1f(i);
    
    ADMKerrSchild(x, y, z, alpha, beta_u, gamma_dd, K_dd);
    z4c.alpha(k,j,i) = alpha();
    adm.psi4(k,j,i) = 1.0 / SQR(alpha());
    for (int a = 0; a < NDIM; ++a) {
      z4c.beta_u(a,k,j,i) = beta_u(a);
      for(int b = 0; b < NDIM; ++b) {
        adm.g_dd(a,b,k,j,i) = gamma_dd(a,b);
        adm.K_dd(a,b,k,j,i) = K_dd(a,b);
      }
    }
  }
}