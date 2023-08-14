//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_utils.cpp
//  \brief Pointwise contractions and expressions for fluxes, sources, closure, etc.
//         Functions to unpack/pack specific variables stored as TensorPointwise

// Athena++ headers
#include "m1.hpp"
#if Z4C_ENABLED
#include "../z4c/z4c.hpp"
#endif
#include "../mesh/mesh.hpp"


using namespace utils;

#define SQ(X) ((X)*(X))

void M1::calc_proj(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
		               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
		               TensorPointwise<Real, Symmetries::NONE, MDIM, 2> & proj_ud)
{
  for (int a = 0; a < MDIM; ++a) {
    for (int b = 0; b < MDIM; ++b) {
      proj_ud(a,b) = tensor::delta(a,b) + u_u(a)*u_d(b);
    }
  }
}

void M1::calc_Pthin(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
		                Real const E,
		                TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		                TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd)
{
  Real const F2 = tensor::dot(g_uu, F_d, F_d);
  Real fac = (F2 > 0 ? E/F2 : 0);
  for (int a = 0; a < MDIM; ++a) {
    for (int b = a; b < MDIM; ++b) {
      P_dd(a,b) = fac * F_d(a) * F_d(b);
    }
  }
}

void M1::calc_Pthick(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
		                 TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
		                 Real const W,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
		                 Real const E,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		                 TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd)
{
  Real const v_dot_F = tensor::dot(g_uu, v_d, F_d);  
  Real const W2 = W*W;
  Real const coef = 1./(2.*W2 + 1.);
  
  // J/3
  Real const Jo3 = coef*((2.*W2 - 1.)*E - 2.*W2*v_dot_F);
  
  // tH = gamma_ud H_d
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> tH_d;
  for (int a = 0; a < MDIM; ++a) {
    tH_d(a) = F_d(a)/W + coef*W*v_d(a)*((4.*W2 + 1.)*v_dot_F - 4.*W2*E);
  }

  for (int a = 0; a < MDIM; ++a) {
    for (int b = a; b < MDIM; ++b) {
      P_dd(a,b)  = Jo3*(4.*W2*v_d(a)*v_d(b) + g_dd(a,b) + n_d(a)*n_d(b));
      P_dd(a,b) += W*(tH_d(a)*v_d(b) + tH_d(b)*v_d(a));
    }
  }
}

void M1::assemble_fnu(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
		                  Real const J,
		                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & H_u,
		                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & fnu_u)
{
  Real const H2 = tensor::dot(H_u, H_u);     // using Euclidean metric!
  for (int a = 0; a < MDIM; ++a) {
    fnu_u(a) = u_u(a) + (J > M1_EPSILON*H2 ? H_u(a)/J : 0);
  }
}

Real compute_Gamma(Real const W,
		               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_u,
		               Real const J, Real const E,
		               TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d)
{
  if (E > rad_E_floor && J > rad_E_floor) {
    Real f_dot_v = std::min(tensor::dot(F_d, v_u)/E, 1 - rad_eps);
    return W*(E/J)*(1 - f_dot_v);
  } else {
    return 1;
  }
}

void M1::assemble_rT(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
		                 Real const J,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & H_d,
		                 TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & K_dd,
		                 TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & rT_dd)
{
  for (int a = 0; a < MDIM; ++a) {
    for (int b = a; b < MDIM; ++b) {
      rT_dd(a,b) = J*u_d(a)*u_d(b) + H_d(a)*u_d(b) + H_d(b)*u_d(a) + K_dd(a,b);
    }
  }
}

// Project out the radiation energy (in any frame)
Real M1::calc_J_from_rT(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & rT_dd,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u)
{
  return tensor::dot(rT_dd, u_u, u_u);
}

void M1::calc_H_from_rT(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & rT_dd,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & H_d)
{
  for (int a = 0; a < MDIM; ++a) {
    H_d(a) = 0.0;
    for (int b = 0; b < MDIM; ++b) {
      for (int c = 0; c < MDIM; ++c) {
	      H_d(a) -= proj_ud(b,a)*u_u(c)*rT_dd(b,c);
      }
    }
  }
}

void M1::calc_K_from_rT(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & rT_dd,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
			                  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & K_dd)
{
  for (int a = 0; a < MDIM; ++a) {
    for (int b = a; b < MDIM; ++b) {
      K_dd(a,b) = 0.0;
      for (int c = 0; c < MDIM; ++c) {
        for (int d = 0; d < MDIM; ++d) {
	        K_dd(a,b) += proj_ud(c,a)*proj_ud(d,b)*rT_dd(c,d);
        }
      }
    }
  }
}

Real M1::calc_E_flux(Real const alp,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
		                 Real const E,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_u,
		                 int const dir)
{
  return alp*F_u(dir) - beta_u(dir)*E;
}

Real M1::calc_F_flux(Real const alp,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & P_ud,
		                 int const dir, int const comp)
{
  return alp*P_ud(dir,comp) - beta_u(dir)*F_d(comp);
}

void M1::calc_rad_sources(Real const eta,
			                    Real const kabs,
			                    Real const kscat,
			                    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
			                    Real const J,
			                    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const H_d,
			                    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & S_d)
{
  for (int a = 0; a < MDIM; ++a) {
    S_d(a) = (eta - kabs*J)*u_d(a) - (kabs + kscat)*H_d(a);
  }
}

Real M1::calc_rE_source(Real const alp,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_u,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & S_d)
{
  return - alp*tensor::dot(n_u, S_d);
}

void M1::calc_rF_source(Real const alp,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const gamma_ud,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & S_d,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & tS_d)
{
  for (int a = 0; a < MDIM; ++a) {
    tS_d(a) = 0.0;
    for (int b = 0; b < MDIM; ++b) {
      tS_d(a) += alp*gamma_ud(b,a)*S_d(b);
    }
  }
}

void M1::apply_floor(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const g_uu,
		                 Real * E,
		                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & F_d)
{
  *E = std::max(rad_E_floor, *E);
  Real const F2 = tensor::dot(g_uu, F_d, F_d);
  Real const lim = (*E)*(*E)*(1 - rad_eps);
  if (F2 > lim) {
    Real fac = lim/F2;
    for (int a = 0; a < MDIM; ++a) {
      F_d(a) *= fac;
    }
  }
}

void M1::uvel(Real alp,
	            Real betax,Real betay,Real betaz,
	            Real w_lorentz,
	            Real velx,Real vely,Real velz,
	            Real * u0, Real * u1, Real * u2, Real * u3)
{
  Real const ialp = 1.0/alp;
  *u0 = w_lorentz*ialp;
  *u1 = w_lorentz*(velx - betax*ialp);
  *u2 = w_lorentz*(vely - betay*ialp);
  *u3 = w_lorentz*(velz - betaz*ialp);
}

//
// Pack/unpack routines
// Note some of these work for both MDIM and NDIM=MDIM-1
// They also turn components of a tensors in NDIM into components of tensors in MDIM-1 and viceversa
// 

void M1::pack_F_d(Real const betax, Real const betay, Real const betaz,
		              Real const Fx, Real const Fy, Real const Fz,
		              TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & F_d)
{
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  F_d(0) = betax*Fx + betay*Fy + betaz*Fz;
  F_d(1) = Fx;
  F_d(2) = Fy;
  F_d(3) = Fz;
}

void M1::unpack_F_d(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		                Real * Fx, Real * Fy, Real * Fz)
{
  *Fx = F_d(1);
  *Fy = F_d(2);
  *Fz = F_d(3);
}

void M1::pack_F_d(Real const Fx, Real const Fy, Real const Fz,
		              TensorPointwise<Real, Symmetries::NONE, MDIM-1, 1> & F_d)
{
  F_d(0) = Fx;
  F_d(1) = Fy;
  F_d(2) = Fz;
}

void M1::unpack_F_d(TensorPointwise<Real, Symmetries::NONE, MDIM-1, 1> const & F_d,
		                Real * Fx, Real * Fy, Real * Fz)
{
  *Fx = F_d(0);
  *Fy = F_d(1);
  *Fz = F_d(2);
}

void M1::pack_H_d(Real const Ht, Real const Hx, Real const Hy, Real const Hz,
		              TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & H_d)
{
  H_d(0) = Ht;
  H_d(1) = Hx;
  H_d(2) = Hy;
  H_d(3) = Hz;
}

void M1::unpack_H_d(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & H_d,
		                Real * Ht, Real * Hx, Real * Hy, Real * Hz)
{
  *Ht = H_d(0);
  *Hx = H_d(1);
  *Hy = H_d(2);
  *Hz = H_d(3);
}

void M1::pack_P_dd(Real const betax, Real const betay, Real const betaz,
		               Real const Pxx, Real const Pxy, Real const Pxz,
		               Real const Pyy, Real const Pyz, Real const Pzz,
		               TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd)
{
  Real const Pbetax = Pxx*betax + Pxy*betay + Pxz*betaz;
  Real const Pbetay = Pxy*betax + Pyy*betay + Pyz*betaz;
  Real const Pbetaz = Pxz*betax + Pyz*betay + Pzz*betaz;
  
  // P_00 = g_0i g_k0 P^ik = beta^i beta^k P_ik
  P_dd(0,0) = Pbetax*betax + Pbetay*betay + Pbetaz*betaz;
  
  // P_0i = g_0j g_ki P^jk = beta_j P_i^j = beta^j P_ij
  P_dd(0,1) = Pbetax;
  P_dd(0,2) = Pbetay;
  P_dd(0,3) = Pbetaz;

  // P_ij
  P_dd(1,1) = Pxx;
  P_dd(1,2) = Pxy;
  P_dd(1,3) = Pxz;
  P_dd(2,2) = Pyy;
  P_dd(2,3) = Pyz;
  P_dd(3,3) = Pzz;
}

void M1::unpack_P_dd(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & P_dd,
		                 Real * Pxx, Real * Pxy, Real * Pxz,
		                 Real * Pyy, Real * Pyz, Real * Pzz)
{
  *Pxx = P_dd(1,1);
  *Pxy = P_dd(1,2);
  *Pxz = P_dd(1,3);
  *Pyy = P_dd(2,2);
  *Pyz = P_dd(2,3);
  *Pzz = P_dd(3,3);
}

void M1::pack_P_dd(Real const Pxx, Real const Pxy, Real const Pxz,
		               Real const Pyy, Real const Pyz, Real const Pzz,
		               TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 2> & P_dd)
{
  P_dd(0,0) = Pxx;
  P_dd(0,1) = Pxy;
  P_dd(0,2) = Pxz;
  P_dd(1,1) = Pyy;
  P_dd(1,2) = Pyz;
  P_dd(2,2) = Pzz;
}

void M1::pack_P_ddd(Real const Pxxx, Real const Pxxy, Real const Pxxz,
                    Real const Pxyy, Real const Pxyz, Real const Pxzz,
                    Real const Pyxx, Real const Pyxy, Real const Pyxz,
                    Real const Pyyy, Real const Pyyz, Real const Pyzz,
                    Real const Pzxx, Real const Pzxy, Real const Pzxz,
                    Real const Pzyy, Real const Pzyz, Real const Pzzz,
                    TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 3> & P_ddd)
{
  P_ddd(0,0,0) = Pxxx;
  P_ddd(0,0,1) = Pxxy;
  P_ddd(0,0,2) = Pxxz;
  P_ddd(0,1,1) = Pxyy;
  P_ddd(0,1,2) = Pxyz;
  P_ddd(0,2,2) = Pxzz;

  P_ddd(1,0,0) = Pyxx;
  P_ddd(1,0,1) = Pyxy;
  P_ddd(1,0,2) = Pyxz;
  P_ddd(1,1,1) = Pyyy;
  P_ddd(1,1,2) = Pyyz;
  P_ddd(1,2,2) = Pyzz;

  P_ddd(2,0,0) = Pzxx;
  P_ddd(2,0,1) = Pzxy;
  P_ddd(2,0,2) = Pzxz;
  P_ddd(2,1,1) = Pzyy;
  P_ddd(2,1,2) = Pzyz;
  P_ddd(2,2,2) = Pzzz;
}

void M1::unpack_P_dd(TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 2> const & P_dd,
		                 Real * Pxx, Real * Pxy, Real * Pxz,
		                 Real * Pyy, Real * Pyz, Real * Pzz)
{
  *Pxx = P_dd(0,0);
  *Pxy = P_dd(0,1);
  *Pxz = P_dd(0,2);
  *Pyy = P_dd(1,1);
  *Pyz = P_dd(1,2);
  *Pzz = P_dd(2,2);
}

void M1::pack_v_u(Real const velx, Real const vely, Real const velz,
		              TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & v_u)
{
    v_u(0) = 0.0;
    v_u(1) = velx;
    v_u(2) = vely;
    v_u(3) = velz;
}

// Lorentz factor from utilde^i
Real M1::GetWLorentz_from_utilde(Real const utx, Real const uty, Real const utz,
				                         Real const gxx, Real const gxy, Real const gxz,
				                         Real const gyy, Real const gyz, Real const gzz,
				                         Real * utlx, Real * utly, Real * utlz,
				                         Real *ut2)
{
  Real _utlx = gxx*utx + gxy*uty + gxz*utz;
  Real _utly = gxy*utx + gyy*uty + gyz*utz;
  Real _utlz = gxz*utx + gyz*uty + gzz*utz; 

  *ut2 = utx*_utlx + uty*_utly + utz*_utlz;
  
  if (utlx) *utlx = _utlx;
  if (utly) *utly = _utly;
  if (utlz) *utlz = _utlz;
  
  return std::sqrt(*ut2 + 1.0);
}

//
// The following routines substitute what THC uses from
//   cpputils/src/utils_tensor.hh
// In particular the main class employed in THC_M1 is
//   tensor::slicing_geometry_const geom
// and the relevant calls are
//   geom.get_metric(ijk, &g_dd);
//   geom.get_extr_curv(ijk, &K_dd);
//   geom.get_inv_metric(ijk, &g_uu);
//   geom.get_normal_form(ijk, &n_d);
//   geom.get_shift_vec(ijk, &beta_u);
//   geom.get_space_proj(ijk, &gamma_ud);
//
// Here, we provide the few relevant routines working with 
// the TensorPointwise objects.
//
// Note this code could be helpful in other parts of Athena++
// and could be separated from M1 and embedded into tensor::
//

void M1::Get4Metric_VC2CCinterp(MeshBlock * pmb,
				                        const int k, const int j, const int i,
                                AthenaArray<Real> & u, 
                                AthenaArray<Real> & u_adm,
                                TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & g_dd,
                                TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & beta_u,
                                TensorPointwise<Real, Symmetries::NONE, MDIM, 0> & alpha)
{
  AthenaArray<Real> vc_z4c_alpha;
  AthenaArray<Real> vc_z4c_beta_u_x, vc_z4c_beta_u_y, vc_z4c_beta_u_z;
  AthenaArray<Real> vc_gamma_xx, vc_gamma_xy, vc_gamma_xz, vc_gamma_yy,
                    vc_gamma_yz, vc_gamma_zz;

  vc_z4c_alpha.InitWithShallowSlice(u, Z4c::I_Z4c_alpha,1);
  vc_z4c_beta_u_x.InitWithShallowSlice(u, Z4c::I_Z4c_betax,1);
  vc_z4c_beta_u_y.InitWithShallowSlice(u, Z4c::I_Z4c_betay,1);
  vc_z4c_beta_u_z.InitWithShallowSlice(u, Z4c::I_Z4c_betaz,1);
  vc_gamma_xx.InitWithShallowSlice(u_adm, Z4c::I_ADM_gxx,1);
  vc_gamma_xy.InitWithShallowSlice(u_adm, Z4c::I_ADM_gxy,1);
  vc_gamma_xz.InitWithShallowSlice(u_adm, Z4c::I_ADM_gxz,1);
  vc_gamma_yy.InitWithShallowSlice(u_adm, Z4c::I_ADM_gyy,1);
  vc_gamma_yz.InitWithShallowSlice(u_adm, Z4c::I_ADM_gyz,1);
  vc_gamma_zz.InitWithShallowSlice(u_adm, Z4c::I_ADM_gzz,1);

  TensorPointwise<Real, Symmetries::SYM2, MDIM, 1> beta_d;
  beta_d.NewTensorPointwise();
  
  // Map from VC to CC gauge and spatial part g_ij
  // Note we go from 3-tensors to 4-tensors (indexes!!!) TODO
  alpha() = VCInterpolation(vc_z4c_alpha,k,j,i);

  beta_u(0)   = 0.;
  beta_u(1) = VCInterpolation(vc_z4c_beta_u_x,k,j,i);
  beta_u(2) = VCInterpolation(vc_z4c_beta_u_y,k,j,i);
  beta_u(3) = VCInterpolation(vc_z4c_beta_u_z,k,j,i);

  // g_ij
  g_dd(1,1) = VCInterpolation(vc_gamma_xx, k, j, i);
  g_dd(1,2) = VCInterpolation(vc_gamma_xy, k, j, i);
  g_dd(1,3) = VCInterpolation(vc_gamma_xz, k, j, i);
  g_dd(2,2) = VCInterpolation(vc_gamma_yy, k, j, i);
  g_dd(2,3) = VCInterpolation(vc_gamma_yz, k, j, i);
  g_dd(3,3) = VCInterpolation(vc_gamma_zz, k, j, i);
  
  // beta_a
  for (int a = 0; a < MDIM; ++a) {
    beta_d(a) = 0.0;  
  }
  for (int a = 1; a < MDIM; ++a) {
    for(int b = 1; a < MDIM; ++a) {
      beta_d(a) += g_dd(a,b) * beta_u(b);
    }
  }
  
  // g_0i
  for (int a = 1; a < MDIM; ++a) {
    g_dd(0,a) = g_dd(a,0) = beta_d(a);
  }

  // g_00
  g_dd(0,0) = - SQ(alpha());
  for (int a = 1; a < MDIM; ++a) {
    g_dd(0,0) += beta_u(a)*beta_d(a);
  }

  beta_d.DeleteTensorPointwise();  
}

// Inverse 4-metric
void M1::Get4Metric_Inv(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
			                  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
			                  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & g_uu)
{
  // g^00
  g_uu(0,0) = - 1.0/SQ(alpha());
  
  // g^0i
  for (int a = 1; a < MDIM; ++a) {
    g_uu(0,a) = g_uu(a,0) =  beta_u(a) * (-g_uu(0,0));
  }

  // g^ij 
  Real const det = SpatialDet(g_dd(1,1), g_dd(1,2), g_dd(1,3),
			                        g_dd(2,2), g_dd(2,3), g_dd(3,3));
  
  SpatialInv(1.0/det,
	           g_dd(1,1), g_dd(1,2), g_dd(1,3),
	           g_dd(2,2), g_dd(2,3), g_dd(3,3),
	           &g_uu(1,1), &g_uu(1,2), &g_uu(1,3),
	           &g_uu(2,2), &g_uu(2,3), &g_uu(3,3));
  
  for (int a = 1; a < MDIM; ++a) {
    for (int b = 1; a < MDIM; ++a) {
      g_uu(a,b) += beta_u(a) * beta_u(b) * g_uu(0,0);      
    }
  }
}

// Return also the inverse 3-metric
void M1::Get4Metric_Inv_Inv3(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
			                       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
			                       TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
			                       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & g_uu,
			                       TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 2> & gam_uu)
{
  // g^00
  g_uu(0,0) = - 1.0/SQ(alpha());
  
  // g^0i
  for (int a = 1; a < MDIM; ++a) {
    g_uu(0,a) = g_uu(a,0) =  beta_u(a) * (-g_uu(0,0));
  }
  // gamma^ij 
  Real const det = SpatialDet(g_dd(1,1), g_dd(1,2), g_dd(1,3),
			                        g_dd(2,2), g_dd(2,3), g_dd(3,3));
  
  SpatialInv(1.0/det,
	           g_dd(1,1), g_dd(1,2), g_dd(1,3),
	           g_dd(2,2), g_dd(2,3), g_dd(3,3),
	           &g_uu(1,1), &g_uu(1,2), &g_uu(1,3),
	           &g_uu(2,2), &g_uu(2,3), &g_uu(3,3));

  gam_uu(0,0) = g_uu(1,1);
  gam_uu(0,1) = g_uu(1,2);
  gam_uu(0,2) = g_uu(1,3);
  gam_uu(1,1) = g_uu(2,2);
  gam_uu(1,2) = g_uu(2,3);
  gam_uu(2,2) = g_uu(3,3);
  
  // g^ij   
  for(int a = 1; a < MDIM; ++a) {
    for(int b = 1; a < MDIM; ++a) {
      g_uu(a,b) += beta_u(a) * beta_u(b) * g_uu(0,0);      
    }
  }
}

void M1::Get4Metric_ExtrCurv_VC2CCinterp(MeshBlock * pmb,
					                               const int k, const int j, const int i,
					                               AthenaArray<Real> & u_adm,
					                               TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & K_dd)
{
  AthenaTensor<Real, TensorSymm::SYM2, MDIM, 2> vc_adm_K_dd;
  vc_adm_K_dd.InitWithShallowSlice(u_adm, Z4c::I_ADM_Kxx);

  // Map K_ij from VC to CC
  // Note we go from 3-tensors to 4-tensors (indexes!!!)
  for (int a=1; a<MDIM; a++) {
    for (int b=a; b<MDIM; b++) {
      K_dd(a,b) = VCInterpolation(vc_adm_K_dd(a-1,b-1),k,j,i);
    }
  }
}

void M1::Get4Metric_Normal(TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
			                     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
			                     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & n_u)
{
  Real const ooalp = 1.0/alpha();
  n_u(0) = ooalp;
  for (int a = 1; a < MDIM; ++a) {
    n_u(a) = - beta_u(a) * ooalp;
  }
}

void M1::Get4Metric_NormalForm(TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
			       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & n_d)
{
  for(int a = 0; a < MDIM; ++a) {
    n_d(a) = 0.;
  }
  n_d(0) = - alpha();
}

void M1::Get4Metric_SpaceProj(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_u,
                              TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
                              TensorPointwise<Real, Symmetries::NONE, MDIM, 2> & gamma_ud)
{
  for (int a = 0; a < MDIM; ++a)
    for (int b = 0; b < MDIM; ++b)
      gamma_ud(a,b) = tensor::delta(a,b) + n_u(a)*n_d(b);
}

Real M1::SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz)
{
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQR(gyz) - SQR(gxy)*gzz + gxx*gyy*gzz;
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SpatialInv(Real const detginv,
//           Real const gxx, Real const gxy, Real const gxz,
//           Real const gyy, Real const gyz, Real const gzz,
//           Real * uxx, Real * uxy, Real * uxz,
//           Real * uyy, Real * uyz, Real * uzz)
// \brief returns inverse of 3-metric
void M1::SpatialInv(Real const detginv,
                    Real const gxx, Real const gxy, Real const gxz,
                    Real const gyy, Real const gyz, Real const gzz,
                    Real * uxx, Real * uxy, Real * uxz,
                    Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
  return;
}

