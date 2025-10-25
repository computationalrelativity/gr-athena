// C headers

// C++ headers
#include <cmath>  // sqrt()

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../parameter_input.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "coordinates.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/floating_point.hpp"

//-----------------------------------------------------------------------------
using namespace gra::aliases;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// GRDynamical Constructor
// Inputs:
//   pmb: pointer to MeshBlock containing this grid
//   pin: pointer to runtime inputs
//   flag: true if object is for coarse grid only in an AMR calculation

GRDynamical::GRDynamical(MeshBlock *pmb, ParameterInput *pin, bool coarse_flag)
  : Coordinates(pmb, pin, coarse_flag)
{
  // ng, nc1 are coarse analogues if `coarse_flag` enabled
  Mesh * pm = pmb->pmy_mesh;
  const int ndim = (pm->f3) ? 3 : (pm->f2) ? 2 : 1;

  // Set up indicial ranges
  const int ill = il - ng;
  const int iuu = iu + ng;

  const int jll = (ndim>1) ? jl-ng: jl;
  const int juu = (ndim>1) ? ju+ng: ju;

  const int kll = (ndim>2) ? kl-ng: kl;
  const int kuu = (ndim>2) ? ku+ng: ku;

  // Initialize volume-averaged coordinates and spacings ----------------------
  for (int i = ill; i <= iuu; ++i) {
    const Real x1_m = x1f(i);
    const Real x1_p = x1f(i + 1);

    // at least 2nd-order accurate
    x1v(i) = 0.5 * (x1_m + x1_p);
  }
  for (int i = ill; i <= iuu - 1; ++i) {
    dx1v(i) = x1v(i + 1) - x1v(i);
  }

  if (ndim > 1)
  {
    for (int j = jll; j <= juu; ++j) {
      const Real x2_m = x2f(j);
      const Real x2_p = x2f(j + 1);
      x2v(j) = 0.5 * (x2_m + x2_p);
    }
    for (int j = jll; j <= juu - 1; ++j) {
      dx2v(j) = x2v(j + 1) - x2v(j);
    }
  }
  else
  {
    const Real x2_m = x2f(jl);
    const Real x2_p = x2f(jl + 1);
    x2v(jl) = 0.5 * (x2_m + x2_p);
    dx2v(jl) = dx2f(jl);
  }

  if (ndim > 2)
  {
    for (int k = kll; k <= kuu; ++k) {
      const Real x2_m = x3f(k);
      const Real x2_p = x3f(k + 1);
      x3v(k) = 0.5 * (x2_m + x2_p);
    }
    for (int k = kll; k <= kuu - 1; ++k) {
      dx3v(k) = x3v(k + 1) - x3v(k);
    }
  }
  else
  {
    const Real x3_m = x3f(kl);
    const Real x3_p = x3f(kl + 1);
    x3v(kl) = 0.5 * (x3_m + x3_p);
    dx3v(kl) = dx3f(kl);
  }


  // Initialize area-averaged coordinates used with MHD AMR
  if (pm->multilevel && MAGNETIC_FIELDS_ENABLED) {

    for (int i = ill; i <= iuu; ++i) {
      x1s2(i) = x1s3(i) = x1v(i);
    }

    if (ndim > 1)
    {
      for (int j = jll; j <= juu; ++j) {
        x2s1(j) = x2s3(j) = x2v(j);
      }
    }
    else
    {
      x2s1(jl) = x2s3(jl) = x2v(jl);
    }

    if (ndim > 2)
    {
      for (int k = kll; k <= kuu; ++k) {
        x3s1(k) = x3s2(k) = x3v(k);
      }
    }
    else
    {
      x3s1(kl) = x3s2(kl) = x3v(kl);
    }
  }


  // Set up finite differencing -----------------------------------------------
  fd_is_defined = true;
  fd_cc = new FiniteDifference::Uniform(
    nc1, nc2, nc3,
    dx1v(0), dx2v(0), dx3v(0)
  );

  fd_cx = new FiniteDifference::Uniform(
    cx_nc1, cx_nc2, cx_nc3,
    dx1v(0), dx2v(0), dx3v(0)
  );

  fd_vc = new FiniteDifference::Uniform(
    nv1, nv2, nv3,
    dx1f(0), dx2f(0), dx3f(0)
  );

  // intergrid interpolators --------------------------------------------------
  // for metric <-> matter sampling conversion
  //
  // if metric_vc then need {vc->fc, vc->cc}
  // if metric_cx then only need cx->fc
  //
  // this motivates the choices of ghosts below

  const int ng_c = (coarse_flag) ? NCGHOST_CX : NGHOST;
  const int ng_v = (coarse_flag) ? NCGHOST : NGHOST;

  int N[] = {nc1 - 2 * ng_c, nc2 - 2 * ng_c, nc3 - 2 * ng_c};

  Real rdx[] = {1. / (x1f(1) - x1f(0)),
                1. / (x2f(1) - x2f(0)),
                1. / (x3f(1) - x3f(0))};

  const int dim = (pmb->pmy_mesh->f3) ? 3 : ((pmb->pmy_mesh->f2) ? 2 : 1);

  ig_is_defined = true;

  ig_1N = new IIG_1N(dim, &N[0], &rdx[0], ng_c, ng_v);
  ig_2N = new IIG_2N(dim, &N[0], &rdx[0], ng_c, ng_v);
  ig_NN = new IIG_NN(dim, &N[0], &rdx[0], ng_c, ng_v);

  // set up various scratches for AddCoordTermsDivergence ---------------------
  if (!coarse_flag)
  {
    // geometry
    sqrt_detgamma_.NewAthenaTensor(nc1);
    detgamma_.NewAthenaTensor(nc1);

    oo_detgamma_.NewAthenaTensor(nc1);

    alpha_.NewAthenaTensor(nc1);
    oo_alpha_.NewAthenaTensor(nc1);
    beta_u_.NewAthenaTensor(nc1);
    gamma_dd_.NewAthenaTensor(nc1);
    gamma_uu_.NewAthenaTensor(nc1);

    K_dd_.NewAthenaTensor(nc1);

    dalpha_d_.NewAthenaTensor(nc1);
    dbeta_du_.NewAthenaTensor(nc1);
    dgamma_ddd_.NewAthenaTensor(nc1);

    // matter
    w_util_u_.NewAthenaTensor(nc1);
    w_util_d_.NewAthenaTensor(nc1);
    W_.NewAthenaTensor(nc1);

    w_hrho_.NewAthenaTensor(nc1);

    // sources
    Stau_.NewAthenaTensor(nc1);
    SS_d_.NewAthenaTensor(nc1);

    // Particular to magnetic fields ------------------------------------------
    if (MAGNETIC_FIELDS_ENABLED)
    {
      oo_sqrt_detgamma_.NewAthenaTensor(nc1);

      beta_d_.NewAthenaTensor(nc1);

      q_scB_u_.NewAthenaTensor(nc1);

      b0_.NewAthenaTensor(nc1);
      b2_.NewAthenaTensor(nc1);

      bi_u_.NewAthenaTensor(nc1);
      bi_d_.NewAthenaTensor(nc1);

      T00.NewAthenaTensor(nc1);
      T0i_u.NewAthenaTensor(nc1);
      T0i_d.NewAthenaTensor(nc1);
      Tij_uu.NewAthenaTensor(nc1);
    }
  }
}

// ----------------------------------------------------------------------------
// Underlying coordinatization is Cartesian, we therefore inherit from
// Coordinates class the following:
//
// EdgeXLength, GetEdgeXLength, CenterWidthX, FaceXArea, GetFaceXArea,
// CellVolume

// ----------------------------------------------------------------------------
// BD: TODO - Double check these expressions
// BD: TODO - Sources are written in two different way, maybe better to pick
//            one...
void GRDynamical::AddCoordTermsDivergence(
  const Real dt,
  const AthenaArray<Real> *flux,
  const AthenaArray<Real> &prim,
#if USETM
  const AthenaArray<Real> &prim_scalar,
#endif
  const AthenaArray<Real> &bb_cc,
  AthenaArray<Real> &cons)
{
  using namespace LinearAlgebra;
  using namespace FloatingPoint;

  MeshBlock * pmb = pmy_block;
  EquationOfState * peos = pmb->peos;
  Hydro * ph = pmb->phydro;

  // regularization factor
  const Real eps_alpha__ = pmb->pz4c->opt.eps_floor;

  // perform variable resampling when required
  Z4c * pz4c = pmy_block->pz4c;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sca adm_alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec adm_beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym adm_K_dd(    pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // BD: TODO - clean this up
  AT_N_sca adm_sqrt_detgamma;
#if FLUID_ENABLED
  adm_sqrt_detgamma.InitWithShallowSlice(
    pz4c->storage.aux_extended,
    Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma
  );
#endif // FLUID_ENABLED

  // Slice matter -------------------------------------------------------------
  AA & ccprim = const_cast<AthenaArray<Real>&>(prim);
  AT_N_sca w_rho(   ccprim, IDN);
  AT_N_sca w_p(     ccprim, IPR);
  AT_N_vec w_util_u(ccprim, IVX);
#if NSCALARS > 0
  AA & ccprim_scalar = const_cast<AthenaArray<Real>&>(prim_scalar);
  AT_S_vec w_r(ccprim_scalar, 0);
#endif

  // --------------------------------------------------------------------------
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    GetGeometricFieldCC(gamma_dd_, adm_gamma_dd, k, j);
    GetGeometricFieldCC(K_dd_,     adm_K_dd,     k, j);
    GetGeometricFieldCC(alpha_,    adm_alpha,    k, j);
    GetGeometricFieldCC(beta_u_,   adm_beta_u,   k, j);

    CC_PCO_ILOOP1(i)
    {
      Real alpha__ = regularize_near_zero(alpha_(i), eps_alpha__);
      oo_alpha_(i) = OO(alpha__);
    }

#if !defined(DBG_FD_CX_COORDDIV) || !defined(Z4C_CX_ENABLED)
    for (int a=0; a<NDIM; ++a)
    {
      GetGeometricFieldDerCC(dgamma_ddd_, adm_gamma_dd, a, k, j);
      GetGeometricFieldDerCC(dalpha_d_,   adm_alpha,    a, k, j);
      GetGeometricFieldDerCC(dbeta_du_,   adm_beta_u,   a, k, j);
    }
#else
    for (int a=0; a<NDIM; ++a)
    CC_PCO_ILOOP1(i)
    {
      dalpha_d_(a,i) = fd_cx->Dx(a, adm_alpha(k,j,i));
    }

    for (int a=0; a<NDIM; ++a)
    for (int b=0; b<NDIM; ++b)
    CC_PCO_ILOOP1(i)
    {
      dbeta_du_(b,a,i) = fd_cx->Dx(b, adm_beta_u(a,k,j,i));
    }

    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    for (int c=0; c<NDIM; ++c)
    CC_PCO_ILOOP1(i)
    {
      dgamma_ddd_(c,a,b,i) = fd_cx->Dx(c, adm_gamma_dd(a,b,k,j,i));
    }
#endif // DBG_FD_CX_COORDDIV

    // Prepare determinant-like
    CC_PCO_ILOOP1(i)
    {
      // detgamma_(i)      = Det3Metric(gamma_dd_, i);
      // sqrt_detgamma_(i) = std::sqrt(detgamma_(i));
      sqrt_detgamma_(i) = adm_sqrt_detgamma(k,j,i);
      detgamma_(i)      = SQR(sqrt_detgamma_(i));

      oo_detgamma_(i)      = OO(detgamma_(i));
#if MAGNETIC_FIELDS_ENABLED
      oo_sqrt_detgamma_(i) = OO(sqrt_detgamma_(i));
#endif
    }

    Inv3Metric(oo_detgamma_, gamma_dd_, gamma_uu_, il, iu);

    // indicial manipulations
    for (int a=0; a<N; ++a)
    CC_PCO_ILOOP1(i)
    {
      w_util_u_(a,i) = w_util_u(a,k,j,i);
    }
    LinearAlgebra::SlicedVecMet3Contraction(w_util_d_, w_util_u_, gamma_dd_,
                                            il, iu);

    // Lorentz factors
    CC_PCO_ILOOP1(i)
    {
      /*
      const Real norm2_utilde_ = InnerProductSlicedVec3Metric(
        w_util_d_, gamma_uu_, i
      );

      W_(i) = std::sqrt(1. + norm2_utilde_);
      */
      W_(i) = pmb->phydro->derived_ms(IX_LOR,k,j,i);
    }

    // Calculate enthalpy (rho*h) NB EOS specific!
    CC_PCO_ILOOP1(i)
    {
#if USETM
      const Real oo_mb = OO(peos->GetEOS().GetBaryonMass());
      Real n = oo_mb * w_rho(k,j,i);
      Real Y[MAX_SPECIES] = {0.0};

#if NSCALARS > 0
      for (int l=0; l<NSCALARS; ++l)
      {
        Y[l] = w_r(l,k,j,i);
      }
#endif

#if defined(Z4C_CX_ENABLED) || defined(Z4C_CC_ENABLED)
      Real T = pmb->phydro->derived_ms(IX_T,k,j,i);
      Real h = pmb->phydro->derived_ms(IX_ETH,k,j,i);
#else
      Real T = peos->GetEOS().GetTemperatureFromP(n, w_p(k,j,i), Y);
      Real h = peos->GetEOS().GetEnthalpy(n, T, Y);
#endif

      w_hrho_(i) = w_rho(k,j,i) * h;
#else
      const Real gamma_adi = peos->GetGamma();
    	w_hrho_(i) = w_rho(k,j,i) + gamma_adi/(gamma_adi-1.0) * w_p(k,j,i);
#endif
    }


    // assemble sources -------------------------------------------------------
    if(MAGNETIC_FIELDS_ENABLED)
    {
      LinearAlgebra::SlicedVecMet3Contraction(beta_d_, beta_u_, gamma_dd_,
                                              il, iu);

      // magnetic field components --------------------------------------------
      for (int a=0; a<N; ++a)
      CC_PCO_ILOOP1(i)
      {
        q_scB_u_(a,i) = bb_cc(IB1+a,k,j,i) * oo_sqrt_detgamma_(i);
      }

      b0_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {
        CC_PCO_ILOOP1(i)
        {
          b0_(i) += q_scB_u_(a, i) * w_util_d_(a, i) * oo_alpha_(i);
        }
      }

      for (int a = 0; a < NDIM; ++a)
      CC_PCO_ILOOP1(i) {
        bi_u_(a, i) = (
          q_scB_u_(a, i) +
          alpha_(i) * b0_(i) * (w_util_u_(a, i) / W_(i) -
                                beta_u_(a, i) * oo_alpha_(i))
        );
      }

      //  bi_d.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {

        CC_PCO_ILOOP1(i)
        {
          bi_d_(a,i) = beta_d_(a,i) * b0_(i);
          //bi_d_(a,i) += bi_u_(b,i)*gamma_dd(a,b,i);
        }

        for (int b=0; b<NDIM; ++b)
        {
          CC_PCO_ILOOP1(i)
          {
            bi_d_(a, i) += gamma_dd_(a, b, i) * bi_u_(b, i);
          }
        }
    	}

      CC_PCO_ILOOP1(i)
      {
        b2_(i) = SQR(alpha_(i))*SQR(b0_(i))/SQR(W_(i));
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      CC_PCO_ILOOP1(i)
      {
        b2_(i) +=
            q_scB_u_(a, i) * q_scB_u_(b, i) * gamma_dd_(a, b, i) / SQR(W_(i));
      }

      // T_dd (space-time) components -----------------------------------------
      CC_PCO_ILOOP1(i)
      {
        T00(i) = ((w_hrho_(i) + b2_(i)) * SQR(W_(i) * oo_alpha_(i)) +
                  (w_p(k, j, i) + b2_(i) / 2.0) * (-1.0 * SQR(oo_alpha_(i))) -
                  b0_(i) * b0_(i));
      }

      for (int a=0; a<NDIM; ++a)
      {
        CC_PCO_ILOOP1(i)
        {
          T0i_u(a, i) =
              ((w_hrho_(i) + b2_(i)) * W_(i) * oo_alpha_(i) *
                   (w_util_u_(a, i) - W_(i) * beta_u_(a, i) * oo_alpha_(i)) +
               (w_p(k, j, i) + b2_(i) / 2.0) * beta_u_(a, i) * SQR(oo_alpha_(i)) -
               b0_(i) * bi_u_(a, i));
        }
      }

      // BD: TODO - why was this loop written like this?
      for (int a=0; a<NDIM; ++a)
      // for (int b=0; b<NDIM; ++b)
      {
        CC_PCO_ILOOP1(i)
        {
          T0i_d(a, i) =
              (((w_hrho_(i) + b2_(i)) * W_(i) * w_util_d_(a, i)) * oo_alpha_(i) -
               b0_(i) * bi_d_(a, i));
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      {
        CC_PCO_ILOOP1(i)
        {
          Tij_uu(a, b, i) =
              ((w_hrho_(i) + b2_(i)) *
                   (w_util_u_(a, i) - W_(i) * beta_u_(a, i) * oo_alpha_(i)) *
                   (w_util_u_(b, i) - W_(i) * beta_u_(b, i) * oo_alpha_(i)) +
               (w_p(k, j, i) + b2_(i) / 2.0) *
                   (gamma_uu_(a, b, i) -
                    beta_u_(a, i) * beta_u_(b, i) * SQR(oo_alpha_(i))) -
               bi_u_(a, i) * bi_u_(b, i));
        }
      }

      // tau-source term ------------------------------------------------------
      Stau_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {
        CC_PCO_ILOOP1(i)
        {
          Stau_(i) += -(T00(i) * beta_u_(a, i) * dalpha_d_(a, i) +
                       T0i_u(a, i) * dalpha_d_(a, i));
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      {
        CC_PCO_ILOOP1(i)
        {
          Stau_(i) += K_dd_(a, b, i) *
                     (T00(i) * beta_u_(a, i) * beta_u_(b, i) +
                      T0i_u(a, i) * (2.0 * beta_u_(b, i)) + Tij_uu(a, b, i));
        }
      }

      // momentum source term -------------------------------------------------
      for (int a=0; a<NDIM; ++a)
      {
        CC_PCO_ILOOP1(i)
        {
          SS_d_(a, i) = -T00(i) * alpha_(i) * dalpha_d_(a, i);
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      {
        CC_PCO_ILOOP1(i)
        {
          SS_d_(a, i) += T0i_d(b, i) * dbeta_du_(a, b, i);
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      {
        CC_PCO_ILOOP1(i)
        {
          SS_d_(a, i) += dgamma_ddd_(a, b, c, i) *
                        (0.5 * T00(i) * (beta_u_(b, i) * beta_u_(c, i)) +
                         T0i_u(b, i) * beta_u_(c, i) + 0.5 * Tij_uu(b, c, i));
        }
      }
    }
    else  // GRHD
    {
      // tau-source term ------------------------------------------------------
      Stau_.ZeroClear();
      for (int a = 0; a < NDIM; ++a)
      {
        CC_PCO_ILOOP1(i)
        {
          Stau_(i) -= w_hrho_(i) * W_(i) * w_util_u_(a, i) * dalpha_d_(a, i) *
            oo_alpha_(i);

        }

        for (int b = 0; b < NDIM; ++b)
        {
          CC_PCO_ILOOP1(i)
          {
            Stau_(i) += (w_hrho_(i) * w_util_u_(a, i) * w_util_u_(b, i) +
                        w_p(k, j, i) * gamma_uu_(a, b, i)) *
                       K_dd_(a, b, i);
          }
        }
      }

      // momentum source term -------------------------------------------------
      for (int a=0; a<NDIM; ++a)
      {
        CC_PCO_ILOOP1(i)
        {
          SS_d_(a, i) = -(w_hrho_(i) * SQR(W_(i)) - w_p(k, j, i)) *
                       dalpha_d_(a, i) * oo_alpha_(i);
        }

        for (int b=0; b<NDIM; ++b)
        {
          CC_PCO_ILOOP1(i)
          {
            SS_d_(a, i) += w_hrho_(i) * W_(i) * w_util_d_(b, i) *
                           dbeta_du_(a, b, i) * oo_alpha_(i);
          }

          for (int c=0; c<NDIM; ++c)
          {
            CC_PCO_ILOOP1(i)
            {
              SS_d_(a, i) += (0.5 *
                              (w_hrho_(i) * w_util_u_(b, i) * w_util_u_(c, i) +
                               w_p(k, j, i) * gamma_uu_(b, c, i)) *
                              dgamma_ddd_(a, b, c, i));
            }
          }
        }
      }
    } // MAGNETIC_FIELDS_ENABLED

    // Add sources
    if (ph->opt_excision.use_taper && ph->opt_excision.excise_hydro_freeze_evo)
    {
      CC_PCO_ILOOP1(i)
      {
        const Real w_vol = dt * alpha_(i) * sqrt_detgamma_(i) *
                           ph->excision_mask(k,j,i);
        cons(IEN,k,j,i) += w_vol * Stau_(i);
        cons(IM1,k,j,i) += w_vol * SS_d_(0,i);
        cons(IM2,k,j,i) += w_vol * SS_d_(1,i);
        cons(IM3,k,j,i) += w_vol * SS_d_(2,i);
      }
    }
    else if (ph->opt_excision.excise_hydro_damping)
    {
      CC_PCO_ILOOP1(i)
      {
        const Real w_vol = dt * alpha_(i) * sqrt_detgamma_(i);

        cons(IEN,k,j,i) += w_vol * Stau_(i);
        cons(IM1,k,j,i) += w_vol * SS_d_(0,i);
        cons(IM2,k,j,i) += w_vol * SS_d_(1,i);
        cons(IM3,k,j,i) += w_vol * SS_d_(2,i);

        // 1 if not excising, 0 if excising
        const Real ef = ph->excision_mask(k,j,i);

        if (ef < 1)
        {
          const Real gam = (1 - ef) * ph->opt_excision.hydro_damping_factor;
          // const Real gam_w_vol = dt * alpha_(i) * sqrt_detgamma_(i) * gam;

          Real gam_w_vol = gam * (
            dt // * alpha_(i) * sqrt_detgamma_(i)
          );

#if USETM
          // scale update to land at or above floor:
          const Real den_new = cons(IDN,k,j,i) - gam_w_vol * ph->u(IDN,k,j,i);
          const Real tau_new = cons(IEN,k,j,i) - gam_w_vol * ph->u(IEN,k,j,i);

          const Real mb = pmb->peos->GetEOS().GetBaryonMass();
          const Real den_flr = (
            mb * pmb->peos->GetEOS().GetDensityFloor() * sqrt_detgamma_(i)
          );
          const Real tau_flr = 0;

          if (den_new < den_flr)
          {
            gam_w_vol = std::min(
              gam_w_vol,
              (cons(IDN,k,j,i) - den_flr) / ph->u(IDN,k,j,i)
            );
          }

          if (tau_new < tau_flr)
          {
            gam_w_vol = std::min(
              gam_w_vol,
              (cons(IEN,k,j,i) - tau_flr) / ph->u(IEN,k,j,i)
            );
          }

#endif // USETM

          cons(IDN,k,j,i) -= gam_w_vol * ph->u(IDN,k,j,i);
          cons(IM1,k,j,i) -= gam_w_vol * ph->u(IM1,k,j,i);
          cons(IM2,k,j,i) -= gam_w_vol * ph->u(IM2,k,j,i);
          cons(IM3,k,j,i) -= gam_w_vol * ph->u(IM3,k,j,i);
          cons(IEN,k,j,i) -= gam_w_vol * ph->u(IEN,k,j,i);

        }
      }
    }
    else
    {
      CC_PCO_ILOOP1(i)
      {
        // BD: TODO - consider embedded pre-factor to avoid zero-div. if alpha->0
        const Real w_vol = dt * alpha_(i) * sqrt_detgamma_(i);
        cons(IEN,k,j,i) += w_vol * Stau_(i);
        cons(IM1,k,j,i) += w_vol * SS_d_(0,i);
        cons(IM2,k,j,i) += w_vol * SS_d_(1,i);
        cons(IM3,k,j,i) += w_vol * SS_d_(2,i);
      }
    }

    // DEBUG FULL EXCISION
    if (ph->opt_excision.use_taper && ph->opt_excision.excise_hydro_taper)
    CC_PCO_ILOOP1(i)
    {
      cons(IDN,k,j,i) = ph->excision_mask(k,j,i) * cons(IDN,k,j,i);
      cons(IM1,k,j,i) = ph->excision_mask(k,j,i) * cons(IM1,k,j,i);
      cons(IM2,k,j,i) = ph->excision_mask(k,j,i) * cons(IM2,k,j,i);
      cons(IM3,k,j,i) = ph->excision_mask(k,j,i) * cons(IM3,k,j,i);
      cons(IEN,k,j,i) = ph->excision_mask(k,j,i) * cons(IEN,k,j,i);
    }

  } // j, k


}

//
// :D
//