// C++ standard headers
#include <iostream>
#include <limits>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../z4c/ahf.hpp"
#include "../z4c/z4c.hpp"
#include "../utils/linear_algebra.hpp"     // Det. & friends
#include "eos.hpp"

// External libraries
// ...

// ----------------------------------------------------------------------------
bool EquationOfState::IsAdmissiblePoint(
  const AA & cons,
  const AA & prim,
  const AT_N_sca & adm_detgamma_,
  const int k, const int j, const int i)
{
  const Real &Dg__ =   cons(IDN,k,j,i);
  /*
  const Real &taug__ = cons(IEN,k,j,i);
  const Real &S_1g__ = cons(IVX,k,j,i);
  const Real &S_2g__ = cons(IVY,k,j,i);
  const Real &S_3g__ = cons(IVZ,k,j,i);
  */
  const Real &w_rho__ = prim(IDN,k,j,i);
  /*
  const Real &w_p__   = prim(IPR,k,j,i);
  const Real &uu1__   = prim(IVX,k,j,i);
  const Real &uu2__   = prim(IVY,k,j,i);
  const Real &uu3__   = prim(IVZ,k,j,i);
  */
  const Real &adm_detgamma__ = adm_detgamma_(i);

  bool is_admissible = true;

  // Check if nan, faster to sum all values and then check if result is nan
  Real sum = 0;
  for (int n=0; n<NHYDRO; ++n)
  {
    sum += cons(n,k,j,i) + prim(n,k,j,i);
  }
  is_admissible = is_admissible && std::isfinite(sum);

  // Now check for positivity
  is_admissible = is_admissible && (adm_detgamma__ >= 0);
  is_admissible = is_admissible && (Dg__ >= 0);
  is_admissible = is_admissible && (w_rho__ >= 0);

  return is_admissible;
}

bool EquationOfState::CanExcisePoint(
  const bool is_slice,
  AT_N_sca & alpha,
  AA & x1,
  AA & x2,
  AA & x3,
  const int i, const int j, const int k)
{
  MeshBlock* pmb = pmy_block_;
  Mesh* pm = pmb->pmy_mesh;
  Hydro * ph = pmb->phydro;

  bool is_admissible = true;

  const bool alpha__ = (is_slice) ? alpha(i) : alpha(k,j,i);

  if(ph->opt_excision.horizon_based || ph->opt_excision.hybrid_hydro)
  {
    Real horizon_radius;
    for (auto pah_f : pm->pah_finder)
    {
      if (not pah_f->ah_found)
        continue;
      horizon_radius = pah_f->rr_min;
      horizon_radius *= ph->opt_excision.horizon_factor;
      const Real r_2 = SQR(x1(i) - pah_f->center[0]) +
                       SQR(x2(j) - pah_f->center[1]) +
                       SQR(x3(k) - pah_f->center[2]);

      if (ph->opt_excision.hybrid_hydro)
      {
        const Real alpha_min__ = (
          ph->opt_excision.hybrid_fac_min_alpha *
          pm->global_extrema.min_adm_alpha
        );


        const bool can_excise = (
          (r_2 < SQR(horizon_radius)) &&
          (alpha__ < alpha_min__)
        );

        is_admissible = is_admissible && !(can_excise);
      }
      else
      {
        // specify alpha cut confined within horzon
        is_admissible = is_admissible && (r_2 > SQR(horizon_radius)) &&
                        (alpha__ > ph->opt_excision.alpha_threshold);
      }
    }
  }
  else if (alpha__ < ph->opt_excision.alpha_threshold)
  {
    // Deal with pure-alpha based excision (if relevant)
    is_admissible = false;
  }

  // "can excise" => !is_admissible
  return !is_admissible;
}

bool EquationOfState::CanExcisePoint(
  Real & excision_factor,
  const bool is_slice,
  AT_N_sca & alpha,
  AA & x1,
  AA & x2,
  AA & x3,
  const int i, const int j, const int k)
{
  MeshBlock* pmb = pmy_block_;
  Mesh* pm = pmb->pmy_mesh;
  Hydro * ph = pmb->phydro;

  bool is_admissible = true;

  const bool alpha__ = (is_slice) ? alpha(i) : alpha(k,j,i);

  if(ph->opt_excision.horizon_based || ph->opt_excision.hybrid_hydro)
  {
    Real max_horizon_radius = -std::numeric_limits<Real>::infinity();
    Real horizon_radius;

    for (auto pah_f : pm->pah_finder)
    {
      if (not pah_f->ah_found)
        continue;

      horizon_radius = pah_f->rr_min;
      horizon_radius *= ph->opt_excision.horizon_factor;

      const Real r_2 = SQR(x1(i) - pah_f->center[0]) +
                       SQR(x2(j) - pah_f->center[1]) +
                       SQR(x3(k) - pah_f->center[2]);

      if (ph->opt_excision.hybrid_hydro)
      {
        const Real alpha_min__ = (
          ph->opt_excision.hybrid_fac_min_alpha *
          pm->global_extrema.min_adm_alpha
        );


        const bool can_excise = (
          (r_2 < SQR(horizon_radius)) &&
          (alpha__ < alpha_min__)
        );

        is_admissible = is_admissible && !(can_excise);
      }
      else
      {
        // specify alpha cut confined within horzon
        is_admissible = is_admissible && (r_2 > SQR(horizon_radius)) &&
                        (alpha__ > ph->opt_excision.alpha_threshold);
      }

      // excision factor based on mollifier -----------------------------------
      if (!is_admissible && (horizon_radius > max_horizon_radius))
      {
        if (ph->opt_excision.use_taper)
        {
          Real dist_ahf = std::sqrt(r_2);

          /*
          // normalized distance
          Real d_norm = dist_ahf / horizon_radius;

          // distance factor [0,1]
          const Real d = (d_norm > 1) ? 1 : d_norm;

          // BD: TODO- add this as par
          const Real s = 1; // how aggressive is cut [1,inf)

          excision_factor = 0.5 * (
            1.0 - std::tanh(s * (2.0 * d - 1.0) / (2.0 * (SQR(d) - d)))
          );
          */

          // enforce (numerically feasible) range
          const Real ef = pmb->pz4c->opt.eps_floor;
          const Real d = std::max(
            std::min(1.0-ef, dist_ahf / horizon_radius),
            ef);

          // shift taper
          const Real a = 0.0;
          const Real b = 1.0;
          const Real p = ph->opt_excision.taper_pow;

          const Real f_arg = (d - a) / (b - a);
          const Real f_d = std::exp(-1.0 / d);

          // compute such that range is [0, 1]
          excision_factor = (f_arg < ef)
            ? 0.
            : (f_arg > 1.0 - ef)
              ? 1.0
              : std::pow(1.0 / (1.0 + std::exp(1.0 + 2.0 / f_arg)), p);
        }
        else
        {
          excision_factor = 0;
        }

        // in-case we have some nested-horizon situation
        max_horizon_radius = horizon_radius;
      }
      else
      {
        excision_factor = 1;
      }
      // ----------------------------------------------------------------------
    }
  }
  else if (alpha__ < ph->opt_excision.alpha_threshold)
  {
    // Deal with pure-alpha based excision (if relevant)
    is_admissible = false;

    // for a basic threshold we take the cut to 0
    excision_factor = 0;
  }

  // "can excise" => !is_admissible
  if (is_admissible)
    excision_factor = 1;

  return !is_admissible;
}

void EquationOfState::SanitizeLoopLimits(
  int & il, int & iu,
  int & jl, int & ju,
  int & kl, int & ku,
  const bool coarse_flag,
  Coordinates *pco)
{
  GRDynamical* pco_gr;
  MeshBlock* pmb = pmy_block_;

  if (coarse_flag)
  {
    pco_gr = static_cast<GRDynamical*>(pmb->pmr->pcoarsec);
  }
  else
  {
    pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  }

  // sanitize loop limits (coarse / fine auto-switched)
  int IL, IU, JL, JU, KL, KU;
  pco_gr->GetGeometricFieldCCIdxRanges(
    IL, IU,
    JL, JU,
    KL, KU);

  // Restrict further the ranges to the input argument
  il = std::max(il, IL);
  iu = std::min(iu, IU);

  jl = std::max(jl, JL);
  ju = std::min(ju, JU);

  kl = std::max(kl, KL);
  ku = std::min(ku, KU);
}

void EquationOfState::GeometryToSlicedCC(
  geom_sliced_cc & gsc,
  const int k, const int j, const int il, const int iu,
  const bool coarse_flag,
  Coordinates *pco)
{
  GRDynamical* pco_gr;
  MeshBlock* pmb = pmy_block_;
  Z4c* pz4c      = pmb->pz4c;

  const int nn1 = (coarse_flag) ? pmb->ncv1 : pmb->nverts1;

  if (coarse_flag) {
    pco_gr = static_cast<GRDynamical*>(pmb->pmr->pcoarsec);
  }
  else {
    pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  }

  // Full 3D metric quantities
  AT_N_sym & sl_adm_gamma_dd      = gsc.sl_adm_gamma_dd;
  AT_N_sca & sl_alpha             = gsc.sl_alpha;
  AT_N_sca & sl_chi               = gsc.sl_chi;
  AT_N_sca & sl_adm_sqrt_detgamma = gsc.sl_adm_sqrt_detgamma;

  // Sliced quantities
  AT_N_sca & alpha_    = gsc.alpha_;
  AT_N_sca & rchi_     = gsc.rchi_;
  AT_N_sym & gamma_dd_ = gsc.gamma_dd_;

  // Prepare inverse + det
  AT_N_sym & gamma_uu_    = gsc.gamma_uu_;
  AT_N_sca & sqrt_det_gamma_ = gsc.sqrt_det_gamma_;
  AT_N_sca & det_gamma_   = gsc.det_gamma_;
  AT_N_sca & oo_det_gamma_= gsc.oo_det_gamma_;

  if (!gsc.is_scratch_allocated)
  {
    alpha_.NewAthenaTensor(nn1);
    gamma_dd_.NewAthenaTensor(nn1);

    gamma_uu_.NewAthenaTensor(nn1);

    sqrt_det_gamma_.NewAthenaTensor(nn1);
    det_gamma_.NewAthenaTensor(nn1);
    oo_det_gamma_.NewAthenaTensor(nn1);
  }

  if (coarse_flag)
  {
    if (!gsc.is_scratch_allocated)
    {
      rchi_.NewAthenaTensor(nn1);
    }

    sl_adm_gamma_dd.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_gxx);
    sl_alpha.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_alpha);
    sl_chi.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_chi);
  }
  else
  {
    sl_adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
    sl_alpha.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_alpha);

    // BD: TODO - clean this up
#if FLUID_ENABLED
    sl_adm_sqrt_detgamma.InitWithShallowSlice(
      pz4c->storage.aux_extended,
      Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma
    );
#endif // FLUID_ENABLED
  }

  // do the required interp. / calculation of derived quantities: -------------
  pco_gr->GetGeometricFieldCC(gamma_dd_, sl_adm_gamma_dd, k, j);
  pco_gr->GetGeometricFieldCC(alpha_, sl_alpha, k, j);

  if (coarse_flag)
  {
    pco_gr->GetGeometricFieldCC(rchi_, sl_chi, k, j);

    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      rchi_(i) = 1./rchi_(i);
    }

    for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      gamma_dd_(a, b, i) = gamma_dd_(a, b, i) * rchi_(i);
    }

  }

  if (!coarse_flag)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      sqrt_det_gamma_(i) = sl_adm_sqrt_detgamma(k,j,i);
      det_gamma_(i) = SQR(sqrt_det_gamma_(i));
      oo_det_gamma_(i) = 1. / det_gamma_(i);
      LinearAlgebra::Inv3Metric(oo_det_gamma_, gamma_dd_, gamma_uu_, i);
    }
  }
  else
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      det_gamma_(i) = LinearAlgebra::Det3Metric(gamma_dd_, i);
      sqrt_det_gamma_(i) = std::sqrt(det_gamma_(i));
      oo_det_gamma_(i) = 1. / det_gamma_(i);
      LinearAlgebra::Inv3Metric(oo_det_gamma_, gamma_dd_, gamma_uu_, i);
    }
  }

  // recycle scratch alloc. on next call
  gsc.is_scratch_allocated = true;
}

void EquationOfState::SetEuclideanCC(geom_sliced_cc & gsc, const int i)
{
  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    gsc.gamma_dd_(a,b,i) = (a==b);
  }

  gsc.sqrt_det_gamma_(i) = 1;
  gsc.det_gamma_(i) = 1;
  gsc.oo_det_gamma_(i) = 1;
}

void EquationOfState::DerivedQuantities(
  AA &derived_ms,
  AA &derived_gs,
  AA &derived_int,
  AA &cons, AA &cons_scalar,
  AA &prim, AA &prim_scalar,
  AA &adm, AA &bcc,
  Coordinates *pco,
  int il, int iu,
  int jl, int ju,
  int kl, int ku,
  int coarseflag,
  bool skip_physical)
{
  MeshBlock* pmb = pmy_block_;
  Hydro * ph     = pmb->phydro;
  Field * pf     = pmb->pfield;

#if USETM
  AA c2p_status;
  c2p_status.InitWithShallowSlice(ph->derived_ms, IX_C2P, 1);

  const Real oo_mb = OO(GetEOS().GetBaryonMass());
  Real Y[MAX_SPECIES] = {0.0};

  AT_N_sym adm_gamma(adm, Z4c::I_ADM_gxx);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    const bool sp_kj = (
      skip_physical &&
      (pmb->js <= j) && (j <= pmb->je) &&
      (pmb->ks <= k) && (k <= pmb->ke)
    );

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (sp_kj && (pmb->is <= i) && (i <= pmb->ie))
      {
        continue;
      }

      // u^a = (W/alpha, util^i)
      const Real W = ph->derived_ms(IX_LOR,k,j,i);
      const Real alp = adm(Z4c::I_ADM_alpha,k,j,i);
      const Real bx = adm(Z4c::I_ADM_betax,k,j,i);
      const Real by = adm(Z4c::I_ADM_betay,k,j,i);
      const Real bz = adm(Z4c::I_ADM_betaz,k,j,i);
      const Real gxx = adm(Z4c::I_ADM_gxx,k,j,i);
      const Real gxy = adm(Z4c::I_ADM_gxy,k,j,i);
      const Real gxz = adm(Z4c::I_ADM_gxz,k,j,i);
      const Real gyy = adm(Z4c::I_ADM_gyy,k,j,i);
      const Real gyz = adm(Z4c::I_ADM_gyz,k,j,i);
      const Real gzz = adm(Z4c::I_ADM_gzz,k,j,i);
      const Real vWx = prim(IVX,k,j,i);
      const Real vWy = prim(IVY,k,j,i);
      const Real vWz = prim(IVZ,k,j,i);
      const Real n = oo_mb * prim(IDN,k,j,i);
      const Real T = derived_ms(IX_T,k,j,i);
#if NSCALARS > 0
      for (int l=0; l<NSCALARS; ++l)
      {
        Y[l] = pmy_block_->pscalars->r(l,k,j,i);
      }
#endif

      ph->derived_ms(IX_U_D_0,k,j,i) = - alp * W +
        bx*vWx*gxx + by*vWy*gyy + bz*vWz*gzz
        + (bx*vWy + by*vWx)*gxy
        + (bx*vWz + bz*vWx)*gxz
        + (by*vWz + bz*vWy)*gyz;

      derived_ms(IX_ETH,k,j,i) = GetEOS().GetEnthalpy(n, T, Y);
      derived_ms(IX_SPB,k,j,i) = GetEOS().GetEntropyPerBaryon(n, T, Y);
      derived_ms(IX_SEN,k,j,i) = GetEOS().GetSpecificInternalEnergy(n, T, Y);
      derived_ms(IX_CS2,k,j,i) = SQR(GetEOS().GetSoundSpeed(n, T, Y));
      derived_ms(IX_CS2,k,j,i) = std::min(
        derived_ms(IX_CS2,k,j,i), max_cs2
      );

      const Real h_inf = GetEOS().GetAsymptoticEnthalpy(Y);
      derived_int(IX_HU0,k,j,i) = derived_ms(IX_ETH,k,j,i)/h_inf*derived_ms(IX_U_D_0,k,j,i);
      derived_int(IX_TR_V1,k,j,i) = alp*vWx/W + bx;
      derived_int(IX_TR_V2,k,j,i) = alp*vWy/W + by;
      derived_int(IX_TR_V3,k,j,i) = alp*vWz/W + bz;

#if MAGNETIC_FIELDS_ENABLED
      pf->derived_ms(IX_B2,k,j,i) = LinearAlgebra::InnerProductVecMetric(
        pf->bcc, adm_gamma, k, j, i
      );
#endif // MAGNETIC_FIELDS_ENABLED
    }

  }
#endif // USETM

}

//
// :D
//
