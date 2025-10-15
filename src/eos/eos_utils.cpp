// C++ standard headers
#include <iomanip>
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
void EquationOfState::StatePrintPoint(
  const std::string & tag,
  MeshBlock *pmb,
  EquationOfState::geom_sliced_cc & gsc,
  const int k, const int j, const int i,
  const bool terminate)
{
  Z4c * pz4c = pmb->pz4c;
  Hydro *ph = pmb->phydro;

  AT_N_sca sc_chi(pz4c->storage.u, Z4c::I_Z4c_chi);


  // conserved hydro
  AT_N_sca sc_cons_D(  ph->u, IDN);
  AT_N_sca sc_cons_tau(ph->u, IEN);
  AT_N_vec sp_cons_S_d(ph->u, IM1);

  // primitive hydro
  AT_N_sca sc_prim_rho(  ph->w, IDN);
  AT_N_sca sc_prim_P(ph->w, IPR);
  AT_N_vec sp_prim_util_u(ph->w, IVX);

  // Auxiliary hydro
  AT_N_sca sc_aux_T(ph->derived_ms, IX_T);
  AT_N_sca sc_aux_W(ph->derived_ms, IX_LOR);
  AT_N_sca sc_aux_h(ph->derived_ms, IX_ETH);

  #pragma omp critical
  {
    std::cout << "eos_utils::DebugState" << std::endl;
    std::cout << "Tag: \n";
    std::cout << tag << "\n";
    std::cout << std::setprecision(14) << std::endl;
    std::cout << "k, j, i:     " << k << ", " << j << ", " << i << "\n";

    std::cout << "x3(k):   " << pmb->pcoord->x3v(k) << "\n";
    std::cout << "x2(j):   " << pmb->pcoord->x2v(j) << "\n";
    std::cout << "x1(i):   " << pmb->pcoord->x1v(i) << "\n";

    std::cout << "geometric fields========================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    gsc.sl_adm_sqrt_detgamma.PrintPoint("gsc.sl_adm_sqrt_detgamma", k,j,i);
    gsc.sl_alpha.PrintPoint("gsc.sl_alpha", k,j,i);
    sc_chi.PrintPoint("sl_chi", k,j,i);

    std::cout << "vec================: " << "\n";
    gsc.sl_beta_u.PrintPoint("gsc.sl_beta_u", k,j,i);

    std::cout << "sym2===============: " << "\n";
    gsc.sl_adm_gamma_dd.PrintPoint("geom.sl_adm_gamma_dd", k,j,i);

    std::cout << "geometric fields [sliced]===============: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    gsc.sqrt_det_gamma_.PrintPoint("gsc.sqrt_det_gamma_", i);
    gsc.det_gamma_.PrintPoint("gsc.det_gamma_", i);
    gsc.alpha_.PrintPoint("gsc.alpha_", i);

    std::cout << "vec================: " << "\n";
    gsc.beta_u_.PrintPoint("gsc.beta_u_", i);

    std::cout << "sym2===============: " << "\n";
    gsc.gamma_dd_.PrintPoint("gsc.gamma_dd_", i);
    gsc.gamma_uu_.PrintPoint("gsc.gamma_uu_", i);

    std::cout << "hydro fields [prim]=====================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    sc_prim_rho.PrintPoint("sc_prim_rho", k,j,i);
    sc_prim_P.PrintPoint("sc_prim_P", k,j,i);

    std::cout << "vec================: " << "\n";
    sp_prim_util_u.PrintPoint("sp_prim_util_u", k,j,i);

    std::cout << "hydro fields [cons]=====================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    sc_cons_D.PrintPoint("sc_cons_D", k,j,i);
    sc_cons_tau.PrintPoint("sc_cons_tau", k,j,i);

    std::cout << "vec================: " << "\n";
    sp_cons_S_d.PrintPoint("sp_cons_S_d", k,j,i);

    std::cout << "hydro (aux) fields======================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    sc_aux_W.PrintPoint("sc_aux_W", k,j,i);
    sc_aux_T.PrintPoint("sc_aux_T", k,j,i);
    sc_aux_h.PrintPoint("sc_aux_h", k,j,i);

  }

  if (terminate)
  {
    assert(false);
  }
}

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
  AT_N_vec & sl_beta_u            = gsc.sl_beta_u;

  // Sliced quantities
  AT_N_sca & alpha_    = gsc.alpha_;
  AT_N_sca & rchi_     = gsc.rchi_;
  AT_N_vec & beta_u_   = gsc.beta_u_;
  AT_N_sym & gamma_dd_ = gsc.gamma_dd_;

  // Prepare inverse + det
  AT_N_sym & gamma_uu_    = gsc.gamma_uu_;
  AT_N_sca & sqrt_det_gamma_ = gsc.sqrt_det_gamma_;
  AT_N_sca & det_gamma_   = gsc.det_gamma_;
  AT_N_sca & oo_det_gamma_= gsc.oo_det_gamma_;

  if (!gsc.is_scratch_allocated)
  {
    alpha_.NewAthenaTensor(nn1);
    beta_u_.NewAthenaTensor(nn1);
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
    sl_beta_u.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_betax);
  }
  else
  {
    sl_adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
    sl_alpha.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_alpha);
    sl_beta_u.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_betax);

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
  pco_gr->GetGeometricFieldCC(beta_u_, sl_beta_u, k, j);

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
  AA &hyd_der_ms,
  AA &hyd_der_int,
  AA &fld_der_ms,
  AA &cons, AA &cons_scalar,
  AA &prim, AA &prim_scalar,
  AA &bcc, geom_sliced_cc & gsc,
  Coordinates *pco,
  int k,
  int j,
  int il, int iu,
  int coarseflag,
  bool skip_physical)
{
  MeshBlock* pmb = pmy_block_;
  Hydro * ph     = pmb->phydro;
  Field * pf     = pmb->pfield;

#if USETM
  const Real oo_mb = OO(GetEOS().GetBaryonMass());
  Real Y[MAX_SPECIES] = {0.0};

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

    for (int l=0; l<NSCALARS; ++l)
    {
      Y[l] = prim_scalar(l,k,j,i);
    }

    // u^a = (W/alpha, util^i)
    const Real x = pco->x1v(i);
    const Real y = pco->x2v(j);

    const Real W = ph->derived_ms(IX_LOR,k,j,i);

    const Real alp = gsc.alpha_(i);
    const Real bx = gsc.beta_u_(0,i);
    const Real by = gsc.beta_u_(1,i);
    const Real bz = gsc.beta_u_(2,i);
    const Real gxx = gsc.gamma_dd_(0,0,i);
    const Real gxy = gsc.gamma_dd_(0,1,i);
    const Real gxz = gsc.gamma_dd_(0,2,i);
    const Real gyy = gsc.gamma_dd_(1,1,i);
    const Real gyz = gsc.gamma_dd_(1,2,i);
    const Real gzz = gsc.gamma_dd_(2,2,i);

    const Real vWx = prim(IVX,k,j,i);
    const Real vWy = prim(IVY,k,j,i);
    const Real vWz = prim(IVZ,k,j,i);

    const Real rho = prim(IDN,k,j,i);
    const Real n = oo_mb * prim(IDN,k,j,i);
    const Real T = hyd_der_ms(IX_T,k,j,i);
    const Real h = hyd_der_ms(IX_ETH,k,j,i);
    const Real h_inf = GetEOS().GetAsymptoticEnthalpy(Y);

    hyd_der_ms(IX_U_d_0,k,j,i) = - alp * W +
      bx*vWx*gxx + by*vWy*gyy + bz*vWz*gzz
      + (bx*vWy + by*vWx)*gxy
      + (bx*vWz + bz*vWx)*gxz
      + (by*vWz + bz*vWy)*gyz;

    hyd_der_ms(IX_SPB,k,j,i) = (T > 0) ? GetEOS().GetEntropyPerBaryon(n, T, Y) : 0;
    hyd_der_ms(IX_SEN,k,j,i) = (T > 0) ? GetEOS().GetSpecificInternalEnergy(n, T, Y) : 0;
    hyd_der_ms(IX_HU_d_0,k,j,i) = h/h_inf * hyd_der_ms(IX_U_d_0,k,j,i);

    hyd_der_ms(IX_CS2,k,j,i) = (T > 0) ? SQR(GetEOS().GetSoundSpeed(n, T, Y)) : 0;
    hyd_der_ms(IX_CS2,k,j,i) = std::min(
      hyd_der_ms(IX_CS2,k,j,i), max_cs2
    );
    hyd_der_ms(IX_OM,k,j,i) = (y*vWx - x*vWy)/std::sqrt(SQR(x) + SQR(y));


#if MAGNETIC_FIELDS_ENABLED
    fld_der_ms(IX_B2,k,j,i) = LinearAlgebra::InnerProductVecSlicedMetric(
      pf->bcc, gsc.gamma_dd_, k, j, i
    );

    const Real v_u_x = vWx / W;
    const Real v_u_y = vWy / W;
    const Real v_u_z = vWz / W;

    const Real oo_sqrt_det_gamma = OO(
      gsc.sqrt_det_gamma_(i)
    );

    fld_der_ms(IX_b0,k,j,i) = oo_sqrt_det_gamma * (W / alp) * (
      gxx * pf->bcc(IB1,k,j,i) * v_u_x +
      gyy * pf->bcc(IB2,k,j,i) * v_u_y +
      gzz * pf->bcc(IB3,k,j,i) * v_u_z +
      gxy * (pf->bcc(IB1,k,j,i) * v_u_y +
              pf->bcc(IB2,k,j,i) * v_u_x) +
      gxz * (pf->bcc(IB1,k,j,i) * v_u_z +
              pf->bcc(IB3,k,j,i) * v_u_x) +
      gyz * (pf->bcc(IB2,k,j,i) * v_u_z +
              pf->bcc(IB3,k,j,i) * v_u_y)
    );

    fld_der_ms(IX_b2,k,j,i) = (
      SQR(alp * fld_der_ms(IX_b0,k,j,i)) +
      SQR(oo_sqrt_det_gamma) * fld_der_ms(IX_B2,k,j,i)
    ) / SQR(W);

    const Real vtil_u_x = v_u_x - bx / alp;
    const Real vtil_u_y = v_u_y - by / alp;
    const Real vtil_u_z = v_u_z - bz / alp;

    fld_der_ms(IX_b_U_1,k,j,i) = (
      oo_sqrt_det_gamma * pf->bcc(IB1,k,j,i) +
      alp * fld_der_ms(IX_b0,k,j,i) * W * vtil_u_x
    ) / W;

    fld_der_ms(IX_b_U_2,k,j,i) = (
      oo_sqrt_det_gamma * pf->bcc(IB2,k,j,i) +
      alp * fld_der_ms(IX_b0,k,j,i) * W * vtil_u_y
    ) / W;

    fld_der_ms(IX_b_U_3,k,j,i) = (
      oo_sqrt_det_gamma * pf->bcc(IB3,k,j,i) +
      alp * fld_der_ms(IX_b0,k,j,i) * W * vtil_u_z
    ) / W;

    fld_der_ms(IX_MAG,k,j,i) = fld_der_ms(IX_B2,k,j,i) / cons(IDN,k,j,i);
    fld_der_ms(IX_BET,k,j,i) = fld_der_ms(IX_b2,k,j,i) / (2.0 * prim(IPR,k,j,i)) ;
    fld_der_ms(IX_MRI,k,j,i) = fld_der_ms(IX_b_U_3,k,j,i) / (std::sqrt(rho) * hyd_der_ms(IX_OM,k,j,i));
    fld_der_ms(IX_ALF,k,j,i) = std::sqrt(fld_der_ms(IX_B2,k,j,i) / (4.0 * M_PI * rho));


#endif // MAGNETIC_FIELDS_ENABLED
  }

#endif // USETM

}

bool EquationOfState::NeighborsEncloseValue(
  const AA &src,
  const int n,
  const int k,
  const int j,
  const int i,
  const AA_B &mask,
  const int num_neighbors,
  const bool exclude_first_extrema,
  const Real fac_min,
  const Real fac_max
)
{
  MeshBlock* pmb = pmy_block_;

  // Calculate average of nearest neighbor vals
  Real avg_val = 0.0;
  int count = 0;

  Real min_val = std::numeric_limits<Real>::max();
  Real max_val = -std::numeric_limits<Real>::max();

  for (int kk = -num_neighbors; kk <= num_neighbors; ++kk)
  for (int jj = -num_neighbors; jj <= num_neighbors; ++jj)
  for (int ii = -num_neighbors; ii <= num_neighbors; ++ii)
  {
    if (ii == 0 && jj == 0 && kk == 0) continue;

    const int i_ix = i + ii;
    const int j_ix = j + jj;
    const int k_ix = k + kk;

    if ((i_ix < 0) || (i_ix > pmb->ncells1-1))
      continue;

    if ((j_ix < 0) || (j_ix > pmb->ncells2-1))
      continue;

    if ((k_ix < 0) || (k_ix > pmb->ncells3-1))
      continue;

    if (!mask(k_ix,j_ix,i_ix)) continue;

    const Real val = src(n,k_ix,j_ix,i_ix);

    min_val = std::min(val, min_val);
    max_val = std::max(val, max_val);
  }

  const Real val = src(n,k,j,i);

  bool nn_enclosing = (fac_min * min_val <= val) && (val <= fac_max * max_val);
  return nn_enclosing;
}

void EquationOfState::NearestNeighborSmooth(
  AA &tar,
  const AA &src,
  const int kl, const int ku,
  const int jl, const int ju,
  const int il, const int iu,
  bool exclude_first_extrema)
{
  MeshBlock* pmb = pmy_block_;
  Hydro * ph     = pmb->phydro;
  Field * pf     = pmb->pfield;

  for (int k = kl; k <= ku; ++k)
  for (int j = jl; j <= ju; ++j)
  {

    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      // Calculate average of nearest neighbor vals
      Real avg_val = 0.0;
      int count = 0;

      Real min_val = std::numeric_limits<Real>::max();
      Real max_val = -std::numeric_limits<Real>::max();

      for (int kk = -1; kk <= 1; ++kk)
      for (int jj = -1; jj <= 1; ++jj)
      for (int ii = -1; ii <= 1; ++ii)
      {
        if (ii == 0 && jj == 0 && kk == 0) continue;

        const int i_ix = i + ii;
        const int j_ix = j + jj;
        const int k_ix = k + kk;

        if ((i_ix < 0) || (i_ix > pmb->ncells1-1))
          continue;

        if ((j_ix < 0) || (j_ix > pmb->ncells2-1))
          continue;

        if ((k_ix < 0) || (k_ix > pmb->ncells3-1))
          continue;

        const Real val = src(k_ix,j_ix,i_ix);

        avg_val += val;

        if (exclude_first_extrema)
        {
          // Track min/max values
          min_val = std::min(min_val, val);
          max_val = std::max(max_val, val);
        }

        count++;
      }

      if (exclude_first_extrema)
      {
        avg_val -= min_val;
        avg_val -= max_val;

        count -= 2;
      }

      avg_val /= count;

      tar(k,j,i) = avg_val;
    }
  }

}

Real EquationOfState::NearestNeighborSmooth(
  const AA &src,
  const int n,
  const int k,
  const int j,
  const int i,
  const AA_B &mask,
  const int num_neighbors,
  const bool keep_base_point,
  const bool exclude_first_extrema,
  const bool use_hybrid_mean_median,
  const Real sigma_frac
)
{
  MeshBlock* pmb = pmy_block_;

  // Collect neighbor values in a fixed-size array
  const int max_neighbors = (2*num_neighbors+1)*(2*num_neighbors+1)*(2*num_neighbors+1);
  Real neighbor_vals[max_neighbors];
  int count = 0;

  Real min_val = std::numeric_limits<Real>::max();
  Real max_val = -std::numeric_limits<Real>::max();

  for (int kk = -num_neighbors; kk <= num_neighbors; ++kk)
  for (int jj = -num_neighbors; jj <= num_neighbors; ++jj)
  for (int ii = -num_neighbors; ii <= num_neighbors; ++ii)
  {
    if (!keep_base_point && ii == 0 && jj == 0 && kk == 0) continue;

    const int i_ix = i + ii;
    const int j_ix = j + jj;
    const int k_ix = k + kk;

    if ((i_ix < 0) || (i_ix >= pmb->ncells1)) continue;
    if ((j_ix < 0) || (j_ix >= pmb->ncells2)) continue;
    if ((k_ix < 0) || (k_ix >= pmb->ncells3)) continue;
    if (!mask(k_ix,j_ix,i_ix)) continue;

    const Real val = src(n,k_ix,j_ix,i_ix);

    neighbor_vals[count++] = val;

    if (exclude_first_extrema)
    {
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }
  }

  if (count == 0) return 0.0;  // nothing to average

  // Exclude first extrema if requested
  if (exclude_first_extrema && count > 2)
  {
    int min_idx = 0;
    int max_idx = 0;
    for (int m = 0; m < count; ++m)
    {
      if (neighbor_vals[m] == min_val) min_idx = m;
      if (neighbor_vals[m] == max_val) max_idx = m;
    }

    // Swap extrema to the end and reduce count
    std::swap(neighbor_vals[min_idx], neighbor_vals[count-1]);
    std::swap(neighbor_vals[max_idx], neighbor_vals[count-2]);
    count -= 2;
  }

  // Hybrid mean-median smoothing
  if (use_hybrid_mean_median)
  {
    // Simple bubble sort for small arrays (fixed-size)
    for (int m = 0; m < count-1; ++m)
      for (int n2 = m+1; n2 < count; ++n2)
        if (neighbor_vals[m] > neighbor_vals[n2]) std::swap(neighbor_vals[m], neighbor_vals[n2]);

    Real median = neighbor_vals[count/2];

    Real sum = 0.0;
    int n_valid = 0;
    for (int m = 0; m < count; ++m)
    {
      if (fabs(neighbor_vals[m] - median) <= sigma_frac*median)
      {
        sum += neighbor_vals[m];
        n_valid++;
      }
    }

    if (n_valid > 0)
      return sum / n_valid;
    else
      return median;  // fallback
  }

  // Default: simple average
  Real avg_val = 0.0;
  for (int m = 0; m < count; ++m) avg_val += neighbor_vals[m];
  return avg_val / count;
}

Real EquationOfState::NearestNeighborSmoothWeighted(
  const AA &src,
  const int n,
  const int k,
  const int j,
  const int i,
  const AA_B &mask,
  const int num_neighbors,
  const bool keep_base_point,
  const bool exclude_first_extrema,
  const bool use_hybrid_mean_median,
  const Real sigma_frac,
  const Real alpha
)
{
  MeshBlock* pmb = pmy_block_;

  const int max_neighbors = (2*num_neighbors+1)*(2*num_neighbors+1)*(2*num_neighbors+1);
  Real neighbor_vals[max_neighbors];
  Real neighbor_weights[max_neighbors];
  int count = 0;

  Real min_val = std::numeric_limits<Real>::max();
  Real max_val = -std::numeric_limits<Real>::max();

  // Collect neighbors and compute distance-based weights
  for (int kk = -num_neighbors; kk <= num_neighbors; ++kk)
  for (int jj = -num_neighbors; jj <= num_neighbors; ++jj)
  for (int ii = -num_neighbors; ii <= num_neighbors; ++ii)
  {
    if (!keep_base_point && ii == 0 && jj == 0 && kk == 0) continue;

    const int i_ix = i + ii;
    const int j_ix = j + jj;
    const int k_ix = k + kk;

    if ((i_ix < 0) || (i_ix >= pmb->ncells1)) continue;
    if ((j_ix < 0) || (j_ix >= pmb->ncells2)) continue;
    if ((k_ix < 0) || (k_ix >= pmb->ncells3)) continue;
    if (!mask(k_ix,j_ix,i_ix)) continue;

    const Real val = src(n,k_ix,j_ix,i_ix);
    neighbor_vals[count] = val;

    Real dist2 = ii*ii + jj*jj + kk*kk;
    Real w = (dist2 == 0.0) ? 1.0 : 1.0/std::sqrt(dist2);
    neighbor_weights[count] = w;

    if (exclude_first_extrema)
    {
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }

    count++;
  }

  if (count == 0) return src(n,k,j,i); // fallback if no valid neighbors

  // Exclude first extrema if requested
  if (exclude_first_extrema && count > 2)
  {
    int min_idx = 0;
    int max_idx = 0;
    for (int m = 0; m < count; ++m)
    {
      if (neighbor_vals[m] == min_val) min_idx = m;
      if (neighbor_vals[m] == max_val) max_idx = m;
    }
    std::swap(neighbor_vals[min_idx], neighbor_vals[count-1]);
    std::swap(neighbor_vals[max_idx], neighbor_vals[count-2]);
    std::swap(neighbor_weights[min_idx], neighbor_weights[count-1]);
    std::swap(neighbor_weights[max_idx], neighbor_weights[count-2]);
    count -= 2;
  }

  // Weighted average
  Real weighted_sum = 0.0;
  Real total_weight = 0.0;

  if (keep_base_point)
  {
    Real base_val = src(n,k,j,i);
    weighted_sum += alpha * base_val;
    total_weight += alpha;
  }

  for (int m = 0; m < count; ++m)
  {
    Real w = neighbor_weights[m] * (keep_base_point ? (1.0 - alpha) : 1.0);
    weighted_sum += w * neighbor_vals[m];
    total_weight += w;
  }

  Real avg_val = weighted_sum / total_weight;

  // Hybrid mean-median option
  if (use_hybrid_mean_median)
  {
    // sort neighbors
    for (int m = 0; m < count-1; ++m)
      for (int n2 = m+1; n2 < count; ++n2)
        if (neighbor_vals[m] > neighbor_vals[n2]) std::swap(neighbor_vals[m], neighbor_vals[n2]);

    Real median = neighbor_vals[count/2];
    Real max_dev = sigma_frac * median;

    avg_val = std::min(std::max(avg_val, median - max_dev), median + max_dev);
  }

  return avg_val;
}

Real EquationOfState::NearestNeighborSmooth(
  const AA &src,
  const int n,
  const int k,
  const int j,
  const int i,
  const AA_B &mask,
  const int num_neighbors,
  const bool keep_base_point,
  const bool exclude_first_extrema,
  const bool use_robust_weights,
  const Real alpha,
  const Real sigma_frac,
  const Real max_dev_frac,
  const Real sigma_s_frac
)
{
  MeshBlock* pmb = pmy_block_;
  Hydro * ph     = pmb->phydro;
  Field * pf     = pmb->pfield;

  Real avg_val = 0.0;
  Real sum_w   = 0.0;

  Real min_val = std::numeric_limits<Real>::max();
  Real max_val = -std::numeric_limits<Real>::max();
  Real w_min = 0.0;
  Real w_max = 0.0;

  // fallback for distance sigma
  Real sigma_s = sigma_s_frac * num_neighbors;
  if (sigma_s <= 0.0) sigma_s = 1.0;

  for (int kk = -num_neighbors; kk <= num_neighbors; ++kk)
  for (int jj = -num_neighbors; jj <= num_neighbors; ++jj)
  for (int ii = -num_neighbors; ii <= num_neighbors; ++ii)
  {
    if (!keep_base_point && ii == 0 && jj == 0 && kk == 0) continue;

    if (!mask(k,j,i)) continue;

    const int i_ix = i + ii;
    const int j_ix = j + jj;
    const int k_ix = k + kk;

    if ((i_ix < 0) || (i_ix >= pmb->ncells1)) continue;
    if ((j_ix < 0) || (j_ix >= pmb->ncells2)) continue;
    if ((k_ix < 0) || (k_ix >= pmb->ncells3)) continue;

    const Real val = src(n,k_ix,j_ix,i_ix);

    // --- distance weighting ---
    Real r2 = static_cast<Real>(ii*ii + jj*jj + kk*kk);
    Real w = std::exp(-r2 / (2.0 * sigma_s * sigma_s));

    avg_val += w * val;
    sum_w   += w;

    if (exclude_first_extrema)
    {
      if (val < min_val) { min_val = val; w_min = w; }
      if (val > max_val) { max_val = val; w_max = w; }
    }
  }

  if (exclude_first_extrema && sum_w > 0.0)
  {
    // Remove min/max contributions (approximate)
    avg_val -= min_val;
    avg_val -= max_val;
    sum_w   -= w_min + w_max;
  }

  if (sum_w > 0.0) avg_val /= sum_w;

  return avg_val;
}


//
// :D
//
