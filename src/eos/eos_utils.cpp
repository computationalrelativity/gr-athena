// C++ standard headers
#include <iostream>
#include <limits>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
// #include "../utils/linear_algebra.hpp"
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
  is_admissible = is_admissible && (!std::isnan(is_admissible));

  // Now check for positivity
  is_admissible = is_admissible && (adm_detgamma__ >= 0);
  is_admissible = is_admissible && (Dg__ >= 0);
  is_admissible = is_admissible && (w_rho__ >= 0);

  return is_admissible;
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
  AT_N_sym & sl_adm_gamma_dd = gsc.sl_adm_gamma_dd;
  AT_N_sca & sl_alpha        = gsc.sl_alpha;
  AT_N_sca & sl_chi          = gsc.sl_chi;

  // Sliced quantities
  AT_N_sca & alpha_    = gsc.alpha_;
  AT_N_sca & rchi_     = gsc.rchi_;
  AT_N_sym & gamma_dd_ = gsc.gamma_dd_;

  // Prepare inverse + det
  AT_N_sym & gamma_uu_    = gsc.gamma_uu_;
  AT_N_sca & det_gamma_   = gsc.det_gamma_;
  AT_N_sca & oo_det_gamma_= gsc.oo_det_gamma_;

  if (!gsc.is_scratch_allocated)
  {
    alpha_.NewAthenaTensor(nn1);
    gamma_dd_.NewAthenaTensor(nn1);

    gamma_uu_.NewAthenaTensor(nn1);
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

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    det_gamma_(i) = LinearAlgebra::Det3Metric(gamma_dd_, i);
    oo_det_gamma_(i) = 1. / det_gamma_(i);
    LinearAlgebra::Inv3Metric(oo_det_gamma_, gamma_dd_, gamma_uu_, i);
  }

  // recycle scratch alloc. on next call
  gsc.is_scratch_allocated = true;
}

void EquationOfState::SetEuclideanCC(AT_N_sym & gamma_dd_, const int i)
{
  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    gamma_dd_(a,b,i) = (a==b);
  }
}

//
// :D
//