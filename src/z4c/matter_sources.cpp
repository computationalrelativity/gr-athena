#include <iostream>
#include <fstream>

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/linear_algebra.hpp"

using namespace gra::aliases;

// ----------------------------------------------------------------------------
// Prepare mat.rho, mat.S_d, and mat.S_dd
#if USETM
void Z4c::GetMatter(
  ::AA & u_mat,
  ::AA & u_adm,
  ::AA & w,
  ::AA & r,
  ::AA & bb_cc)
#else
void Z4c::GetMatter(::AA & u_mat, ::AA & u_adm, ::AA & w, ::AA & bb_cc)
#endif
{

#if defined(Z4C_WITH_HYDRO_ENABLED)

  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;
  Hydro * phydro = pmb->phydro;
#if USETM
  PassiveScalars * pscalars = pmb->pscalars;
#endif

  EquationOfState * peos = pmb->peos;

  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);

  Matter_vars mat;
  SetMatterAliases(u_mat, mat);

  ADM_vars adm;
  Z4c_vars z4c;

  SetZ4cAliases(storage.u,   z4c);
  SetADMAliases(storage.adm, adm);

  // set up some parameters ---------------------------------------------------
#if USETM
  Real mb = peos->GetEOS().GetBaryonMass();
#else
  Real gamma_adi = peos->GetGamma();
#endif
  // --------------------------------------------------------------------------

  // quantities we have (sliced below)
  AthenaArray<Real> & sl_w = (
    (opt.fix_admsource == 0) ? w : phydro->w_init
  );
#if USETM
  AthenaArray<Real> & sl_scalars = r;
#endif

  AT_N_sca sl_w_rho(   sl_w, IDN);
  AT_N_sca sl_w_p(     sl_w, IPR);
  AT_N_vec sl_w_util_u(sl_w, IVX);
#if USETM && NSCALARS>0
  // valence 1 object with non-vector idx
  AT_S_vec sl_scalars_r(r, 0);
#endif

#if MAGNETIC_FIELDS_ENABLED
  AT_N_vec sl_bb;  // BD: TODO this is not properly populated?
  if(opt.fix_admsource==0)
  {
    sl_bb.InitWithShallowSlice(bb_cc, IB1);
    // BD: TODO check this (? what to do for fixed source)
  }
#endif

  // AT_N_sca alpha( pz4c->storage.u, Z4c::I_Z4c_alpha);
  // AT_N_vec beta_u(pz4c->storage.u, Z4c::I_Z4c_betax);

  if(opt.Tmunuinterp==0)
  {
    AT_N_sca w_hrho(   mbi.nn1);
    AT_N_sca W(        mbi.nn1);  // Lorentz factor

    AT_N_sca detgamma(mbi.nn1);
    AT_N_sca bsq(     mbi.nn1);
    AT_N_sca b0_u(    mbi.nn1);

    AT_N_vec v_d(mbi.nn1);
    AT_N_vec v_u(mbi.nn1);

#if MAGNETIC_FIELDS_ENABLED
    AT_N_vec bb_u(  mbi.nn1);
    AT_N_vec bi_u(  mbi.nn1);
    AT_N_vec bi_d(  mbi.nn1);
    AT_N_vec beta_d(mbi.nn1);
#endif

    ILOOP2(k,j)
    {
      pco_gr->GetMatterField(w_rho,      sl_w_rho,    k, j);
      pco_gr->GetMatterField(w_p,        sl_w_p,      k, j);
      pco_gr->GetMatterField(w_utilde_u, sl_w_util_u, k, j);
#if USETM && NSCALARS>0
      pco_gr->GetMatterField(w_r,        sl_scalars_r,k, j);
#endif
#if MAGNETIC_FIELDS_ENABLED
      pco_gr->GetMatterField(bb,         sl_bb,       k, j);
#endif

      ILOOP1(i)
      {
        // NB specific to EOS
#if USETM
        Real n = w_rho(i) / mb;
        // FIXME: Generalize to work with EOSes accepting particle fractions.
        Real Y[MAX_SPECIES] = {0.0};
#if NSCALARS>0
        for (int l=0; l<NSCALARS; l++)
        {
          Y[l] = w_r(l,i);
        }
#endif
        Real T = peos->GetEOS().GetTemperatureFromP(n, w_p(i), Y);

        Real Wvu[3] = { };
        for (int ix=0; ix<3; ++ix)
        {
          Wvu[ix] = w_utilde_u(ix,i);
        }

        peos->GetEOS().ApplyPrimitiveFloor(n, Wvu, w_p(i), T, Y);

        w_hrho(i) = w_rho(i)*peos->GetEOS().GetEnthalpy(n, T, Y);
#else
        w_hrho(i) = w_rho(i) + gamma_adi/(gamma_adi-1.0) * w_p(i);
#endif

        // compute Lorenz factors
        Real const norm2_utilde = InnerProductSlicedVec3Metric(w_utilde_u,
                                                               adm.g_dd,
                                                               k, j, i);
        W(i) = std::sqrt(1.0 + norm2_utilde);
        detgamma(i) = Det3Metric(adm.g_dd, k, j, i);
      }

      // extract 3-velocity
      for (int a=0; a<NDIM; ++a)
      {
        ILOOP1(i)
        {
          v_u(a,i) = w_utilde_u(a,i) / W(i);
        }
      }

      ILOOP1(i)
      {
        SlicedVecMet3Contraction(v_d, v_u, adm.g_dd, k, j, i);

#if MAGNETIC_FIELDS_ENABLED
        Real const r_sqrt_detgamma_i = 1.0/std::sqrt(detgamma(i));
        bb_u(0,i) = bb(0,i)*r_sqrt_detgamma_i;
        bb_u(1,i) = bb(1,i)*r_sqrt_detgamma_i;
        bb_u(2,i) = bb(2,i)*r_sqrt_detgamma_i;
#endif
      }

#if MAGNETIC_FIELDS_ENABLED
      b0_u.ZeroClear();
      for(int a=0;a<NDIM;++a)
      {
        ILOOP1(i)
        {
          b0_u(i) += W(i)*bb_u(a,i)*v_d(a,i)/adm.alpha(k,j,i);
        }
      }

      beta_d.ZeroClear();
      for(int a=0;a<NDIM;++a)
      {
    	  for(int b=0;b<NDIM;++b)
        {
    	    ILOOP1(i)
          {
    	      beta_d(a,i) += adm.g_dd(a,b,k,j,i) * adm.beta_u(b,k,j,i);
    	    }
	      }
      }

      for(int a=0;a<NDIM;++a)
      {
        ILOOP1(i)
        {
          bi_u(a,i) = (bb_u(a,i)+adm.alpha(k,j,i)*b0_u(i)*W(i)*(v_u(a,i)-
                       adm.beta_u(a,k,j,i)/adm.alpha(k,j,i)))/W(i);
        }
      }

    	for(int a=0;a<NDIM;++a)
      {
        ILOOP1(i)
        {
          bi_d(a,i) = beta_d(a,i) * b0_u(i);
        }

        for(int b=0;b<NDIM;++b)
        {
          ILOOP1(i)
          {
            bi_d(a,i) += adm.g_dd(a,b,k,j,i)*bi_u(b,i);
          }
        }
    	}

      ILOOP1(i)
      {
        bsq(i) = adm.alpha(k,j,i)*adm.alpha(k,j,i)*b0_u(i)*b0_u(i) / SQR(W(i));
      }

      for(int a=0;a<NDIM;++a)
      {
        for(int b=0;b<NDIM;++b)
        {
          ILOOP1(i)
          {
            bsq(i) += bb_u(a,i)*bb_u(b,i)*adm.g_dd(a,b,k,j,i) / SQR(W(i));
          }
        }
      }
#endif

    //  Update matter variables
#if MAGNETIC_FIELDS_ENABLED
      ILOOP1(i)
      {
        Real const wb_fac = (w_hrho(i)+bsq(i))*SQR(W(i));
        Real const pb_sum = (w_p(i)+bsq(i)/2.0);

        mat.rho(k,j,i) = (wb_fac-pb_sum -
                          adm.alpha(k,j,i)*adm.alpha(k,j,i)*b0_u(i)*b0_u(i));

        mat.S_d(0,k,j,i) = wb_fac*v_d(0,i)-b0_u(i)*bi_d(0,i)*adm.alpha(k,j,i);
        mat.S_d(1,k,j,i) = wb_fac*v_d(1,i)-b0_u(i)*bi_d(1,i)*adm.alpha(k,j,i);
        mat.S_d(2,k,j,i) = wb_fac*v_d(2,i)-b0_u(i)*bi_d(2,i)*adm.alpha(k,j,i);

        mat.S_dd(0,0,k,j,i) = (wb_fac*v_d(0,i)*v_d(0,i) +
                               pb_sum*adm.g_dd(0,0,k,j,i)-bi_d(0,i)*bi_d(0,i));
        mat.S_dd(0,1,k,j,i) = (wb_fac*v_d(0,i)*v_d(1,i) +
                               pb_sum*adm.g_dd(0,1,k,j,i)-bi_d(0,i)*bi_d(1,i));
        mat.S_dd(0,2,k,j,i) = (wb_fac*v_d(0,i)*v_d(2,i) +
                               pb_sum*adm.g_dd(0,2,k,j,i)-bi_d(0,i)*bi_d(2,i));
        mat.S_dd(1,1,k,j,i) = (wb_fac*v_d(1,i)*v_d(1,i) +
                               pb_sum*adm.g_dd(1,1,k,j,i)-bi_d(1,i)*bi_d(1,i));
        mat.S_dd(1,2,k,j,i) = (wb_fac*v_d(1,i)*v_d(2,i) +
                               pb_sum*adm.g_dd(1,2,k,j,i)-bi_d(1,i)*bi_d(2,i));
        mat.S_dd(2,2,k,j,i) = (wb_fac*v_d(2,i)*v_d(2,i) +
                               pb_sum*adm.g_dd(2,2,k,j,i)-bi_d(2,i)*bi_d(2,i));
      }
#else

      ILOOP1(i)
      {
        mat.rho(k,j,i) = w_hrho(i)*SQR(W(i)) - w_p(i);

        // Real oo_sqrtdg = 1.0 / std::sqrt(detgamma(i));
        // Real D   = phydro->u(IDN,k,j,i);
        // Real tau = phydro->u(IEN,k,j,i);
        // mat.rho(k,j,i) = oo_sqrtdg * (D + tau);
      }
      for (int a=0; a<NDIM; ++a)
      {
        ILOOP1(i)
        {
          mat.S_d(a,k,j,i) = w_hrho(i)*SQR(W(i))*v_d(a,i);
          // Real oo_sqrtdg = 1.0 / std::sqrt(detgamma(i));
          // mat.S_d(a,k,j,i) = phydro->u(IM1+a,k,j,i) *oo_sqrtdg;
        }
        for (int b=a; b<NDIM; ++b)
        {
          ILOOP1(i)
          {
            mat.S_dd(a,b,k,j,i) = (w_hrho(i)*SQR(W(i))*v_d(a,i)*v_d(b,i)+
                                   w_p(i)*adm.g_dd(a,b,k,j,i));

            // mat.S_dd(a,b,k,j,i) = (mat.S_d(a,k,j,i)*v_d(b,i)+
            //                        w_p(i)*adm.g_dd(a,b,k,j,i));

          }
        }
      }

#endif
    }

  }

#endif // Z4C_WITH_HYDRO_ENABLED
}
