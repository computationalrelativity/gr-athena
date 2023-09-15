#include <iostream>
#include <fstream>

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/linear_algebra.hpp"

#include "../utils/interp_intergrid.hpp" //SB FIXME imported from matter_tracker_extrema

// ----------------------------------------------------------------------------
// Prepare mat.rho, mat.S_d, and mat.S_dd
void Z4c::GetMatter(
  AthenaArray<Real> & u_mat,
  AthenaArray<Real> & u_adm,
  AthenaArray<Real> & w,
  AthenaArray<Real> & bb_cc)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;
  Hydro * phydro = pmb->phydro;
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);

  Matter_vars mat;
  SetMatterAliases(u_mat, mat);

  // set up some parameters ---------------------------------------------------
#if USETM
  Real mb = pmb->peos->GetEOS().GetBaryonMass();
#else
  Real gamma_adi = pmb->peos->GetGamma(); //NB specific to EOS
#endif
  // --------------------------------------------------------------------------

  // quantities we have (sliced below)
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> sl_rho;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> sl_pgas;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> sl_utilde;

  AthenaArray<Real> & sl_w = (
    (opt.fix_admsource == 0) ? w : phydro->w_init
  );

  sl_rho.InitWithShallowSlice(   sl_w, IDN);
  sl_pgas.InitWithShallowSlice(  sl_w, IPR);
  sl_utilde.InitWithShallowSlice(sl_w, IVX);

#if MAGNETIC_FIELDS_ENABLED
  if(opt.fix_admsource==0)
  {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> sl_bb;
    sl_bb.InitWithShallowSlice(bb_cc, IB1);
    // BD: TODO check this (? what to do for fixed source)
  }
#endif

  AthenaArray<Real> alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u, v_u, v_d;

  ADM_vars adm;
  Z4c_vars z4c;
  SetZ4cAliases(storage.u,z4c);

  if(opt.fix_admsource==0)
  {
    SetADMAliases(u_adm,adm);
  }
  else if (opt.fix_admsource==1)
  {
    SetADMAliases(u_adm,adm);
    // BD: TODO 
    //SetADMAliases(storage.adm_init,adm);
  }

  if(opt.Tmunuinterp==0)
  {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> wgas, gamma_lor, detgamma, bsq, b0_u;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_d, v_u, bb_u, bi_u, bi_d, beta_d;

#if MAGNETIC_FIELDS_ENABLED
    bb_u.NewAthenaTensor(mbi.nn1);
    bi_u.NewAthenaTensor(mbi.nn1);
    bi_d.NewAthenaTensor(mbi.nn1);
#endif
    wgas.NewAthenaTensor(mbi.nn1);
    gamma_lor.NewAthenaTensor(mbi.nn1);
    detgamma.NewAthenaTensor(mbi.nn1);
    bsq.NewAthenaTensor(mbi.nn1);
    b0_u.NewAthenaTensor(mbi.nn1);
    v_d.NewAthenaTensor(mbi.nn1);
    v_u.NewAthenaTensor(mbi.nn1);
    beta_d.NewAthenaTensor(mbi.nn1);

    alpha.InitWithShallowSlice(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha, 1);

    ILOOP2(k,j)
    {
      pco_gr->GetMatterField(rho,    sl_rho,    k, j);
      pco_gr->GetMatterField(pgas,   sl_pgas,   k, j);
      pco_gr->GetMatterField(utilde, sl_utilde, k, j);

#if MAGNETIC_FIELDS_ENABLED
      pco_gr->GetMatterField(bb,     sl_bb,     k, j);
#endif

      ILOOP1(i)
      {
        // NB specific to EOS
#if USETM
        Real n = rho(i)/mb;
        // FIXME: Generalize to work with EOSes accepting particle fractions.
        Real Y[MAX_SPECIES] = {0.0};
        Real T = pmb->peos->GetEOS().GetTemperatureFromP(n, pgas(i), Y);
        wgas(i) = n*pmb->peos->GetEOS().GetEnthalpy(n, T, Y);
#else
        wgas(i) = rho(i) + gamma_adi/(gamma_adi-1.0) * pgas(i);
#endif

        // compute Lorenz factors
        Real const norm_utilde = InnerProductSlicedVec3Metric(utilde, adm.g_dd,
                                                              k, j, i);
        gamma_lor(i) = sqrt(1.0+norm_utilde);
        detgamma(i) = Det3Metric(adm.g_dd, k, j, i);
      }

      // extract 3-velocity
      for (int a=0; a<NDIM; ++a)
      {
        ILOOP1(i)
        {
          v_u(a,i) = utilde(a,i) / gamma_lor(i);
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
      // b0_u = 0.0;
#if MAGNETIC_FIELDS_ENABLED
      b0_u.ZeroClear();
      for(int a=0;a<NDIM;++a)
      {
        ILOOP1(i)
        {
          b0_u(i) += gamma_lor(i)*bb_u(a,i)*v_d(a,i)/alpha(k,j,i);
        }
      }
#endif

      beta_d.ZeroClear();
      for(int a=0;a<NDIM;++a)
      {
    	  for(int b=0;b<NDIM;++b)
        {
    	    ILOOP1(i)
          {
    	      beta_d(a,i) += adm.g_dd(a,b,k,j,i) * z4c.beta_u(b,k,j,i);
    	    }
	      }
      }

#if MAGNETIC_FIELDS_ENABLED
      for(int a=0;a<NDIM;++a)
      {
        ILOOP1(i)
        {
          bi_u(a,i) = (bb_u(a,i)+alpha(k,j,i)*b0_u(i)*gamma_lor(i)*(v_u(a,i)-
                       z4c.beta_u(a,k,j,i)/alpha(k,j,i)))/gamma_lor(i);
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
        bsq(i) = (alpha(k,j,i)*alpha(k,j,i)*b0_u(i)*b0_u(i) /
                  (gamma_lor(i)*gamma_lor(i)));
      }

      for(int a=0;a<NDIM;++a)
      {
        for(int b=0;b<NDIM;++b)
        {
          ILOOP1(i)
          {
            bsq(i) += (bb_u(a,i)*bb_u(b,i)*adm.g_dd(a,b,k,j,i) /
                       (gamma_lor(i)*gamma_lor(i)));
          }
        }
      }
#endif

#if MAGNETIC_FIELDS_ENABLED
      ILOOP1(i)
      {
        Real const gam_lor2 = SQR(gamma_lor(i));
        Real const wb_fac = (wgas(i)+bsq(i))*gam_lor2;
        Real const pb_sum = (pgas(i)+bsq(i)/2.0);

        mat.rho(k,j,i) = (wb_fac-pb_sum -
                          alpha(k,j,i)*alpha(k,j,i)*b0_u(i)*b0_u(i));

        mat.S_d(0,k,j,i) = wb_fac*v_d(0,i)-b0_u(i)*bi_d(0,i)*alpha(k,j,i);
        mat.S_d(1,k,j,i) = wb_fac*v_d(1,i)-b0_u(i)*bi_d(1,i)*alpha(k,j,i);
        mat.S_d(2,k,j,i) = wb_fac*v_d(2,i)-b0_u(i)*bi_d(2,i)*alpha(k,j,i);

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
        mat.rho(k,j,i) = wgas(i)*SQR(gamma_lor(i)) - pgas(i);
      }
      for (int a=0; a<NDIM; ++a)
      {
        ILOOP1(i)
        {
          mat.S_d(a,k,j,i) = wgas(i)*SQR(gamma_lor(i))*v_d(a,i);
        }
        for (int b=a; b<NDIM; ++b)
        {
          ILOOP1(i)
          {
            mat.S_dd(a,b,k,j,i) = (wgas(i)*SQR(gamma_lor(i))*v_d(a,i)*v_d(b,i)+
                                   pgas(i)*adm.g_dd(a,b,k,j,i));
          }
        }
      }

#endif
    }
  }


// cleanup
  if(opt.Tmunuinterp==0)
  {
#if MAGNETIC_FIELDS_ENABLED
    bb_u.DeleteAthenaTensor();
    bi_u.DeleteAthenaTensor();
    bi_d.DeleteAthenaTensor();
#endif
    wgas.DeleteAthenaTensor();
    gamma_lor.DeleteAthenaTensor();
    detgamma.DeleteAthenaTensor();
    bsq.DeleteAthenaTensor();
    b0_u.DeleteAthenaTensor();
    v_d.DeleteAthenaTensor();
    v_u.DeleteAthenaTensor();
    beta_d.DeleteAthenaTensor();
  }

}