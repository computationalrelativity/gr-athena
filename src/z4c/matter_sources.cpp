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


  int nn1 = pmb->ncells1;
  int nn2 = pmb->ncells2;
  int nn3 = pmb->ncells3;

  AthenaArray<Real> alpha;

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> W_lor, rhoadm; //lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u, v_u, v_d, Siadm_d, utilde_u; //lapse
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd, Sijadm_dd; //lapse

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
    //SetADMAliases(storage.adm_init,adm);
  }

  // Cell centred hydro vars //SB: TODO remove/cleanup *ccvars (L510 z4c.hpp)
  if(opt.fix_admsource==0)
  {
    // rhocc.InitWithShallowSlice(    w,IDN,1);
    // pgascc.InitWithShallowSlice(   w,IPR,1);
    // utilde1cc.InitWithShallowSlice(w,IVX,1);
    // utilde2cc.InitWithShallowSlice(w,IVY,1);
    // utilde3cc.InitWithShallowSlice(w,IVZ,1);
#if MAGNETIC_FIELDS_ENABLED
    bb1cc.InitWithShallowSlice(bb_cc,IB1,1);
    bb2cc.InitWithShallowSlice(bb_cc,IB2,1);
    bb3cc.InitWithShallowSlice(bb_cc,IB3,1);   //check this!
#endif
  }
  else if(opt.fix_admsource==1)
  {
    // rhocc.InitWithShallowSlice(    phydro->w_init,IDN,1);
    // pgascc.InitWithShallowSlice(   phydro->w_init,IPR,1);
    // utilde1cc.InitWithShallowSlice(phydro->w_init,IVX,1);
    // utilde2cc.InitWithShallowSlice(phydro->w_init,IVY,1);
    // utilde3cc.InitWithShallowSlice(phydro->w_init,IVZ,1);
  }




  if(opt.Tmunuinterp==0)
  {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> bb1vc,bb2vc,bb3vc, tmp, wgas, gamma_lor, v1,v2,v3, detgamma, detg, bsq, b0_u;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_d, v_u,  bb_u, bi_u, bi_d, utildevc_u, beta_d;
    // rhovc.NewAthenaTensor(nn1);
    // pgasvc.NewAthenaTensor(nn1);
    // epsvc.NewAthenaTensor(nn1);
    // utilde1vc.NewAthenaTensor(nn1);
    // utilde2vc.NewAthenaTensor(nn1);
    // utilde3vc.NewAthenaTensor(nn1);

#if MAGNETIC_FIELDS_ENABLED
    bb1vc.NewAthenaTensor(nn1);
    bb2vc.NewAthenaTensor(nn1);
    bb3vc.NewAthenaTensor(nn1);

    bb_u.NewAthenaTensor(nn1);
    bi_u.NewAthenaTensor(nn1);
    bi_d.NewAthenaTensor(nn1);
#endif
    v1.NewAthenaTensor(nn1);
    v2.NewAthenaTensor(nn1);
    v3.NewAthenaTensor(nn1);
    tmp.NewAthenaTensor(nn1);
    wgas.NewAthenaTensor(nn1);
    gamma_lor.NewAthenaTensor(nn1);
    detgamma.NewAthenaTensor(nn1);
    detg.NewAthenaTensor(nn1);
    bsq.NewAthenaTensor(nn1);
    b0_u.NewAthenaTensor(nn1);
    v_d.NewAthenaTensor(nn1);
    v_u.NewAthenaTensor(nn1);
    beta_d.NewAthenaTensor(nn1);

    utildevc_u.NewAthenaTensor(nn1);
    alpha.InitWithShallowSlice(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha, 1);

    ILOOP2(k,j)
    {
      pco_gr->GetMatterField(rho,    sl_rho,    k, j);
      pco_gr->GetMatterField(pgas,   sl_pgas,   k, j);
      pco_gr->GetMatterField(utilde, sl_utilde, k, j);

      ILOOP1(i)
      {
        // pco_gr->GetMatterField(rho, rhocc, k, j);
#ifdef HYBRID_INTERP
    	  // rhovc(i) = CCInterpolation(rhocc, k, j, i);
        // pgasvc(i) = CCInterpolation(pgascc, k, j, i);

        // utilde1vc(i) = CCInterpolation(utilde1cc, k, j, i);
        // utilde2vc(i) = CCInterpolation(utilde2cc, k, j, i);
        // utilde3vc(i) = CCInterpolation(utilde3cc, k, j, i);
#if MAGNETIC_FIELDS_ENABLED
        bb1vc(i) = CCInterpolation(bb1cc, k, j, i);
        bb2vc(i) = CCInterpolation(bb2cc, k, j, i);
        bb3vc(i) = CCInterpolation(bb3cc, k, j, i);
#endif // MAGNETIC_FIELDS_ENABLED
#else
        // rhovc(i) = ig->map3d_CC2VC(rhocc(k,j,i));
        // pgas(i) = ig->map3d_CC2VC(pgascc(k,j,i));

        // utilde1vc(i) = ig->map3d_CC2VC(utilde1cc(k,j,i));
        // utilde2vc(i) = ig->map3d_CC2VC(utilde2cc(k,j,i));
        // utilde3vc(i) = ig->map3d_CC2VC(utilde3cc(k,j,i));
#if MAGNETIC_FIELDS_ENABLED
        bb1vc(i)     = ig->map3d_CC2VC(bb1cc(k,j,i));
        bb2vc(i)     = ig->map3d_CC2VC(bb2cc(k,j,i));
        bb3vc(i)     = ig->map3d_CC2VC(bb3cc(k,j,i));
#endif
#endif // HYBRID_INTERP

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
        tmp(i) = utilde(0,i)*utilde(0,i)*adm.g_dd(0,0,k,j,i)
               + utilde(1,i)*utilde(1,i)*adm.g_dd(1,1,k,j,i)
               + utilde(2,i)*utilde(2,i)*adm.g_dd(2,2,k,j,i)
               + 2.0*utilde(0,i)*utilde(1,i)*adm.g_dd(0,1,k,j,i)
               + 2.0*utilde(0,i)*utilde(2,i)*adm.g_dd(0,2,k,j,i)
               + 2.0*utilde(1,i)*utilde(2,i)*adm.g_dd(1,2,k,j,i);

        gamma_lor(i) = sqrt(1.0+tmp(i));
        //   convert to 3-velocity
        v1(i) = utilde(0,i)/gamma_lor(i);
        v2(i) = utilde(1,i)/gamma_lor(i);
        v3(i) = utilde(2,i)/gamma_lor(i);

        v_d(0,i) = v1(i) * adm.g_dd(0,0,k,j,i) + v2(i)*adm.g_dd(0,1,k,j,i) + v3(i)*adm.g_dd(0,2,k,j,i);
        v_d(1,i) = v1(i) * adm.g_dd(0,1,k,j,i) + v2(i)*adm.g_dd(1,1,k,j,i) + v3(i)*adm.g_dd(1,2,k,j,i);
        v_d(2,i) = v1(i) * adm.g_dd(0,2,k,j,i) + v2(i)*adm.g_dd(1,2,k,j,i) + v3(i)*adm.g_dd(2,2,k,j,i);

        detgamma(i) = Det3Metric(
          adm.g_dd(0,0,k,j,i), adm.g_dd(0,1,k,j,i), adm.g_dd(0,2,k,j,i),
          adm.g_dd(1,1,k,j,i), adm.g_dd(1,2,k,j,i), adm.g_dd(2,2,k,j,i));
        detg(i) = alpha(k,j,i)*detgamma(i);

#if MAGNETIC_FIELDS_ENABLED
        bb_u(0,i) = bb1vc(i)/std::sqrt(detgamma(i));
        bb_u(1,i) = bb2vc(i)/std::sqrt(detgamma(i));
        bb_u(2,i) = bb3vc(i)/std::sqrt(detgamma(i));
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
    	      beta_d(a,i) += adm.g_dd(a,b,k,j,i)*z4c.beta_u(b,k,j,i);
    	    }
	      }
      }

      ILOOP1(i)
      {
        utildevc_u(0,i) = utilde(0,i);
        utildevc_u(1,i) = utilde(1,i);
        utildevc_u(2,i) = utilde(2,i);
        v_u(0,i)        = v1(i);
        v_u(1,i)        = v2(i);
        v_u(2,i)        = v3(i);
      }

#if MAGNETIC_FIELDS_ENABLED
      for(int a=0;a<NDIM;++a)
      {
        ILOOP1(i)
        {
          bi_u(a,i) = (bb_u(a,i) + alpha(k,j,i)*b0_u(i)*gamma_lor(i)*(v_u(a,i) - z4c.beta_u(a,k,j,i)/alpha(k,j,i)))/gamma_lor(i);
        }
      }
      /*
      bi_d.ZeroClear();
      for(int a=0;a<NDIM;++a)
      {
        for(int b=0;b<NDIM;++b)
        {
          ILOOP1(i)
          {
            bi_d(a,i) += bi_u(b,i)*adm.g_dd(a,b,k,j,i);
          }
        }
      }
      */

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
        bsq(i) = alpha(k,j,i)*alpha(k,j,i)*b0_u(i)*b0_u(i)/(gamma_lor(i)*gamma_lor(i));
      }

      for(int a=0;a<NDIM;++a)
      {
        for(int b=0;b<NDIM;++b)
        {
          ILOOP1(i)
          {
            bsq(i) += bb_u(a,i)*bb_u(b,i)*adm.g_dd(a,b,k,j,i)/(gamma_lor(i)*gamma_lor(i));
          }
        }
      }
#endif

#if MAGNETIC_FIELDS_ENABLED
      ILOOP1(i)
      {
        mat.rho(k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i)) - (pgas(i) + bsq(i)/2.0) - alpha(k,j,i)*alpha(k,j,i)*b0_u(i)*b0_u(i);
        mat.S_d(0,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(0,i) - b0_u(i)*bi_d(0,i)*alpha(k,j,i);
        mat.S_d(1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(1,i) - b0_u(i)*bi_d(1,i)*alpha(k,j,i);
        mat.S_d(2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))*v_d(2,i) - b0_u(i)*bi_d(2,i)*alpha(k,j,i);
        mat.S_dd(0,0,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(0,i) + (pgas(i)+bsq(i)/2.0)*adm.g_dd(0,0,k,j,i) - bi_d(0,i)*bi_d(0,i);
        mat.S_dd(0,1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(1,i) + (pgas(i)+bsq(i)/2.0)*adm.g_dd(0,1,k,j,i) - bi_d(0,i)*bi_d(1,i);
        mat.S_dd(0,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(0,i)*v_d(2,i) + (pgas(i)+bsq(i)/2.0)*adm.g_dd(0,2,k,j,i) - bi_d(0,i)*bi_d(2,i);
        mat.S_dd(1,1,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(1,i)*v_d(1,i) + (pgas(i)+bsq(i)/2.0)*adm.g_dd(1,1,k,j,i) - bi_d(1,i)*bi_d(1,i);
        mat.S_dd(1,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(1,i)*v_d(2,i) + (pgas(i)+bsq(i)/2.0)*adm.g_dd(1,2,k,j,i) - bi_d(1,i)*bi_d(2,i);
        mat.S_dd(2,2,k,j,i) = (wgas(i)+bsq(i))*SQR(gamma_lor(i))* v_d(2,i)*v_d(2,i) + (pgas(i)+bsq(i)/2.0)*adm.g_dd(2,2,k,j,i) - bi_d(2,i)*bi_d(2,i);
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
            mat.S_dd(a,b,k,j,i) = (wgas(i)*SQR(gamma_lor(i))*v_d(a,i)*v_d(b,i) +
                                   pgas(i)*adm.g_dd(a,b,k,j,i));
          }
        }
      }

#endif
    }
  }

}