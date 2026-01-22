#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>

// Athena++ headers
#include "smagorinsky.hpp"
#include "../athena.hpp"                   // enums, macros
#include "../athena_aliases.hpp"        // type definitions as AthenaTensor... 
#include "../hydro/hydro.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "../utils/linear_algebra.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"
#include "../parameter_input.hpp"

// -----------------------------------------------------------------------------------
// Functions needed to compute lmix (Kiuchi)
static Real kiuchi_lmix_fit(
        Real const rho,
        Real const stretch = 1.0,
        Real const ampl = 1.0) {
    // Fitting coefficients
    Real const lrho_0 = -8.49697276793;
    Real const a      = 0.0151145023166;
    Real const b      = -0.425383267966;

    Real const xi = (std::log10(rho) - lrho_0)/stretch;
    if(xi < 0) {
        return 0.0;
    }
    else {
        return a*ampl*xi*std::exp(-std::pow(std::abs(xi*b), 2.5));
    }
}

static Real kiuchi2_lmix_fit(
        Real const rho,
        Real const lrho0,
        Real const lrho1,
        Real const a,
        Real const b,
        Real const c) {
    Real const lrho = std::max(std::log10(rho), lrho0);
    assert(std::isfinite(lrho));
    if (lrho <= lrho1) {
        return std::pow(10.0, a + b*lrho);
    }
    else {
        return std::pow(10.0, a + b*lrho1 + c*(lrho - lrho1));
    }
}
// -------------------------------------------------------------------------------


// SmagoSG methods implementation

//Constructor
SmagoSG::SmagoSG(MeshBlock *pmb_in, ParameterInput *pin):
        pz4c(pmb_in->pz4c), adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx) {
            pmb = pmb_in;
            nx1 = pmb->ncells1;
            nx2 = pmb->ncells2;
            nx3 = pmb->ncells3;
            TurbStressTensor_dd.NewAthenaTensor(nx3, nx2, nx1);
            g_uu.NewAthenaTensor(nx3,nx2,nx1);

            Gamma_ddd.NewAthenaTensor(nx1);
            Gamma_udd.NewAthenaTensor(nx1);
            dg_ddd.NewAthenaTensor(nx1);
            detg.NewAthenaTensor(nx1);
            w_util_d.NewAthenaTensor(nx3,nx2,nx1);
            Dv_dd.NewAthenaTensor(nx1);


            //Parameters from parfile
            visc_type       = pin->GetOrAddString("SmagoSG", "visc_type", "const");
            lmix            = pin->GetOrAddReal("SmagoSG", "lmix", 1.0);
            kiuchi2_lrho0   = pin->GetOrAddReal("SmagoSG", "kiuchi2_lrho0", 1.0);
            kiuchi2_lrho1   = pin->GetOrAddReal("SmagoSG", "kiuchi2_lrho1", 1.0);
            kiuchi2_a       = pin->GetOrAddReal("SmagoSG", "kiuchi2_a", 1.0);
            kiuchi2_b       = pin->GetOrAddReal("SmagoSG", "kiuchi2_b", 1.0);
            kiuchi2_c       = pin->GetOrAddReal("SmagoSG", "kiuchi2_c", 1.0);
            kiuchi_stretch  = 1.0;
            kiuchi_ampl     = 1.0;
        }

//--------------------------------------------------------------------------------

// Some GR definitions needed 
void SmagoSG::Christoffel_calc (int k, int j) {

    Gamma_ddd.ZeroClear();
    Gamma_udd.ZeroClear();
    dg_ddd.ZeroClear();
    detg.ZeroClear();
    
    // Use extended range to cover ghost cells needed for tensor calculation
    // Avoids using ILOOP1 which only covers active domain
    int is = pmb->is, ie = pmb->ie;
    int il_ext = (is > 0) ? is - 1 : 0;
    int iu_ext = (ie < nx1 - 1) ? ie + 1 : nx1 - 1;
    
    // first derivatives of g
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
        for (int i=il_ext; i<=iu_ext; ++i) {
             dg_ddd(c,a,b,i) = pmb->pcoord->fd_cx->Dx(c, adm_gamma_dd(a,b,k,j,i));
        }
    }
    
    // -----------------------------------------------------------------------------------
    // inverse metric

    for (int i=il_ext; i<=iu_ext; ++i) {
        detg(i) = LinearAlgebra::Det3Metric(adm_gamma_dd,k,j,i);
        LinearAlgebra::Inv3Metric(1./detg(i),
            adm_gamma_dd(0,0,k,j,i), adm_gamma_dd(0,1,k,j,i), adm_gamma_dd(0,2,k,j,i),
            adm_gamma_dd(1,1,k,j,i), adm_gamma_dd(1,2,k,j,i), adm_gamma_dd(2,2,k,j,i),
            &g_uu(0,0,k,j,i), &g_uu(0,1,k,j,i), &g_uu(0,2,k,j,i),
            &g_uu(1,1,k,j,i), &g_uu(1,2,k,j,i), &g_uu(2,2,k,j,i));
    }

// -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
        for (int i=il_ext; i<=iu_ext; ++i) {
             Gamma_ddd(c,a,b,i) = 0.5*(dg_ddd(a,b,c,i) + dg_ddd(b,a,c,i) - dg_ddd(c,a,b,i));
        }
    }

    Gamma_udd.ZeroClear();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
        for (int i=il_ext; i<=iu_ext; ++i) {
             Gamma_udd(c,a,b,i) += g_uu(c,d,k,j,i)*Gamma_ddd(d,a,b,i);
        }
    }

}
// -----------------------------------------------------------------------------------End GR definitions


//Turbolent viscosity    
//Function that computes the turbolent viscosity starting from the density rho and the speed of sound in a specific point 
Real SmagoSG::compute_turb_visc (Real rho, Real csound) {    
    if (visc_type == "alpha") {
        nu_turb = lmix * csound;
        return nu_turb;
    }

    else if (visc_type == "const") {
        nu_turb = lmix;
        return nu_turb;
    }

    else if (visc_type == "kiuchi") {
        Real const lmix = kiuchi_lmix_fit(rho,
                kiuchi_stretch, kiuchi_ampl);
        nu_turb = lmix * csound;
        return nu_turb;

    }

    else if (visc_type == "kiuchi2") {
        Real const lmix = kiuchi2_lmix_fit(rho,
                kiuchi2_lrho0, kiuchi2_lrho1,
                kiuchi2_a, kiuchi2_b, kiuchi2_c);
        nu_turb = lmix * csound;
        return nu_turb;
    }

    else {
        std::stringstream msg;
        msg << "### FATAL ERROR in SmagoSG::compute_turb_visc" << std::endl
            << "visc_type = " << visc_type << " not recognized!" << std::endl;
        ATHENA_ERROR(msg);
    }

    return nu_turb;

}

//--------------------------------------------------------------------------------

//Turbolent stress tensor dd_components
//Function that computes the turbolent stress tensor tau_ij at every point in 3D space
void SmagoSG::TurTensorCalculator (){
    
    //Matter and Lorentz factor
    ph = pmb->phydro;
    ps = pmb->pscalars;
    peos = pmb->peos;
    AA & ccprim = const_cast<AthenaArray<Real>&>(ph->w);
    AT_N_sca w_rho(   ccprim, IDN);
    AT_N_vec w_util_u(ccprim, IVX);
    AT_N_sca w_p(     ccprim, IPR);
    #if NSCALARS > 0
        AA & ccprim_scalar = const_cast<AthenaArray<Real>&>(ps->r);
        AT_S_vec w_r(ccprim_scalar, 0);
    #endif

    Real W_lor;
    Real w_hrho;
    Real h;

    // DEFINE SAFE INDICES 
    // w_util_d on the EXTENDED domain (including ghosts)
    // so that can take derivatives of it inside the active domain.
    
    // Full allocated range (0 to N-1)
    int i_full_start = 0; 
    int i_full_end = nx1 - 1;
    int j_full_start = 0; 
    int j_full_end = (nx2 > 1) ? nx2 - 1 : 0;
    int k_full_start = 0; 
    int k_full_end = (nx3 > 1) ? nx3 - 1 : 0;

    // Active domain indices 
    int ks = pmb->ks, ke = pmb->ke;
    int js = pmb->js, je = pmb->je;
    int is = pmb->is, ie = pmb->ie;

    // Range for Tensor Calculation 
    int k_calc_start = (nx3 > 1) ? std::max(ks - NGHOST, 0) : ks;
    int k_calc_end   = (nx3 > 1) ? std::min(ke + NGHOST, nx3 - 1) : ke;

    int j_calc_start = (nx2 > 1) ? std::max(js - NGHOST, 0) : js;
    int j_calc_end   = (nx2 > 1) ? std::min(je + NGHOST, nx2 - 1) : je;

    int i_calc_start = std::max(is - NGHOST, 0);
    int i_calc_end   = std::min(ie + NGHOST, nx1 - 1);


    //Turbolent stress tensor calculation
    TurbStressTensor_dd.ZeroClear(); 
    w_util_d.ZeroClear();
    Dv_dd.ZeroClear();
    Real csound;
    
    
    // w_util_d computation on FULL domain

    for (int k=k_full_start; k<=k_full_end; ++k) {
        for (int j=j_full_start; j<=j_full_end; ++j) {
            for (int i=i_full_start; i<=i_full_end; ++i) { // Manual loop on full X range
                for (int a=0; a<NDIM; ++a) {
                    for(int c=0; c<3; c++) {
                        w_util_d(a,k,j,i) += w_util_u(c,k,j,i) * adm_gamma_dd(a,c,k,j,i);
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------------------
    // Tensor Calculation on SAFE domain
    // Covers Active Domain + 1 Ghost layer, ensuring memory bounds
    // --------------------------------------------------------------------------------
    for (int k=k_calc_start; k<=k_calc_end; ++k)
    for (int j=j_calc_start; j<=j_calc_end; ++j)
    {
        Christoffel_calc(k, j); // Uses internal safe range
        
        for (int a=0; a<NDIM; ++a)
        for (int b=a; b<NDIM; ++b)
        {
            for (int i=i_calc_start; i<=i_calc_end; ++i)
            {
                //Lorentz Factor at k,j,i
                W_lor = ph->derived_ms(IX_LOR,k,j,i);

                // Calculate enthalpy (w_hrho = rho*h) NB EOS specific!    
                #if USETM // tabulated gas
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
                    w_hrho = w_rho(k,j,i) * h;
                    //Local speed of sound (at point k,j,i)
                    csound = peos->GetEOS().GetSoundSpeed(n, T, Y);

                #else // ideal gas
                    const Real gamma_adi = peos->GetGamma();
                    w_hrho = w_rho(k,j,i) + gamma_adi/(gamma_adi-1.0) * w_p(k,j,i);
                    Real cs_sq = (gamma_adi * w_p(k,j,i)) / w_hrho;
                    csound = std::sqrt(std::max(0.0, cs_sq));
                #endif
                    

                

                // Check csound value
                if (!std::isfinite(csound) || csound <= 0.0) {
                    std::stringstream msg;
                    msg << "### FATAL ERROR in SmagoSG: csound not valid" << std::endl
                        << "Value: " << csound << std::endl
                        << "Position (k,j,i): (" << k << "," << j << "," << i << ")" << std::endl
                        << "Primitives (Rho, Press): " << w_rho(k,j,i) << ", " << w_p(k,j,i) << std::endl;
                    ATHENA_ERROR(msg);
                }
                                 
                //Gradient of velocities, mixed components
                // Dx uses i+1 and i-1, which is safe because i >= 1 and i <= N-2
                Dv_dd(a,b,i) = pmb->pcoord->fd_cc->Dx(a, w_util_d(b,k,j,i));   
                
                for (int c = 0; c < 3; c++) {
                    Dv_dd(a,b,i) -= Gamma_udd(c,a,b,i) * w_util_d(c,k,j,i); 
                }

                //Trace of the gradient of the velocity
                Real Tr_Dv = 0;
                for(int aa = 0; aa < 3; aa++) { // local indexes aa,bb to not overwrite external ones
                    for(int bb = 0; bb < 3; bb++) {
                        Tr_Dv += g_uu(aa,bb,k,j,i) * Dv_dd(aa,bb,i); 
                    }
                }
                
                TurbStressTensor_dd (a,b,k,j,i) = -2 * compute_turb_visc(w_rho(k,j,i), csound) * w_hrho * SQR(W_lor) * ( 0.5 * (Dv_dd(a,b,i) + Dv_dd(b,a,i)) - (1./3.) * Tr_Dv * adm_gamma_dd(a,b,k,j,i));
            }
        }
    }
}


// --------------------------------------------------------------------------------
// Compute TurbStressTensor_uu on the fly
// --------------------------------------------------------------------------------
Real SmagoSG::Get_TurbStressTensor_uu(int a, int b, int k, int j, int i) {
    Real val = 0.0;
    // T^ab = g^ac * g^bd * T_cd
    for (int c = 0; c < NDIM; ++c) {
        for (int d = 0; d < NDIM; ++d) {
             val += g_uu(a,c,k,j,i) * g_uu(b,d,k,j,i) * TurbStressTensor_dd(c,d,k,j,i);
        }
    }
    return val;
}

// --------------------------------------------------------------------------------
// Compute TurbStressTensor_du on the fly
// --------------------------------------------------------------------------------
Real SmagoSG::Get_TurbStressTensor_du(int m, int a, int k, int j, int i) {
    Real val = 0.0;
    // T^m_a = g^mc * T_ca
    for (int c = 0; c < NDIM; ++c) {
        val += g_uu(m,c,k,j,i) * TurbStressTensor_dd(c,a,k,j,i);
    }
    return val;
}

// --------------------------------------------------------------------------------
// Compute divergence of TurbStressTensor_du
// --------------------------------------------------------------------------------
Real SmagoSG::Get_d_TurbStressTensor_d(int a, int k, int j, int i) {
    
    // 1/(2*dx) factors for finite difference
    Real idx[3] = {0.5/pmb->pcoord->dx1v(i), 
                   0.5/pmb->pcoord->dx2v(j), 
                   0.5/pmb->pcoord->dx3v(k)}; 

    Real div_val = 0.0;

    // Loop on the three directions (c = x, y, z)
    for (int c = 0; c < NDIM; ++c) { 
        int k_off = (c==2);
        int j_off = (c==1);
        int i_off = (c==0);

        // Compute right neighbour
        // Indexes
        int kR=k+k_off, jR=j+j_off, iR=i+i_off;
        
        // Call the function for TurbStressTensor_du
        Real T_mix_R = Get_TurbStressTensor_du(c, a, kR, jR, iR);

        // Compute left neighbour
        // Indexes
        int kL=k-k_off, jL=j-j_off, iL=i-i_off;

        // Call the function for TurbStressTensor_du
        Real T_mix_L = Get_TurbStressTensor_du(c, a, kL, jL, iL);


        // Centered finite difference
        div_val += (T_mix_R - T_mix_L) * idx[c];
    }
    
    return div_val;
}

