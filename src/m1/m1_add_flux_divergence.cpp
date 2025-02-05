// c++
// ...

// Athena++ headers
#include "m1.hpp"
#include "m1_macro.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::MulAddFluxDivergence(AthenaArray<Real> & u_inh, const Real fac)
{
  /*
  More compact with something like
  const int dir = 0;
  AthenaArray<Real> flux;

  InitShallowSlice:
    storage.flux[dir]
    ix_map_GS(G, S, ixn_Lab::N)
    ixn_Lab::N

  ... probably not more performant
  */

  vars_Lab I   { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, I);

  AA & dflx_ = scratch.dflx_;

  M1_MLOOP2(k,j)
  {
    dflx_.ZeroClear();

    {
      const int ix_d = 0;

      for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        AT_C_sca & F_nG  = fluxes.sc_nG( ix_g,ix_s,ix_d);
        AT_C_sca & F_E   = fluxes.sc_E(  ix_g,ix_s,ix_d);
        AT_N_vec & F_f_d = fluxes.sp_F_d(ix_g,ix_s,ix_d);

        M1_MLOOP1(i)
        if (MaskGet(k, j, i))
        if (MaskGetHybridize(k,j,i))
        {
          dflx_(ix_g,ix_s,ixn_Lab::nG,i) += (
            F_nG(k,j,i+1) - F_nG(k,j,i)
          ) / mbi.dx1(i);

          dflx_(ix_g,ix_s,ixn_Lab::E,i) += (
            F_E(k,j,i+1) - F_E(k,j,i)
          ) / mbi.dx1(i);
        }

        for (int a=0; a<N; ++a)
        M1_MLOOP1(i)
        if (MaskGet(k, j, i))
        if (MaskGetHybridize(k,j,i))
        {
          dflx_(ix_g,ix_s,ixn_Lab::F_x+a,i) += (
            F_f_d(a,k,j,i+1) - F_f_d(a,k,j,i)
          ) / mbi.dx1(i);
        }
      }
    }

    if (mbi.nn2 > 1)
    {
      const int ix_d = 1;

      for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        AT_C_sca & F_nG  = fluxes.sc_nG( ix_g,ix_s,ix_d);
        AT_C_sca & F_E   = fluxes.sc_E(  ix_g,ix_s,ix_d);
        AT_N_vec & F_f_d = fluxes.sp_F_d(ix_g,ix_s,ix_d);

        M1_MLOOP1(i)
        if (MaskGet(k, j, i))
        if (MaskGetHybridize(k,j,i))
        {
          dflx_(ix_g,ix_s,ixn_Lab::nG,i) += (
            F_nG(k,j+1,i) - F_nG(k,j,i)
          ) / mbi.dx2(j);

          dflx_(ix_g,ix_s,ixn_Lab::E,i) += (
            F_E(k,j+1,i) - F_E(k,j,i)
          ) / mbi.dx2(j);
        }

        for (int a=0; a<N; ++a)
        M1_MLOOP1(i)
        if (MaskGet(k, j, i))
        if (MaskGetHybridize(k,j,i))
        {
          dflx_(ix_g,ix_s,ixn_Lab::F_x+a,i) += (
            F_f_d(a,k,j+1,i) - F_f_d(a,k,j,i)
          ) / mbi.dx2(j);
        }
      }
    }

    if (mbi.nn3 > 1)
    {
      const int ix_d = 2;

      for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        AT_C_sca & F_nG  = fluxes.sc_nG( ix_g,ix_s,ix_d);
        AT_C_sca & F_E   = fluxes.sc_E(  ix_g,ix_s,ix_d);
        AT_N_vec & F_f_d = fluxes.sp_F_d(ix_g,ix_s,ix_d);

        M1_MLOOP1(i)
        if (MaskGet(k, j, i))
        if (MaskGetHybridize(k,j,i))
        {
          dflx_(ix_g,ix_s,ixn_Lab::nG,i) += (
            F_nG(k+1,j,i) - F_nG(k,j,i)
          ) / mbi.dx3(k);

          dflx_(ix_g,ix_s,ixn_Lab::E,i) += (
            F_E(k+1,j,i) - F_E(k,j,i)
          ) / mbi.dx3(k);
        }

        for (int a=0; a<N; ++a)
        M1_MLOOP1(i)
        if (MaskGet(k, j, i))
        if (MaskGetHybridize(k,j,i))
        {
          dflx_(ix_g,ix_s,ixn_Lab::F_x+a,i) += (
            F_f_d(a,k+1,j,i) - F_f_d(a,k,j,i)
          ) / mbi.dx3(k);
        }
      }
    }

    // update RHS
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      M1_MLOOP1(i)
      if (MaskGet(k, j, i))
      if (MaskGetHybridize(k,j,i))
      {
        I.sc_nG(ix_g,ix_s)(k,j,i) -= fac * dflx_(ix_g,ix_s,ixn_Lab::nG,i);
        I.sc_E(ix_g,ix_s)(k,j,i)  -= fac * dflx_(ix_g,ix_s,ixn_Lab::E,i);
      }

      for (int a=0; a<N; ++a)
      M1_MLOOP1(i)
      if (MaskGet(k, j, i))
      if (MaskGetHybridize(k,j,i))
      {
        I.sp_F_d(ix_g,ix_s)(a,k,j,i) -= fac * dflx_(ix_g,ix_s,
                                                    ixn_Lab::F_x+a,i);
      }
    }


  }
}

void M1::SubFluxDivergence(AthenaArray<Real> & u_inh)
{
  MulAddFluxDivergence(u_inh, -1.0);
}

//-----------------------------------------------------------------------------
// Add the flux divergence to the RHS (see analogous Hydro method)
void M1::AddFluxDivergence(AthenaArray<Real> & u_inh)
{
  MulAddFluxDivergence(u_inh, 1.0);
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//