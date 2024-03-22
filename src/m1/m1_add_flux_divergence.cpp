// c++
// ...

// Athena++ headers
#include "m1.hpp"
#include "m1_macro.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

//-----------------------------------------------------------------------------
// Add the flux divergence to the RHS (see analogous Hydro method)
void M1::AddFluxDivergence(AthenaArray<Real> & u_rhs)
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

  M1_ILOOP2(k,j)
  {
    scratch.dflx.ZeroClear();

    {
      const int ix_d = 0;

      for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        AT_N_sca & F_E   = fluxes.E(  ix_g,ix_s,ix_d);
        AT_N_vec & F_f_d = fluxes.F_d(ix_g,ix_s,ix_d);
        AT_N_sca & F_nG  = fluxes.nG( ix_g,ix_s,ix_d);

        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::E,i) += (
            F_E(k,j,i+1) - F_E(k,j,i)
          ) / mbi.dx1(i);
        }

        for (int a=0; a<N; ++a)
        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::Fx+a,i) += (
            F_f_d(a,k,j,i+1) - F_f_d(a,k,j,i)
          ) / mbi.dx1(i);
        }

        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::nG,i) += (
            F_nG(k,j,i+1) - F_nG(k,j,i)
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
        AT_N_sca & F_E   = fluxes.E(  ix_g,ix_s,ix_d);
        AT_N_vec & F_f_d = fluxes.F_d(ix_g,ix_s,ix_d);
        AT_N_sca & F_nG  = fluxes.nG( ix_g,ix_s,ix_d);

        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::E,i) += (
            fluxes.E(ix_g,ix_s,ix_d)(k,j+1,i) -
            fluxes.E(ix_g,ix_s,ix_d)(k,j,i)
          ) / mbi.dx2(i);
        }

        for (int a=0; a<N; ++a)
        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::Fx+a,i) += (
            fluxes.F_d(ix_g,ix_s,ix_d)(a,k,j+1,i) -
            fluxes.F_d(ix_g,ix_s,ix_d)(a,k,j,i)
          ) / mbi.dx2(i);
        }

        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::nG,i) += (
            fluxes.nG(ix_g,ix_s,ix_d)(k,j+1,i) -
            fluxes.nG(ix_g,ix_s,ix_d)(k,j,i)
          ) / mbi.dx2(i);
        }
      }
    }

    if (mbi.nn3 > 1)
    {
      const int ix_d = 2;

      for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
      for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
      {
        AT_N_sca & F_E   = fluxes.E(  ix_g,ix_s,ix_d);
        AT_N_vec & F_f_d = fluxes.F_d(ix_g,ix_s,ix_d);
        AT_N_sca & F_nG  = fluxes.nG( ix_g,ix_s,ix_d);

        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::E,i) += (
            fluxes.E(ix_g,ix_s,ix_d)(k+1,j,i) -
            fluxes.E(ix_g,ix_s,ix_d)(k,j,i)
          ) / mbi.dx3(i);
        }

        for (int a=0; a<N; ++a)
        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::Fx+a,i) += (
            fluxes.F_d(ix_g,ix_s,ix_d)(a,k+1,j,i) -
            fluxes.F_d(ix_g,ix_s,ix_d)(a,k,j,i)
          ) / mbi.dx3(i);
        }

        M1_ILOOP1(i)
        {
          scratch.dflx(ix_g,ix_s,ixn_Lab::nG,i) += (
            fluxes.nG(ix_g,ix_s,ix_d)(k+1,j,i) -
            fluxes.nG(ix_g,ix_s,ix_d)(k,j,i)
          ) / mbi.dx3(i);
        }
      }
    }

    // update RHS
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      M1_ILOOP1(i)
      {
        rhs.E(ix_g,ix_s)(k,j,i) -= scratch.dflx(ix_g,ix_s,ixn_Lab::E,i);
      }

      for (int a=0; a<N; ++a)
      M1_ILOOP1(i)
      {
        rhs.F_d(ix_g,ix_s)(a,k,j,i) -= scratch.dflx(ix_g,ix_s,ixn_Lab::Fx+a,i);
      }

      M1_ILOOP1(i)
      {
        rhs.nG(ix_g,ix_s)(k,j,i) -= scratch.dflx(ix_g,ix_s,ixn_Lab::nG,i);
      }

    }


  }
  return;
}


// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//