//! \file interp_intergrid.cpp
//  \brief namespace containing vertex-centered <-> cell-centered conversion

// C headers

// C++ headers

// Athena++ headers
#include "interp_intergrid.hpp"

void InterpIntergrid::var_map_VC2CC(
  const AthenaArray<Real> & var_vc,
  AthenaArray<Real> & var_cc,
  const int dim,
  const int ll, const int lu,
  const int ml, const int mu,
  const int nl, const int nu
)
// var_map_VC2CC:
// Interpolate vertex-centered to cell-centered data.
//
// Indices to be provided for interpolation ranges are CC and EP are included.
//
// The variable NGRCV_HSZ must be defined and corresponds to half the
// number of nodes used in the polynomial interpolant.
//
// See also:
// var_map_CC2VC
{
  // deal with differing ghosts
  const int dg = VC_NGHOST-CC_NGHOST;

  if(dim==3)
  {
    for(int n=nl; n<=nu; ++n)
    for(int m=ml; m<=mu; ++m)
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i+1;
        const int i_u = l+i;
        const int lix = NGRCV_HSZ+i-1;
        Real const lci = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];

        for(int j=1; j<=NGRCV_HSZ; ++j)
        {
          const int j_l = m-j+1;
          const int j_u = m+j;
          const int ljx = NGRCV_HSZ+j-1;

          Real const lcj = \
            InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ljx];
          for(int k=1; k<=NGRCV_HSZ; ++k)
          {
            const int k_l = n-k+1;
            const int k_u = n+k;
            const int lkx = NGRCV_HSZ+k-1;
            Real const lck = \
              InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lkx];

            Real const f_uuu = var_vc(k_u+dg, j_u+dg, i_u+dg);
            Real const f_lll = var_vc(k_l+dg, j_l+dg, i_l+dg);
            Real const f_lul = var_vc(k_l+dg, j_u+dg, i_l+dg);
            Real const f_ulu = var_vc(k_u+dg, j_l+dg, i_u+dg);
            Real const f_ull = var_vc(k_l+dg, j_l+dg, i_u+dg);
            Real const f_luu = var_vc(k_u+dg, j_u+dg, i_l+dg);
            Real const f_llu = var_vc(k_u+dg, j_l+dg, i_l+dg);
            Real const f_uul = var_vc(k_l+dg, j_u+dg, i_u+dg);

            var_cc(n,m,l) += lck * lcj * lci * (
              (f_uuu + f_lll) + (f_lul + f_ulu) +
              (f_ull + f_luu) + (f_llu + f_uul)
            );

          }

        }
      }
  }
  else if(dim==2)
  {
    for(int m=ml; m<=mu; ++m)
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i+1;
        const int i_u = l+i;
        const int lix = NGRCV_HSZ+i-1;
        Real const lci = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];

        for(int j=1; j<=NGRCV_HSZ; ++j)
        {
          const int j_l = m-j+1;
          const int j_u = m+j;
          const int ljx = NGRCV_HSZ+j-1;
          Real const lcj = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ljx];

          Real const f_uu = var_vc(j_u+dg, i_u+dg);
          Real const f_ll = var_vc(j_l+dg, i_l+dg);
          Real const f_ul = var_vc(j_l+dg, i_u+dg);
          Real const f_lu = var_vc(j_u+dg, i_l+dg);

          var_cc(m,l) += lcj * lci * (
            ((f_uu+f_ll) +(f_ul+f_lu))
          );
        }
      }
  }
  else if(dim==1)
  {
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i+1;
        const int i_u = l+i;
        const int lix = NGRCV_HSZ+i-1;

        Real const lc = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];
        var_cc(l) += lc*(var_vc(i_l+dg)+var_vc(i_u+dg));
      }
  }

}

void InterpIntergrid::var_map_CC2VC(
  const AthenaArray<Real> & var_cc,
  AthenaArray<Real> & var_vc,
  const int dim,
  const int ll, const int lu,
  const int ml, const int mu,
  const int nl, const int nu
)
// var_map_CC2VC:
// Interpolate cell-centered to vertex-centered data.
//
// Indices to be provided for interpolation ranges are VC and EP are included.
//
// The variable NGRCV_HSZ must be defined and corresponds to half the
// number of nodes used in the polynomial interpolant.
//
// See also:
// var_map_VC2CC
{
  // deal with differing ghosts
  const int dg = CC_NGHOST-VC_NGHOST;

  if(dim==3)
  {
    for(int n=nl; n<=nu; ++n)
    for(int m=ml; m<=mu; ++m)
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i;
        const int i_u = l+i-1;
        const int lix = NGRCV_HSZ+i-1;
        Real const lci = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];

        for(int j=1; j<=NGRCV_HSZ; ++j)
        {
          const int j_l = m-j;
          const int j_u = m+j-1;
          const int ljx = NGRCV_HSZ+j-1;

          Real const lcj = \
            InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ljx];
          for(int k=1; k<=NGRCV_HSZ; ++k)
          {
            const int k_l = n-k;
            const int k_u = n+k-1;
            const int lkx = NGRCV_HSZ+k-1;
            Real const lck = \
              InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lkx];

            Real const f_uuu = var_cc(k_u+dg, j_u+dg, i_u+dg);
            Real const f_lll = var_cc(k_l+dg, j_l+dg, i_l+dg);
            Real const f_lul = var_cc(k_l+dg, j_u+dg, i_l+dg);
            Real const f_ulu = var_cc(k_u+dg, j_l+dg, i_u+dg);
            Real const f_ull = var_cc(k_l+dg, j_l+dg, i_u+dg);
            Real const f_luu = var_cc(k_u+dg, j_u+dg, i_l+dg);
            Real const f_llu = var_cc(k_u+dg, j_l+dg, i_l+dg);
            Real const f_uul = var_cc(k_l+dg, j_u+dg, i_u+dg);

            var_vc(n,m,l) += lck * lcj * lci * (
              (f_uuu + f_lll) + (f_lul + f_ulu) +
              (f_ull + f_luu) + (f_llu + f_uul)
            );

          }

        }
      }
  }
  else if(dim==2)
  {
    for(int m=ml; m<=mu; ++m)
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i;
        const int i_u = l+i-1;
        const int lix = NGRCV_HSZ+i-1;
        Real const lci = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];
        for(int j=1; j<=NGRCV_HSZ; ++j)
        {
          const int j_l = m-j;
          const int j_u = m+j-1;
          const int ljx = NGRCV_HSZ+j-1;
          Real const lcj = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ljx];

          Real const f_uu = var_cc(j_u+dg, i_u+dg);
          Real const f_ll = var_cc(j_l+dg, i_l+dg);
          Real const f_ul = var_cc(j_l+dg, i_u+dg);
          Real const f_lu = var_cc(j_u+dg, i_l+dg);

          var_vc(m,l) += lcj * lci * (
            ((f_uu+f_ll) +(f_ul+f_lu))
          );
        }
      }
  }
  else if(dim==1)
  {
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i;
        const int i_u = l+i-1;
        const int lix = NGRCV_HSZ+i-1;

        Real const lc = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];
        var_vc(l) += lc*(var_cc(i_l+dg)+var_cc(i_u+dg));
      }
  }

}
