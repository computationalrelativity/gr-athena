//! \file interp_intergrid.cpp
//  \brief namespace containing vertex-centered <-> cell-centered conversion

// C headers

// C++ headers

// Athena++ headers
#include "interp_intergrid.hpp"

// base implementations -------------------------------------------------------
void InterpIntergrid::var_map_VC2CC(
  const AthenaArray<Real> & var_vc,
  AthenaArray<Real> & var_cc,
  const int dim,
  const int fl, const int fu,
  const int ll, const int lu,
  const int ml, const int mu,
  const int nl, const int nu
)
// var_map_VC2CC:
// Interpolate vertex-centered to cell-centered data on uniform grids.
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
    for(int f=fl; f<=fu; ++f)
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

            Real const f_uuu = var_vc(f, k_u+dg, j_u+dg, i_u+dg);
            Real const f_lll = var_vc(f, k_l+dg, j_l+dg, i_l+dg);
            Real const f_lul = var_vc(f, k_l+dg, j_u+dg, i_l+dg);
            Real const f_ulu = var_vc(f, k_u+dg, j_l+dg, i_u+dg);
            Real const f_ull = var_vc(f, k_l+dg, j_l+dg, i_u+dg);
            Real const f_luu = var_vc(f, k_u+dg, j_u+dg, i_l+dg);
            Real const f_llu = var_vc(f, k_u+dg, j_l+dg, i_l+dg);
            Real const f_uul = var_vc(f, k_l+dg, j_u+dg, i_u+dg);

            var_cc(f,n,m,l) += lck * lcj * lci * (
              (f_uuu + f_lll) + (f_lul + f_ulu) +
              (f_ull + f_luu) + (f_llu + f_uul)
            );

          }

        }
      }
  }
  else if(dim==2)
  {
    for(int f=fl; f<=fu; ++f)
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

          Real const f_uu = var_vc(f, j_u+dg, i_u+dg);
          Real const f_ll = var_vc(f, j_l+dg, i_l+dg);
          Real const f_ul = var_vc(f, j_l+dg, i_u+dg);
          Real const f_lu = var_vc(f, j_u+dg, i_l+dg);

          var_cc(f,m,l) += lcj * lci * (
            ((f_uu+f_ll) +(f_ul+f_lu))
          );
        }
      }
  }
  else if(dim==1)
  {
    for(int f=fl; f<=fu; ++f)
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i+1;
        const int i_u = l+i;
        const int lix = NGRCV_HSZ+i-1;

        Real const lc = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];
        var_cc(f,l) += lc*(var_vc(f,i_l+dg)+var_vc(f,i_u+dg));
      }
  }

}

void InterpIntergrid::var_map_CC2VC(
  const AthenaArray<Real> & var_cc,
  AthenaArray<Real> & var_vc,
  const int dim,
  const int fl, const int fu,
  const int ll, const int lu,
  const int ml, const int mu,
  const int nl, const int nu
)
// var_map_CC2VC:
// Interpolate cell-centered to vertex-centered data on uniform grids.
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
    for(int f=fl; f<=fu; ++f)
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

            Real const f_uuu = var_cc(f, k_u+dg, j_u+dg, i_u+dg);
            Real const f_lll = var_cc(f, k_l+dg, j_l+dg, i_l+dg);
            Real const f_lul = var_cc(f, k_l+dg, j_u+dg, i_l+dg);
            Real const f_ulu = var_cc(f, k_u+dg, j_l+dg, i_u+dg);
            Real const f_ull = var_cc(f, k_l+dg, j_l+dg, i_u+dg);
            Real const f_luu = var_cc(f, k_u+dg, j_u+dg, i_l+dg);
            Real const f_llu = var_cc(f, k_u+dg, j_l+dg, i_l+dg);
            Real const f_uul = var_cc(f, k_l+dg, j_u+dg, i_u+dg);

            var_vc(f,n,m,l) += lck * lcj * lci * (
              (f_uuu + f_lll) + (f_lul + f_ulu) +
              (f_ull + f_luu) + (f_llu + f_uul)
            );

          }

        }
      }
  }
  else if(dim==2)
  {
    for(int f=fl; f<=fu; ++f)
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

          Real const f_uu = var_cc(f, j_u+dg, i_u+dg);
          Real const f_ll = var_cc(f, j_l+dg, i_l+dg);
          Real const f_ul = var_cc(f, j_l+dg, i_u+dg);
          Real const f_lu = var_cc(f, j_u+dg, i_l+dg);

          var_vc(f,m,l) += lcj * lci * (
            ((f_uu+f_ll) +(f_ul+f_lu))
          );
        }
      }
  }
  else if(dim==1)
  {
    for(int f=fl; f<=fu; ++f)
    for(int l=ll; l<=lu; ++l)
      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i;
        const int i_u = l+i-1;
        const int lix = NGRCV_HSZ+i-1;

        Real const lc = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];
        var_vc(f,l) += lc*(var_cc(f,i_l+dg)+var_cc(f,i_u+dg));
      }
  }

}

void InterpIntergrid::var_map_VC2CC_Taylor1(
    const AthenaArray<Real> & rds,
    const AthenaArray<Real> & var_vc,
    AthenaArray<Real> & dvar_cc,
    const int dim,
    const int ll, const int lu,
    const int ml, const int mu,
    const int nl, const int nu
  )
// var_map_VC2CC_Taylor1:
// Embed interpolation within derivative computation transferring VC to CC.
//
// Indices to be provided for interpolation ranges are CC and EP are included.
//
// Underlying expressions are based on Taylor Series at first order and ordered
// so as to mitigate subtractive cancellation.
//
// See also:
// var_map_VC2CC_Taylor1
{
  // deal with differing ghosts
  const int dg = VC_NGHOST-CC_NGHOST;

  if(dim==3)
  {
    for(int n=nl; n<=nu; ++n)
    {
      const int k = 1;

      const int k_u = dg+n+k;
      const int k_l = dg+n-k+1;

      for(int m=ml; m<=mu; ++m)
      {
        const int j = 1;

        const int j_u = dg+m+j;
        const int j_l = dg+m-j+1;

        for(int l=ll; l<=lu; ++l)
        {
          const int i = 1;               // i in range(1, N_I+1) with N_I = 1

          const int i_u = dg+l+i;
          const int i_l = dg+l-i+1;

          // Derivative[0,0,0][.]
          dvar_cc(0,n,m,l) = 0.125*(
            (var_vc(k_u,j_u,i_u) + var_vc(k_l,j_l,i_l)) +
            (var_vc(k_u,j_u,i_l) + var_vc(k_l,j_l,i_u)) +
            (var_vc(k_l,j_u,i_l) + var_vc(k_u,j_l,i_u)) +
            (var_vc(k_l,j_u,i_u) + var_vc(k_u,j_l,i_l))
          );

          // Derivative[1,0,0][.]
          dvar_cc(1,n,m,l) = 0.25*rds(0)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_l,j_l,i_l)) -
             (var_vc(k_u,j_u,i_l) - var_vc(k_l,j_l,i_u))) -
            ((var_vc(k_l,j_u,i_l) - var_vc(k_u,j_l,i_u)) -
             (var_vc(k_l,j_u,i_u) - var_vc(k_u,j_l,i_l)))
          );

          // Derivative[0,1,0][.]
          dvar_cc(2,n,m,l) = 0.25*rds(1)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_l,j_l,i_l)) +
             (var_vc(k_u,j_u,i_l) - var_vc(k_l,j_l,i_u))) +
            ((var_vc(k_l,j_u,i_l) - var_vc(k_u,j_l,i_u)) +
             (var_vc(k_l,j_u,i_u) - var_vc(k_u,j_l,i_l)))
          );

          // Derivative[0,0,1][.]
          dvar_cc(3,n,m,l) = 0.25*rds(2)*(
            (
              (var_vc(k_u,j_u,i_u) - var_vc(k_l,j_l,i_l)) -
              (var_vc(k_l,j_u,i_l) - var_vc(k_u,j_l,i_u))
            ) +
            (
              (var_vc(k_u,j_u,i_l) - var_vc(k_l,j_l,i_u)) -
              (var_vc(k_l,j_u,i_u) - var_vc(k_u,j_l,i_l))
            )
          );

          // Derivative[1,1,0][.]
          dvar_cc(4,n,m,l) = 0.5*rds(0)*rds(1)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_u,j_u,i_l)) -
             (var_vc(k_u,j_l,i_u) - var_vc(k_u,j_l,i_l))) +
            ((var_vc(k_l,j_u,i_u) - var_vc(k_l,j_u,i_l)) -
             (var_vc(k_l,j_l,i_u) - var_vc(k_l,j_l,i_l)))
          );

          // Derivative[0,1,1][.]
          dvar_cc(5,n,m,l) = 0.5*rds(1)*rds(2)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_l,j_u,i_u)) -
             (var_vc(k_u,j_l,i_u) - var_vc(k_l,j_l,i_u))) +
            ((var_vc(k_u,j_u,i_l) - var_vc(k_l,j_u,i_l)) -
             (var_vc(k_u,j_l,i_l) - var_vc(k_l,j_l,i_l)))
          );

          // Derivative[1,0,1][.]
          dvar_cc(6,n,m,l) = 0.5*rds(0)*rds(2)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_l,j_u,i_u)) -
             (var_vc(k_u,j_u,i_l) - var_vc(k_l,j_u,i_l))) +
            ((var_vc(k_u,j_l,i_u) - var_vc(k_l,j_l,i_u)) -
             (var_vc(k_u,j_l,i_l) - var_vc(k_l,j_l,i_l)))
          );

          // Derivative[1,1,1][.]
          dvar_cc(7,n,m,l) = rds(0)*rds(1)*rds(2)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_l,j_u,i_u)) -
             (var_vc(k_u,j_l,i_u) - var_vc(k_l,j_u,i_l))) +
            ((var_vc(k_l,j_l,i_u) - var_vc(k_l,j_l,i_l)) -
             (var_vc(k_u,j_u,i_l) - var_vc(k_u,j_l,i_l)))
          );


        }
      }
    }
  }
  else if(dim==2)
  {
    for(int m=ml; m<=mu; ++m)
    {
      const int j = 1;

      const int j_u = dg+m+j;
      const int j_l = dg+m-j+1;

      for(int l=ll; l<=lu; ++l)
      {
        const int i = 1;               // i in range(1, N_I+1) with N_I = 1

        const int i_u = dg+l+i;
        const int i_l = dg+l-i+1;

        // Derivative[0,0][.]
        dvar_cc(0,m,l) = 0.25*(
          (var_vc(j_l,i_l) + var_vc(j_u,i_u)) +
          (var_vc(j_u,i_l) + var_vc(j_l,i_u))
        );

        // Derivative[1,0][.]
        dvar_cc(1,m,l) = 0.5*rds(0)*(
          (var_vc(j_l,i_u) - var_vc(j_l,i_l)) +
          (var_vc(j_u,i_u) - var_vc(j_u,i_l))
        );

        // Derivative[0,1][.]
        dvar_cc(2,m,l) = 0.5*rds(1)*(
          (var_vc(j_u,i_u) - var_vc(j_l,i_u)) +
          (var_vc(j_u,i_l) - var_vc(j_l,i_l))
        );
        // circ. perm
        // dvar_cc(2,m,l) = 0.5*rds(1)*(
        //   (var_vc(j_u,i_l) - var_vc(j_l,i_l)) +
        //   (var_vc(j_u,i_u) - var_vc(j_l,i_u))
        // );

        // Derivative[1,1][.]
        dvar_cc(3,m,l) = rds(0)*rds(1)*(
          (var_vc(j_u,i_u) - var_vc(j_l,i_u)) -
          (var_vc(j_u,i_l) - var_vc(j_l,i_l))
        );

      }
    }
  }
  else if(dim==1)
  {
    for(int l=ll; l<=lu; ++l)
    {
      const int i = 1;               // i in range(1, N_I+1) with N_I = 1

      const int i_u = dg+l+i;
      const int i_l = dg+l-i+1;

      // Derivative[0][.]
      dvar_cc(0,l) = 0.5*(var_vc(i_l)+var_vc(i_u));

      // Derivative[1][.]
      dvar_cc(1,l) = rds(0)*(var_vc(i_u)-var_vc(i_l));
    }
  }
}

// implementations for local calculation and vectorization --------------------
InterpIntergridLocal::InterpIntergridLocal(
  const int dim, const int *N, const Real *rds)
{
  this->dim = dim;

  this->rds = new Real[dim];
  this->N = new int[dim];
  this->strides = new int[2*dim];

  for(int i=0; i<dim; ++i)
  {
    this->rds[i] = rds[i];
    this->N[i] = N[i];
    if(i==0)
    {
      this->strides[i] = 1;
      this->strides[dim+i] = 1;
    }
    else
    {
      this->strides[i] = this->strides[i-1] * (
        VC_NGHOST + N[i-1] + 1 + VC_NGHOST);
      this->strides[dim+i] = this->strides[dim+i-1] * (
        CC_NGHOST + N[i-1] + CC_NGHOST);
    }
  }

}

InterpIntergridLocal::~InterpIntergridLocal()
{
  delete[] rds;
  delete[] N;
  delete[] strides;
}

void InterpIntergridLocal::var_map_VC2CC(
  const AthenaArray<Real> & var_vc,
  AthenaArray<Real> & var_cc,
  const int fl, const int fu,
  const int ll, const int lu,
  const int ml, const int mu,
  const int nl, const int nu
)
// See InterpIntergrid::var_map_VC2CC
{
  if(dim==3)
  {
    for(int f=fl; f<=fu; ++f)
    for(int n=nl; n<=nu; ++n)
    for(int m=ml; m<=mu; ++m)
    {
#pragma omp simd
      for(int l=ll; l<=lu; ++l)
        var_cc(f,n,m,l) = _map3d_var(var_vc(f,0,0,0),l,m,n,0,1);
    }
  }
  else if(dim==2)
  {
    for(int f=fl; f<=fu; ++f)
    for(int m=ml; m<=mu; ++m)
    {
#pragma omp simd
      for(int l=ll; l<=lu; ++l)
        var_cc(f,m,l) = _map2d_var(var_vc(f,0,0),l,m,0,1);
    }
  }
  else if(dim==1)
  {
    for(int f=fl; f<=fu; ++f)
    {
#pragma omp simd
      for(int l=ll; l<=lu; ++l)
        var_cc(f,l) = _map1d_var(var_vc(f,0),l,0,1);
    }
  }

}

void InterpIntergridLocal::var_map_CC2VC(
  const AthenaArray<Real> & var_cc,
  AthenaArray<Real> & var_vc,
  const int fl, const int fu,
  const int ll, const int lu,
  const int ml, const int mu,
  const int nl, const int nu
)
// See InterpIntergrid::var_map_CC2VC
{
  if(dim==3)
  {
    for(int f=fl; f<=fu; ++f)
    for(int n=nl; n<=nu; ++n)
    for(int m=ml; m<=mu; ++m)
    {
#pragma omp simd
      for(int l=ll; l<=lu; ++l)
        var_vc(f,n,m,l) = _map3d_var(var_cc(f,0,0,0),l,m,n,-1,-1);

    }
  }
  else if(dim==2)
  {
    for(int f=fl; f<=fu; ++f)
    for(int m=ml; m<=mu; ++m)
    {
#pragma omp simd
      for(int l=ll; l<=lu; ++l)
        var_vc(f,m,l) = _map2d_var(var_cc(f,0,0),l,m,-1,-1);
    }
  }
  else if(dim==1)
  {
    for(int f=fl; f<=fu; ++f)
    {
#pragma omp simd
      for(int l=ll; l<=lu; ++l)
        var_vc(f,l) = _map1d_var(var_cc(f,0),l,-1,-1);
    }
  }

}

void InterpIntergridLocal::var_map_VC2CC_Taylor1(
    const AthenaArray<Real> & rds,
    const AthenaArray<Real> & var_vc,
    AthenaArray<Real> & dvar_cc,
    const int ll, const int lu,
    const int ml, const int mu,
    const int nl, const int nu
  )
// See InterpIntergrid::var_map_VC2CC_Taylor1
{
  const int dg = VC_NGHOST-CC_NGHOST;

  if(dim==3)
  {
    for(int n=nl; n<=nu; ++n)
    {
      const int k_l = dg+n;
      const int k_u = dg+n+1;

      for(int m=ml; m<=mu; ++m)
      {
        const int j_l = dg+m;
        const int j_u = dg+m+1;

        for(int l=ll; l<=lu; ++l)
        {
          const int i_l = dg+l;
          const int i_u = dg+l+1;

          // Derivative[1,0,0][.]
          dvar_cc(0,n,m,l) = 0.25*rds(0)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_l,j_l,i_l)) -
             (var_vc(k_u,j_u,i_l) - var_vc(k_l,j_l,i_u))) -
            ((var_vc(k_l,j_u,i_l) - var_vc(k_u,j_l,i_u)) -
             (var_vc(k_l,j_u,i_u) - var_vc(k_u,j_l,i_l)))
          );

          // Derivative[0,1,0][.]
          dvar_cc(1,n,m,l) = 0.25*rds(1)*(
            ((var_vc(k_u,j_u,i_u) - var_vc(k_l,j_l,i_l)) +
             (var_vc(k_u,j_u,i_l) - var_vc(k_l,j_l,i_u))) +
            ((var_vc(k_l,j_u,i_l) - var_vc(k_u,j_l,i_u)) +
             (var_vc(k_l,j_u,i_u) - var_vc(k_u,j_l,i_l)))
          );

          // Derivative[0,0,1][.]
          dvar_cc(2,n,m,l) = 0.25*rds(2)*(
            (
              (var_vc(k_u,j_u,i_u) - var_vc(k_l,j_l,i_l)) -
              (var_vc(k_l,j_u,i_l) - var_vc(k_u,j_l,i_u))
            ) +
            (
              (var_vc(k_u,j_u,i_l) - var_vc(k_l,j_l,i_u)) -
              (var_vc(k_l,j_u,i_u) - var_vc(k_u,j_l,i_l))
            )
          );

        }
      }
    }
  }
  else if(dim==2)
  {
    for(int m=ml; m<=mu; ++m)
    {
      const int j_l = dg+m;
      const int j_u = dg+m+1;

      for(int l=ll; l<=lu; ++l)
      {
        const int i_l = dg+l;
        const int i_u = dg+l+1;

        // Derivative[1,0][.]
        dvar_cc(0,m,l) = 0.5*rds(0)*(
          (var_vc(j_l,i_u) - var_vc(j_l,i_l)) +
          (var_vc(j_u,i_u) - var_vc(j_u,i_l))
        );

        // Derivative[0,1][.]
        dvar_cc(1,m,l) = 0.5*rds(1)*(
          (var_vc(j_u,i_u) - var_vc(j_l,i_u)) +
          (var_vc(j_u,i_l) - var_vc(j_l,i_l))
        );

      }
    }
  }
  else if(dim==1)
  {
    for(int l=ll; l<=lu; ++l)
    {
      const int i_l = dg+l;
      const int i_u = dg+l+1;

      // Derivative[1][.]
      dvar_cc(0,l) = rds(0)*(var_vc(i_u)-var_vc(i_l));
    }
  }
}