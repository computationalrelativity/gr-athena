#ifndef INTERP_INTERGRID_HPP_
#define INTERP_INTERGRID_HPP_
//! \file interp_intergrid.hpp
//  \brief prototypes of utility functions to pack/unpack buffers

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "interp_univariate.hpp"

namespace InterpIntergrid {

  void var_map_VC2CC(
    const AthenaArray<Real> & var_vc,
    AthenaArray<Real> & var_cc,
    const int dim,
    const int fl, const int fu,
    const int ll, const int lu,
    const int ml, const int mu,
    const int nl, const int nu
  );

  void var_map_CC2VC(
    const AthenaArray<Real> & var_cc,
    AthenaArray<Real> & var_vc,
    const int dim,
    const int fl, const int fu,
    const int ll, const int lu,
    const int ml, const int mu,
    const int nl, const int nu
  );

  void var_map_VC2CC_Taylor1(
    const AthenaArray<Real> & rds,
    const AthenaArray<Real> & var_vc,
    AthenaArray<Real> & dvar_cc,
    const int dim,
    const int ll, const int lu,
    const int ml, const int mu,
    const int nl, const int nu
  );

}; // namespace InterpIntergrid


class InterpIntergridLocal {
  public:
    InterpIntergridLocal(int dim, const int * strides, const Real * rds);
    virtual ~InterpIntergridLocal();

    // instance (vectorizable) for nd->1d -----------------
    inline Real map1d_VC2CC(const Real & in)
    {
      return _map1d_var(in,0,0,1);
    }

    inline Real map2d_VC2CC(const Real & in)
    {
      return _map2d_var(in,0,0,0,1);
    }

    inline Real map3d_VC2CC(const Real & in)
    {
      return _map3d_var(in,0,0,0,0,1);
    }

    inline Real map1d_CC2VC(const Real & in)
    {
      return _map1d_var(in,0,-1,-1);
    }

    inline Real map2d_CC2VC(const Real & in)
    {
      return _map2d_var(in,0,0,-1,-1);
    }

    inline Real map3d_CC2VC(const Real & in)
    {
      return _map3d_var(in,0,0,0,-1,-1);
    }

    inline Real map1d_VC2CC_der(const int dir, Real & in)
    {
      return _map1d_varD1TS(in, 0);
    }

    inline Real map2d_VC2CC_der(const int dir, Real & in)
    {
      return _map2d_varD1TS(in, dir, 0, 0);
    }

    inline Real map3d_VC2CC_der(const int dir, Real & in)
    {
      return _map3d_varD1TS(in, dir, 0, 0, 0);
    }


    // interfaces mimicking base implementation -------------------------------
    void var_map_VC2CC(
      const AthenaArray<Real> & var_vc,
      AthenaArray<Real> & var_cc,
      const int fl, const int fu,
      const int ll, const int lu,
      const int ml, const int mu,
      const int nl, const int nu
    );

    void var_map_CC2VC(
      const AthenaArray<Real> & var_cc,
      AthenaArray<Real> & var_vc,
      const int fl, const int fu,
      const int ll, const int lu,
      const int ml, const int mu,
      const int nl, const int nu
    );

    void var_map_VC2CC_Taylor1(
      const AthenaArray<Real> & rds,
      const AthenaArray<Real> & var_vc,
      AthenaArray<Real> & dvar_cc,
      const int ll, const int lu,
      const int ml, const int mu,
      const int nl, const int nu
    );

  private:
    int dim;
    int * N;
    int * strides;
    Real * rds;

    // VC<->CC ----------------------------------------------------------------
    inline Real _map3d_var(const Real & in,
      const int l, const int m, const int n,
      const int os, const int ph_dg)
    {
      const Real * pin = &in;
      const int dg = ph_dg*(VC_NGHOST-CC_NGHOST);
      Real out(0.);

      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i+1+os;
        const int i_u = l+i+os;
        const int lix = NGRCV_HSZ+i-1;

        const int I_L = (i_l+dg)*strides[0-os*dim];
        const int I_U = (i_u+dg)*strides[0-os*dim];

        Real const lci = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];

        for(int j=1; j<=NGRCV_HSZ; ++j)
        {
          const int j_l = m-j+1+os;
          const int j_u = m+j+os;
          const int ljx = NGRCV_HSZ+j-1;

          const int J_L = (j_l+dg)*strides[1-os*dim];
          const int J_U = (j_u+dg)*strides[1-os*dim];

          Real const lcj = \
            InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ljx];

          for(int k=1; k<=NGRCV_HSZ; ++k)
          {
            const int k_l = n-k+1+os;
            const int k_u = n+k+os;
            const int lkx = NGRCV_HSZ+k-1;

            const int K_L = (k_l+dg)*strides[2-os*dim];
            const int K_U = (k_u+dg)*strides[2-os*dim];

            Real const lck = \
              InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lkx];

            Real const f_uuu = pin[K_U+J_U+I_U];
            Real const f_lll = pin[K_L+J_L+I_L];
            Real const f_lul = pin[K_L+J_U+I_L];
            Real const f_ulu = pin[K_U+J_L+I_U];
            Real const f_ull = pin[K_L+J_L+I_U];
            Real const f_luu = pin[K_U+J_U+I_L];
            Real const f_llu = pin[K_U+J_L+I_L];
            Real const f_uul = pin[K_L+J_U+I_U];

            out += lck * lcj * lci * (
              (f_uuu + f_lll) + (f_lul + f_ulu) +
              (f_ull + f_luu) + (f_llu + f_uul)
            );

          }

        }
      }

      return out;
    }

    inline Real _map2d_var(const Real & in, const int l, const int m,
      const int os, const int ph_dg)
    {
      const Real * pin = &in;
      const int dg = ph_dg*(VC_NGHOST-CC_NGHOST);
      Real out(0.);

      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i+1+os;
        const int i_u = l+i+os;
        const int lix = NGRCV_HSZ+i-1;
        Real const lci = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];

        const int I_L = (i_l+dg)*strides[0-os*dim];
        const int I_U = (i_u+dg)*strides[0-os*dim];

        for(int j=1; j<=NGRCV_HSZ; ++j)
        {
          const int j_l = m-j+1+os;
          const int j_u = m+j+os;
          const int ljx = NGRCV_HSZ+j-1;
          Real const lcj = \
            InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ljx];

          const int J_L = (j_l+dg)*strides[1-os*dim];
          const int J_U = (j_u+dg)*strides[1-os*dim];

          Real const f_uu = pin[J_U+I_U];
          Real const f_ll = pin[J_L+I_L];
          Real const f_ul = pin[J_L+I_U];
          Real const f_lu = pin[J_U+I_L];

          out += lcj * lci * (
            ((f_uu+f_ll) +(f_ul+f_lu))
          );
        }
      }
      return out;
    }

    inline Real _map1d_var(const Real & in, const int l,
      const int os, const int ph_dg)
    {
      const Real * pin = &in;
      const int dg = ph_dg*(VC_NGHOST-CC_NGHOST);
      Real out(0.);

      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = l-i+1+os;
        const int i_u = l+i+os;
        const int lix = NGRCV_HSZ+i-1;

        const int I_L = (i_l+dg)*strides[0-os*dim];
        const int I_U = (i_u+dg)*strides[0-os*dim];

        Real const lc = InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][lix];
        out += lc*(pin[I_L]+pin[I_U]);
      }
      return out;
    }

    // VC->D[CC] --------------------------------------------------------------
    inline Real _map1d_varD1TS(const Real & in, const int l)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int i_l = dg+l;
      const int i_u = dg+l+1;

      return rds[0]*(pin[i_u]-pin[i_l]);
    }

    inline Real _map2d_varD1TS(const Real & in,
      const int dir, const int l, const int m)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int j_l = (dg+m)*strides[1];
      const int j_u = (dg+m+1)*strides[1];

      const int i_l = (dg+l);
      const int i_u = (dg+l+1);

      const Real fac = 0.5*rds[dir];

      if (dir==0)
      {
        return fac*(
            (pin[j_l+i_u] - pin[j_l+i_l]) +
            (pin[j_u+i_u] - pin[j_u+i_l])
          );
      }
      return fac*(
          (pin[j_u+i_u] - pin[j_l+i_u]) +
          (pin[j_u+i_l] - pin[j_l+i_l])
        );

    }

    inline Real _map3d_varD1TS(const Real & in,
      const int dir, const int l, const int m, const int n)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int k_l = (dg+n)*strides[2];
      const int k_u = (dg+n+1)*strides[2];

      const int j_l = (dg+m)*strides[1];
      const int j_u = (dg+m+1)*strides[1];

      const int i_l = dg+l;
      const int i_u = dg+l+1;

      const Real fac = 0.25*rds[dir];

      if (dir==0)
      {
        return fac*(
          ((pin[k_u+j_u+i_u] - pin[k_l+j_l+i_l]) -
           (pin[k_u+j_u+i_l] - pin[k_l+j_l+i_u])) -
          ((pin[k_l+j_u+i_l] - pin[k_u+j_l+i_u]) -
           (pin[k_l+j_u+i_u] - pin[k_u+j_l+i_l]))
        );
      }
      else if (dir==1)
      {
        return fac*(
          ((pin[k_u+j_u+i_u] - pin[k_l+j_l+i_l]) +
           (pin[k_u+j_u+i_l] - pin[k_l+j_l+i_u])) +
          ((pin[k_l+j_u+i_l] - pin[k_u+j_l+i_u]) +
           (pin[k_l+j_u+i_u] - pin[k_u+j_l+i_l]))
        );
      }
      return fac*(
        (
          (pin[k_u+j_u+i_u] - pin[k_l+j_l+i_l]) -
          (pin[k_l+j_u+i_l] - pin[k_u+j_l+i_u])
        ) +
        (
          (pin[k_u+j_u+i_l] - pin[k_l+j_l+i_u]) -
          (pin[k_l+j_u+i_u] - pin[k_u+j_l+i_l])
        )
      );
    }

};


#endif // INTERP_INTERGRID_HPP_
