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

// Provide centred stencils for VC->D[CC]
// Odd 'der' take negative values on the left part of the stencil
// uniform grid assumed.
template<int der_, int half_stencil_size_>
class InterpolateVC2DerCC {
  public:
    // order of convergence (in spacing)
    enum {order = 2 * half_stencil_size_ - 1};
    enum {N_I = half_stencil_size_};
    static Real const coeff[N_I];
};


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
      #if NGRCV_HSZ == 1
        return _map1d_varD1TS(dir, in);
      #else
        return _map1d_varD1HO(dir, in);
      #endif
    }

    inline Real map2d_VC2CC_der(const int dir, Real & in)
    {
      #if NGRCV_HSZ == 1
        return _map2d_varD1TS(dir, in);
      #else
        return _map2d_varD1HO(dir, in);
      #endif
    }

    inline Real map3d_VC2CC_der(const int dir, Real & in)
    {
      #if NGRCV_HSZ == 1
        return _map3d_varD1TS(dir, in);
      #else
        return _map3d_varD1HO(dir, in);
      #endif
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

    // Taylor series 1st order based
    inline Real _map1d_varD1TS(const int dir, const Real & in)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int i_l = dg*strides[dir];
      const int i_u = (dg+1)*strides[dir];

      return rds[dir]*(pin[i_u]-pin[i_l]);
    }

    // Higher order
    inline Real _map1d_varD1HO(const int dir, const Real & in)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      Real out = 0.;

      for(int i=1; i<=NGRCV_HSZ; ++i)
      {
        const int i_l = (dg-i+1)*strides[dir];
        const int i_u = (dg+i  )*strides[dir];

        const Real li = InterpolateVC2DerCC<1, NGRCV_HSZ>::coeff[i-1];

        out += li*(pin[i_u]-pin[i_l]);
      }

      return rds[dir]*out;
    }


    // Taylor series 1st order based
    inline Real _map2d_varD1TS(const int dir, const Real & in)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int ix_a = dg*strides[(dir+1)%2];
      const int ix_b = (dg+1)*strides[(dir+1)%2];

      return 0.5*(
        _map1d_varD1TS(dir, pin[ix_a]) +
        _map1d_varD1TS(dir, pin[ix_b])
      );
    }

    // Higher order
    inline Real _map2d_varD1HO(const int dir, const Real & in)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int b_is_0 = (dir == 0);  // (dir+1)%2
      const int b_is_1 = (dir == 1);  // (dir+0)%2

      Real out = 0.;

      for(int j=1; j<=NGRCV_HSZ; ++j)
      {
        const int j_l = (dg-j+1)*strides[1];
        const int j_u = (dg+j  )*strides[1];

        for(int i=1; i<=NGRCV_HSZ; ++i)
        {
          const int i_l = (dg-i+1)*strides[0];
          const int i_u = (dg+i  )*strides[0];

          const int ix_c = (i-1) * b_is_0 + (j-1) * b_is_1;
          const int ix_l = (i-1) * b_is_1 + (j-1) * b_is_0 + NGRCV_HSZ;

          const Real lag = \
            InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ix_l];
          const Real cdr = \
            InterpolateVC2DerCC<1, NGRCV_HSZ>::coeff[ix_c];

          const Real lcji = cdr*lag;

          const int ix_uu = j_u+i_u;
          const int ix_ll = j_l+i_l;
          const int ix_lu = j_l+i_u;
          const int ix_ul = j_u+i_l;

          const Real f_uu = pin[ix_uu];
          const Real f_ll = pin[ix_ll];
          const Real f_lu = pin[ix_lu*b_is_0+ix_ul*b_is_1];
          const Real f_ul = pin[ix_ul*b_is_0+ix_lu*b_is_1];

          out += lcji*((f_uu-f_ul) +
                       (f_lu-f_ll));

        }
      }

      return rds[dir]*out;
    }


    // Taylor series 1st order based
    inline Real _map3d_varD1TS(const int dir, const Real & in)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int ix_a = ((dg+1)*strides[(dir+1)%3] + (dg+1)*strides[(dir+2)%3]);
      const int ix_b = (dg*strides[(dir+1)%3] + dg*strides[(dir+2)%3]);
      const int ix_c = (dg*strides[(dir+1)%3] + (dg+1)*strides[(dir+2)%3]);
      const int ix_d = ((dg+1)*strides[(dir+1)%3] + dg*strides[(dir+2)%3]);

      return 0.25*(
        _map1d_varD1TS(dir, pin[ix_a]) +
        _map1d_varD1TS(dir, pin[ix_b]) +
        _map1d_varD1TS(dir, pin[ix_c]) +
        _map1d_varD1TS(dir, pin[ix_d])
      );
    }

    // Higher order
    inline Real _map3d_varD1HO(const int dir, const Real & in)
    {
      const Real * pin = &in;
      const int dg = VC_NGHOST-CC_NGHOST;

      const int b_is_0 = (dir == 0);  // (1+dir)%3%2
      const int b_is_1 = (dir == 1);  // (2+dir)%2
      const int b_is_2 = (dir == 2);  // ((3-dir)%3)%2

      Real out = 0.;
      for(int k=1; k<=NGRCV_HSZ; ++k)
      {
        const int k_l = (dg-k+1)*strides[2];
        const int k_u = (dg+k  )*strides[2];

        for(int j=1; j<=NGRCV_HSZ; ++j)
        {
          const int j_l = (dg-j+1)*strides[1];
          const int j_u = (dg+j  )*strides[1];

          for(int i=1; i<=NGRCV_HSZ; ++i)
          {
            const int i_l = (dg-i+1)*strides[0];
            const int i_u = (dg+i  )*strides[0];

            const int ix_c = (i-1) * b_is_0 + (j-1) * b_is_1 + (k-1) * b_is_2;
            const int ix_l1 = (
              (i-1) * b_is_2 + (j-1) * b_is_0 + (k-1) * b_is_1
              + NGRCV_HSZ);
            const int ix_l2 = (
              (i-1) * b_is_1 + (j-1) * b_is_2 + (k-1) * b_is_0
              + NGRCV_HSZ);

            const Real lag1 = \
              InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ix_l1];
            const Real lag2 = \
              InterpolateLagrangeUniform<NGRCV_HSZ>::coeff[1][ix_l2];
            const Real cdr = \
              InterpolateVC2DerCC<1, NGRCV_HSZ>::coeff[ix_c];

            const Real lckji = cdr*lag1*lag2;

            const int ix_uuu = k_u+j_u+i_u;
            const int ix_lll = k_l+j_l+i_l;
            const int ix_lul = k_l+j_u+i_l;
            const int ix_ulu = k_u+j_l+i_u;
            const int ix_ull = k_u+j_l+i_l;
            const int ix_luu = k_l+j_u+i_u;
            const int ix_llu = k_l+j_l+i_u;
            const int ix_uul = k_u+j_u+i_l;

            const int I_lul = ix_lul*b_is_0+ix_ull*b_is_1+ix_llu*b_is_2;
            const int I_ulu = ix_ulu*b_is_0+ix_luu*b_is_1+ix_uul*b_is_2;
            const int I_ull = ix_ull*b_is_0+ix_llu*b_is_1+ix_lul*b_is_2;
            const int I_luu = ix_luu*b_is_0+ix_uul*b_is_1+ix_ulu*b_is_2;
            const int I_llu = ix_llu*b_is_0+ix_lul*b_is_1+ix_ull*b_is_2;
            const int I_uul = ix_uul*b_is_0+ix_ulu*b_is_1+ix_luu*b_is_2;

            const Real f_uuu = pin[ix_uuu];
            const Real f_lll = pin[ix_lll];
            const Real f_lul = pin[I_lul];
            const Real f_ulu = pin[I_ulu];
            const Real f_ull = pin[I_ull];
            const Real f_luu = pin[I_luu];
            const Real f_llu = pin[I_llu];
            const Real f_uul = pin[I_uul];

            out += lckji*((f_uuu - f_uul) +
                          (f_ulu - f_ull) +
                          (f_llu - f_lll) +
                          (f_luu - f_lul));

          }
        }
      }
      return rds[dir]*out;
    }

};


#endif // INTERP_INTERGRID_HPP_
