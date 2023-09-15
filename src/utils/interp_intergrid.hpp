#ifndef INTERP_INTERGRID_HPP_
#define INTERP_INTERGRID_HPP_
//! \file interp_intergrid.hpp
//  \brief prototypes of utility functions to pack/unpack buffers

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"

// ----------------------------------------------------------------------------
// New impl. here
namespace InterpIntergrid {

// Provide centred stencils for VC->D[CC]
// Reversed stencil entry order (smaller values first)
template<int der_, int half_stencil_size_>
class InterpolateVC2DerCC {
  public:
    // order of convergence (in spacing)
    enum {order = 2 * half_stencil_size_ - 1};
    enum {N_I = half_stencil_size_};
    static Real const coeff[N_I];
};

// templated on data type and number of nodes utilized either-side of target
// base-points
template <typename dtype, int H_SZ>
class InterpIntergrid
{
  public:
    InterpIntergrid(const int ndim,
                    const int * N,
                    const dtype * rds,
                    const int NG_CC,
                    const int NG_VC);
    ~InterpIntergrid();

  public:
    // interfaces for interpolation; output is into 1d scratch
    template <TensorSymm TSYM, int DIM, int NVAL>
    void VC2CC(
      AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
      const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
      const int cc_k,
      const int cc_j);

    void VC2CC(
      AthenaArray<       dtype> & tar,
      const  AthenaArray<dtype> & src,
      const int cc_k,
      const int cc_j);

    template <TensorSymm TSYM, int DIM, int NVAL>
    void CC2VC(
      AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
      const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
      const int vc_k,
      const int vc_j);

    void CC2VC(
      AthenaArray<       dtype> & tar,
      const  AthenaArray<dtype> & src,
      const int vc_k,
      const int vc_j);

    // Mapping to face-centers
    // -----------------------
    // Consider ordering (k,j,i)
    //
    // Vertex-centered maps to FC where FC is aligned on:
    //
    // axis 0:
    // (VC,VC,VC) -> (CC,CC,VC)
    //
    // axis 1:
    // (VC,VC,VC) -> (CC,VC,CC)
    //
    // axis 2:
    // (VC,VC,VC) -> (VC,CC,CC)
    //
    // Cell-centered maps to FC where FC is aligned on:
    //
    // axis 0:
    // (CC,CC,CC) -> (CC,CC,VC)
    //
    // axis 1:
    // (CC,CC,CC) -> (CC,VC,CC)
    //
    // axis 2:
    // (CC,CC,CC) -> (VC,CC,CC)
    //
    // N.B. as is seen above the character of the target indices changes
    // based on the direction FC is aligned.
    void VC2FC(
      AthenaArray<       dtype> & tar,
      const  AthenaArray<dtype> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j,
      const int tr_il,
      const int tr_iu);

    void CC2FC(
      AthenaArray<       dtype> & tar,
      const  AthenaArray<dtype> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j,
      const int tr_il,
      const int tr_iu);

    template <TensorSymm TSYM, int DIM, int NVAL>
    void VC2FC(
      AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
      const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j,
      const int tr_il,
      const int tr_iu);

    template <TensorSymm TSYM, int DIM, int NVAL>
    void CC2FC(
      AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
      const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j,
      const int tr_il,
      const int tr_iu);

    // 1d scratch equivalents
    void VC2FC(
      AthenaArray<       dtype> & tar,
      const  AthenaArray<dtype> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j);

    void CC2FC(
      AthenaArray<       dtype> & tar,
      const  AthenaArray<dtype> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j);

    template <TensorSymm TSYM, int DIM, int NVAL>
    void VC2FC(
      AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
      const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j);

    template <TensorSymm TSYM, int DIM, int NVAL>
    void CC2FC(
      AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
      const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
      const int dir,
      const int tr_k,  // target grid idxs
      const int tr_j);


    template <TensorSymm TSYM, int DIM, int NVAL>
    void VC2CC_D1(
      AthenaTensor<       dtype, TSYM, DIM, NVAL+1> & tar,
      const  AthenaTensor<dtype, TSYM, DIM, NVAL  > & src,
      const int dir,
      const int cc_k,
      const int cc_j);

    void VC2CC_D1(
      AthenaArray<       dtype> & tar,
      const  AthenaArray<dtype> & src,
      const int dir,
      const int cc_k,
      const int cc_j);

  private:
    const int dim;
    const int NG_CC;
    const int NG_VC;
    const int dg;
    const int dc;
    const int dv;

    int * N;           // number of _physical cells_  (vertices will be + 1)
    int * ncells;      // these include ghosts
    int * nverts;      // these include ghosts
    int * strides_cc;
    int * strides_vc;
    dtype * rds;

  public:
    // expose maximal target iteration indices ...
    int cc_il = 0, cc_iu = 0;
    int cc_jl = 0, cc_ju = 0;
    int cc_kl = 0, cc_ku = 0;

    int vc_il = 0, vc_iu = 0;
    int vc_jl = 0, vc_ju = 0;
    int vc_kl = 0, vc_ku = 0;
};

}  // namespace InterpIntergrid

// implementation details (for templates) =====================================
#include "interp_intergrid.tpp"
// ============================================================================

// ----------------------------------------------------------------------------
// Direct impl. here for reference
// Suggestion: use the newer, they are more flexible and contain these...
namespace InterpIntergridDirect {

//----------------------------------------------------------------------------------------
// \!fn Real CCInterpolation(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform cubic interpolation from cell-centered grid to vertex.
inline Real CCInterpolation(AthenaArray<Real> &in, int k, int j, int i) {
  // interpolation coefficients
  // ordering is (i, j, k) = +/- (1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1),
  //                             (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2)
  const Real coeff[8] = {729.0, -81.0, -81.0, 9.0, -81.0, 9.0, 9.0, -1.0};
  const Real den   = 4096.0;
  // Cells are located to the right of their corresponding vertices. So, i=1 corresponds
  // to a physical cell index a=0.
  const int off = 1;
  Real sum = 0;
  // Loop in reverse; the higher-index coefficients tend to have smaller contributions,
  // so this *may* limit round-off error.
  for (int c = 2; c > 0; --c) {
    for (int b = 2; b > 0; --b) {
      for (int a = 2; a > 0; --a) {
        int index = (a - off) + 2*((b - off) + 2*(c - off));
        // Grab the cube around vertex and add it up.
        Real lll = in(k - c, j - b, i - a);
        Real llu = in(k - c, j - b, i + a - off);
        Real lul = in(k - c, j + b - off, i - a);
        Real luu = in(k - c, j + b - off, i + a - off);
        Real ull = in(k + c - off, j - b, i - a);
        Real ulu = in(k + c - off, j - b, i + a - off);
        Real uul = in(k + c - off, j + b - off, i - a);
        Real uuu = in(k + c - off, j + b - off, i + a - off);
        // Attempt to add up the cube in a way that preserves symmetry.
        sum += coeff[index]*( ((lll + uuu) + (lul + ulu)) + ((llu + uul) + (luu + ull)) );
      }
    }
  }
  return sum/den;
}
//---------------------------------------------------------------------------------------
// \!fn Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform linear interpolation to the desired cell-centered grid index.
inline Real VCInterpolation(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.125*(((in(k, j, i) + in(k + 1, j + 1, i + 1)) // lower-left-front to upper-right-back
               + (in(k+1, j+1, i) + in(k, j, i+1))) // upper-left-back to lower-right-front
               +((in(k, j+1, i) + in(k + 1, j, i+1)) // lower-left-back to upper-right-front
               + (in(k+1, j, i) + in(k, j+1, i+1)))); // upper-left-front to lower-right-back
}
//---------------------------------------------------------------------------------------
// \!fn Real VCReconstruct(AthenaArray<Real> &in, int k, int j, int i)
// \brief Perform linear interpolation to the desired face-centered grid index.
template<int dir>
inline Real VCReconstruct(AthenaArray<Real> &in, int k, int j, int i);
template<>
inline Real VCReconstruct<0>(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.25*((in(k, j, i) + in(k + 1, j + 1, i)) +
               (in(k + 1, j, i) + in(k, j + 1, i)));
}
template<>
inline Real VCReconstruct<1>(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.25*((in(k, j, i) + in(k + 1, j, i + 1)) +
               (in(k + 1, j, i) + in(k, j, i + 1)));
}
template<>
inline Real VCReconstruct<2>(AthenaArray<Real> &in, int k, int j, int i) {
  return 0.25*((in(k, j, i) + in(k, j + 1, i + 1)) +
               (in(k, j + 1, i) + in(k, j, i + 1)));
}
inline Real VCReconstruct(int dir, AthenaArray<Real> &in, int k, int j, int i) {
  switch(dir) {
    case 0:
      return VCReconstruct<0>(in, k, j, i);
    case 1:
      return VCReconstruct<1>(in, k, j, i);
    case 2:
      return VCReconstruct<2>(in, k, j, i);
    default:
      abort();
  }
}
//---------------------------------------------------------------------------------------
// \!fn Real VCDiff(AthenaArray<Real> &in, int k, int j, int i)
// \brief Evaluates the undivided derivative of a vertex centered variable at cell centers.
template<int dir>
inline Real VCDiff(AthenaArray<Real> &in, int k, int j, int i);
template<>
inline Real VCDiff<0>(AthenaArray<Real> &in, int k, int j, int i) {
  const Real coeff[2] = {9./8., -1./24.};
  Real stencil[4]; // values at the cell faces
  int off = 1;
  for (int a = -1; a < 3; ++a) {
    stencil[a + off] = 0.25*((in(k, j, i + a) + in(k + 1, j + 1, i + a)) +
                             (in(k + 1, j, i + a) + in(k, j + 1, i + a)));
  }
  return coeff[1]*(stencil[3] - stencil[0]) + coeff[0]*(stencil[2] - stencil[1]);
}
template<>
inline Real VCDiff<1>(AthenaArray<Real> &in, int k, int j, int i) {
  const Real coeff[2] = {9./8., -1./24.};
  Real stencil[4]; // values at the cell faces
  int off = 1;
  for (int a = -1; a < 3; ++a) {
    stencil[a + off] = 0.25*((in(k, j + a, i) + in(k + 1, j + a, i + 1)) +
                             (in(k + 1, j + a, i) + in(k, j + a, i + 1)));
  }
  return coeff[1]*(stencil[3] - stencil[0]) + coeff[0]*(stencil[2] - stencil[1]);
}
template<>
inline Real VCDiff<2>(AthenaArray<Real> &in, int k, int j, int i) {
  const Real coeff[2] = {9./8., -1./24.};
  Real stencil[4]; // values at the cell faces
  int off = 1;
  for (int a = -1; a < 3; ++a) {
    stencil[a + off] = 0.25*((in(k + a, j, i) + in(k + a, j + 1, i + 1)) +
                             (in(k + a, j + 1, i) + in(k + a, j, i + 1)));
  }
  return coeff[1]*(stencil[3] - stencil[0]) + coeff[0]*(stencil[2] - stencil[1]);
}
inline Real VCDiff(int dir, AthenaArray<Real> &in, int k, int j, int i) {
  switch(dir) {
    case 0:
      return VCDiff<0>(in, k, j, i);
    case 1:
      return VCDiff<1>(in, k, j, i);
    case 2:
      return VCDiff<2>(in, k, j, i);
    default:
      abort();
  }
}


} // namespace InterpIntergridDirect

//
// :D
//

#endif // INTERP_INTERGRID_HPP_
