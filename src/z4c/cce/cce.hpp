#ifndef CCE_HPP
#define CCE_HPP

#include <string>
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"

class Mesh;
class MeshBlock;
class ParameterInput;
namespace decomp_decompose {class decomp_info;};

// max absolute value of spin in spin weighted Ylm
#define MAX_SPIN (2)

class CCE
{
  private:
    Real rin;  // inner radius of shell
    Real rout; // outer radius of shell
    Real ncycle; // num. of cycle(iter)
    Real *ifield; // interpolated values of the given field
    std::string fieldname; // field name that used for pittnull code
    std::string filename; // h5 file name
    Mesh *pm;             // mesh
    ParameterInput *pin;  // param file
    const decomp_decompose::decomp_info **dinfo_pp; // decomposition info
    int num_mu_points;  // number of points in theta direction(polar)
    int num_phi_points; // number of points in phi direction(azimuthal)
    int num_x_points;   // number of points in radius between the two shells
    int num_l_modes;    // number of l modes in -2Ylm (m modes calculated automatically)
    int num_n_modes;    // radial modes
    int nangle;         // num_mu_points*num_phi_points
    int npoint;         // num_mu_points*num_phi_points*num_x_points
    int spin;     // it's 0 and really not used
    Real *xb; // Cart. x coords. for spherical coords.
    Real *yb; // Cart. y coords. for spherical coords.
    Real *zb; // Cart. z coords. for spherical coords.
    
  public:
    CCE(Mesh *const pm, ParameterInput *const pin, std::string fname, int n);
    ~CCE();
    void InterpolateSphToCart(MeshBlock *const pmb);
    void ReduceInterpolation();
    void Decompose();
    void Write();
};

#endif

