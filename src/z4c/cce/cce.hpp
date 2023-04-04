#ifndef CCE_HPP
#define CCE_HPP

class Mesh;
class ParameterInput;

class CCE
{
  private:
    Real Rin;  // inner radius of shell
    Real Rout; // outer radius of shell
    Real ncycle; // num. of cycle(iter)
    std::string fieldname; // field name
    std::string filename; // h5 file name
    Mesh *pm;             // mesh
    ParameterInput *pin;  // param file
    decomp_info **dinfo_p; // decomposition info
    int num_mu_points;  // number of points in theta direction(polar)
    int num_phi_points; // number of points in phi direction(azimuthal)
    int num_x_points;   // number of points in radius between the two shells
    int num_l_modes;    // number of l modes in -2Ylm (m modes calculated automatically)
    int num_n_modes;    // radial modes
    int nangle;         // num_mu_points*num_phi_points
    int npoint;         // num_mu_points*num_phi_points*num_x_points
    int max_spin; // max absolute value of spin in spin weighted Ylm, = 2
    int spin;     // it's 0 and really not used
    Real *xb; // Cart. x coords. for spherical coords.
    Real *yb; // Cart. y coords. for spherical coords.
    Real *zb; // Cart. z coords. for spherical coords.
    
  public:
    CCE(Mesh *const pm, ParameterInput *const pin, std::string fname, int n);
    ~CCE();
}

#endif

