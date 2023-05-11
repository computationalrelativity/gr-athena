#ifndef Z4c_AMR_HPP
#define Z4c_AMR_HPP

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../utils/finite_differencing.hpp"
#include "z4c_macro.hpp"

//! \class Z4c_AMR
//  \brief managing AMR for Z4c simulations
class Z4c_AMR
{
  private:
    Real ref_x1min;  // x1 min of the region of interest for the refinement
    Real ref_x1max;  // x1 max of the region of interest for the refinement
    Real ref_x2min;  // x2 min of the region of interest for the refinement
    Real ref_x2max;  // x2 max of the region of interest for the refinement
    Real ref_x3min;  // x3 min of the region of interest for the refinement
    Real ref_x3max;  // x3 max of the region of interest for the refinement
    Real ref_gwh;    // resolution required for GW extraction
    Real ref_gwr;    // max radius among the gw extraction radii
    int ref_deriv;   // order of derivative to compute error
    int ref_pow;     // power of the derivative
    bool verbose;    // turn on/off print
    
    // returning the L2 norm of: 
    // h^6 * ( (d^n fld/dx^n)^p + (d^n fld/dy^n)^p + (d^n fld/dz^n)^p )
    Real amr_err_L2_derive_chi_pow(MeshBlock *const pmb, const int deriv_order, 
                                   const int p);

    // returning the max err of: 
    // h^6 * ( (d^n fld/dx^n)^p + (d^n fld/dy^n)^p + (d^n fld/dz^n)^p )
    Real amr_err_Linf_derive_chi_pow(MeshBlock *const pmb, const int deriv_order, 
                                   const int p);
    
  public:
    Z4c *pz4c;               // ptr to z4c
    ParameterInput *pin;     // ptr to parameter
    std::string ref_method;  // method of refinement
    Real mb_radius;  // the length of the line from the origin to the meshblock's center
    Real ref_hmax;   // max grid-space in hx, hy, hz
    Real ref_hpow;   // power of the grid space
    Real ref_FD_r1_inn, ref_FD_r1_out; // 1st annulus of refinement
    Real ref_FD_r2_inn, ref_FD_r2_out; // 2nd annulus of refinement
    Real ref_tol1, dref_tol1; // 1st annulus of refinement range
    Real ref_tol2, dref_tol2; // 2n annulus of refinement range
    Real ref_PrerefTime;    // preref if the time is less than
    bool ref_IsPreref_Linf; // pre-refine with Linf?
    bool ref_IsPreref_L2;   // pre-refine with L2?

    Z4c_AMR(MeshBlock *pmb,ParameterInput *pin);
    int ShouldIRefine(MeshBlock *pmb); // should I refine?
    ~Z4c_AMR();
    // using the FD error as an approximation for the error in the meshblock.
    int FDErrorApprox(MeshBlock *pmb, Real dref_tol, Real ref_tol); 
    int LinfBoxInBox(MeshBlock *pmb); // Linf box in box
    int L2SphereInSphere(MeshBlock *pmb); // L2 Sphere in Sphere
};

#endif // Z4c_AMR_HPP
