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
    Z4c *pz4c;       // ptr to z4c
    Real ref_tol;    // refinment tolerance
    Real dref_tol;   // derefinment tolerance
    Real ref_x1min;  // x1 min of the region of interest for the refinement
    Real ref_x1max;  // x1 max of the region of interest for the refinement
    Real ref_x2min;  // x2 min of the region of interest for the refinement
    Real ref_x2max;  // x2 max of the region of interest for the refinement
    Real ref_x3min;  // x3 min of the region of interest for the refinement
    Real ref_x3max;  // x3 max of the region of interest for the refinement
    int ref_deriv;   // order of derivative to compute error
    int ref_pow;     // power of the derivative
    bool verbose;    // turn on/off print
    
    // returning the L2 norm of: 
    // h^6 * ( (d^n fld/dx^n)^p + (d^n fld/dy^n)^p + (d^n fld/dz^n)^p )
    Real amr_err_L2_derive_chi_pow(MeshBlock *const pmb, const int deriv_order, 
                                   const int p);
    
  public:
    std::string ref_method;  // method of refinement
    explicit Z4c_AMR(MeshBlock *pmb);
    ~Z4c_AMR();
    int FDErrorApprox(MeshBlock *pmb); // using the FD error as an approximation for
                                       // the error in the meshblock.
};

#endif // Z4c_AMR_HPP
