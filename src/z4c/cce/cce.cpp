#include "cce.hpp"
#include "../../parameter_input.hpp"

using namespace decomp_matrix_class;
using namespace decomp_sYlm;
using namespace decomp_decompose;

CCE::CCE(Mesh *const pm, ParameterInput *const pin, std::string name, int n):
    pm(pm),
    pin(pin),
    fieldname(name),
    max_spin(2), // max absolute value of spin in spin weighted Ylm
    spin(0),
    dinfo_pp(nullptr)
{
  rin  = pin->GetReal("cce", "rin_"  + std::to_string(n));
  rout = pin->GetReal("cce", "rout_" + std::to_string(n));
  num_mu_points  = pin->GetOrAddInteger("cce","num_theta",41);
  num_phi_points = pin->GetOrAddInteger("cce","num_phi",82);
  num_x_points   = pin->GetOrAddInteger("cce","num_r_inshell",28);
  num_l_modes    = pin->GetOrAddInteger("cce","num_l_modes",7);
  num_n_modes    = pin->GetOrAddInteger("cce","num_radial_modes",7);
  nangle = num_mu_points*num_phi_points;
  npoint = nangle*num_x_points;

  double *radius    = nullptr;
  double *mucolloc  = nullptr;
  double *phicolloc = nullptr;
  myassert (ABS(spin) <= max_spin);

  const int nlmmodes = num_l_modes*(num_l_modes+2*ABS(max_spin));
  dinfo_pp = new const decomp_info* [2*max_spin+1];
  myassert(dinfo_pp);
  for (int s=-max_spin; s<=max_spin; s++)
  {
    dinfo_pp[s+max_spin] = nullptr;
  }

  radius = new double [num_x_points];
  xb = new double [nangle*num_x_points];
  yb = new double [nangle*num_x_points];
  zb = new double [nangle*num_x_points];
  mucolloc = new double [nangle];
  phicolloc = new double [nangle];

  myassert(radius);
  myassert(xb);
  myassert(yb);
  myassert(zb);
  myassert(mucolloc);
  myassert(phicolloc);

  if (! dinfo_pp[max_spin + spin])
  {
    dinfo_pp[max_spin + spin] =  initialize(spin, num_l_modes, num_n_modes,
                                   num_mu_points, num_phi_points, num_x_points);
    myassert(dinfo_pp[max_spin + spin]);
  }

  dinfo_pp[max_spin + spin]->get_ncolloc(num_x_points, radius);
  dinfo_pp[max_spin + spin]->get_mucolloc(nangle, mucolloc);
  dinfo_pp[max_spin + spin]->get_phicolloc(nangle, phicolloc);

  for (int k=0; k < num_x_points; k++)
  {
    double xk = radius[k];
    radius[k] = 0.5 * ( (rout - rin) * xk + (rout + rin) );
  }

  for (int k=0; k < num_x_points; k++)
  {
    for (int i=0; i < nangle; i++)
    {
      const double phi = phicolloc[i];
      const double mu = mucolloc[i];

      const double sph = sin(phi);
      const double cph = cos(phi);
      const double cth = mu;
      const double sth = sqrt(1.0 - mu*mu);

      const int indx = i + k*nangle;
      
      xb[indx] = radius[k] * sth*cph;
      yb[indx] = radius[k] * sth*sph;
      zb[indx] = radius[k] * cth;
    
    }
  }
    
  // free mem.
  delete [] mucolloc;
  delete [] phicolloc;
  delete [] radius;
}

CCE::~CCE()
{
  delete [] xb;
  delete [] yb;
  delete [] zb;
}
