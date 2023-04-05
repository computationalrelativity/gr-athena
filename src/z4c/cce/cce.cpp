#include <algorithm> // for fill
#include <cmath>     // for NAN
#include <stdexcept>
#include <sstream>
#include "cce.hpp"
#include "matrix.hpp"
#include "sYlm.hpp"
#include "myassert.hpp"
#include "decomp.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../utils/lagrange_interp.hpp"
#include "../z4c.hpp"

using namespace decomp_matrix_class;
using namespace decomp_sYlm;
using namespace decomp_decompose;

#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

CCE::CCE(Mesh *const pm, ParameterInput *const pin, std::string name, int n):
    pm(pm),
    pin(pin),
    fieldname(name),
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
  myassert (ABS(spin) <= MAX_SPIN);

  const int nlmmodes = num_l_modes*(num_l_modes+2*ABS(MAX_SPIN));
  dinfo_pp = new const decomp_info* [2*MAX_SPIN+1];
  myassert(dinfo_pp);
  for (int s=-MAX_SPIN; s<=MAX_SPIN; s++)
  {
    dinfo_pp[s+MAX_SPIN] = nullptr;
  }

  // alloc
  radius = new double [num_x_points];
  xb = new double [nangle*num_x_points];
  yb = new double [nangle*num_x_points];
  zb = new double [nangle*num_x_points];
  mucolloc = new double [nangle];
  phicolloc = new double [nangle];
  ifield = new double [nangle*num_x_points]();
 
  myassert(radius);
  myassert(xb);
  myassert(yb);
  myassert(zb);
  myassert(mucolloc);
  myassert(phicolloc);
  myassert(ifield);

  std::fill(radius, radius + (num_x_points),NAN); // init to nan
  std::fill(xb, xb + (nangle*num_x_points),NAN); // init to nan
  std::fill(yb, yb + (nangle*num_x_points),NAN); // init to nan
  std::fill(zb, zb + (nangle*num_x_points),NAN); // init to nan
  std::fill(mucolloc,  mucolloc  + (nangle),NAN); // init to nan
  std::fill(phicolloc, phicolloc + (nangle),NAN); // init to nan

  if (! dinfo_pp[MAX_SPIN + spin])
  {
    dinfo_pp[MAX_SPIN + spin] =  initialize(spin, num_l_modes, num_n_modes,
                                   num_mu_points, num_phi_points, num_x_points);
    myassert(dinfo_pp[MAX_SPIN + spin]);
  }

  dinfo_pp[MAX_SPIN + spin]->get_ncolloc(num_x_points, radius);
  dinfo_pp[MAX_SPIN + spin]->get_mucolloc(nangle, mucolloc);
  dinfo_pp[MAX_SPIN + spin]->get_phicolloc(nangle, phicolloc);

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
  delete [] ifield;
  delete [] xb;
  delete [] yb;
  delete [] zb;

  for (int i = 0; i < 2*MAX_SPIN+1; ++i)
    delete [] dinfo_pp[i];
  delete [] dinfo_pp;
}

void CCE::InterpolateSphToCart(MeshBlock *const pmb)
{
  const int Npoints = nangle*num_x_points;
  Real const origin[3] = {pmb->pcoord->x1f(0),
                          pmb->pcoord->x2f(0),
                          pmb->pcoord->x3f(0)};
  Real const delta[3]  = {pmb->pcoord->dx1f(0),
                          pmb->pcoord->dx2f(0),
                          pmb->pcoord->dx3f(0)};
  int const size[3]    = {pmb->nverts1,
                          pmb->nverts2,
                          pmb->nverts3};

  // find the src field
  Real *src_field = nullptr;
  Z4c *const pz4c = pmb->pz4c;
  Z4c::Z4c_vars z4c;
  Z4c::ADM_vars adm;
  pz4c->SetZ4cAliases(pz4c->storage.u, z4c);
  pz4c->SetADMAliases(pz4c->storage.adm, adm);
  
  if (fieldname == "gxx")
  {
    src_field = &adm.g_dd(0,0, 0,0,0);
  }
  else if (fieldname == "gxy")
  {
    src_field = &adm.g_dd(0,1, 0,0,0);
  }
  else if (fieldname == "gxz")
  {
    src_field = &adm.g_dd(0,2, 0,0,0);
  }
  else if (fieldname == "gyy")
  {
    src_field = &adm.g_dd(1,1, 0,0,0);
  }
  else if (fieldname == "gyz")
  {
    src_field = &adm.g_dd(1,2, 0,0,0);
  }
  else if (fieldname == "gzz")
  {
    src_field = &adm.g_dd(2,2, 0,0,0);
  }
  else if (fieldname == "betax")
  {
    src_field = &z4c.beta_u(0, 0, 0, 0);
  }
  else if (fieldname == "betay")
  {
    src_field = &z4c.beta_u(1, 0, 0, 0);
  }
  else if (fieldname == "betaz")
  {
    src_field = &z4c.beta_u(2, 0, 0, 0);
  }
  else if (fieldname == "alp")
  {
    src_field = &z4c.alpha(0, 0, 0);
  }
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CCE interpolation" << std::endl;
    msg << "Could not find '" << fieldname << "' for interpolation!";
    throw std::runtime_error(msg.str().c_str());
  }
  
  
  for (int p = 0; p < Npoints; ++p)
  {
    double coord[3] = {xb[p], yb[p], zb[p]};
    if (pmb->PointContained(coord[0], coord[1], coord[2]))
    {
      LagrangeInterpND<2*NGHOST-1, 3> linterp(origin, delta, size, coord);

// note: the point may take place at the boundary of two meshblocks
#pragma omp atomic write
      ifield[p] = linterp.eval(src_field);
      
      // printf("f(%g,%g,%g) = %g\n",coord[0],coord[1],coord[2],ifield[p]);
    }
  }
}

void CCE::ReduceInterpolation(){}
void CCE::Decompose(){}
void CCE::Write(){}
