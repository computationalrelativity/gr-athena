//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <unistd.h>
#include <cmath> // NAN

static const double fact35[] = {
  1.,
  1.,
  2.,
  6.,
  24.,
  120.,
  720.,
  5040.,
  40320.,
  362880.,
  3628800., 
  39916800.,
  479001600.,
  6227020800.,
  87178291200.,
  1307674368000.,
  20922789888000.,
  355687428096000.,
  6402373705728000.,
  121645100408832000.,
  2432902008176640000.,
  51090942171709440000.,
  1124000727777607680000.,
  25852016738884976640000.,
  620448401733239439360000.,
  15511210043330985984000000.,
  403291461126605635584000000.,
  10888869450418352160768000000.,
  304888344611713860501504000000.,
  8841761993739701954543616000000.,
  265252859812191058636308480000000.,
  8222838654177922817725562880000000.,
  263130836933693530167218012160000000.,
  8683317618811886495518194401280000000.,
  295232799039604140847618609643520000000.,
  10333147966386144929666651337523200000000.};

Real Factorial(const int n) {
  if (n < 0){
    std::stringstream msg;
    msg << "### FATAL ERROR in WaveExtractRWZ::Factorial" << std::endl
        << "factorial requires integer nonnegeative argument " << n << std::endl;
    ATHENA_ERROR(msg);
  } else if (n <= 35){
    return fact35[n];
  } else {
    return ((Real)n) * Factorial(n-1);
  }
}

Real SphHarm_Plm(const int l, const int m, const Real x) {

  Real pmm = 1.0;
    
  if (m>=0) {
    Real somx2 = std::sqrt((1.0-x)*(1.0+x));
    Real fact = 1.0;
    for (int i=1; i<=m; ++i) {
      pmm  = -pmm*fact*somx2;
      fact += 2.0;
    }
  }
  else {
    std::stringstream msg;
    msg << "### FATAL ERROR in Utils::SphHarm_Plm" << std::endl
        << "SphHarm_Plm requires nonnegeative m argument " << m << std::endl;
    ATHENA_ERROR(msg);
  }
  
  if (l == m) {
    return pmm;
  }
  
  Real pmmp1 = x*(2.0*m + 1.0)*pmm;
  
  if (l == (m+1)) { 
    return pmmp1;
  }
  
  for (int i=m+2; i<=l; ++i) {
    Real pll = ( x* ((Real)(2*i-1)) * pmmp1 - ((Real)(i+m-1)) * pmm )/((Real)(i-m));
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pmmp1;    
}


void SphHarm_Ylm(const int l, const int m, const Real theta, const Real phi,
				 Real * YlmR, Real * YlmI) {
  
  const int abs_m = std::abs(m);
  const Real fact_norm = Factorial(l+abs_m)/Factorial(l-abs_m);
    
  //TODO Old code, test and remove -----
  Real fac = 1.0;  
  for (int i=(l-abs_m+1); i<=(l+abs_m); ++i) {
    fac *= (Real)(i);
  } 
  if (std::fabs(fac - fact_norm) > 1e-10){
     printf("l = %s, m = %s, theta = %s, phi = %s \n", l, m, theta, phi);
     printf("fac = %s, fact_norm = %s, diff = %s \n", fac, fact_norm, fac - fact_norm); 
  } 
  //assert(std::fabs(fac - fact_norm)>1e-10);
  //fac = 1.0/fac //divide once below
  // -------------
  
  const Real a = std::sqrt((Real)(2*l+1)/(4.0*PI*fact_norm));
  //const int mfac = (m>0)? std::pow(-1.0,abs_m) : 1.0; //FIXME: this is the original, but it should be:
  const int mfac = (m<0)? std::pow(-1.0, m) : 1.0; //FIXME: this is the original, but it should be:
  //const int mfac = (m==0)? 1.0 : std::pow(-1.0,m); 
  const Real Plm = mfac * a * SphHarm_Plm(l,abs_m,std::cos(theta));

  *YlmR = Plm * std::cos((Real)(m)*phi);
  *YlmI = Plm * std::sin((Real)(m)*phi);
  
}

// \brief compute vector spherical harmonics basis and functions Xlm and Wlm
void SphHarm_Ylm_a(const int l_, const int m_, const Real theta, const Real phi,
				   Real * YthR, Real * YthI, Real * YphR, Real * YphI,
				   Real * XR, Real * XI, Real * WR, Real * WI) {
  
  const Real l = (Real)l_;
  const Real m = (Real)m_;

  const Real div_sin_theta = 1.0/(std::sin(theta));
  const Real cot_theta = std::cos(theta) * div_sin_theta;
  
  const Real a = -(l+1.0) * cot_theta;
  const Real b = std::sqrt((SQR(l+1.0)-SQR(m))*(l+0.5)/(l+1.5)) * div_sin_theta;

  Real YR,YI; // l,m
  SphHarm_Ylm(l,m,theta,phi,&YR,&YI);
  
  Real YplusR,YplusI; // l+1,m
  SphHarm_Ylm(l+1,m,theta,phi,&YplusR,&YplusI);

  const Real _YthR = a * YR + b * YplusR;
  const Real _YthI = a * YI + b * YplusI;

  const Real c = - 2.0*cot_theta;
  const Real d = (2.0*SQR(m*div_sin_theta) - l*(l+1.0));
  
  *YthR = _YthR;
  *YthI = _YthI;
  
  *YphR = - m * YI;
  *YphI =   m * YR;

  *WR =  c * (*YthR) + d * YR;
  *WI =  c * (*YthI) + d * YI;

  *XR = 2.0 * m * (cot_theta*YI - _YthI);
  *XI = 2.0 * m * (_YthR - cot_theta*YR);    
}

