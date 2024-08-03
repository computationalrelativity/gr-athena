// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <string>     // c_str()

// Athena++ classes headers
#include "../athena.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "task_list.hpp"
#include "time_integrators.hpp"

#include "../wave/wave.hpp"

// ----------------------------------------------------------------------------
using namespace TaskLists::Integrators;

Butcher::Butcher(ParameterInput *pin, Mesh *pm)
{

  std::string integrator;
  integrator = pin->GetOrAddString("time", "integrator", "vl2");

  cfl_limit = std::numeric_limits<Real>::infinity();

  if (integrator == "bt_rk2")
  {
    nstages = 2;

    bt_a.resize(nstages);

    bt_a[0] = {0, 0};
    bt_a[1] = {1, 0};
    bt_b    = {0, 1};
    bt_c    = {0, 1. / 3., 2. / 3.};
  }
  else if (integrator == "bt_rk4")
  {
    nstages = 4;

    bt_a.resize(nstages);

    bt_a[0] = {0,   0,   0, 0};
    bt_a[1] = {0.5, 0,   0, 0};
    bt_a[2] = {0,   0.5, 0, 0};
    bt_a[3] = {0,   0,   1, 0};
    bt_b    = {1. / 6., 1. / 3., 1. / 3., 1. / 6.};
    bt_c    = {0, 0.5, 0.5, 1};
  }
  else if (integrator == "bt_rk5_6_lawson")
  {
    // An Order Five Runge Kutta Process with Extended Region of Stability,
    // J. Douglas Lawson, Siam Journal on Numerical Analysis, Vol. 3, No. 4,
    // (Dec., 1966) pages 593-597.

    nstages = 6;

    bt_a.resize(nstages);

    // bt_a[0] = {0        , 0      , 0      , 0      , 0      ,  0};
    // bt_a[1] = {1./12.   , 0      , 0      , 0      , 0      ,  0};
    // bt_a[2] = {-1./8.   , 3./8.  , 0      , 0      , 0      ,  0};
    // bt_a[3] = {3./5.    , -9./10., 4./5.  , 0      , 0      ,  0};
    // bt_a[4] = {39./80.  , -9./20., 3./20. , 9./16. , 0      ,  0};
    // bt_a[5] = {-59./35. , 66./35., 48./35., -12./7., 8./7.  ,  0};

    // bt_b    = {7./90., 0., 16./45., 2./15., 16./45., 7./90.};
    // bt_c    = {0, 1./12., 1./4., 1./2., 3./4., 1.};

    for (int n=0; n<nstages; ++n)
    {
      bt_a[n] = {0, 0, 0, 0, 0, 0};
    }
    bt_b = {0, 0, 0, 0, 0, 0};
    bt_c = {0, 0, 0, 0, 0, 0};


    bt_a[2-1][1-1]=1./12.;
    bt_a[3-1][1-1]=-1./8.;
    bt_a[3-1][2-1]=3./8.;
    bt_a[4-1][1-1]=3./5.;
    bt_a[4-1][2-1]=-9./10.;
    bt_a[4-1][3-1]=4./5.;
    bt_a[5-1][1-1]=39./80.;
    bt_a[5-1][2-1]=-9./20.;
    bt_a[5-1][3-1]=3./20.;
    bt_a[5-1][4-1]=9./16.;
    bt_a[6-1][1-1]=-59./35.;
    bt_a[6-1][2-1]=66./35.;
    bt_a[6-1][3-1]=48./35.;
    bt_a[6-1][4-1]=-12./7.;
    bt_a[6-1][5-1]=8./7.;

    bt_b[1-1]=7./90.;
    bt_b[2-1]=0.;
    bt_b[3-1]=16./45.;
    bt_b[4-1]=2./15.;
    bt_b[5-1]=16./45.;
    bt_b[6-1]=7./90.;

    bt_c[2-1]=1./12.;
    bt_c[3-1]=1./4.;
    bt_c[4-1]=1./2.;
    bt_c[5-1]=3./4.;
    bt_c[6-1]=1.;

  }
  else if (integrator == "bt_rk6_7_tsi_pap")
  {
    // Cheap Error Estimation for Runge-Kutta methods, by Ch. Tsitouras and
    // S.N. Papakostas, Siam Journal on Scientific Computing,
    // Vol. 20, Issue 6, Nov 1999.

    nstages = 7;

    bt_a.resize(nstages);

    for (int n=0; n<nstages; ++n)
    {
      bt_a[n] = {0, 0, 0, 0, 0, 0, 0};
    }
    bt_b = {0, 0, 0, 0, 0, 0, 0};
    bt_c = {0, 0, 0, 0, 0, 0, 0};

    bt_a[2-1][1-1]=122./825.;
    bt_a[3-1][1-1]=61./1100.;
    bt_a[3-1][2-1]=183./1100.;
    bt_a[4-1][1-1]=369717480605./1944575538588.;
    bt_a[4-1][2-1]=-338008281875./648191846196.;
    bt_a[4-1][3-1]=368505651250./486143884647.;
    bt_a[5-1][1-1]=93214174153328./340919077080285.;
    bt_a[5-1][2-1]=-152702000./555274449.;
    bt_a[5-1][3-1]=1851003107866000./22506857615972997.;
    bt_a[5-1][4-1]=83558471950495568./137541907653168315.;
    bt_a[6-1][1-1]=-1965847328391587593./15119418421435820580.;
    bt_a[6-1][2-1]=880622875./1818346284.;
    bt_a[6-1][3-1]=405093066521302476836750./2976253186608002415460743.;
    bt_a[6-1][4-1]=-63703602474558645489436./7476885456589129904746665.;
    bt_a[6-1][5-1]=16702879434846059./57945970286473161.;
    bt_a[7-1][1-1]=45053227812514909./138126271460184260.;
    bt_a[7-1][2-1]=-6119317875./10073470676.;
    bt_a[7-1][3-1]=169126321772072545991000./380255815114811302413181.;
    bt_a[7-1][4-1]=224483491181217551025101184./389215442486773620500705065.;
    bt_a[7-1][5-1]=-465033719274534591489./846209446845346083994.;
    bt_a[7-1][6-1]=92269971576997526970./114009981780519605573.;
    // bt_a[8-1][1-1]=-79047738172285352895991./104406674665036144815360.;
    // bt_a[8-1][2-1]=48160647374125./83448631079984.;
    // bt_a[8-1][3-1]=121222613235537082394556512125./56519166026227705328020901904.;
    // bt_a[8-1][4-1]=-1160593758240764869503758536187./503406154571662014742841768880.;
    // bt_a[8-1][5-1]=920561234320198236972923./686647274587702364481792.;
    // bt_a[8-1][6-1]=0.;
    // bt_a[8-1][7-1]=0.;

    bt_b[1-1]=957591139./13163409600.;
    bt_b[2-1]=0.;
    bt_b[3-1]=3106181514970703125./10856998549728201696.;
    bt_b[4-1]=42389973878019739820811./222948632808913396985600.;
    bt_b[5-1]=174493329774903./1559109847896640.;
    bt_b[6-1]=535612536001611./2028157860601600.;
    bt_b[7-1]=2518367669./33571977600.;
    // bt_b[8-1]=0.;

    bt_c[2-2]=122./825.;
    bt_c[3-2]=61./275.;
    bt_c[4-2]=3355./7863.;
    bt_c[5-2]=64./93.;
    bt_c[6-2]=67./87.;
    bt_c[7-2]=1.;
    bt_c[8-2]=1.;

  }
  else if (integrator == "bt_rk8_11_cop_ver")
  {
    // Some Explicit Runge-Kutta Methods of High Order, by
    // G. J. Cooper and J. H. Verner, SIAM Journal on Numerical Analysis,
    // Vol. 9, No. 3, (September 1972), pages 389 to 405

    nstages = 11;

    bt_a.resize(nstages);

    for (int n=0; n<nstages; ++n)
    {
      bt_a[n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
    bt_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    bt_c = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const Real sqrt_21 = std::sqrt(21.);

    bt_a[2-1][1-1]=1./2.;
    bt_a[3-1][1-1]=1./4.;
    bt_a[3-1][2-1]=1./4.;
    bt_a[4-1][1-1]=1./7.;
    bt_a[4-1][2-1]=-1./14.+3./98.*sqrt_21;
    bt_a[4-1][3-1]=3./7.-5./49.*sqrt_21;
    bt_a[5-1][1-1]=11./84.-1./84.*sqrt_21;
    bt_a[5-1][2-1]=0;
    bt_a[5-1][3-1]=2./7.-4./63.*sqrt_21;
    bt_a[5-1][4-1]=1./12.+1./252.*sqrt_21;
    bt_a[6-1][1-1]=5./48.-1./48.*sqrt_21;
    bt_a[6-1][2-1]=0;
    bt_a[6-1][3-1]=1./4.-1./36.*sqrt_21;
    bt_a[6-1][4-1]=-77./120.-7./180.*sqrt_21;
    bt_a[6-1][5-1]=63./80.+7./80.*sqrt_21;
    bt_a[7-1][1-1]=5./21.+1./42.*sqrt_21;
    bt_a[7-1][2-1]=0;
    bt_a[7-1][3-1]=-48./35.-92./315.*sqrt_21;
    bt_a[7-1][4-1]=211./30.+29./18.*sqrt_21;
    bt_a[7-1][5-1]=-36./5.-23./14.*sqrt_21;
    bt_a[7-1][6-1]=9./5.+13./35.*sqrt_21;
    bt_a[8-1][1-1]=1./14.;
    bt_a[8-1][2-1]=0;
    bt_a[8-1][3-1]=0;
    bt_a[8-1][4-1]=0;
    bt_a[8-1][5-1]=1./9.+1./42.*sqrt_21;
    bt_a[8-1][6-1]=13./63.+1./21.*sqrt_21;
    bt_a[8-1][7-1]=1./9.;
    bt_a[9-1][1-1]=1./32.;
    bt_a[9-1][2-1]=0;
    bt_a[9-1][3-1]=0;
    bt_a[9-1][4-1]=0;
    bt_a[9-1][5-1]=91./576.+7./192.*sqrt_21;
    bt_a[9-1][6-1]=11./72.;
    bt_a[9-1][7-1]=-385./1152.+25./384.*sqrt_21;
    bt_a[9-1][8-1]=63./128.-13./128.*sqrt_21;
    bt_a[10-1][1-1]=1./14.;
    bt_a[10-1][2-1]=0;
    bt_a[10-1][3-1]=0;
    bt_a[10-1][4-1]=0;
    bt_a[10-1][5-1]=1./9.;
    bt_a[10-1][6-1]=-733./2205.+1./15.*sqrt_21;
    bt_a[10-1][7-1]=515./504.-37./168.*sqrt_21;
    bt_a[10-1][8-1]=-51./56.+11./56.*sqrt_21;
    bt_a[10-1][9-1]=132./245.-4./35.*sqrt_21;
    bt_a[11-1][1-1]=0;
    bt_a[11-1][2-1]=0;
    bt_a[11-1][3-1]=0;
    bt_a[11-1][4-1]=0;
    bt_a[11-1][5-1]=-7./3.-7./18.*sqrt_21;
    bt_a[11-1][6-1]=-2./5.-28./45.*sqrt_21;
    bt_a[11-1][7-1]=-91./24.+53./72.*sqrt_21;
    bt_a[11-1][8-1]=301./72.-53./72.*sqrt_21;
    bt_a[11-1][9-1]=28./45.+28./45.*sqrt_21;
    bt_a[11-1][10-1]=49./18.+7./18.*sqrt_21;

    bt_b[1-1]=1./20.;
    bt_b[2-1]=0;
    bt_b[3-1]=0;
    bt_b[4-1]=0;
    bt_b[5-1]=0;
    bt_b[6-1]=0;
    bt_b[7-1]=0;
    bt_b[8-1]=49./180.;
    bt_b[9-1]=16./45.;
    bt_b[10-1]=49./180.;
    bt_b[11-1]=1./20.;

    bt_c[2-2]=1./2.;
    bt_c[3-2]=1./2.;
    bt_c[4-2]=1./2.-1./14.*sqrt_21;
    bt_c[5-2]=1./2.-1./14.*sqrt_21;
    bt_c[6-2]=1./2.;
    bt_c[7-2]=1./2.+1./14.*sqrt_21;
    bt_c[8-2]=1./2.+1./14.*sqrt_21;
    bt_c[9-2]=1./2.;
    bt_c[10-2]=1./2.-1./14.*sqrt_21;
    bt_c[11-2]=1.;

  }
  else
  {
  std::stringstream msg;
  msg << "### FATAL ERROR in Butcher constructor" << std::endl
      << "integrator=" << integrator
      << " not valid time integrator" << std::endl;
  ATHENA_ERROR(msg);
  }

  // Set cfl_number based on user input
  Real cfl_number = pin->GetReal("time", "cfl_number");

  // Save to Mesh class
  pm->cfl_number = cfl_number;
}

void Butcher::PrepareStageScratch(const int stage,
                                  MeshBlock * pmb,
                                  std::vector<AA> & bt_k,
                                  const AthenaArray<Real> & u,
                                  AthenaArray<Real> & rhs)
{
  const int nn1 = u.GetDim1();
  const int nn2 = u.GetDim2();
  const int nn3 = u.GetDim3();
  const int N   = u.GetDim4();

  if (bt_k.size() != nstages + 1)
  {
    for (int n=0; n<=nstages; ++n)
    {
      AthenaArray<Real> k(N,nn3,nn2,nn1);
      bt_k.push_back(k);
    }
  }

  // rhs = k_s
  rhs.InitWithShallowSlice(bt_k[stage-1]);

  if (stage==1)
  {
    // last scratch stores u_n
    PutScratchBT_u(pmb, bt_k, u);
  }
}

void Butcher::PutScratchBT_u(MeshBlock * pmb,
                             std::vector<AA> & bt_k,
                             const AthenaArray<Real> & u)
{
  static const int SCRATCH_SLOT_U = 0;

  const int nn1 = u.GetDim1();
  const int nn2 = u.GetDim2();
  const int nn3 = u.GetDim3();
  const int N   = u.GetDim4();

  const int il = 0, iu = nn1-1;
  const int jl = 0, ju = nn2-1;
  const int kl = 0, ku = nn3-1;

  // Grab scratch
  AthenaArray<Real> u_scr;
  u_scr.InitWithShallowSlice(bt_k[nstages+SCRATCH_SLOT_U]);

  for (int n=0; n<N; ++n)
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      u_scr(n,k,j,i) = u(n,k,j,i);
    }
  }
}

void Butcher::SumBT_ak(MeshBlock * pmb,
                       const int stage,
                       const Real dt,
                       std::vector<AA> & bt_k,
                       AthenaArray<Real> & u_out)
{
  static const int SCRATCH_SLOT_U = 0;

  const int nn1 = u_out.GetDim1();
  const int nn2 = u_out.GetDim2();
  const int nn3 = u_out.GetDim3();
  const int N   = u_out.GetDim4();

  const int il = 0, iu = nn1-1;
  const int jl = 0, ju = nn2-1;
  const int kl = 0, ku = nn3-1;

  // Grab scratch for sum of arguments to current stage k
  AthenaArray<Real> u;
  u.InitWithShallowSlice((bt_k[nstages+SCRATCH_SLOT_U]));

  for (int n=0; n<N; ++n)
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      u_out(n,k,j,i) = u(n,k,j,i);
    }
  }

  for (int s=1; s<=stage; ++s)
  {
    const Real a = bt_a[stage-1][s-1];
    if (a!=0)
    {
      const Real a_dt = a*dt;

      for (int n=0; n<N; ++n)
      for (int k=kl; k<=ku; ++k)
      for (int j=jl; j<=ju; ++j)
      {
        #pragma omp simd
        for (int i=il; i<=iu; ++i)
        {
          u_out(n,k,j,i) += a_dt * bt_k[s-1](n,k,j,i);  // k_s(n,k,j,i)
        }
      }

    }
  }
}

void Butcher::SumBT_bk(MeshBlock * pmb,
                       const Real dt,
                       std::vector<AA> & bt_k,
                       AthenaArray<Real> & u_out)
{
  const int SCRATCH_SLOT_U = 0;

  const int nn1 = u_out.GetDim1();
  const int nn2 = u_out.GetDim2();
  const int nn3 = u_out.GetDim3();
  const int N   = u_out.GetDim4();

  const int il = 0, iu = nn1-1;
  const int jl = 0, ju = nn2-1;
  const int kl = 0, ku = nn3-1;

  // Grab scratch for sum of arguments to current stage k
  AthenaArray<Real> u;
  u.InitWithShallowSlice(bt_k[nstages+SCRATCH_SLOT_U]);

  for (int n=0; n<N; ++n)
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      u_out(n,k,j,i) = u(n,k,j,i);
    }
  }

  for (int s=1; s<=nstages; ++s)
  {
    const Real b = bt_b[s-1];

    if (b!=0)
    {
      const Real b_dt = b*dt;

      for (int n=0; n<N; ++n)
      for (int k=kl; k<=ku; ++k)
      for (int j=jl; j<=ju; ++j)
      {
        #pragma omp simd
        for (int i=il; i<=iu; ++i)
        {
          u_out(n,k,j,i) += b_dt * bt_k[s-1](n,k,j,i);  // k_s(n,k,j,i)
        }
      }

    }
  }

}

//
// :D
//
