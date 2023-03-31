#include <cstdio>

#define N (5)

void Test_SphericalHarmonicDecomp_DumpMetric(int iter);

int main(void)
{
   for (int i = 0; i < N; ++i)
   {
      printf("Calling Test_SphericalHarmonicDecomp_DumpMetric(%d) ...\n",i);
      Test_SphericalHarmonicDecomp_DumpMetric(i);
   }
   
}
