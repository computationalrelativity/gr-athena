#include <cstdio>

#define Code_mesh  void *mb

void SphericalHarmonicDecomp_DumpMetric(Code_mesh);

int main(void)
{
   printf("Calling SphericalHarmonicDecomp_DumpMetric ...\n");
   SphericalHarmonicDecomp_DumpMetric(nullptr);
}
