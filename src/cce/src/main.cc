#include <cstdio>

#define Code_mesh       const MeshBlock *mb
class MeshBlock;

void SphericalHarmonicDecomp_DumpMetric(Code_mesh);

int main(void)
{
   printf("Calling SphericalHarmonicDecomp_DumpMetric ...\n");
   SphericalHarmonicDecomp_DumpMetric(0);
}
