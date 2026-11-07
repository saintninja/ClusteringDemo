#include "test_support.hpp"

int main() {
  clustering::tests::TestSuite suite;
  clustering::tests::AddLinearAlgebraTests(suite);
  clustering::tests::AddDistanceTests(suite);
  clustering::tests::AddQualityMetricTests(suite);
  clustering::tests::AddInitializerTests(suite);
  clustering::tests::AddAlgorithmTests(suite);
  clustering::tests::AddInputOutputTests(suite);
  clustering::tests::AddPipelineTests(suite);
  return suite.Run();
}
