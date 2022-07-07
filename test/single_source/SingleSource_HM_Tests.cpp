#include "benchmark/performance_host_device_memory.cpp"
#include "scenarios/host_device_memory.cpp"

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}