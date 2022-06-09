#include <gtest/gtest.h>
#include "scenarios/managed_memory.cpp"
#include "scenarios/host_device_memory.cpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}