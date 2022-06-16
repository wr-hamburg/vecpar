#ifndef VECPAR_TIMETEST_HPP
#define VECPAR_TIMETEST_HPP

#include <chrono>
#include <gtest/gtest.h>

class TimeTest : public testing::Test {
protected:
  void SetUp() override {
    const testing::TestInfo *const test_info =
        testing::UnitTest::GetInstance()->current_test_info();

    printf("[%s] %s \n", test_info->test_suite_name(), test_info->name());
    start_time = std::chrono::steady_clock::now();
  }
  void TearDown() override {
    end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Time = " << diff.count() << " s\n";
  }

  std::chrono::time_point<std::chrono::steady_clock> start_time;
  std::chrono::time_point<std::chrono::steady_clock> end_time;
};

#endif // VECPAR_TIMETEST_HPP
