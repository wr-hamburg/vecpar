#!/bin/sh

spack load llvm@14.0.0
spack load googletest@1.10.0
spack load eigen@3.4.0
spack load cmake@3.22.2
spack load cuda@11.6
echo "Spack environment loaded!"

# the number of repetitions
N=$1

# the number of threads
TH=$2

# move to test folder
cd $3

# the destination folder for the results
FILE_F=$4

# max threads limit
export OMP_THREAD_LIMIT=($TH+1)
export OMP_NUM_THREADS=$TH

############## CPU ####################
./hm_tests_compiled_for_cpu --gtest_repeat=$N  --gtest_filter=PerformanceTest_HostDevice*
./mm_tests_compiled_for_cpu --gtest_repeat=$N  --gtest_filter=PerformanceTest_ManagedMemory*

############## GPU ####################
./hm_tests_compiled_for_gpu --gtest_repeat=$N  --gtest_filter=PerformanceTest_HostDevice*
./mm_tests_compiled_for_gpu --gtest_repeat=$N  --gtest_filter=PerformanceTest_ManagedMemory*

# move csv to results folder
mkdir -p $FILE_F
mv *.csv $FILE_F
echo "Copy results to $FILE_F completed!"

# clear environment
spack unload llvm@14.0.0
spack unload googletest@1.10.0
spack unload eigen@3.4.0
spack unload cmake@3.22.2
spack unload cuda@11.6
echo "Spack environment unloaded!"
