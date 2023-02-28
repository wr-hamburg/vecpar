#!/bin/sh

# $1 = repetitions
# $2 = omp threads count
# $3 = location test executables
# $4 = destination folder
# e.g. ./saxpy_and_chain.sh 20 16 ~/vecpar-release-build/test/ ~/results/

spack load llvm@14.0.0
spack load googletest@1.10.0
spack load eigen@3.4.0
spack load cmake@3.22.3 %gcc@11.2.1
spack load cuda@11.6.2
spack load vecmem@0.22.0
echo "Spack environment loaded!"

echo "clang version: "
clang --version

# the number of repetitions
N=$1

# the number of omp threads
TH=$2

# the destination folder for the results
FILE_F=$4

# max threads limit
export OMP_THREAD_LIMIT=($TH+1)
export OMP_NUM_THREADS=$TH

echo "Starting tests for vecpar native"
# move to test folder
cd $3/single_source

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

echo "Starting tests for vecpar ompt"
cd ../ompt

############## CPU ####################
./tests_ompt_cpu --gtest_repeat=$N  --gtest_filter=PerformanceTest_HostDevice*

############## GPU ####################
./tests_ompt_gpu --gtest_repeat=$N  --gtest_filter=PerformanceTest_HostDevice*

# move csv to results folder
mkdir -p $FILE_F
mv *.csv $FILE_F
echo "Copy results to $FILE_F completed!"

unset OMP_THREAD_LIMIT
unset OMP_NUM_THREADS

# cleanup
spack unload llvm@14.0.0
spack unload googletest@1.10.0
spack unload eigen@3.4.0
spack unload cmake@3.22.3 %gcc@11.2.1
spack unload cuda@11.6.2
spack unload vecmem@0.22.0
echo "Spack environment unloaded!"
