#ifndef VECPAR_CHAIN_HPP
#define VECPAR_CHAIN_HPP

#include "main.hpp"

#include "orchestrator/default_chain.hpp"

#if defined(__CUDA__) && defined(__clang__)
#include "orchestrator/gpu_chain.hpp"
#endif

#endif // VECPAR_CHAIN_HPP
