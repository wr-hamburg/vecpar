#ifndef VECPAR_CONFIG_HPP
#define VECPAR_CONFIG_HPP
#include <stddef.h>

namespace vecpar {

class config {

public:
  config() = default;

  config(int grid_size, int block_size)
      : m_gridSize(grid_size), m_blockSize(block_size) {}

  config(int grid_size, int block_size, size_t ext_memory)
      : m_gridSize(grid_size), m_blockSize(block_size),
        m_memorySize(ext_memory) {}

  static constexpr bool isEmpty(config c) { return (c.m_gridSize == 0 || c.m_blockSize == 0); }

  int m_gridSize = 0;
  int m_blockSize = 0;
  size_t m_memorySize = 0;
};
} // namespace vecpar

#endif // VECPAR_CONFIG_HPP
