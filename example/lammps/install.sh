

# -D Kokkos_ARCH_Hopper70=ON \ # A800
# -D Kokkos_ARCH_ADA89=ON \    # 4090
# -D Kokkos_ARCH_HSX90=ON \    # H100

# If the gcc version is too old, specify it manually
# -D CMAKE_C_COMPILER=/usr/bin/gcc-13 \
# -D CMAKE_CXX_COMPILER=/usr/bin/g++-13 \
# -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \

cmake -C kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd) \
  -D CMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -D CMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -D BUILD_MPI=ON \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_ADA89=ON \
  ../cmake
