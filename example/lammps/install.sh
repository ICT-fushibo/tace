

# Select the architecture matching the deployment GPU:
# -D Kokkos_ARCH_AMPERE80=ON \  # A100/A800
# -D Kokkos_ARCH_ADA89=ON \     # RTX 4090
# -D Kokkos_ARCH_HOPPER90=ON \  # H100/H200/H20

# If the gcc version is too old, specify it manually
# -D CMAKE_C_COMPILER=/usr/bin/gcc-13 \
# -D CMAKE_CXX_COMPILER=/usr/bin/g++-13 \
# -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \

# -D Python_EXECUTABLE="$(which python)" \
# -D Python3_EXECUTABLE="$(which python)" \

cmake -C kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd) \
  -D BUILD_MPI=ON \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_ADA89=ON \
  ../cmake
