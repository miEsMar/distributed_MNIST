cmake_minimum_required(VERSION 3.5)
project(dist_mnist)

find_package(MPI REQUIRED)
find_package(Torch REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH} ${CMAKE_PREFIX_PATH}/include)

add_executable(dist_mnist main.cpp)
target_link_libraries(dist_mnist ${TORCH_LIBRARIES})
target_link_libraries(dist_mnist ${MPI_LIBRARIES})
# target_link_libraries(dist_mnist ${CMAKE_PREFIX_PATH}/lib/libc10d.a)

add_executable(serial_mnist main.cpp)
target_compile_definitions(serial_mnist PUBLIC SERIALISED_VERSION)
target_link_libraries(serial_mnist ${TORCH_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(dist_mnist PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(dist_mnist PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
