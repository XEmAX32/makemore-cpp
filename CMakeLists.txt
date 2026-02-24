cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main LANGUAGES CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(ROOT REQUIRED COMPONENTS Core Hist Graf Gpad RIO)

add_executable(main main.cpp)
target_link_libraries(main
  ${TORCH_LIBRARIES}
  ROOT::Core
  ROOT::Hist
  ROOT::Graf
  ROOT::Gpad
  ROOT::RIO
)

set_property(TARGET main PROPERTY CXX_STANDARD 20)
set_property(TARGET main PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET main PROPERTY CXX_EXTENSIONS OFF)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)