cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app LANGUAGES CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")

set_property(TARGET example-app PROPERTY CXX_STANDARD 20)
set_property(TARGET example-app PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET example-app PROPERTY CXX_EXTENSIONS OFF)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)