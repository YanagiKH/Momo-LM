cmake_minimum_required(VERSION 3.20)
project(momo_core VERSION 0.3.0 LANGUAGES C CXX)

option(MOMO_ENABLE_SANITIZERS "Build native tests with address/undefined sanitizers" OFF)

add_library(momo_core SHARED src/tensor.c src/attention.c src/runtime.cpp)
target_include_directories(momo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(momo_core PRIVATE MOMO_BUILDING_DLL=1)
target_compile_features(momo_core PRIVATE c_std_11 cxx_std_17)
set_target_properties(momo_core PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

if(MSVC)
  target_compile_options(momo_core PRIVATE /O2 /W4)
else()
  target_compile_options(momo_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_link_libraries(momo_core PRIVATE m)
endif()

add_executable(momo_native_smoke tests/smoke.cpp)
target_compile_features(momo_native_smoke PRIVATE cxx_std_17)
target_link_libraries(momo_native_smoke PRIVATE momo_core)

if(MOMO_ENABLE_SANITIZERS AND NOT MSVC)
  foreach(target momo_core momo_native_smoke)
    target_compile_options(${target} PRIVATE -fno-omit-frame-pointer -fsanitize=address,undefined)
    target_link_options(${target} PRIVATE -fno-omit-frame-pointer -fsanitize=address,undefined)
  endforeach()
endif()

enable_testing()
add_test(NAME momo_native_smoke COMMAND momo_native_smoke)

install(TARGETS momo_core
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES include/momo_core.h DESTINATION include)
