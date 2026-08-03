cmake_minimum_required(VERSION 3.20)
project(momo_core VERSION 0.2.0 LANGUAGES C CXX)

add_library(momo_core SHARED src/tensor.c src/runtime.cpp)
target_include_directories(momo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(momo_core PRIVATE MOMO_BUILDING_DLL=1)
target_compile_features(momo_core PRIVATE c_std_11 cxx_std_17)

if(MSVC)
  target_compile_options(momo_core PRIVATE /O2 /W4)
else()
  target_compile_options(momo_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
  target_link_libraries(momo_core PRIVATE m)
endif()

add_executable(momo_native_smoke tests/smoke.cpp)
target_compile_features(momo_native_smoke PRIVATE cxx_std_17)
target_link_libraries(momo_native_smoke PRIVATE momo_core)

enable_testing()
add_test(NAME momo_native_smoke COMMAND momo_native_smoke)

install(TARGETS momo_core
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES include/momo_core.h DESTINATION include)
