add_subdirectory(${PROJECT_SOURCE_DIR}/ext/Catch2 Catch2)
add_executable(pathed_test
  test.cpp

  test_distribution.cu
)
target_link_libraries(pathed_test Catch2::Catch2 rays)
