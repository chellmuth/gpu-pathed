add_subdirectory(${PROJECT_SOURCE_DIR}/ext/Catch2 Catch2)
add_executable(pathed_test
  test.cpp

  test_distribution.cu
  test_ply_parser.cpp
)
target_link_libraries(pathed_test Catch2::Catch2 rays)
