add_executable(clustering_tests
  algorithm_tests.cpp
  distance_tests.cpp
  initializer_tests.cpp
  input_output_tests.cpp
  linear_algebra_tests.cpp
  main.cpp
  pipeline_tests.cpp
  quality_metric_tests.cpp
)
target_link_libraries(clustering_tests PRIVATE clustering_lib)
target_include_directories(clustering_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(clustering_tests PRIVATE
  "CLUSTERING_DATA_DIR=\"${PROJECT_SOURCE_DIR}/data\"")
clustering_configure_target(clustering_tests)

add_test(NAME clustering_tests COMMAND clustering_tests)
