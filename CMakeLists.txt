cmake_minimum_required(VERSION 3.20)
project(clustering VERSION 1.0.0 LANGUAGES CXX)

set(CLUSTERING_CXX_STANDARD 23 CACHE STRING "C++ standard used by the project")
set_property(CACHE CLUSTERING_CXX_STANDARD PROPERTY STRINGS 23 26)
option(CLUSTERING_BUILD_TESTS "Build the unit tests" ON)
option(CLUSTERING_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

function(clustering_configure_target target)
  target_compile_features(${target} PUBLIC cxx_std_${CLUSTERING_CXX_STANDARD})
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(CLUSTERING_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    if(CLUSTERING_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

add_library(clustering_lib STATIC
  src/algorithms/affinity_propagation.cpp
  src/algorithms/clusterer.cpp
  src/algorithms/dbscan.cpp
  src/algorithms/gaussian_mixture.cpp
  src/algorithms/kmeans.cpp
  src/algorithms/nearest_center.cpp
  src/app/clusterer_factory.cpp
  src/app/clustering_problem.cpp
  src/app/command_line.cpp
  src/app/pipeline.cpp
  src/app/report_formatter.cpp
  src/core/cholesky_factor.cpp
  src/core/linear_algebra.cpp
  src/core/matrix.cpp
  src/distance/euclidean_distance.cpp
  src/distance/mahalanobis_distance.cpp
  src/distance/manhattan_distance.cpp
  src/distance/metric_factory.cpp
  src/evaluation/adjusted_mutual_information.cpp
  src/evaluation/adjusted_rand_index.cpp
  src/evaluation/cluster_indexing.cpp
  src/evaluation/contingency_table.cpp
  src/evaluation/davies_bouldin_index.cpp
  src/evaluation/quality_evaluator.cpp
  src/evaluation/silhouette_coefficient.cpp
  src/initialization/forgy_initializer.cpp
  src/initialization/initializer_factory.cpp
  src/initialization/kmeans_plus_plus_initializer.cpp
  src/initialization/seeded_initializer.cpp
  src/initialization/training_labels.cpp
  src/initialization/training_split.cpp
  src/io/clustered_csv_writer.cpp
  src/io/csv_table.cpp
  src/io/dataset_loader.cpp
  src/io/label_encoder.cpp
  src/preprocessing/standardizer.cpp
)
target_include_directories(clustering_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
clustering_configure_target(clustering_lib)

add_executable(clustering_cli src/main.cpp)
target_link_libraries(clustering_cli PRIVATE clustering_lib)
clustering_configure_target(clustering_cli)

install(TARGETS clustering_cli RUNTIME DESTINATION bin)

if(CLUSTERING_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
