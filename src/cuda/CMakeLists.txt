if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES native)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

  add_executable(cuda-function-qualifiers function-qualifiers/main.cu)
  target_link_libraries(cuda-function-qualifiers GTest::gtest_main)
  gtest_discover_tests(cuda-function-qualifiers)

  add_executable(cuda-hello hello/main.cu)
  target_link_libraries(cuda-hello GTest::gtest_main)
  gtest_discover_tests(cuda-hello)

  add_executable(cuda-vector-add vector-add/main.cu)
  target_link_libraries(cuda-vector-add GTest::gtest_main)
  gtest_discover_tests(cuda-vector-add)

  add_executable(cuda-matrix-mul matrix-mul/main.cu)
  target_link_libraries(cuda-matrix-mul GTest::gtest_main)
  gtest_discover_tests(cuda-matrix-mul)

  add_executable(cuda-reduction reduction/main.cu)
  target_link_libraries(cuda-reduction GTest::gtest_main)
  gtest_discover_tests(cuda-reduction)

  add_executable(cuda-shared-memory shared-memory/main.cu)
  target_link_libraries(cuda-shared-memory GTest::gtest_main)
  gtest_discover_tests(cuda-shared-memory)

  add_executable(cuda-streams streams/main.cu)
  target_link_libraries(cuda-streams GTest::gtest_main)
  gtest_discover_tests(cuda-streams)

  add_executable(cuda-memory-benchmark memory-benchmark/main.cu)
  target_link_libraries(cuda-memory-benchmark GTest::gtest_main)
  gtest_discover_tests(cuda-memory-benchmark)

  add_executable(cuda-coop-groups-basic coop-groups-basic/main.cu)
  target_link_libraries(cuda-coop-groups-basic GTest::gtest_main)
  gtest_discover_tests(cuda-coop-groups-basic)

  add_executable(cuda-coop-groups-reduce coop-groups-reduce/main.cu)
  target_link_libraries(cuda-coop-groups-reduce GTest::gtest_main)
  gtest_discover_tests(cuda-coop-groups-reduce)

  add_executable(cuda-coop-groups-tile coop-groups-tile/main.cu)
  target_link_libraries(cuda-coop-groups-tile GTest::gtest_main)
  gtest_discover_tests(cuda-coop-groups-tile)

  add_executable(cuda-coop-groups-coalesced coop-groups-coalesced/main.cu)
  target_link_libraries(cuda-coop-groups-coalesced GTest::gtest_main)
  gtest_discover_tests(cuda-coop-groups-coalesced)

  add_executable(cuda-coop-groups-ptx coop-groups-ptx/main.cu)
  target_link_libraries(cuda-coop-groups-ptx GTest::gtest_main)
  gtest_discover_tests(cuda-coop-groups-ptx)

  add_executable(cuda-coop-groups-cheatsheet coop-groups-cheatsheet/main.cu)
  target_link_libraries(cuda-coop-groups-cheatsheet GTest::gtest_main)
  gtest_discover_tests(cuda-coop-groups-cheatsheet)

  add_executable(cuda-memory-visibility-primitives memory-visibility-primitives/main.cu)
  target_link_libraries(cuda-memory-visibility-primitives GTest::gtest_main)
  gtest_discover_tests(cuda-memory-visibility-primitives)

  add_executable(cuda-memory-visibility-libcu memory-visibility-libcu/main.cu)
  target_link_libraries(cuda-memory-visibility-libcu GTest::gtest_main)
  gtest_discover_tests(cuda-memory-visibility-libcu)

  add_executable(cuda-memory-visibility-ptx memory-visibility-ptx/main.cu)
  target_link_libraries(cuda-memory-visibility-ptx GTest::gtest_main)
  gtest_discover_tests(cuda-memory-visibility-ptx)

  add_executable(cuda-memory-visibility-patterns memory-visibility-patterns/main.cu)
  target_link_libraries(cuda-memory-visibility-patterns GTest::gtest_main)
  gtest_discover_tests(cuda-memory-visibility-patterns)

  add_executable(cuda-memory-visibility-barrier memory-visibility-barrier/main.cu)
  target_link_libraries(cuda-memory-visibility-barrier GTest::gtest_main)
  gtest_discover_tests(cuda-memory-visibility-barrier)

  add_executable(cuda-pipeline pipeline/main.cu)
  target_link_libraries(cuda-pipeline GTest::gtest_main)
  gtest_discover_tests(cuda-pipeline)

  add_executable(cuda-p2p-nvlink p2p-nvlink/main.cu)
  target_link_libraries(cuda-p2p-nvlink GTest::gtest_main)
  gtest_discover_tests(cuda-p2p-nvlink)

  add_executable(cuda-p2p-ipc p2p-ipc/main.cu)
  target_link_libraries(cuda-p2p-ipc GTest::gtest_main)
  gtest_discover_tests(cuda-p2p-ipc)

  add_executable(cuda-vmm vmm/main.cu)
  target_link_libraries(cuda-vmm GTest::gtest_main cuda)
  gtest_discover_tests(cuda-vmm)

  add_executable(cuda-libcuxx-atomic libcuxx-atomic/main.cu)
  target_link_libraries(cuda-libcuxx-atomic GTest::gtest_main)
  gtest_discover_tests(cuda-libcuxx-atomic)

  add_executable(cuda-libcuxx-barrier libcuxx-barrier/main.cu)
  target_link_libraries(cuda-libcuxx-barrier GTest::gtest_main)
  gtest_discover_tests(cuda-libcuxx-barrier)

  add_executable(cuda-libcuxx-semaphore libcuxx-semaphore/main.cu)
  target_link_libraries(cuda-libcuxx-semaphore GTest::gtest_main)
  gtest_discover_tests(cuda-libcuxx-semaphore)

  add_executable(cuda-libcuxx-latch libcuxx-latch/main.cu)
  target_link_libraries(cuda-libcuxx-latch GTest::gtest_main)
  gtest_discover_tests(cuda-libcuxx-latch)
endif()
