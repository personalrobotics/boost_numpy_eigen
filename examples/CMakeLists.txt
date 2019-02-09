add_library(hello SHARED EXCLUDE_FROM_ALL hello.cc hello_mod.cc)
target_link_libraries(hello boost_numpy_eigen)
set_target_properties(hello PROPERTIES
  SUFFIX ".so"  # python uses '.so' extension even on macOS
)
add_custom_command(TARGET hello
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy libhello*.so ${CMAKE_SOURCE_DIR}
)
