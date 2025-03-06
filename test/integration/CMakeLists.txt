# Drake integration test
ka_add_test(
  NAME test_drake_integration
  LABEL integration
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../test_drake_integration.cpp
  DEPENDS kinematic_arbiter_core drake::drake
)
