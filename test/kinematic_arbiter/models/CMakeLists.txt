# Rigid body state model test
ka_add_test(
  NAME rigid_body_state_model_test
  LABEL models
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../models/rigid_body_state_model_test.cpp
  DEPENDS kinematic_arbiter_core drake::drake
)
