# Add kinematic arbiter test subdirectories
add_subdirectory(core)
add_subdirectory(models)

# Only add the model test if not already defined
if(NOT TARGET rigid_body_state_model_test)
  ka_add_test(
    NAME rigid_body_state_model_test
    LABEL models
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../models/rigid_body_state_model_test.cpp
    DEPENDS kinematic_arbiter_core drake::drake
  )
endif()

# You can add other test subdirectories here as needed, for example:
# add_subdirectory(filters)
# add_subdirectory(io)
# add_subdirectory(visualization)
