# Statistical utilities test
ka_add_test(
  NAME statistical_utils_test
  LABEL core
  SOURCES statistical_utils_test.cpp
  DEPENDS kinematic_arbiter_core
)

# Add other core component tests here if needed
# For example:
# ka_add_test(
#   NAME mediated_kalman_filter_test
#   LABEL core
#   SOURCES mediated_kalman_filter_test.cpp
#   DEPENDS kinematic_arbiter_core
# )
