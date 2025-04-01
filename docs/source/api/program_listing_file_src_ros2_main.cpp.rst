
.. _program_listing_file_src_ros2_main.cpp:

Program Listing for File main.cpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_ros2_main.cpp>` (``src/ros2/main.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #include "rclcpp/rclcpp.hpp"
   #include "kinematic_arbiter/ros2/kinematic_arbiter_node.hpp"

   int main(int argc, char * argv[])
   {
     // Initialize ROS
     rclcpp::init(argc, argv);

     // Create node
     auto node = std::make_shared<kinematic_arbiter::ros2::KinematicArbiterNode>();

     // Spin the node
     rclcpp::spin(node);

     // Shutdown
     rclcpp::shutdown();

     return 0;
   }
