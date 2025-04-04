
.. _program_listing_file_src_single_dof_demo_ros2_nodes_signal_generator_node.py:

Program Listing for File signal_generator_node.py
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_ros2_nodes_signal_generator_node.py>` (``src/single_dof_demo/ros2/nodes/signal_generator_node.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   #!/usr/bin/env python3

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """Signal generator node for the simplified demo."""

   import rclpy
   from geometry_msgs.msg import PointStamped, Point
   from std_msgs.msg import Header
   from rcl_interfaces.msg import (
       FloatingPointRange,
       ParameterDescriptor,
       SetParametersResult,
   )
   from rclpy.callback_groups import ReentrantCallbackGroup
   from rclpy.node import Node
   from std_srvs.srv import Trigger
   from rclpy.parameter import Parameter

   from kinematic_arbiter.single_dof_demo.core.signal_generator import (
       SignalParams,
       SingleDofSignalGenerator,
   )


   class SignalGeneratorNode(Node):
       """Signal generator node for the simplified demo."""

       def __init__(self):
           """Initialize the signal generator node."""
           super().__init__("signal_generator")

           # Use ReentrantCallbackGroup to allow concurrent callbacks
           self.callback_groupcallback_group = ReentrantCallbackGroup()

           # Default parameter values
           self.default_paramsdefault_params = {
               "publishing_rate": 20.0,
               "max_frequency": 1.0,
               "max_amplitude": 1.0,
               "noise_level": 0.2,
               "number_of_signals": 10,
           }

           # Parameters
           self.declare_parameters(
               namespace="",
               parameters=[
                   (
                       "publishing_rate",
                       self.default_paramsdefault_params["publishing_rate"],
                       self._create_float_descriptor_create_float_descriptor(
                           0.1, 100.0, "Rate at which signals are published (Hz)"
                       ),
                   ),
                   (
                       "max_frequency",
                       self.default_paramsdefault_params["max_frequency"],
                       self._create_float_descriptor_create_float_descriptor(
                           0.1, 100.0, "Maximum frequency for signal components"
                       ),
                   ),
                   (
                       "max_amplitude",
                       self.default_paramsdefault_params["max_amplitude"],
                       self._create_float_descriptor_create_float_descriptor(
                           0.0, 10.0, "Maximum amplitude for signal components"
                       ),
                   ),
                   (
                       "noise_level",
                       self.default_paramsdefault_params["noise_level"],
                       self._create_float_descriptor_create_float_descriptor(
                           0.0, 5.0, "Standard deviation of Gaussian noise"
                       ),
                   ),
                   (
                       "number_of_signals",
                       self.default_paramsdefault_params["number_of_signals"],
                       ParameterDescriptor(
                           description="Number of sinusoidal components"
                       ),
                   ),
               ],
           )

           # Add parameter callback
           self.add_on_set_parameters_callback(self.parameters_callbackparameters_callback)

           # Publishers
           self.noisy_publishernoisy_publisher = self.create_publisher(
               PointStamped, "raw_measurements", 10
           )
           self.clean_publisherclean_publisher = self.create_publisher(
               PointStamped, "true_signal", 10
           )

           # Initialize signal generator
           self._init_signal_generator_init_signal_generator()

           # Initialize time
           self.initial_timeinitial_time = self.get_clock().now()

           # Create timer for signal publishing
           self.timertimer = None
           self._create_timer_create_timer()

           # Services
           self.reset_servicereset_service = self.create_service(
               Trigger,
               "~/reset_generator",
               self.handle_resethandle_reset,
               callback_group=self.callback_groupcallback_group,
           )

           self.reset_params_servicereset_params_service = self.create_service(
               Trigger,
               "~/reset_parameters",
               self.handle_reset_parametershandle_reset_parameters,
               callback_group=self.callback_groupcallback_group,
           )

           self.get_logger().info("Signal generator node initialized")

       def _create_float_descriptor(self, min_val, max_val, description):
           """Create a float parameter descriptor."""
           return ParameterDescriptor(
               floating_point_range=[
                   FloatingPointRange(from_value=min_val, to_value=max_val)
               ],
               description=description,
           )

       def parameters_callback(self, params):
           """Handle parameter changes."""
           result = SetParametersResult(successful=True)
           timer_update_needed = False

           for param in params:
               try:
                   if param.name == "publishing_rate":
                       if param.value <= 0.0:
                           raise ValueError("Publishing rate must be positive")
                       timer_update_needed = True
                       self.get_logger().info(
                           f"Updated publishing rate to {param.value}"
                       )
                   elif param.name == "max_frequency":
                       if param.value <= 0.0:
                           raise ValueError("Max frequency must be positive")
                       self.signal_paramssignal_params.max_frequency = param.value
                       self.get_logger().info(
                           f"Updated max frequency to {param.value}"
                       )
                   elif param.name == "max_amplitude":
                       if param.value < 0.0:
                           raise ValueError("Max amplitude must be non-negative")
                       self.signal_paramssignal_params.max_amplitude = param.value
                       self.get_logger().info(
                           f"Updated max amplitude to {param.value}"
                       )
                   elif param.name == "noise_level":
                       if param.value < 0.0:
                           raise ValueError("Noise level must be non-negative")
                       self.noise_levelnoise_level = param.value
                       self.get_logger().info(
                           f"Updated noise level to {param.value}"
                       )
                   elif param.name == "number_of_signals":
                       if param.value <= 0:
                           raise ValueError("Number of signals must be positive")
                       self.signal_paramssignal_params.number_of_signals = param.value
                       self.get_logger().info(
                           f"Updated number of signals to {param.value}"
                       )
               except Exception as e:
                   self.get_logger().error(
                       f"Error setting parameter {param.name}: {str(e)}"
                   )
                   result.successful = False
                   result.reason = str(e)
                   return result

           # If any parameters changed that require reinitializing the generator
           if any(
               param.name
               in ["max_frequency", "max_amplitude", "number_of_signals"]
               for param in params
           ):
               self._init_signal_generator_init_signal_generator()

           # If publishing rate changed, update the timer
           if timer_update_needed:
               self._create_timer_create_timer()

           return result

       def _init_signal_generator(self):
           """Initialize the signal generator with current parameters."""
           self.signal_paramssignal_params = SignalParams()
           self.signal_paramssignal_params.max_frequency = self.get_parameter(
               "max_frequency"
           ).value
           self.signal_paramssignal_params.max_amplitude = self.get_parameter(
               "max_amplitude"
           ).value
           self.signal_paramssignal_params.number_of_signals = self.get_parameter(
               "number_of_signals"
           ).value

           # Store noise level separately
           self.noise_levelnoise_level = self.get_parameter("noise_level").value

           # Create signal generator
           self.signal_generatorsignal_generator = SingleDofSignalGenerator(self.signal_paramssignal_params)
           self.get_logger().info("Signal generator reinitialized")

       def _create_timer(self):
           """Create or update the timer for signal publishing."""
           # Cancel existing timer if it exists
           if self.timertimer:
               self.timertimer.cancel()

           # Create new timer with current publishing rate
           publishing_rate = self.get_parameter("publishing_rate").value
           timer_period = 1.0 / publishing_rate
           self.timertimer = self.create_timer(timer_period, self.timer_callbacktimer_callback)
           self.get_logger().info(
               f"Timer updated with period {timer_period:.4f}s"
           )

       def timer_callback(self):
           """Publish clean and noisy measurements."""
           # Update current time
           current_time = self.get_clock().now() - self.initial_timeinitial_time

           # Generate signals
           clean_signal, noisy_signal = self.signal_generatorsignal_generator.generate_signal(
               current_time.nanoseconds * 1e-9, noise_level=self.noise_levelnoise_level
           )

           # Get current ROS time for message timestamp
           current_stamp = self.get_clock().now().to_msg()

           # Create and publish signal messages
           clean_msg = PointStamped(
               header=Header(stamp=current_stamp, frame_id="world"),
               point=Point(x=clean_signal, y=0.0, z=0.0),
           )
           self.clean_publisherclean_publisher.publish(clean_msg)

           noisy_msg = PointStamped(
               header=Header(stamp=current_stamp, frame_id="world"),
               point=Point(x=noisy_signal, y=0.0, z=0.0),
           )
           self.noisy_publishernoisy_publisher.publish(noisy_msg)

       def handle_reset(self, request, response):
           """Handle signal generator reset requests."""
           # Reinitialize the signal generator with a new random seed
           self._init_signal_generator_init_signal_generator()

           # Reset the signal components with a new random seed
           new_seed = self.signal_generatorsignal_generator.reset()
           self.get_logger().info(
               f"Signal generator reset with new seed: {new_seed}"
           )

           # Reset the time reference
           self.initial_timeinitial_time = self.get_clock().now()

           response.success = True
           response.message = (
               f"Signal generator reset successful with seed {new_seed}"
           )
           return response

       def handle_reset_parameters(self, request, response):
           """Reset all parameters to their default values."""
           try:
               # Set parameters directly
               parameters = []
               for name, value in self.default_paramsdefault_params.items():
                   param_type = Parameter.Type.DOUBLE
                   if name == "number_of_signals":
                       param_type = Parameter.Type.INTEGER

                   parameters.append(Parameter(name, param_type, value))

               self.set_parameters(parameters)

               for name, value in self.default_paramsdefault_params.items():
                   self.get_logger().info(f"Reset {name} to {value}")

               # Reinitialize signal generator with default values
               self._init_signal_generator_init_signal_generator()

               # Update timer with default publishing rate
               self._create_timer_create_timer()

               self.get_logger().info("All parameters reset to default values")
               response.success = True
               response.message = "Parameters reset successful"
           except Exception as e:
               self.get_logger().error(f"Error resetting parameters: {str(e)}")
               response.success = False
               response.message = f"Error: {str(e)}"

           return response


   def main(args=None):
       """Run the signal generator node."""
       rclpy.init(args=args)
       node = SignalGeneratorNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()


   if __name__ == "__main__":
       main()
