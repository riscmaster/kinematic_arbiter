
.. _program_listing_file_src_single_dof_demo_display_gui.py:

Program Listing for File display_gui.py
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_display_gui.py>` (``src/single_dof_demo/display_gui.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """Interactive visualization tool for comparing Kalman filters."""

   import matplotlib.pyplot as plt
   import numpy as np
   from core.signal_generator import SingleDofSignalGenerator, SignalParams
   from core.kalman_filter import KalmanFilter
   from core.mediated_kalman_filter import MediatedKalmanFilter, Mediation
   from matplotlib.lines import Line2D
   from matplotlib.widgets import Button, RadioButtons, Slider


   class DisplayGui(object):
       """Interactive GUI for visualizing and comparing filter performance."""

       def __init__(self):
           """Initialize the display GUI with plots and interactive controls."""
           self.fig, (self.mkf_ax, self.kf_ax) = plt.subplots(2, 1, sharex=True)
           custom_lines = [
               Line2D([0], [0], color="b"),
               Line2D([0], [0], color="g"),
               Line2D([0], [0], color="r"),
               Line2D(
                   [0],
                   [0],
                   marker="o",
                   color="w",
                   label="Scatter",
                   markerfacecolor="k",
               ),
           ]
           self.kf_ax.legend(
               custom_lines,
               ["Truth", "Measurement", "Estimate", "Mediation"],
               loc="best",
           )
           self.mkf_ax.set_title("Mediated Kalman Filter")
           self.kf_ax.set_title("Standard Kalman Filter")
           self.mkf_ax.margins(y=0)
           self.kf_ax.margins(y=0)

           plt.subplots_adjust(left=0.25, bottom=0.25)
           signal_params = SignalParams()
           signal_generator = SingleDofSignalGenerator(signal_params)
           self.signal, self.noisy_signal, self.signal_timesignal_time = (
               signal_generator.generate_complete_signal()
           )

           self.mkf_signal_estimatemkf_signal_estimate = np.zeros(len(self.signal_timesignal_time))
           self.mkf_signal_boundmkf_signal_bound = np.zeros(len(self.signal_timesignal_time))
           self.mkf_measurement_boundmkf_measurement_bound = np.zeros(len(self.signal_timesignal_time))
           self.kf_signal_estimatekf_signal_estimate = np.zeros(len(self.signal_timesignal_time))
           self.kf_signal_boundkf_signal_bound = np.zeros(len(self.signal_timesignal_time))
           self.kf_measurement_boundkf_measurement_bound = np.zeros(len(self.signal_timesignal_time))
           self.mediationmediation = np.zeros(len(self.signal_timesignal_time))

           # Initialize tuning parameters
           self.initial_mkf_tuninginitial_mkf_tuning = 1.5
           self.initial_process_noiseinitial_process_noise = 0.025
           self.initial_measurement_noiseinitial_measurement_noise = 0.05
           self.mkf_mediationmkf_mediation = Mediation.ADJUST_STATE

           # Create the filters
           self.initialize_filtersinitialize_filters()

           # Generate initial data
           self.update_dataupdate_data()

           # Plot setup for Mediated Kalman Filter
           (self.mkf_noisy_plot_val,) = self.mkf_ax.plot(
               self.signal_timesignal_time, self.noisy_signal, "g", alpha=0.5
           )
           (self.mkf_noisy_plot_up,) = self.mkf_ax.plot(
               self.signal_timesignal_time,
               self.noisy_signal + self.mkf_measurement_boundmkf_measurement_bound,
               "g--",
               alpha=0.5,
               lw=0.5,
           )
           (self.mkf_noisy_plot_low,) = self.mkf_ax.plot(
               self.signal_timesignal_time,
               self.noisy_signal - self.mkf_measurement_boundmkf_measurement_bound,
               "g--",
               alpha=0.5,
               lw=0.5,
           )
           (self.mkf_est_plot_val,) = self.mkf_ax.plot(
               self.signal_timesignal_time, self.mkf_signal_estimatemkf_signal_estimate, "r"
           )
           (self.mediation_plot,) = self.mkf_ax.plot(
               self.signal_timesignal_time, self.mediationmediation, "ko"
           )
           (self.mkf_est_plot_up,) = self.mkf_ax.plot(
               self.signal_timesignal_time,
               self.mkf_signal_estimatemkf_signal_estimate + self.mkf_signal_boundmkf_signal_bound,
               "r--",
               alpha=0.5,
               lw=0.5,
           )
           (self.mkf_est_plot_low,) = self.mkf_ax.plot(
               self.signal_timesignal_time,
               self.mkf_signal_estimatemkf_signal_estimate - self.mkf_signal_boundmkf_signal_bound,
               "r--",
               alpha=0.5,
               lw=0.5,
           )
           (self.mkf_plot_true_val,) = self.mkf_ax.plot(
               self.signal_timesignal_time, self.signal, "b"
           )

           # Plot setup for Standard Kalman Filter
           (self.kf_noisy_plot_val,) = self.kf_ax.plot(
               self.signal_timesignal_time, self.noisy_signal, "g", alpha=0.5
           )
           (self.kf_noisy_plot_up,) = self.kf_ax.plot(
               self.signal_timesignal_time,
               self.noisy_signal + self.kf_measurement_boundkf_measurement_bound,
               "g--",
               alpha=0.5,
               lw=0.5,
           )
           (self.kf_noisy_plot_low,) = self.kf_ax.plot(
               self.signal_timesignal_time,
               self.noisy_signal - self.kf_measurement_boundkf_measurement_bound,
               "g--",
               alpha=0.5,
               lw=0.5,
           )
           (self.kf_est_plot_val,) = self.kf_ax.plot(
               self.signal_timesignal_time, self.kf_signal_estimatekf_signal_estimate, "r"
           )
           (self.kf_est_plot_up,) = self.kf_ax.plot(
               self.signal_timesignal_time,
               self.kf_signal_estimatekf_signal_estimate + self.kf_signal_boundkf_signal_bound,
               "r--",
               alpha=0.5,
               lw=0.5,
           )
           (self.kf_est_plot_low,) = self.kf_ax.plot(
               self.signal_timesignal_time,
               self.kf_signal_estimatekf_signal_estimate - self.kf_signal_boundkf_signal_bound,
               "r--",
               alpha=0.5,
               lw=0.5,
           )
           (self.kf_plot_true_val,) = self.kf_ax.plot(
               self.signal_timesignal_time, self.signal, "b"
           )

           # MKF Tuner
           axcolor = "lightgoldenrodyellow"
           ax_mkf_tuner = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
           self.mkf_tunermkf_tuner = Slider(
               ax_mkf_tuner,
               "MKF PM Ratio",
               0.0,
               1.0,
               valinit=np.log10((self.initial_mkf_tuninginitial_mkf_tuning + 1.0)),
               valstep=0.01,
           )
           self.mkf_tunermkf_tuner.on_changed(self.update_mkf_plotupdate_mkf_plot)

           # KF Tuner
           self.kf_process_tunerkf_process_tuner = Slider(
               plt.axes(
                   [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
               ),
               "KF Process",
               0.0,
               0.1,
               valinit=np.log10(self.initial_process_noiseinitial_process_noise + 1.0),
               valstep=0.001,
           )
           self.kf_process_tunerkf_process_tuner.on_changed(self.update_kf_plotupdate_kf_plot)

           self.kf_measurement_tunerkf_measurement_tuner = Slider(
               plt.axes(
                   [0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow"
               ),
               "KF Measurement",
               0.0,
               1.0,
               valinit=np.log10(self.initial_measurement_noiseinitial_measurement_noise + 1.0),
               valstep=0.01,
           )
           self.kf_measurement_tunerkf_measurement_tuner.on_changed(self.update_kf_plotupdate_kf_plot)

           # Reset button
           resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
           self.buttonbutton = Button(
               resetax, "Reset", color=axcolor, hovercolor="0.975"
           )
           self.buttonbutton.on_clicked(self.resetreset)

           # Radio buttons for mediation method
           rax = plt.axes([0.015, 0.65, 0.185, 0.15], facecolor=axcolor)
           self.radioradio = RadioButtons(
               rax, ("State", "Meas", "Reject Meas", "Nothing"), active=0
           )
           self.radioradio.on_clicked(self.update_mkf_methodupdate_mkf_method)

       def initialize_filters(self):
           """Initialize both filters with current parameters."""
           # Create MKF filter
           self.mkfmkf = MediatedKalmanFilter(
               process_to_measurement_ratio=self.initial_mkf_tuninginitial_mkf_tuning,
               sample_window=40,
               mediation=self.mkf_mediationmkf_mediation,
           )
           self.mkfmkf.process_variance = self.initial_process_noiseinitial_process_noise
           self.mkfmkf.measurement_variance = self.initial_measurement_noiseinitial_measurement_noise
           self.mkfmkf.state_variance = 0.1

           # Create standard KF filter
           self.kfkf = KalmanFilter(
               process_noise=self.initial_process_noiseinitial_process_noise,
               measurement_noise=self.initial_measurement_noiseinitial_measurement_noise,
           )
           self.kfkf.process_variance = self.initial_process_noiseinitial_process_noise
           self.kfkf.measurement_variance = self.initial_measurement_noiseinitial_measurement_noise
           self.kfkf.state_variance = 0.1

       def update_data(self):
           """Update all data using current filter settings."""
           # Reinitialize the filters instead of trying to reset them
           self.initialize_filtersinitialize_filters()

           # Process all data points
           for i, (t, s) in enumerate(zip(self.signal_timesignal_time, self.noisy_signal)):
               # Process through MKF
               mkf_output = self.mkfmkf.update(measurement=s, t=t)
               self.mkf_signal_estimatemkf_signal_estimate[i] = mkf_output.final.state.value
               self.mkf_signal_boundmkf_signal_bound[i] = 3.0 * np.sqrt(self.mkfmkf.state_variance)
               self.mkf_measurement_boundmkf_measurement_bound[i] = 3.0 * np.sqrt(
                   self.mkfmkf.measurement_variance
               )

               # Process through standard KF
               kf_output = self.kfkf.update(measurement=s)
               self.kf_signal_estimatekf_signal_estimate[i] = kf_output.final.state.value
               self.kf_signal_boundkf_signal_bound[i] = 3.0 * np.sqrt(self.kfkf.state_variance)
               self.kf_measurement_boundkf_measurement_bound[i] = 3.0 * np.sqrt(
                   self.kfkf.measurement_variance
               )

               # Store mediation points
               self.mediationmediation[i] = (
                   self.mkf_signal_estimatemkf_signal_estimate[i]
                   if self.mkfmkf.mediation
                   else float("NaN")
               )

       def update_mkf_method(self, label):
           """Update the mediation method based on radio button selection."""
           if label == "State":
               self.mkf_mediationmkf_mediation = Mediation.ADJUST_STATE
           elif label == "Meas":
               self.mkf_mediationmkf_mediation = Mediation.ADJUST_MEASUREMENT
           elif label == "Reject Meas":
               self.mkf_mediationmkf_mediation = Mediation.REJECT_MEASUREMENT
           elif label == "Nothing":
               self.mkf_mediationmkf_mediation = Mediation.NO_ACTION
           self.update_mkf_plotupdate_mkf_plot(None)

       def update_mkf_plot(self, _):
           """Update the mediated Kalman filter plot based on slider values."""
           # Convert slider value to actual ratio
           ratio_value = -1.0 + 10**self.mkf_tunermkf_tuner.val
           self.mkf_tunermkf_tuner.valtext.set_text(ratio_value)

           # Re-initialize MKF with new parameters
           self.mkfmkf = MediatedKalmanFilter(
               process_to_measurement_ratio=ratio_value,
               sample_window=40,
               mediation=self.mkf_mediationmkf_mediation,
           )
           self.mkfmkf.process_variance = self.initial_process_noiseinitial_process_noise
           self.mkfmkf.measurement_variance = self.initial_measurement_noiseinitial_measurement_noise

           # Process all data through the filter
           for i, (t, s) in enumerate(zip(self.signal_timesignal_time, self.noisy_signal)):
               mkf_output = self.mkfmkf.update(measurement=s, t=t)
               self.mkf_signal_estimatemkf_signal_estimate[i] = mkf_output.final.state.value
               self.mkf_signal_boundmkf_signal_bound[i] = 3.0 * np.sqrt(self.mkfmkf.state_variance)
               self.mkf_measurement_boundmkf_measurement_bound[i] = 3.0 * np.sqrt(
                   self.mkfmkf.measurement_variance
               )
               self.mediationmediation[i] = (
                   self.mkf_signal_estimatemkf_signal_estimate[i]
                   if self.mkfmkf.mediation
                   else float("NaN")
               )

           # Update plot data
           self.mkf_noisy_plot_val.set_ydata(self.noisy_signal)
           self.mkf_noisy_plot_up.set_ydata(
               self.noisy_signal + self.mkf_measurement_boundmkf_measurement_bound
           )
           self.mkf_noisy_plot_low.set_ydata(
               self.noisy_signal - self.mkf_measurement_boundmkf_measurement_bound
           )
           self.mkf_est_plot_val.set_ydata(self.mkf_signal_estimatemkf_signal_estimate)
           self.mediation_plot.set_ydata(self.mediationmediation)
           self.mkf_est_plot_up.set_ydata(
               self.mkf_signal_estimatemkf_signal_estimate + self.mkf_signal_boundmkf_signal_bound
           )
           self.mkf_est_plot_low.set_ydata(
               self.mkf_signal_estimatemkf_signal_estimate - self.mkf_signal_boundmkf_signal_bound
           )
           self.mkf_plot_true_val.set_ydata(self.signal)
           self.fig.canvas.draw_idle()

       def update_kf_plot(self, _):
           """Update the standard Kalman filter plot based on slider values."""
           # Convert slider values to actual noise values
           process_noise = -1.0 + 10**self.kf_process_tunerkf_process_tuner.val
           self.kf_process_tunerkf_process_tuner.valtext.set_text(process_noise)
           measurement_noise = -1.0 + 10**self.kf_measurement_tunerkf_measurement_tuner.val
           self.kf_measurement_tunerkf_measurement_tuner.valtext.set_text(measurement_noise)

           # Re-initialize KF with new parameters
           self.kfkf = KalmanFilter(
               process_noise=process_noise, measurement_noise=measurement_noise
           )

           # Process all data through the filter
           for i, s in enumerate(self.noisy_signal):
               kf_output = self.kfkf.update(measurement=s)
               self.kf_signal_estimatekf_signal_estimate[i] = kf_output.final.state.value
               self.kf_signal_boundkf_signal_bound[i] = 3.0 * np.sqrt(self.kfkf.state_variance)
               self.kf_measurement_boundkf_measurement_bound[i] = 3.0 * np.sqrt(
                   self.kfkf.measurement_variance
               )

           # Update plot data
           self.kf_noisy_plot_val.set_ydata(self.noisy_signal)
           self.kf_noisy_plot_up.set_ydata(
               self.noisy_signal + self.kf_measurement_boundkf_measurement_bound
           )
           self.kf_noisy_plot_low.set_ydata(
               self.noisy_signal - self.kf_measurement_boundkf_measurement_bound
           )
           self.kf_est_plot_val.set_ydata(self.kf_signal_estimatekf_signal_estimate)
           self.kf_est_plot_up.set_ydata(
               self.kf_signal_estimatekf_signal_estimate + self.kf_signal_boundkf_signal_bound
           )
           self.kf_est_plot_low.set_ydata(
               self.kf_signal_estimatekf_signal_estimate - self.kf_signal_boundkf_signal_bound
           )
           self.kf_plot_true_val.set_ydata(self.signal)
           self.fig.canvas.draw_idle()

       def reset(self, _):
           """Reset all sliders to their initial values."""
           self.mkf_tunermkf_tuner.reset()
           self.kf_measurement_tunerkf_measurement_tuner.reset()
           self.kf_process_tunerkf_process_tuner.reset()
           # Reset filters to default states
           self.initialize_filtersinitialize_filters()
           self.update_dataupdate_data()
           # Update plots
           self.update_mkf_plotupdate_mkf_plot(None)
           self.update_kf_plotupdate_kf_plot(None)

       @staticmethod
       def run():
           """Run the display GUI by showing the plot window."""
           plt.show()
