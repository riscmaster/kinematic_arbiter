# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Interactive visualization tool for comparing Kalman filters."""

import matplotlib.pyplot as plt
import numpy as np
from src.single_dof_demo.core.signal_generator import (
    SignalParams,
    generate_signals,
)
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
        params = SignalParams()
        self.signal, self.noisy_signal, self.signal_time = generate_signals(
            params=params
        )

        self.mkf_signal_estimate = np.zeros(len(self.signal_time))
        self.mkf_signal_bound = np.zeros(len(self.signal_time))
        self.mkf_measurement_bound = np.zeros(len(self.signal_time))
        self.kf_signal_estimate = np.zeros(len(self.signal_time))
        self.kf_signal_bound = np.zeros(len(self.signal_time))
        self.kf_measurement_bound = np.zeros(len(self.signal_time))
        self.mediation = np.zeros(len(self.signal_time))
        initial_mkf_tuning = 0.25
        initial_process_noise = 0.025
        initial_measurement_noise = 0.05
        self.mkf_mediation = Mediation.ADJUST_STATE
        mkf = MediatedKalmanFilter(
            process_to_measurement_ratio=initial_mkf_tuning,
            window_time=0.01,
            mediation=self.mkf_mediation,
        )
        kf = KalmanFilter(
            process_noise=initial_process_noise,
            measurement_noise=initial_measurement_noise,
        )
        for i, (t, s) in enumerate(zip(self.signal_time, self.noisy_signal)):
            self.mkf_signal_estimate[i] = mkf.update(
                measurement=s, t=t
            ).final.state.value
            self.mkf_signal_bound[i] = 3.0 * np.sqrt(mkf.state_variance)
            self.mkf_measurement_bound[i] = 3.0 * np.sqrt(
                mkf.measurement_variance
            )
            self.kf_signal_estimate[i] = kf.update(
                measurement=s
            ).final.state.value
            self.kf_signal_bound[i] = 3.0 * np.sqrt(kf.state_variance)
            self.kf_measurement_bound[i] = 3.0 * np.sqrt(
                kf.measurement_variance
            )
            self.mediation[i] = (
                self.mkf_signal_estimate[i] if mkf.mediation else float("NaN")
            )

        (self.mkf_noisy_plot_val,) = self.mkf_ax.plot(
            self.signal_time, self.noisy_signal, "g", alpha=0.5
        )
        (self.mkf_noisy_plot_up,) = self.mkf_ax.plot(
            self.signal_time,
            self.noisy_signal + self.mkf_measurement_bound,
            "g--",
            alpha=0.5,
            lw=0.5,
        )
        (self.mkf_noisy_plot_low,) = self.mkf_ax.plot(
            self.signal_time,
            self.noisy_signal - self.mkf_measurement_bound,
            "g--",
            alpha=0.5,
            lw=0.5,
        )
        (self.mkf_est_plot_val,) = self.mkf_ax.plot(
            self.signal_time, self.mkf_signal_estimate, "r"
        )
        (self.mediation_plot,) = self.mkf_ax.plot(
            self.signal_time, self.mediation, "ko"
        )
        (self.mkf_est_plot_up,) = self.mkf_ax.plot(
            self.signal_time,
            self.mkf_signal_estimate + self.mkf_signal_bound,
            "r--",
            alpha=0.5,
            lw=0.5,
        )
        (self.mkf_est_plot_low,) = self.mkf_ax.plot(
            self.signal_time,
            self.mkf_signal_estimate - self.mkf_signal_bound,
            "r--",
            alpha=0.5,
            lw=0.5,
        )
        (self.mkf_plot_true_val,) = self.mkf_ax.plot(
            self.signal_time, self.signal, "b"
        )

        (self.kf_noisy_plot_val,) = self.kf_ax.plot(
            self.signal_time, self.noisy_signal, "g", alpha=0.5
        )
        (self.kf_noisy_plot_up,) = self.kf_ax.plot(
            self.signal_time,
            self.noisy_signal + self.kf_measurement_bound,
            "g--",
            alpha=0.5,
            lw=0.5,
        )
        (self.kf_noisy_plot_low,) = self.kf_ax.plot(
            self.signal_time,
            self.noisy_signal - self.kf_measurement_bound,
            "g--",
            alpha=0.5,
            lw=0.5,
        )
        (self.kf_est_plot_val,) = self.kf_ax.plot(
            self.signal_time, self.kf_signal_estimate, "r"
        )
        (self.kf_est_plot_up,) = self.kf_ax.plot(
            self.signal_time,
            self.kf_signal_estimate + self.kf_signal_bound,
            "r--",
            alpha=0.5,
            lw=0.5,
        )
        (self.kf_est_plot_low,) = self.kf_ax.plot(
            self.signal_time,
            self.kf_signal_estimate - self.kf_signal_bound,
            "r--",
            alpha=0.5,
            lw=0.5,
        )
        (self.kf_plot_true_val,) = self.kf_ax.plot(
            self.signal_time, self.signal, "b"
        )

        # MKF Tuner
        axcolor = "lightgoldenrodyellow"
        ax_mkf_tuner = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
        self.mkf_tuner = Slider(
            ax_mkf_tuner,
            "MKF PM Ratio",
            0.0,
            1.0,
            valinit=np.log10((initial_mkf_tuning + 1.0)),
            valstep=1e-4,
        )
        self.mkf_tuner.on_changed(self.update_mkf_plot)

        # KF Tuners
        self.kf_process_tuner = Slider(
            plt.axes(
                [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
            ),
            "KF Process",
            0.0,
            0.1,
            valinit=np.log10(initial_process_noise + 1.0),
            valstep=1e-4,
        )
        self.kf_process_tuner.on_changed(self.update_kf_plot)

        self.kf_measurement_tuner = Slider(
            plt.axes(
                [0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow"
            ),
            "KF Measurement",
            0.0,
            1.0,
            valinit=np.log10(initial_measurement_noise + 1.0),
            valstep=1e-4,
        )
        self.kf_measurement_tuner.on_changed(self.update_kf_plot)

        # Reset Button
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(
            resetax, "Reset", color=axcolor, hovercolor="0.975"
        )
        self.button.on_clicked(self.reset)

        rax = plt.axes([0.015, 0.65, 0.185, 0.15], facecolor=axcolor)
        self.radio = RadioButtons(
            rax, ("State", "Meas", "Reject Meas", "Nothing"), active=0
        )

        self.radio.on_clicked(self.update_mkf_method)

    def update_mkf_method(self, label):
        """Update the mediation method based on radio button selection."""
        if label == "State":
            self.mkf_mediation = Mediation.ADJUST_STATE
        elif label == "Meas":
            self.mkf_mediation = Mediation.ADJUST_MEASUREMENT
        elif label == "Reject Meas":
            self.mkf_mediation = Mediation.REJECT_MEASUREMENT
        elif label == "Nothing":
            self.mkf_mediation = Mediation.NO_ACTION
        self.update_mkf_plot(None)

    def update_mkf_plot(self, _):
        """Update the mediated Kalman filter plot based on slider values."""
        bar_value = -(self.mkf_tuner.val - 1)
        pm_ratio = 100.0 if (bar_value == 0) else -np.log10(bar_value)
        self.mkf_tuner.valtext.set_text(pm_ratio)
        mkf = MediatedKalmanFilter(
            process_to_measurement_ratio=pm_ratio,
            window_time=0.01,
            mediation=self.mkf_mediation,
        )
        for i, (t, s) in enumerate(zip(self.signal_time, self.noisy_signal)):
            self.mkf_signal_estimate[i] = mkf.update(
                measurement=s, t=t
            ).final.state.value
            self.mkf_signal_bound[i] = 3.0 * np.sqrt(mkf.state_variance)
            self.mkf_measurement_bound[i] = 3.0 * np.sqrt(
                mkf.measurement_variance
            )
            self.mediation[i] = (
                self.mkf_signal_estimate[i] if mkf.mediation else float("NaN")
            )

        self.mkf_noisy_plot_val.set_ydata(self.noisy_signal)
        self.mkf_noisy_plot_up.set_ydata(
            self.noisy_signal + self.mkf_measurement_bound
        )
        self.mkf_noisy_plot_low.set_ydata(
            self.noisy_signal - self.mkf_measurement_bound
        )
        self.mkf_est_plot_val.set_ydata(self.mkf_signal_estimate)
        self.mediation_plot.set_ydata(self.mediation)
        self.mkf_est_plot_up.set_ydata(
            self.mkf_signal_estimate + self.mkf_signal_bound
        )
        self.mkf_est_plot_low.set_ydata(
            self.mkf_signal_estimate - self.mkf_signal_bound
        )
        self.mkf_plot_true_val.set_ydata(self.signal)
        self.fig.canvas.draw_idle()

    def update_kf_plot(self, _):
        """Update the standard Kalman filter plot based on slider values."""
        process_noise = -1.0 + 10**self.kf_process_tuner.val
        self.kf_process_tuner.valtext.set_text(process_noise)
        measurement_noise = -1.0 + 10**self.kf_measurement_tuner.val
        self.kf_measurement_tuner.valtext.set_text(measurement_noise)
        kf = KalmanFilter(
            process_noise=process_noise, measurement_noise=measurement_noise
        )
        for i, s in enumerate(self.noisy_signal):
            self.kf_signal_estimate[i] = kf.update(
                measurement=s
            ).final.state.value
            self.kf_signal_bound[i] = 3.0 * np.sqrt(kf.state_variance)
            self.kf_measurement_bound[i] = 3.0 * np.sqrt(
                kf.measurement_variance
            )

        self.kf_noisy_plot_val.set_ydata(self.noisy_signal)
        self.kf_noisy_plot_up.set_ydata(
            self.noisy_signal + self.kf_measurement_bound
        )
        self.kf_noisy_plot_low.set_ydata(
            self.noisy_signal - self.kf_measurement_bound
        )
        self.kf_est_plot_val.set_ydata(self.kf_signal_estimate)
        self.kf_est_plot_up.set_ydata(
            self.kf_signal_estimate + self.kf_signal_bound
        )
        self.kf_est_plot_low.set_ydata(
            self.kf_signal_estimate - self.kf_signal_bound
        )
        self.kf_plot_true_val.set_ydata(self.signal)
        self.fig.canvas.draw_idle()

    def reset(self, _):
        """Reset all sliders to their initial values."""
        self.mkf_tuner.reset()
        self.kf_measurement_tuner.reset()
        self.kf_process_tuner.reset()

    @staticmethod
    def run():
        """Run the display GUI by showing the plot window."""
        plt.show()
