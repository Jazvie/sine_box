import sys
import numpy as np
import simpleaudio as sa
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSlider, QLabel, QComboBox, QPushButton)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
from sequential_time_generator import SequentialTimeGenerator
from formant_generator import vowel_formants

class SequentialTimeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequential Time-Domain Formant Generator")
        self.setGeometry(100, 100, 1200, 800)

        # Audio settings
        self.Fs = 16000  # Hz (fixed)
        self.display_duration = 0.05  # 50ms for display (show more detail)
        self.display_samples = int(self.Fs * self.display_duration)
        self.audio_buffer_size = 1024  # Smaller buffer for more responsive updates
        self.play_obj = None
        self.is_playing = False
        self.last_samples = np.zeros(self.display_samples)  # For smoother display updates
        
        # Initialize the generator
        self.generator = SequentialTimeGenerator(fs=self.Fs)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setYRange(-1.2, 1.2)
        self.plot_widget.setXRange(0, self.display_duration)
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setLabel('left', 'Amplitude')
        layout.addWidget(self.plot_widget)

        # Create parameter sliders
        self.create_parameter_sliders(layout)

        # Create vowel selector and play controls
        control_layout = QHBoxLayout()
        
        # Vowel selector
        vowel_label = QLabel("Vowel:")
        self.vowel_selector = QComboBox()
        self.vowel_selector.addItems(list(vowel_formants.keys()))
        control_layout.addWidget(vowel_label)
        control_layout.addWidget(self.vowel_selector)
        
        # Play/Stop button
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_audio)
        control_layout.addWidget(self.play_button)
        
        layout.addLayout(control_layout)

        # Set up the plot line
        pen = pg.mkPen(color='b', width=2)
        self.plot_line = self.plot_widget.plot([], [], pen=pen)
        
        # Connect controls to update function
        self.vowel_selector.currentTextChanged.connect(self.update_parameters)
        
        # Set up timers
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(50)  # 20 Hz update rate for display

        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.update_audio)
        self.audio_timer.start(20)  # 50 Hz update rate for audio

    def create_parameter_sliders(self, layout):
        # Display fixed sample rate
        fs_layout = QHBoxLayout()
        fs_label = QLabel("Sampling Frequency (Hz):")
        fs_value = QLabel(f"{self.Fs} (fixed)")
        fs_layout.addWidget(fs_label)
        fs_layout.addWidget(fs_value)
        layout.addLayout(fs_layout)

        # Fundamental frequency (f0)
        f0_layout = QHBoxLayout()
        f0_label = QLabel("Fundamental Frequency (Hz):")
        self.f0_slider = QSlider(Qt.Orientation.Horizontal)
        self.f0_slider.setMinimum(50)
        self.f0_slider.setMaximum(400)
        self.f0_slider.setValue(155)
        self.f0_value = QLabel("155")
        f0_layout.addWidget(f0_label)
        f0_layout.addWidget(self.f0_slider)
        f0_layout.addWidget(self.f0_value)
        layout.addLayout(f0_layout)
        self.f0_slider.valueChanged.connect(self.update_parameters)

        # Wave morph
        morph_layout = QHBoxLayout()
        morph_label = QLabel("Wave Morph (0:Sine, 1:Triangle, 2:Square, 3:Sawtooth):")
        self.morph_slider = QSlider(Qt.Orientation.Horizontal)
        self.morph_slider.setMinimum(0)
        self.morph_slider.setMaximum(300)  # 0 to 3 with 100 steps per unit
        self.morph_slider.setValue(0)
        self.morph_value = QLabel("0.00")
        morph_layout.addWidget(morph_label)
        morph_layout.addWidget(self.morph_slider)
        morph_layout.addWidget(self.morph_value)
        layout.addLayout(morph_layout)
        self.morph_slider.valueChanged.connect(self.update_parameters)

    def update_parameters(self):
        """Update generator parameters based on current control values"""
        # Update labels
        self.f0_value.setText(str(self.f0_slider.value()))
        self.morph_value.setText(f"{self.morph_slider.value() / 100:.2f}")

        # Update generator parameters
        self.generator.update_parameters(
            f0=self.f0_slider.value(),
            morph=self.morph_slider.value() / 100.0,
            formants=vowel_formants[self.vowel_selector.currentText()]
        )

    def update_display(self):
        """Update the waveform display with smoothing"""
        # Generate samples for display
        samples = self.generator.generate_samples(self.display_samples)
        
        # Smooth display updates
        alpha = 0.7  # Smoothing factor
        smoothed = alpha * samples + (1 - alpha) * self.last_samples
        self.last_samples = smoothed
        
        # Update plot with anti-aliased line
        t = np.linspace(0, self.display_duration, len(smoothed))
        self.plot_line.setData(t, smoothed, antialias=True)

    def update_audio(self):
        """Generate and play audio samples if audio is enabled"""
        if not self.is_playing:
            return
            
        # Generate new samples
        samples = self.generator.generate_samples(self.audio_buffer_size)
        
        # Apply smooth normalization
        max_val = np.max(np.abs(samples))
        if max_val > 1e-6:
            # Use soft normalization to prevent sudden changes
            target_gain = 0.95 / max_val
            self.current_gain = getattr(self, 'current_gain', target_gain)
            self.current_gain = 0.95 * self.current_gain + 0.05 * target_gain
            normalized = samples * self.current_gain
        else:
            normalized = samples
        
        # Convert to 16-bit integers with safety bounds
        bounded = np.clip(normalized, -1.0, 1.0)
        audio_data = (bounded * 32767).astype(np.int16)
        
        # Play audio
        if self.play_obj is not None and self.play_obj.is_playing():
            return
        self.play_obj = sa.play_buffer(audio_data, 1, 2, self.Fs)

    def toggle_audio(self, checked):
        """Toggle audio playback"""
        self.is_playing = checked
        self.play_button.setText("Stop" if checked else "Play")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = SequentialTimeViewer()
    viewer.show()
    sys.exit(app.exec())
