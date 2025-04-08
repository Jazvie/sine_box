import sys
import numpy as np
import simpleaudio as sa
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSlider, QLabel, QComboBox)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
from sequential_generator import SequentialGenerator
from formant_generator import vowel_formants

class SequentialViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequential Formant Generator")
        self.setGeometry(100, 100, 1000, 800)

        # Audio settings
        self.Fs = 16000  # Hz (fixed)
        self.display_duration = 0.1  # 100ms for display
        self.display_samples = int(self.Fs * self.display_duration)
        self.play_obj = None
        self.buffer_size = 2048  # Number of samples to generate at once
        
        # Initialize the sequential generator
        self.generator = SequentialGenerator(fs=self.Fs)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        self.buffer_position = 0

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
        layout.addWidget(self.plot_widget)

        # Create parameter sliders
        self.create_parameter_sliders(layout)

        # Create vowel selector
        vowel_layout = QHBoxLayout()
        vowel_label = QLabel("Vowel:")
        self.vowel_selector = QComboBox()
        self.vowel_selector.addItems(list(vowel_formants.keys()))
        vowel_layout.addWidget(vowel_label)
        vowel_layout.addWidget(self.vowel_selector)
        layout.addLayout(vowel_layout)

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

        # Initialize display
        self.update_display()

    def create_parameter_sliders(self, layout):
        # Display fixed sample rate
        fs_layout = QHBoxLayout()
        fs_label = QLabel("Sampling Frequency (Hz):")
        fs_value = QLabel("16000 (fixed)")
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

        # Maximum harmonic
        harmonic_layout = QHBoxLayout()
        harmonic_label = QLabel("Max Harmonic:")
        self.harmonic_slider = QSlider(Qt.Orientation.Horizontal)
        self.harmonic_slider.setMinimum(1)
        self.harmonic_slider.setMaximum(100)
        self.harmonic_slider.setValue(25)
        self.harmonic_value = QLabel("25")
        harmonic_layout.addWidget(harmonic_label)
        harmonic_layout.addWidget(self.harmonic_slider)
        harmonic_layout.addWidget(self.harmonic_value)
        layout.addLayout(harmonic_layout)
        self.harmonic_slider.valueChanged.connect(self.update_parameters)

        # Wave morph
        morph_layout = QHBoxLayout()
        morph_label = QLabel("Wave Morph:")
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
        """Update generator parameters based on current slider values"""
        # Update labels
        self.f0_value.setText(str(self.f0_slider.value()))
        self.harmonic_value.setText(str(self.harmonic_slider.value()))
        self.morph_value.setText(f"{self.morph_slider.value() / 100:.2f}")

        # Update generator parameters
        self.generator.update_parameters(
            f0=self.f0_slider.value(),
            morph=self.morph_slider.value() / 100.0,
            max_harmonic=self.harmonic_slider.value(),
            formants=vowel_formants[self.vowel_selector.currentText()]
        )

    def update_display(self):
        """Update the waveform display"""
        # Generate samples for display
        samples = self.generator.generate_samples(self.display_samples)
        
        # Update plot
        t = np.linspace(0, self.display_duration, len(samples))
        self.plot_line.setData(t, samples)

    def update_audio(self):
        """Generate and play audio samples"""
        # Generate new samples
        samples = self.generator.generate_samples(self.buffer_size)
        
        # Normalize and convert to 16-bit integers
        normalized = samples / max(abs(samples.max()), abs(samples.min()))
        audio_data = (normalized * 32767).astype(np.int16)
        
        # Play audio
        if self.play_obj is not None and self.play_obj.is_playing():
            self.play_obj.stop()
        self.play_obj = sa.play_buffer(audio_data, 1, 2, self.Fs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = SequentialViewer()
    viewer.show()
    sys.exit(app.exec())
