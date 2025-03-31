import sys
import numpy as np
import simpleaudio as sa
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSlider, QLabel, QComboBox)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
from scipy.signal import lfilter
from formant_generator import excitation_fourier, create_combined_filter, vowel_formants

class FormantViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Formant Generator Viewer")
        self.setGeometry(100, 100, 1000, 800)

        # Audio settings
        self.Fs = 16000  # Hz (fixed)
        self.display_duration = 0.1  # 100ms for display
        self.display_samples = int(self.Fs * self.display_duration)
        self.play_obj = None
        self.last_wave = None
        self.audio_needs_update = True

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
        self.vowel_selector.currentTextChanged.connect(self.mark_audio_update)
        
        # Set up timer for audio updates
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.update_audio)
        self.audio_timer.start(50)  # 20 Hz update rate for audio

        # Initialize
        self.update_audio()

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
        self.f0_slider.setMaximum(170)
        self.f0_slider.setValue(155)
        self.f0_value = QLabel("155")
        f0_layout.addWidget(f0_label)
        f0_layout.addWidget(self.f0_slider)
        f0_layout.addWidget(self.f0_value)
        layout.addLayout(f0_layout)
        self.f0_slider.valueChanged.connect(self.mark_audio_update)

        # Number of periods
        periods_layout = QHBoxLayout()
        periods_label = QLabel("Number of Periods:")
        self.periods_slider = QSlider(Qt.Orientation.Horizontal)
        self.periods_slider.setMinimum(1)
        self.periods_slider.setMaximum(1000)
        self.periods_slider.setValue(50)
        self.periods_value = QLabel("50")
        periods_layout.addWidget(periods_label)
        periods_layout.addWidget(self.periods_slider)
        periods_layout.addWidget(self.periods_value)
        layout.addLayout(periods_layout)
        self.periods_slider.valueChanged.connect(self.mark_audio_update)

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
        self.harmonic_slider.valueChanged.connect(self.mark_audio_update)

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
        self.morph_slider.valueChanged.connect(self.mark_audio_update)

    def mark_audio_update(self):
        """Mark that audio needs to be updated and update labels"""
        self.audio_needs_update = True
        
        # Update labels

        self.f0_value.setText(str(self.f0_slider.value()))
        self.periods_value.setText(str(self.periods_slider.value()))
        self.harmonic_value.setText(str(self.harmonic_slider.value()))
        self.morph_value.setText(f"{self.morph_slider.value() / 100:.2f}")

    def update_audio(self):
        """Update the audio output"""
        if not self.audio_needs_update and self.play_obj and self.play_obj.is_playing():
            return

        # Get current parameters
        Fs = self.Fs  # Fixed at 16000 Hz
        f0 = self.f0_slider.value()
        num_periods = self.periods_slider.value()
        max_harmonic = self.harmonic_slider.value()
        wave_morph = self.morph_slider.value() / 100.0
        vowel = self.vowel_selector.currentText()

        # Generate excitation signal
        excitation = excitation_fourier(Fs, f0, num_periods, max_harmonic, wave_morph)

        # Create and apply formant filter
        formants = vowel_formants[vowel]
        a_combined = create_combined_filter(Fs, formants)
        b_coeff = [1.0]

        # Apply the filter
        wave = np.real(np.array(excitation))  # Ensure the input is real
        wave = wave / np.max(np.abs(wave))  # Normalize before filtering
        wave = np.array(wave, dtype=np.float64)  # Convert to float64 for filtering
        wave = np.real(wave)  # Ensure the array is real
        filtered_wave = np.real(lfilter(b_coeff, a_combined, wave))
        
        # Normalize the filtered wave
        filtered_wave = filtered_wave / np.max(np.abs(filtered_wave))
        
        # Convert to 16-bit integers
        audio_data = (filtered_wave * 32767).astype(np.int16)
        
        # Stop previous playback if any
        if self.play_obj is not None and self.play_obj.is_playing():
            self.play_obj.stop()
        
        # Start new playback
        self.play_obj = sa.play_buffer(audio_data, 1, 2, Fs)
        
        # Update plot
        samples_to_show = min(len(filtered_wave), self.display_samples)
        t = np.linspace(0, self.display_duration, samples_to_show)
        self.plot_line.setData(t, filtered_wave[:samples_to_show])
        
        self.last_wave = filtered_wave
        self.audio_needs_update = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = FormantViewer()
    viewer.show()
    sys.exit(app.exec())
