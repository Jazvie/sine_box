import sys
import numpy as np
import simpleaudio as sa
from collections import deque
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
        self.display_duration = .05  # 50ms for display (show more detail)
        self.display_samples = int(self.Fs * self.display_duration)
        self.frame_size = 512  # Frame size for processing
        self.hop_size = 128  # 75% overlap for smoother transitions
        self.play_obj = None
        self.is_playing = False
        self.last_samples = np.zeros(self.display_samples)  # For smoother display updates
        self.audio_queue = deque(maxlen=4)  # Small queue for audio continuity
        
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
        
        # Create update timer
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(8)  # 125 Hz for audio/display updates rate for consistent audio

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
        self.f0_slider.setMinimum(80)
        self.f0_slider.setMaximum(255)
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
        """Update the waveform display using audio samples"""
        if not self.is_playing:
            return

        # Generate new samples for both audio and display
        new_samples = self.generator.generate_samples(self.hop_size)
        self.audio_queue.append(new_samples)
        
        # Update display with the latest audio data
        if self.audio_queue:
            # Get the most recent samples for display
            display_data = np.concatenate(list(self.audio_queue))
            if len(display_data) > self.display_samples:
                display_data = display_data[-self.display_samples:]
            elif len(display_data) < self.display_samples:
                display_data = np.pad(display_data, (0, self.display_samples - len(display_data)))
            
            # Smooth display updates
            alpha = 0.7  # Smoothing factor
            smoothed = alpha * display_data + (1 - alpha) * self.last_samples
            self.last_samples = smoothed
            
            # Update plot with anti-aliased line
            t = np.linspace(0, self.display_duration, len(smoothed))
            self.plot_line.setData(t, smoothed, antialias=True)
        
        # Play audio if we have enough samples
        if len(self.audio_queue) >= 3 and (self.play_obj is None or not self.play_obj.is_playing()):
            # Concatenate chunks with overlap
            chunks = list(self.audio_queue)
            output = np.concatenate(chunks[:-1])
            
            # Convert to 16-bit integers
            bounded = np.clip(output, -1.0, 1.0)
            audio_data = (bounded * 32767).astype(np.int16)
            
            # Play audio
            self.play_obj = sa.play_buffer(audio_data, 1, 2, self.Fs)
            
            # Keep last chunk for continuity
            self.audio_queue.clear()
            self.audio_queue.append(chunks[-1])

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
