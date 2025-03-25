import sys
import numpy as np
import simpleaudio as sa
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSlider, QLabel)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

class WaveViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wave Interpolator")
        self.setGeometry(100, 100, 800, 600)

        # Audio settings
        self.sample_rate = 44100  # Hz
        self.display_duration = 0.1  # 100ms for display
        self.display_samples = int(self.sample_rate * self.display_duration)
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

        # Create wave type slider
        wave_layout = QHBoxLayout()
        wave_label = QLabel("Wave Type:")
        self.wave_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_slider.setMinimum(0)
        self.wave_slider.setMaximum(300)  # 0 to 3 with 100 steps per unit
        self.wave_slider.setValue(0)
        self.wave_type_value = QLabel("Sine")
        wave_layout.addWidget(wave_label)
        wave_layout.addWidget(self.wave_slider)
        wave_layout.addWidget(self.wave_type_value)
        layout.addLayout(wave_layout)

        # Create frequency slider
        freq_layout = QHBoxLayout()
        freq_label = QLabel("Frequency (Hz):")
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(20)    # 2 Hz minimum
        self.freq_slider.setMaximum(2000)  # 200 Hz maximum
        self.freq_slider.setValue(100)      # Start at 10 Hz
        self.freq_value = QLabel("10.0")
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_slider)
        freq_layout.addWidget(self.freq_value)
        layout.addLayout(freq_layout)

        # Create volume slider
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)  # 0 to 1 with 100 steps
        self.volume_slider.setValue(50)     # Start at 0.5
        self.volume_value = QLabel("0.50")
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_value)
        layout.addLayout(volume_layout)

        # Set up the plot line
        pen = pg.mkPen(color='b', width=2)
        self.plot_line = self.plot_widget.plot([], [], pen=pen)

        # Connect sliders to update function
        self.wave_slider.valueChanged.connect(self.mark_audio_update)
        self.freq_slider.valueChanged.connect(self.mark_audio_update)
        self.volume_slider.valueChanged.connect(self.mark_audio_update)

        # Set up timer for visualization
        self.display_phase = 0.0
        self.vis_timer = QTimer()
        self.vis_timer.timeout.connect(self.update_plot)
        self.vis_timer.start(16)  # ~60 FPS

        # Set up timer for audio updates
        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.update_audio)
        self.audio_timer.start(50)  # 20 Hz update rate for audio

        # Initialize
        self.update_audio()

    def get_wave_type_name(self, wave_type):
        """Get the name of the current wave type"""
        if wave_type < 1:
            return "Sine"
        elif wave_type < 2:
            return "Triangle"
        elif wave_type < 3:
            return "Sawtooth"
        else:
            return "Square"

    def generate_waveform(self, t, wave_type):
        """Generate interpolated waveform"""
        # Generate base waveforms
        sine_wave = np.sin(2 * np.pi * t)
        triangle_wave = 2 * np.abs(2 * (t % 1) - 1) - 1
        sawtooth_wave = 2 * (t % 1) - 1
        square_wave = np.sign(sine_wave)
        
        if wave_type < 1:  # Sine to Triangle
            alpha = wave_type
            return (1 - alpha) * sine_wave + alpha * triangle_wave
        elif wave_type < 2:  # Triangle to Sawtooth
            alpha = wave_type - 1
            return (1 - alpha) * triangle_wave + alpha * sawtooth_wave
        elif wave_type < 3:  # Sawtooth to Square
            alpha = wave_type - 2
            return (1 - alpha) * sawtooth_wave + alpha * square_wave
        else:  # Perfect square wave at 3
            return square_wave

    def mark_audio_update(self):
        """Mark that audio needs to be updated"""
        self.audio_needs_update = True
        
        # Update labels
        wave_type = self.wave_slider.value() / 100.0
        freq = self.freq_slider.value() / 10.0
        volume = self.volume_slider.value() / 100.0
        
        self.wave_type_value.setText(self.get_wave_type_name(wave_type))
        self.freq_value.setText(f"{freq:.1f}")
        self.volume_value.setText(f"{volume:.2f}")

    def update_audio(self):
        """Update the audio output"""
        if not self.audio_needs_update and self.play_obj and self.play_obj.is_playing():
            return

        wave_type = self.wave_slider.value() / 100.0
        freq = self.freq_slider.value() / 10.0
        volume = self.volume_slider.value() / 100.0

        # Calculate exact number of samples for one complete cycle
        samples_per_cycle = int(round(self.sample_rate / freq))
        
        # Generate time points for exactly one cycle
        t = np.linspace(0, 1, samples_per_cycle, endpoint=False)
        
        # Generate one cycle of the waveform
        wave = self.generate_waveform(t, wave_type)
        
        # Apply volume
        wave = wave * volume
        
        # Convert to 16-bit integers
        audio_data = (wave * 32767).astype(np.int16)
        
        # Stop previous playback if any
        if self.play_obj is not None and self.play_obj.is_playing():
            self.play_obj.stop()
        
        # Create a longer buffer by repeating the cycle
        num_repeats = 100  # This creates a longer buffer but maintains perfect periodicity
        audio_data = np.tile(audio_data, num_repeats)
        
        # Start new playback
        self.play_obj = sa.play_buffer(audio_data, 1, 2, self.sample_rate)
        self.last_wave = wave
        self.audio_needs_update = False

    def update_plot(self):
        """Update the visualization"""
        if self.last_wave is None:
            return
            
        # Update display phase
        self.display_phase += 0.016  # 16ms
        self.display_phase %= self.display_duration
        
        # Generate time points for display
        t = np.linspace(0, self.display_duration, self.display_samples)
        
        # Calculate how many cycles to show
        freq = self.freq_slider.value() / 10.0
        cycles_to_show = freq * self.display_duration
        
        # Create visualization by repeating the cycle
        wave_samples = len(self.last_wave)
        repeats = int(np.ceil(cycles_to_show))
        full_wave = np.tile(self.last_wave, repeats)
        
        # Interpolate to match display samples
        x_original = np.linspace(0, repeats, len(full_wave))
        x_new = np.linspace(0, repeats, self.display_samples)
        full_wave = np.interp(x_new, x_original, full_wave)
        
        # Roll the array to create animation
        shift = int(self.display_phase * self.sample_rate)
        full_wave = np.roll(full_wave, -shift)
        
        # Update plot
        self.plot_line.setData(t, full_wave)

    def closeEvent(self, event):
        """Clean up audio when closing"""
        if self.play_obj is not None and self.play_obj.is_playing():
            self.play_obj.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = WaveViewer()
    viewer.show()
    sys.exit(app.exec())
