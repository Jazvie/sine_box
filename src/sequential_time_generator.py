import numpy as np
from scipy.signal import lfilter
from collections import deque

class SequentialTimeGenerator:
    def __init__(self, fs=16000):
        """
        Initialize the sequential generator with time-domain synthesis and overlap-add filtering
        fs: sampling frequency in Hz
        """
        self.fs = fs
        
        # Current state variables
        self.phase = 0.0  # Phase accumulator for continuity
        self.current_f0 = 155.0  # Current fundamental frequency
        self.current_morph = 0.0  # Current wave morph value
        self.current_max_harmonic = 25  # Maximum number of harmonics
        
        # Filter parameters
        self.current_formants = [(730, 80), (1090, 90), (2440, 120)]  # Default to 'a'
        
        # Overlap-add parameters
        self.frame_size = 2048  # Larger frame size for better frequency resolution
        self.hop_size = 128     # Smaller hop size for smoother transitions (93.75% overlap)
        self.overlap_factor = self.frame_size // self.hop_size
        
        # Initialize buffers
        self.input_buffer = np.zeros(self.frame_size * 2)  # Double buffer for input
        self.output_buffer = np.zeros(self.frame_size * 2)  # Double buffer for output
        self.buffer_position = 0
        self.prev_window = np.zeros(self.frame_size)  # Store previous window for smooth transitions
        
        # Initialize filter state
        self._update_filter_coeffs()
        self.filter_states = []
        for _ in range(len(self.current_formants)):
            self.filter_states.append(np.zeros(2))  # 2nd order filter states

    def _update_filter_coeffs(self):
        """Update filter coefficients for each formant using improved resonator design"""
        self.filter_coeffs = []
        for F, B in self.current_formants:
            # Convert frequency and bandwidth to radians
            w0 = 2 * np.pi * F / self.fs
            bw = 2 * np.pi * B / self.fs
            
            # Compute filter coefficients for resonator
            R = np.exp(-bw/2)  # Pole radius
            theta = w0  # Pole angle
            
            # Create second-order section with improved numerator
            b = np.array([1 - R*R, 0.0, 0.0])  # Modified numerator for better response
            a = np.array([1.0, 
                         -2 * R * np.cos(theta),  # First-order term
                         R * R])  # Second-order term
            
            # Normalize gain at center frequency
            w = np.exp(1j * w0)
            freq_resp = (b[0] + b[1]*w + b[2]*w*w) / (a[0] + a[1]*w + a[2]*w*w)
            gain = np.abs(freq_resp)
            b = b / gain  # Normalize to unity gain at center frequency
            
            # Add resonator to cascade
            self.filter_coeffs.append((b, a))

    def _generate_base_waveform(self, t, morph):
        """Generate base waveform using bandlimited synthesis"""
        phase = 2 * np.pi * self.current_f0 * t
        result = np.zeros_like(t)
        nyquist = self.fs / 2
        max_harmonic = min(int(nyquist / self.current_f0), 50)  # Limit harmonics to prevent aliasing
        
        def sine_wave(phase):
            return np.sin(phase)
        
        def triangle_wave(phase, n_harmonics):
            wave = np.zeros_like(phase)
            for n in range(1, n_harmonics + 1, 2):
                coef = 8 / (np.pi * np.pi * n * n) * ((-1)**((n-1)//2))
                wave += coef * np.sin(n * phase)
            return wave
        
        def square_wave(phase, n_harmonics):
            wave = np.zeros_like(phase)
            for n in range(1, n_harmonics + 1, 2):
                coef = 4 / (np.pi * n)
                wave += coef * np.sin(n * phase)
            return wave
        
        def sawtooth_wave(phase, n_harmonics):
            wave = np.zeros_like(phase)
            for n in range(1, n_harmonics + 1):
                coef = 2 / (np.pi * n) * ((-1)**(n+1))
                wave += coef * np.sin(n * phase)
            return wave
        
        # Determine which waveforms to blend
        lower = int(np.floor(morph))
        upper = int(np.ceil(morph))
        frac = morph - lower
        
        # Generate bandlimited waveforms
        if lower == 0:
            wave_lower = sine_wave(phase)
        elif lower == 1:
            wave_lower = triangle_wave(phase, max_harmonic)
        elif lower == 2:
            wave_lower = square_wave(phase, max_harmonic)
        else:
            wave_lower = sawtooth_wave(phase, max_harmonic)
            
        if upper == 0:
            wave_upper = sine_wave(phase)
        elif upper == 1:
            wave_upper = triangle_wave(phase, max_harmonic)
        elif upper == 2:
            wave_upper = square_wave(phase, max_harmonic)
        else:
            wave_upper = sawtooth_wave(phase, max_harmonic)
        
        # Blend waveforms and apply amplitude compensation
        blended = (1 - frac) * wave_lower + frac * wave_upper
        
        # Apply anti-aliasing filter
        cutoff = 0.8 * nyquist  # Leave some headroom
        if self.current_f0 * max_harmonic > cutoff:
            # Simple lowpass filter
            alpha = 0.2
            blended = alpha * blended + (1 - alpha) * np.roll(blended, 1)
        
        return blended

    def _apply_formant_filters(self, frame):
        """Apply formant filters in series with gain compensation"""
        output = frame.copy()
        
        # Apply each formant filter in series
        for i, (b, a) in enumerate(self.filter_coeffs):
            # Apply filter and update state
            filtered, self.filter_states[i] = lfilter(b, a, output, zi=self.filter_states[i])
            
            # Apply gain compensation (resonators tend to reduce amplitude)
            gain = 1.0 / (1.0 - abs(a[1]))  # Compensate based on pole location
            output = filtered * gain
            
            # Safety check for instability
            if np.any(np.abs(output) > 1e3):
                output = frame  # Revert to input if unstable
                self.filter_states[i] = np.zeros_like(self.filter_states[i])
        
        return output

    def generate_next_sample(self):
        """Generate the next sample using overlap-add processing"""
        # Generate new samples if buffer is depleted
        if self.buffer_position >= self.hop_size:
            # Shift buffers
            self.input_buffer[:-self.hop_size] = self.input_buffer[self.hop_size:]
            self.output_buffer[:-self.hop_size] = self.output_buffer[self.hop_size:]
            
            # Generate new frame
            t = np.arange(self.hop_size) / self.fs + self.phase / self.fs
            new_samples = self._generate_base_waveform(t, self.current_morph)
            self.input_buffer[-self.frame_size:] = np.pad(new_samples, (self.frame_size - self.hop_size, 0), mode='constant')
            
            # Update phase
            self.phase += self.hop_size
            self.phase %= self.fs  # Use modulo for cleaner phase wrapping
            
            # Generate smooth window transition
            window = np.hanning(self.frame_size)
            
            # Apply window to current frame
            frame = self.input_buffer[-self.frame_size:] * window
            
            # Apply formant filters
            filtered_frame = self._apply_formant_filters(frame)
            
            # Overlap-add with proper normalization
            # Store current window for next frame
            self.output_buffer[-self.frame_size:] = self.prev_window + filtered_frame * window
            self.prev_window = filtered_frame * window
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(self.output_buffer[-self.frame_size:]))
            if max_val > 1e-6:
                self.output_buffer[-self.frame_size:] /= max_val
            
            # Reset buffer position
            self.buffer_position = 0
        
        # Get next sample and increment position
        sample = self.output_buffer[self.buffer_position]
        if np.isnan(sample) or np.isinf(sample):
            sample = 0.0
        self.buffer_position += 1
        
        return sample

    def generate_samples(self, num_samples):
        """Generate multiple samples"""
        return np.array([self.generate_next_sample() for _ in range(num_samples)])

    def update_parameters(self, f0=None, morph=None, formants=None, max_harmonic=None):
        """Update generator parameters smoothly"""
        if f0 is not None:
            # Adjust phase to maintain continuity when frequency changes
            self.phase = self.phase * (f0 / self.current_f0)
            self.current_f0 = f0
            
        if morph is not None:
            self.current_morph = morph
            
        if max_harmonic is not None:
            self.current_max_harmonic = max_harmonic
            
        if formants is not None:
            self.current_formants = formants
            self._update_filter_coeffs()
            # Reset filter states when formants change
            self.filter_states = [np.zeros(2) for _ in range(len(self.current_formants))]
