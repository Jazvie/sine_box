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
        self.filter_coeffs = []  # Will store (b, a) coefficients
        self.filter_states = []  # Will store filter states
        
        # Streaming parameters
        self.frame_size = 512  # Frame size for processing
        self.hop_size = 128   # 75% overlap for smoother transitions
        
        # Initialize buffers
        self.input_buffer = np.zeros(self.frame_size)
        self.output_buffer = np.zeros(self.frame_size)
        self.prev_output = np.zeros(self.hop_size)  # For overlap-add
        
        # Create window for overlap-add
        self.window = np.hanning(self.frame_size)
        # Normalize window for perfect reconstruction
        overlap_sum = np.zeros(self.frame_size)
        for i in range(0, self.frame_size - self.hop_size + 1, self.hop_size):
            overlap_sum[i:i+self.hop_size] += self.window[i:i+self.hop_size] ** 2
        self.window = self.window / np.sqrt(overlap_sum + 1e-10)
        
        # Global gain smoothing with slower changes
        self.current_gain = 1.0
        self.target_gain = 1.0
        self.gain_smoothing = 0.995  # Smooth gain changes
        
        # Initialize filter state
        self._update_filter_coeffs()
        self.filter_states = []
        for _ in range(len(self.current_formants)):
            self.filter_states.append(np.zeros(2))  # 2nd order filter states

    def _update_filter_coeffs(self):
        """Update filter coefficients for each formant using the specified transfer function
        H_i(z) = 1/(1-2e^{-pib_i}cos(2pi f_i)z^{-1} + e^{-2pi b_i}z^{-2})
        with stability checks and normalization
        """
        self.filter_coeffs = []
        for F, B in self.current_formants:
            # Normalize frequencies with safety limits
            f_i = np.clip(F / self.fs, 0.001, 0.499)  # Avoid 0 and Nyquist
            b_i = np.clip(B / self.fs, 0.001, 0.499)  # Ensure positive bandwidth
            
            # Calculate filter coefficients
            exp_term = np.exp(-np.pi * b_i)
            cos_term = np.cos(2 * np.pi * f_i)
            
            # Numerator and denominator coefficients
            b = np.array([1.0])  # Numerator is just 1
            a = np.array([1.0, 
                         -2 * exp_term * cos_term,  # First-order term
                         exp_term * exp_term])  # Second-order term
            
            # Check filter stability
            poles = np.roots(a)
            if np.any(np.abs(poles) >= 1.0):
                # If unstable, adjust coefficients to ensure stability
                max_pole = np.max(np.abs(poles))
                a[1] /= max_pole
                a[2] /= max_pole * max_pole
            
            # Normalize gain at DC
            dc_gain = np.sum(b) / np.sum(a)
            b = b / np.abs(dc_gain)
            
            # Add filter to cascade
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
        """Apply formant filters with overlap-add processing"""
        frame_len = len(frame)
        
        # Ensure input buffer has enough space
        if len(self.input_buffer) < frame_len + self.hop_size:
            self.input_buffer = np.zeros(frame_len + self.hop_size)
        
        # Shift input buffer and add new frame
        self.input_buffer[:-frame_len] = self.input_buffer[frame_len:]
        self.input_buffer[-frame_len:] = frame
        
        # Apply window
        windowed = self.input_buffer[:self.frame_size] * self.window
        
        # Apply filters in series with careful state handling
        filtered = windowed.copy()
        for i, (b, a) in enumerate(self.filter_coeffs):
            # Apply filter with current state
            filtered, new_state = lfilter(b, a, filtered, zi=self.filter_states[i])
            
            # Smooth state updates and handle instability
            if np.any(np.abs(filtered) > 10.0) or np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                # Gradually recover from instability
                self.filter_states[i] = 0.9 * self.filter_states[i]
                filtered = windowed  # Reset to input
            else:
                # Smooth state update
                self.filter_states[i] = 0.95 * self.filter_states[i] + 0.05 * new_state
        
        # Apply smooth gain control
        max_val = np.max(np.abs(filtered))
        if max_val > 1e-6:
            self.target_gain = min(1.0, 0.95 / max_val)
        self.current_gain = 0.995 * self.current_gain + 0.005 * self.target_gain
        filtered *= self.current_gain
        
        # Extract output with overlap-add
        output = np.zeros(frame_len)
        if frame_len <= self.hop_size:
            output = filtered[:frame_len]
        else:
            # Add previous overlap
            output[:self.hop_size] = self.prev_output + filtered[:self.hop_size]
            # Copy remaining samples
            output[self.hop_size:] = filtered[self.hop_size:frame_len]
            # Save overlap for next frame
            if frame_len + self.hop_size <= len(filtered):
                self.prev_output = filtered[frame_len:frame_len + self.hop_size]
            else:
                self.prev_output = np.zeros(self.hop_size)
        
        return output

    def generate_samples(self, num_samples):
        """Generate samples with streaming formant filtering"""
        output = np.zeros(num_samples)
        current_pos = 0
        
        while current_pos < num_samples:
            # Generate in small chunks
            chunk_size = min(self.hop_size, num_samples - current_pos)
            
            # Generate base waveform
            t = np.arange(chunk_size) / self.fs + self.phase / (2 * np.pi * self.current_f0)
            chunk = self._generate_base_waveform(t, self.current_morph)
            
            # Apply formant filtering
            filtered = self._apply_formant_filters(chunk)
            
            # Add to output
            output[current_pos:current_pos + chunk_size] = filtered[:chunk_size]
            
            # Update phase and position
            self.phase = (self.phase + 2 * np.pi * self.current_f0 * chunk_size / self.fs) % (2 * np.pi)
            current_pos += chunk_size
        
        return output

    def update_parameters(self, f0=None, morph=None, formants=None):
        """Update synthesis parameters"""
        if f0 is not None:
            self.current_f0 = f0
        if morph is not None:
            self.current_morph = morph
        if formants is not None:
            self.current_formants = formants
            self._update_filter_coeffs()
            # Reset filter states when formants change
            self.filter_states = [np.zeros(2) for _ in range(len(self.current_formants))]
