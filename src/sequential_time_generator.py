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
        self.frame_size = 1024  # Frame size for good frequency resolution
        self.hop_size = 256     # 75% overlap for smooth transitions
        self.overlap_factor = self.frame_size // self.hop_size
        
        # Initialize buffers for overlap-add
        self.input_buffer = deque(maxlen=self.frame_size)
        self.output_buffer = np.zeros(self.frame_size)
        self.overlap_buffer = np.zeros(self.frame_size)
        
        # Window function for overlap-add
        self.window = np.hanning(self.frame_size)
        
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
        """Apply formant filters in series using overlap-add method with robust stability checks"""
        # Apply window to input frame
        windowed_frame = frame * self.window
        
        # Apply each formant filter in series
        filtered = windowed_frame.copy()
        for i, (b, a) in enumerate(self.filter_coeffs):
            # Apply filter and update state
            filtered, self.filter_states[i] = lfilter(b, a, filtered, zi=self.filter_states[i])
            
            # Check for instability or overflow
            if np.any(np.abs(filtered) > 10.0) or np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                # Reset filter state and use a simple one-pole smoothing filter
                self.filter_states[i] = np.zeros_like(self.filter_states[i])
                alpha = 0.99
                filtered = alpha * filtered[:-1] + (1 - alpha) * windowed_frame[1:]
                filtered = np.append(filtered[0], filtered)
            
            # Normalize to prevent cumulative gain
            max_val = np.max(np.abs(filtered))
            if max_val > 1e-6:
                filtered = filtered / max_val
        
        return filtered

            
    def generate_samples(self, num_samples):
        """Generate the next batch of samples with formant filtering using overlap-add"""
        # Generate time points
        t = np.arange(num_samples) / self.fs + self.phase / (2 * np.pi * self.current_f0)
        
        # Generate base waveform
        new_samples = self._generate_base_waveform(t, self.current_morph)
        
        # Normalize input samples
        max_val = np.max(np.abs(new_samples))
        if max_val > 1e-6:
            new_samples = new_samples / max_val
        
        # Update phase for continuity
        self.phase = (self.phase + 2 * np.pi * self.current_f0 * num_samples / self.fs) % (2 * np.pi)
        
        # Add new samples to input buffer
        for sample in new_samples:
            self.input_buffer.append(sample)
        
        # Process if we have enough samples
        if len(self.input_buffer) >= self.frame_size:
            # Get frame and apply filtering
            frame = np.array(self.input_buffer)
            filtered_frame = self._apply_formant_filters(frame)
            
            # Overlap-add with normalization
            self.output_buffer[:self.frame_size-self.hop_size] = self.output_buffer[self.hop_size:]
            self.output_buffer[self.frame_size-self.hop_size:] = 0
            self.output_buffer += filtered_frame
            
            # Normalize output buffer
            max_val = np.max(np.abs(self.output_buffer))
            if max_val > 1e-6:
                self.output_buffer = self.output_buffer / max_val
            
            # Clear processed samples from input buffer
            for _ in range(self.hop_size):
                self.input_buffer.popleft()
        
        # Return the next batch of samples
        output = self.output_buffer[:num_samples].copy()
        self.output_buffer = np.roll(self.output_buffer, -num_samples)
        self.output_buffer[-num_samples:] = 0
        
        # Final safety check
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            output = np.zeros_like(output)
        
        return output

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
