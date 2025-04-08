import numpy as np
from scipy.signal import lfilter
from collections import deque

class SequentialGenerator:
    def __init__(self, fs=16000):
        """
        Initialize the sequential generator with:
        fs: sampling frequency in Hz
        """
        self.fs = fs
        
        # Current state variables
        self.phase = 0.0  # Phase accumulator for continuity
        self.current_f0 = 155.0  # Current fundamental frequency
        self.current_morph = 0.0  # Current wave morph value
        self.current_max_harmonic = 25  # Maximum number of harmonics
        
        # Filter state
        self.filter_state = None
        self.current_formants = [(730, 80), (1090, 90), (2440, 120)]  # Default to 'a'
        self.current_filter_coeffs = self._create_filter_coeffs()
        
        # Buffer for filter state continuity
        self.signal_buffer = deque(maxlen=10)  # Keep last 10 samples for continuity

    def _create_filter_coeffs(self):
        """Create filter coefficients for the current formants"""
        a_combined = np.array([1.0])
        for (F, B) in self.current_formants:
            f = F / self.fs
            b = B / self.fs
            a1 = 2 * np.exp(-np.pi * b) * np.cos(2 * np.pi * f)
            a2 = np.exp(-2 * np.pi * b)
            a_coeff = np.array([1.0, -a1, a2])
            a_combined = np.convolve(a_combined, a_coeff)
        return (np.array([1.0]), a_combined)  # (b, a) coefficients

    def _get_harmonic_coefficient(self, n, morph):
        """Get interpolated harmonic coefficient for given harmonic number and morph value"""
        def coeff_sine(n):
            return 1.0 if n == 1 else 0.0

        def coeff_triangle(n):
            if n % 2 == 1:
                return 8 / np.pi**2 * ((-1)**((n-1)//2)) / (n**2)
            return 0.0

        def coeff_square(n):
            if n % 2 == 1:
                return 4 / (np.pi * n)
            return 0.0

        def coeff_sawtooth(n):
            return 2 / (np.pi * n) * ((-1)**(n+1))

        lower = int(np.floor(morph))
        upper = int(np.ceil(morph))
        frac = morph - lower

        coef_funcs = [coeff_sine, coeff_triangle, coeff_square, coeff_sawtooth]
        coef_lower = coef_funcs[min(lower, 3)](n)
        coef_upper = coef_funcs[min(upper, 3)](n)

        return (1 - frac) * coef_lower + frac * coef_upper

    def generate_next_sample(self):
        """Generate the next sample based on current parameters"""
        sample = 0.0
        
        # Generate sample using additive synthesis
        for n in range(1, self.current_max_harmonic + 1):
            coef = self._get_harmonic_coefficient(n, self.current_morph)
            harmonic_freq = n * self.current_f0
            sample += coef * np.sin(2 * np.pi * harmonic_freq * self.phase / self.fs)

        # Update phase
        self.phase += 1.0
        if self.phase >= self.fs:  # Wrap phase to prevent floating point errors
            self.phase -= self.fs

        # Apply formant filter
        if self.filter_state is None:
            self.filter_state = np.zeros(len(self.current_filter_coeffs[1]) - 1)
            
        filtered_sample, self.filter_state = lfilter(*self.current_filter_coeffs, 
                                                   [sample], 
                                                   zi=self.filter_state)
        
        return filtered_sample[0]

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
            self.current_filter_coeffs = self._create_filter_coeffs()
            # Reset filter state when formants change
            self.filter_state = None

    def generate_samples(self, num_samples):
        """Generate multiple samples"""
        return np.array([self.generate_next_sample() for _ in range(num_samples)])
