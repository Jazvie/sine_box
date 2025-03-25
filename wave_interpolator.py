import numpy as np
import matplotlib.pyplot as plt

def generate_waveform(t, wave_type):
    """
    Generate a waveform that interpolates between sine, square, triangle, and sawtooth waves.
    
    Parameters:
    t (array): Time array
    wave_type (float): Value between 0 and 4 that determines the wave shape
        0: Pure sine wave
        1: Pure square wave
        2: Pure triangle wave
        3: Pure sawtooth wave
        4: Back to sine wave
        (Intermediate values create mixed waveforms)
    
    Returns:
    array: The generated waveform
    """
    # Ensure wave_type is between 0 and 4
    wave_type = wave_type % 4
    
    # Generate base waveforms
    sine_wave = np.sin(2 * np.pi * t)
    square_wave = np.sign(sine_wave)
    
    # Triangle wave
    triangle_wave = 2 * np.abs(2 * (t % 1) - 1) - 1
    
    # Sawtooth wave
    sawtooth_wave = 2 * (t % 1) - 1
    
    # Determine which waves to interpolate between
    if wave_type < 1:  # Sine to Square
        alpha = wave_type
        return (1 - alpha) * sine_wave + alpha * square_wave
    elif wave_type < 2:  # Square to Triangle
        alpha = wave_type - 1
        return (1 - alpha) * square_wave + alpha * triangle_wave
    elif wave_type < 3:  # Triangle to Sawtooth
        alpha = wave_type - 2
        return (1 - alpha) * triangle_wave + alpha * sawtooth_wave
    else:  # Sawtooth to Sine
        alpha = wave_type - 3
        return (1 - alpha) * sawtooth_wave + alpha * sine_wave

# Example usage
if __name__ == "__main__":
    # Generate time array
    duration = 1.0
    sample_rate = 1000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create subplots to show different interpolation stages
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    wave_types = [0, 0.5, 1, 2, 3, 4]
    
    for i, wave_type in enumerate(wave_types):
        row = i // 2
        col = i % 2
        wave = generate_waveform(t, wave_type)
        axs[row, col].plot(t, wave)
        axs[row, col].set_title(f'Wave Type: {wave_type}')
        axs[row, col].grid(True)
        axs[row, col].set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()
