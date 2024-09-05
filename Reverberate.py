import numpy as np
import scipy.signal as signal
import soundfile as sf
from pathlib import Path


def apply_reverb_multichannel(input_wav, output_wav, rt60=0.7,
                              room_dimensions=[5,5,5],  # Add room dimensions
                              source_position=[2,3,2],  # Add source position
                              microphone_positions=[(2.5, 2.5, 1.5), (2.5, 2.5, 1.0)]):  # Add microphone positions
    audio, sr = sf.read(input_wav)

    # Check if audio is multichannel
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=1)
    num_channels = audio.shape[1]

    # Generate impulse response for each microphone
    impulse_responses = []
    for mic_pos in microphone_positions:
        impulse_response = generate_impulse_response(
            room_dimensions, source_position, mic_pos, rt60, sr
        )
        impulse_responses.append(impulse_response)

    # Apply impulse responses to each channel
    reverberated_audio = np.zeros_like(audio)
    for channel in range(num_channels):
        reverberated_audio[:, channel] = signal.convolve(
            audio[:, channel], impulse_responses[channel], mode='full'
        )[: len(audio)]

    # Normalize the output to prevent clipping
    reverberated_audio /= np.max(np.abs(reverberated_audio))

    # Write the output audio file
    sf.write(output_wav, reverberated_audio, sr)


def generate_impulse_response(room_dimensions, source_position, mic_position,
                              rt60, sr):
    """
    Generates an impulse response using the image method.

    Args:
        room_dimensions: Room dimensions (length, width, height) in meters.
        source_position: Position of the sound source (x, y, z) in meters.
        mic_position: Position of the microphone (x, y, z) in meters.
        rt60: Reverberation time (in seconds).
        sr: Sample rate.

    Returns:
        Impulse response as a NumPy array.
    """
    # Calculate room parameters
    c = 343  # Speed of sound in meters per second
    volume = np.prod(room_dimensions)
    absorption_coefficient = -volume * np.log(0.01) / (rt60 * c)

    # Create image sources
    image_sources = create_image_sources(room_dimensions, source_position)

    # Calculate distances and time delays
    distances = [np.linalg.norm(mic_position - image_source) for image_source in image_sources]
    time_delays = np.array(distances) / 343  # Speed of sound in m/s

    # Dynamically determine impulse response length
    max_delay = np.max(time_delays)
    impulse_response_length = int(max_delay * sr) + 1  # Add 1 for safety
    impulse_response = np.zeros(impulse_response_length)

    for delay in time_delays:
        # Apply decay based on absorption
        decay_factor = np.exp(-absorption_coefficient * delay)
        impulse_response[int(delay * sr)] += decay_factor

    return impulse_response


def create_image_sources(room_dimensions, source_position):
    """
    Creates image sources for the image method.

    Args:
        room_dimensions: Room dimensions (length, width, height) in meters.
        source_position: Position of the sound source (x, y, z) in meters.

    Returns:
        List of image source positions.
    """

    image_sources = []
    for x_sign in [-1, 1]:
        for y_sign in [-1, 1]:
            for z_sign in [-1, 1]:
                image_source = np.array(
                    [
                        x_sign * source_position[0],
                        y_sign * source_position[1],
                        z_sign * source_position[2],
                    ]
                )
                image_sources.append(image_source)

    return image_sources


