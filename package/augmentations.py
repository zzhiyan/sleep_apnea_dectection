import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

def random_noise(data, noise_level=0.2):
    noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=noise_level, dtype=tf.float32)
    return data + noise

def random_scaling(data, scaling_factor=0.1):
    scale = tf.random.uniform(shape=tf.shape(data), minval=1-scaling_factor, maxval=1+scaling_factor, dtype=tf.float32)
    return data * scale

def random_offset(data, offset_factor=0.01):
    offset = tf.random.uniform(shape=tf.shape(data), minval=-offset_factor, maxval=offset_factor, dtype=tf.float32)
    return data + offset

def blockout_augmentation(dataset, proportion=0.1):
    dataset = np.squeeze(dataset)
    n_sequences, time_length = dataset.shape
    block_size = int(proportion * time_length)
    start_indices = np.random.randint(0, time_length - block_size + 1, size=n_sequences)
    block_indices = np.arange(block_size)
    mask = np.zeros_like(dataset, dtype=bool)
    mask[np.arange(n_sequences)[:, None], start_indices[:, None] + block_indices] = True
    dataset[mask] = 0    # Apply the blockout
    dataset = np.expand_dims(dataset, axis=-1)
    return dataset

def local_time_warping(series, window_size=50, sigma=0.5):
    series = np.squeeze(series)
    num_samples, num_features = series.shape
    warped_series = series.copy()
    for i in range(num_samples):
        start = np.random.randint(0, num_features - window_size)
        end = start + window_size
        orig_time = np.linspace(0, 1, window_size)
        random_factors = np.random.normal(loc=1.0, scale=sigma, size=window_size)
        warped_time = np.cumsum(random_factors)
        warped_time = (warped_time - warped_time.min()) / (warped_time.max() - warped_time.min())
        interpolation = interp1d(warped_time, series[ i, start:end], kind='linear', fill_value="extrapolate")
        warped_series[ i, start: end]= interpolation(orig_time)
    warped_series = np.expand_dims(warped_series, axis=-1)
    return warped_series

def left_to_right_flip(time_series):
    flipped_series = np.flip(time_series, axis=1)
    return flipped_series

def bidirectional_flip(time_series):
    flipped_series = -time_series
    return flipped_series

def smoothed_filter(signal, window_size=9):
    kernel = np.ones(window_size) / window_size
    pad_width = len(kernel) // 2
    padded_signal = np.pad(signal, pad_width=((0, 0), (pad_width, pad_width), (0, 0)), mode='edge')
    filtered_x = np.apply_along_axis(lambda m: np.convolve(m.flatten(), kernel, mode='valid'), axis=1,
                                     arr=padded_signal)
    smoothed_signal = filtered_x.reshape(-1, 500, 1)
    return smoothed_signal






