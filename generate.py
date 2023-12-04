import os
import pickle
import numpy as np
import soundfile as sf

from generator import SoundGenerator
from vae import VAE
from train import SPECTROGRAMS_PATH


HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "fsdd/min_max_values.pkl"


def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # load trained vae and initialize sound generator object
    vae = VAE.load("model_soundofai")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)
    # load spectrograms and minmax values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    spectrograms, file_paths = load_fsdd(SPECTROGRAMS_PATH)
    # sample some data
    sampled_spectrograms, sampled_min_max_values = select_spectrograms(spectrograms,
                                                       file_paths,
                                                       min_max_values, 
                                                       num_spectrograms=5)
    # generate audios
    generated_waveforms, _ = sound_generator.generate(sampled_spectrograms, sampled_min_max_values)
    # convert spectrograms to audio waveforms
    original_waveforms = sound_generator.convert_spectrograms_to_audio(sampled_spectrograms, sampled_min_max_values)
    # save generated and original audios
    save_signals(generated_waveforms, SAVE_DIR_GENERATED)
    save_signals(original_waveforms, SAVE_DIR_ORIGINAL)
    