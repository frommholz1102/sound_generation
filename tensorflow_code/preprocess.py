import os
import pickle
import librosa
import numpy as np

class Loader:
    """
    Responsible for loading the audio wav file.
    """
    def __init__(self, sample_rate:int, duration:float, mono:bool):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, filepath:str):
        # pass all object attributes to librosa load function
        # returns signal itself and sample rate
        signal, _ = librosa.load(filepath, sr=self.sample_rate, duration=self.duration, mono=self.mono)
        return signal
    

class Padder:
    """
    Responsible for applying padding to the audio signal if the signal is too short.
    Modes of padding are: constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric, wrap
    """
    def __init__(self, mode:str="constant") -> None:
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array
    
    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """
    Extracts the spectrogram in dB from the time-series audio signal.
    The arguments are used to determine the STFT parameters.
    """
    def __init__(self, frame_size:int, hop_length:int):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        # apply short-time Fourier transform from librosa
        # result has shape (1+framesize/2, timesteps) -> dropping last item
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]
        # convert complex numbers to absolute values
        spectrogram = np.abs(stft)
        # convert to dB
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormalizer:
    """
    Applikes min-max normalization to the spectrogram.
    """
    def __init__(self, min_val:float=0.0, max_val:float=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, array):
        # normalize to [0, 1]
        norm_array = (array - array.min()) / (array.max() - array.min())
        # scale to desired range
        scaled_array = norm_array * (self.max_val - self.min_val) + self.min_val
        return scaled_array
    
    def denormalize(self, norm_array, original_min, original_max):
        # scale back to [0, 1]
        array = (norm_array - self.min_val) / (self.max_val - self.min_val)
        # scale back to original range
        denorm_array = array * (original_max - original_min) + original_min
        return denorm_array


class Saver:
    """
    Saving features and min-max values.
    """
    def __init__(self, feature_save_dir:str, min_max_values_save_dir:str):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path:str):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path
    
    def save_min_max_values(self, min_max_values:dict):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def _generate_save_path(self, filename:str):
        # split path and filename
        file_name = os.path.split(filename)[1]
        # create savepath
        save_path = os.path.join(self.feature_save_dir, file_name[:-4] + ".npy")
        return save_path
    

class PreprocessingPipeline:
    """
    Preprocessing pipeline for audio files and saving them in a directory.
    1. load the audio data
    2. pad if necessary
    3. extract log spectrogram
    4. normalize
    5. save spectrogram
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None


    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        # every time the loader is initialized, the number of expected samples is set as welöl
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
        

    def preprocess(self, audio_files_directory):

        for root, _, files in os.walk(audio_files_directory):
            for file in files:
                # load
                file_path = os.path.join(root, file)
                self._process_file(file_path)
        # save all stored min-max values
        self.saver.save_min_max_values(self.min_max_values)
                

    def _process_file(self, file_path):
        # load
        signal = self.loader.load(file_path)
        # pad
        if self._do_padding(signal): 
            signal = self._apply_padder(signal)
        # extract
        feature = self.extractor.extract(signal)
        # normalize
        norm_feature = self.normalizer.normalize(feature)
        # save
        save_path = self.saver.save_feature(norm_feature, file_path)
        # store min-max values
        self._store_min_max_values(save_path, feature.min(), feature.max())


    def _do_padding(self, signal):
        if len(signal) < self._num_expected_samples:
            return True


    def _apply_padder(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    

    def _store_min_max_values(self, save_path, min_val, max_val):
        # key of dictionary is save path itself
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74  # in seconds
    SAMPLE_RATE = 22050
    MONO = True


    SPECTROGRAMS_SAVE_DIR = "fsdd/spectrograms/"
    MIN_MAX_VALUES_SAVE_DIR = "fsdd/"
    FILES_DIR = "fsdd/audio/"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normalizer = min_max_normalizer
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.preprocess(FILES_DIR)