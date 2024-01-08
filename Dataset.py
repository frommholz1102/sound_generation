# IMPORTS
import torch
import torchaudio as ta
from torch.utils.data import Dataset
from torchaudio_augmentations import *
from utils import *

print('Backend: ', ta.list_audio_backends())

class FSDD(Dataset):
    """
    Dataset class for Free Spoken Digit Dataset (FSDD).
    """
    
    def __init__(self, 
                    data_folder: str, 
                    sample_rate: int, 
                    duration: float,
                    transforms = None):
        
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_expected_samples = int(self.sample_rate * self.duration)
        self.transforms = transforms
        self.audio_paths = get_wav_list(data_folder)
        print(f"Found {len(self.audio_paths)} audio files in {data_folder}")

        # define preprocessing classes
        self.loader = Loader(sample_rate=self.sample_rate, duration=self.duration)
        self.padder = Padder()
        self.extractor = LogMelSpectrogramExtractor(sample_rate=self.sample_rate)
        self.normalizer = MinMaxNormalizer()

    def __len__(self):
        """
        Returns the number of audio files in the dataset.
        """
        return len(self.audio_paths)


    def __getitem__(self, idx):
        """
        Returns the spectrogram and target spectrogram (same) of the audio file at the given index.
        A number of preprecoessing steps are applied to the audio file:
            1. load the audio data
            2. pad if necessary
            3. extract log spectrogram (using STFT)
            4. normalize

        Args:
            idx (int): Index of the audio file within the dataset

        Returns:
            tuple: (tensor, tensor) input and target spectrogram (same contents) for generative model
        """

        def _apply_padding(signal):
            """
            Checking if padding is required, calculating number of missing items (based on difference
            between expected number of samples and actual number of samples in the signal tensor) 
            and applying padding.
            This function is implemented separately from Padder class since it has no access to the signal.
            """
            if signal.shape[1] < self.num_expected_samples:
                num_missing_items = self.num_expected_samples - signal.shape[1]
                padded_signal = self.padder.right_pad(signal, num_missing_items)
                print(f"Applied padding to signal with old shape {signal.shape}, new shape {padded_signal.shape}")
            
            return padded_signal


        # load wav and resample to target sample rate
        audio_tensor = self.loader.load(self.audio_paths[idx])

        # apply padding if necessary
        audio_tensor = _apply_padding(audio_tensor)

        # convert to LogMelSpectrogram
        log_mel_spectrogram = self.extractor.extract(audio_tensor)

        # normalize
        norm_log_mel_spectrogram = self.normalizer.normalize(log_mel_spectrogram)

        # apply augmentation
        if self.transforms:
            norm_log_mel_spectrogram = self.transforms(norm_log_mel_spectrogram)
        
        return norm_log_mel_spectrogram, norm_log_mel_spectrogram
    

    def collate_function(batch):
        """
        This function is used by the dataloader to stack the input and target tensors.

        Args:
            batch (list): list of tuples (input, target)

        Returns:
            torch.tensor: stacked input tensor
            torch.tensor: stacked target tensor
        """

        inputs, targets = [], []
        for input, target in batch:
            inputs += [input]
            targets += [target]

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        return inputs, targets

    

        


class Loader:
    """
    Class for loading audio files.
    """
    def __init__(self, sample_rate: int = 22050, duration: float = 0.74,):
        self.sample_rate = sample_rate
        self.duration = duration

    def load(self, file_path: str):
        """
        Loads an audio file and returns the signal as a torch tensor.

        Args:
            file_path (str): Path to the audio file

        Returns:
            torch.tensor: Audio signal
        """
        # load audio file
        signal, sr = ta.load(file_path, format="wav")
        # resample if necessary
        if sr != self.sample_rate:
            signal = ta.transforms.Resample(sr, self.sample_rate)(signal)
            print(f"Resampled file {file_path} from {sr} Hz to {self.sample_rate} Hz")
            print(f"Signal shape: {signal.shape}")

        return signal
    

class Padder:
    """
    Responsible for applying padding to the audio signal if the signal is too short.
    Modes of padding are: constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric, wrap
    """
    def __init__(self, mode:str="constant"):
        self.mode = mode
        
    def right_pad(self, signal, num_missing_items):
        padded_signal = torch.nn.functional.pad(signal, (0, num_missing_items), mode=self.mode)
        return padded_signal
    

class LogMelSpectrogramExtractor:
    """
    Extracts the log Mel spectrogram from the time-series audio signal.
    The arguments are used to determine the STFT parameters.
    """
    def __init__(self, sample_rate, n_fft: int = 512, hop_length: int = 256, n_mels: int = 64):
        self.sample_rate = sample_rate  
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.sepc_transform = ta.transforms.Spectrogram(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            power=1.0)
        self.mel_spec_transform = ta.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels, 
            power=1.0)
        

    def extract(self, signal, mel: bool = True):
        """
        Extracts the log Mel spectrogram from the time-series audio signal.

        Args:
            signal (torch.tensor): Audio signal
            mel (bool, optional): Whether to calculate a Mel spectrogram. Defaults to True.

        Returns:
            torch.tensor: Log Mel spectrogram [channel, frequency_bins, time_bins]
        """
        # apply short-time Fourier transform from torchaudio
        # result has shape (1+framesize/2, timesteps) -> dropping last item
        
        # calculate normal or mel spectrogram
        # convert to dB
        if mel:
            spectrogram = self.mel_spec_transform(signal)
            log_spectrogram = ta.transforms.AmplitudeToDB()(spectrogram)
        else:
            spectrogram = self.sepc_transform(signal)
            log_spectrogram = ta.transforms.AmplitudeToDB()(spectrogram)
        
        print(f"Log mel spectrogram shape: {log_spectrogram.shape}")

        return log_spectrogram
    

class MinMaxNormalizer:
    """
    Applies min-max normalization to the spectrogram. For denormalization the original min and max values
    are stored in the same location as the audio data.
    """
    def __init__(self, min_val:float=0.0, max_val:float=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, signal):
        # normalize to [0, 1]
        norm_signal = (signal - signal.min()) / (signal.max() - signal.min())
        # scale to desired range
        scaled_signal = norm_signal * (self.max_val - self.min_val) + self.min_val
        print("Signal with min/max values: ", signal.min(), signal.max())
        print("Normalized signal with min/max values: ", scaled_signal.min(), scaled_signal.max())
        print("Normalized signal shape: ", scaled_signal.shape)
        return scaled_signal
    
    def denormalize(self, norm_signal, original_min, original_max):
        # scale back to [0, 1]
        signal = (norm_signal - self.min_val) / (self.max_val - self.min_val)
        # scale back to original range
        denorm_signal = signal * (original_max - original_min) + original_min
        return denorm_signal


if __name__ == "__main__":

    test_dataset = FSDD(data_folder="fsdd/audio", sample_rate=22050, duration=0.74)
    test_item = test_dataset[0]