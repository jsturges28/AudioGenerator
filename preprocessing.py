import os
import numpy as np 
import librosa
import pickle

'''
Preprocessing file -- used for converting .wav and .mp4 files
into mel spectrograms that the model can ingest

Steps:

1. Load file

2. Pad signal

3. Extract mel spectrogram from signal

4. Normalize spectrogram

5. Save normalized spectrogram
'''

class Loader:
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, filepath):
        signal = librosa.load(filepath, 
                            sr=self.sample_rate, 
                            duration=self.duration, 
                            mono=self.mono)[0]
        
        return signal
    
class Padder:

    def __init__(self, mode='constant'):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, 
                              (num_missing_items, 0),
                              mode=self.mode)  # np.pad(arr, (left, right), mode='constant')
        
        return padded_array
    
    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, 
                              (0, num_missing_items),
                              mode=self.mode)  # np.pad(arr, (left, right), mode='constant')
        
        return padded_array
    
class MelSpectrogramExtractor:
    def __init__(self, sampling_rate, hop_length):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def convert_to_mel_spectrogram(self, signal):
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, 
                                                         sr=self.sampling_rate,
                                                         hop_length=self.hop_length)
        
        return mel_spectrogram
        
class MinMaxNormalizer:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min()) # makes the largest value in norm_array (max) = 1 and the smallest = 0
        norm_array = norm_array * (self.max - self.min) + self.min # makes the miniumum value self.min and the maximum value self.max
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min) # reverses the operations form normalize
        array = array * (original_max - original_min) + original_min
        return array
    
class Saver:
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, filepath): # save the spectrograms as .npy extension
        save_path = self._generate_save_path(filepath)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        np.save(save_path, feature) 
        return save_path
    
    def save_min_max_values(self, min_max_values): # pickle the min_max_values
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path

'''
Put it all together with the PreprocessingPipeline class
1. Create loader 
'''

class PreprocessingPipeline:

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values = {}

        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):       # Ties into Loader class (which returns a librosa signal)
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir): # path_to_audio_files_dir, subdirectories, files
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file: {file}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.convert_to_mel_spectrogram(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_min_max_value(self, save_path, min, max):
        self.min_max_values[save_path] = {"min": min,
                                          "max": max}
        

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74 # seconds
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = os.path.abspath("C:/Users/stur8980/Documents/GitHub/AudioGenerator/mel_spectrograms/")
    MIN_MAX_VALUES_SAVE_DIR = os.path.abspath("C:/Users/stur8980/Documents/GitHub/AudioGenerator/")
    FILES_DIR = "./recordings"

    # instantiate the objects

    preprocessing_pipeline = PreprocessingPipeline()

    preprocessing_pipeline.loader = Loader(SAMPLE_RATE, DURATION, MONO)
    preprocessing_pipeline.padder = Padder()
    preprocessing_pipeline.extractor = MelSpectrogramExtractor(SAMPLE_RATE, HOP_LENGTH)
    preprocessing_pipeline.normalizer = MinMaxNormalizer(0, 1)
    preprocessing_pipeline.saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline.process(FILES_DIR)
        
            
