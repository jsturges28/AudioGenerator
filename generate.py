import numpy as np
import os
import random
import soundfile as sf
import pickle

from preprocessing import MinMaxNormalizer
from soundgenerator import SoundGenerator
from unet import UNET
from train import SPECTROGRAMS_PATH

NUM_SAMPLES = 5
HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
SAVE_DIR_MODIFIED = "samples/modified/"
MIN_MAX_VALUES_PATH = "./min_max_values.pkl"


'''
generate.py generates audio from randomly selected spectrograms
and saves them into .wav format
'''

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
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    #print(file_paths)
    #print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values

def create_modified_specs(sampled_specs, sampled_min_max_values, min_mod=-10, max_mod=10):
    normalizer = MinMaxNormalizer(0, 1)

    # denormalize the selected spectrograms
    denormalized_specs = np.array([normalizer.denormalize(spec, values["min"], values["max"]) 
                                   for spec, values in zip(sampled_specs, sampled_min_max_values)])

    # apply the random modifications
    modified_denormalized_specs = denormalized_specs + np.random.uniform(min_mod, max_mod, denormalized_specs.shape)

    # compute new min and max values
    new_min_values = np.min(modified_denormalized_specs, axis=(1, 2))
    new_max_values = np.max(modified_denormalized_specs, axis=(1, 2))

    # normalize the modified spectrograms
    modified_sample_specs = np.array([normalizer.normalize(spec) 
                                      for spec in modified_denormalized_specs])
    
    # create a dictionary for new min max values
    new_min_max_values = [{"min": min_val, "max": max_val} 
                          for min_val, max_val in zip(new_min_values, new_max_values)]

    return modified_sample_specs, new_min_max_values

def save_signals(signals, save_dir, identifier, extension=".wav", sample_rate=22050):
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for i, signal, in enumerate(signals):
        save_path = os.path.join(save_dir, "{}_{}{}".format(i, identifier, extension))
        sf.write(save_path, signal, sample_rate)


if __name__ == '__main__':
    #print(select_spectrograms(SPECTROGRAM_PATH, 5))

    unet = UNET.load("model")
    sound_generator = SoundGenerator(unet, HOP_LENGTH)

    # load spectrograms and min max values
    with open(MIN_MAX_VALUES_PATH, 'rb') as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    sampled_specs, sampled_min_max_values = select_spectrograms(specs, 
                                                                file_paths, 
                                                                min_max_values,
                                                                NUM_SAMPLES)

    modified_sample_specs, modified_min_max_values = create_modified_specs(sampled_specs=sampled_specs,
                                                  sampled_min_max_values=sampled_min_max_values)

    unet_signals = sound_generator.generate_unet_signals(sampled_specs,
                                                         sampled_min_max_values)
    
    modified_unet_signals = sound_generator.generate_unet_signals(modified_sample_specs,
                                                                  modified_min_max_values)
    
    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs,
                                                                     sampled_min_max_values)
    
    # save audio signals
    save_signals(unet_signals, SAVE_DIR_GENERATED, "unet_generated")
    save_signals(modified_unet_signals, SAVE_DIR_MODIFIED, "unet_modified")
    save_signals(original_signals, SAVE_DIR_ORIGINAL, "original")


