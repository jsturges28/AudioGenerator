from preprocessing import MinMaxNormalizer
import librosa

class Soundgenerator:
    '''
    Generate audio from mel spectrograms
    '''

    def __init__(self, model, hop_length):
        self.model = model
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormalizer(0,1)

    def generate_unet_signals(self, spectrograms, min_max_values):
        generated_spectrograms = self.model.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []

        for spectrogram, min_max_value in zip(spectrograms, min_max_values):

            # convert generated spectrogram back to 2D
            new_spectrogram = spectrogram[:, :, 0]
            # denormalize spectrogram
            denorm_spectrogram = self._min_max_normalizer.denormalize(new_spectrogram, min_max_value["min"], min_max_value["max"])
            # inverse mel function
            spec = librosa.feature.inverse.mel_to_stft(denorm_spectrogram)
            signal = librosa.feature.griffinlim(spec, hop_length=self.hop_length)
            signals.append(signal)

        return signals