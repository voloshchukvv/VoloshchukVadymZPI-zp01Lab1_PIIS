import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt
import tensorflow as tf

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

model = models.load_model("finish_model")

def predict():
    x = 'data/audio/stop/0fa1e7a9_nohash_1.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000, )
    x = tf.squeeze(x, axis=-1)
    x = get_spectrogram(x)
    x = x[tf.newaxis, ...]

    prediction = model(x)
    label_prediction = np.argmax(prediction, axis=1)
    print(commands[label_prediction[0]])
    x_labels = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
    plt.bar(x_labels, tf.nn.softmax(prediction[0]))
    plt.show()


if __name__ == "__main__":
    predict()

