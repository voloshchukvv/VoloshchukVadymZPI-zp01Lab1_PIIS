import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_audio = 'data/audio'

if __name__ == '__main__':
    commands = np.array(tf.io.gfile.listdir(str('data/audio')))
    commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
    print('Commands:', commands)

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(directory=data_audio,batch_size=64,validation_split=0.2,seed=0,output_sequence_length=16000,
        subset='both')

    label_names = np.array(train_ds.class_names)
    print("label names:", label_names)

    train_ds.element_spec

    def squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    def get_spectrogram(waveform):
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def make_spec_ds(ds):
        return ds.map(
            map_func=lambda audio, label: (get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)

    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break

    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

    input_shape = example_spectrograms.shape[1:]
    num_labels = len(label_names)
    norm_layer = layers.Normalization()
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 10
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    x = tf.io.read_file(str('data/audio/stop/1b755c65_nohash_1.wav'))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000, )
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_spectrogram(x)
    x = x[tf.newaxis]

    prediction = model(x)
    x_labels = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
    plt.bar(x_labels, tf.nn.softmax(prediction[0]))
    plt.show()