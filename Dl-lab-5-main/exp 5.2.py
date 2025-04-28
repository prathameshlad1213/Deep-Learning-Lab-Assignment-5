import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import random
import sys

# Load and preprocess text
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Create mapping
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Create training examples
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Split input-target pairs
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Model architecture
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = Sequential([
        Embedding(vocab_size, embedding_dim), # Remove batch_input_shape
        LSTM(rnn_units, return_sequences=True, stateful=True,
             recurrent_initializer='glorot_uniform'),
        Dropout(0.2),
        LSTM(rnn_units, return_sequences=True, stateful=True,
             recurrent_initializer='glorot_uniform'),
        Dropout(0.2),
        Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

# Custom callback for text generation during training
def on_epoch_end(epoch, logs):
    print(f'\nGenerating text after epoch {epoch+1}')
    start_string = "ROMEO: "
    num_generate = 300
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 0.7
    
    # Reset the states of the LSTM layers individually
    for layer in model.layers:
        if isinstance(layer, LSTM):  # Check if the layer is an LSTM
            layer.reset_states()
            
    for i in range(num_generate):
        predictions = model(input_eval)
        # Reshape the predictions before squeezing
        predictions = tf.reshape(predictions, (1, -1, vocab_size))  
        predictions = tf.squeeze(predictions, 0) 
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    
    print(start_string + ''.join(text_generated))


# Train model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
history = model.fit(dataset, epochs=30, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
