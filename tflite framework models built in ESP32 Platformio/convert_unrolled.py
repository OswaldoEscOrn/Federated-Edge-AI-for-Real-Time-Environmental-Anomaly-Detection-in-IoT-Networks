import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization

# Original model path
input_h5 = r"D:\Oswaldo's surf project\DR O's database\models\lstm_autoencoder_hybrid100k_data.h5"
# Output TFLite file path
output_tflite = r"D:\PlatformIO\Projects\attempt3\lstm_100k_model_unrolled.tflite"

# Model parameters (consistent with training script)
TIMESTEPS = 24
N_FEATURES = 4
LSTM_UNITS = 128
ENCODING_DIM = 32

# Build model with unroll=True (identical structure to original, all LSTM layers add unroll=True)
def build_unrolled_lstm_autoencoder(sequence_length, n_features):
    inputs = Input(shape=(sequence_length, n_features), name='lstm_input')

    # Encoder
    encoded = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                   name='encoder_lstm1', unroll=True)(inputs)  # Added unroll=True
    encoded = BatchNormalization(name='encoder_bn1')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout1')(encoded)

    encoded = LSTM(LSTM_UNITS // 2, activation='tanh', return_sequences=True,
                   name='encoder_lstm2', unroll=True)(encoded)
    encoded = BatchNormalization(name='encoder_bn2')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout2')(encoded)

    encoded = LSTM(LSTM_UNITS // 4, activation='tanh', return_sequences=False,
                   name='encoder_lstm3', unroll=True)(encoded)

    # Bottleneck layer
    encoded = Dense(ENCODING_DIM, activation='tanh', name='bottleneck')(encoded)

    # Decoder
    decoded = RepeatVector(sequence_length, name='repeat_vector')(encoded)

    decoded = LSTM(LSTM_UNITS // 4, activation='tanh', return_sequences=True,
                   name='decoder_lstm1', unroll=True)(decoded)
    decoded = BatchNormalization(name='decoder_bn1')(decoded)
    decoded = Dropout(0.2, name='decoder_dropout1')(decoded)

    decoded = LSTM(LSTM_UNITS // 2, activation='tanh', return_sequences=True,
                   name='decoder_lstm2', unroll=True)(decoded)
    decoded = BatchNormalization(name='decoder_bn2')(decoded)
    decoded = Dropout(0.2, name='encoder_dropout2')(decoded)   # Note: name mismatch, but kept as original

    decoded = LSTM(LSTM_UNITS, activation='tanh', return_sequences=True,
                   name='decoder_lstm3', unroll=True)(decoded)
    decoded = BatchNormalization(name='decoder_bn3')(decoded)

    outputs = TimeDistributed(Dense(n_features, activation='linear'),
                              name='output')(decoded)

    autoencoder = Model(inputs, outputs, name='lstm_autoencoder')
    return autoencoder

print("Building new model with unroll=True...")
new_model = build_unrolled_lstm_autoencoder(TIMESTEPS, N_FEATURES)

print(f"Loading original weights: {input_h5}")
# Load weights using load_weights (requires exact layer name matching, which is satisfied)
new_model.load_weights(input_h5)

print("Converting...")
# Define inference function with fixed batch size=1
@tf.function
def predict(x):
    return new_model(x, training=False)

concrete_func = predict.get_concrete_function(tf.TensorSpec([1, TIMESTEPS, N_FEATURES], tf.float32))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_lower_tensor_list_ops = True
tflite_model = converter.convert()

# Save TFLite model
os.makedirs(os.path.dirname(output_tflite), exist_ok=True)
with open(output_tflite, 'wb') as f:
    f.write(tflite_model)

print(f"Conversion completed, model saved to: {output_tflite}")
