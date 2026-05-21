""""" Info: Models dosyası, model tanımlamalarını ve eğitimi için kullanılır."""

from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_baseline_model():
    """
    Baseline olarak Logistic Regression modeli oluşturur.
    """
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    return model


def build_sigmoid_autoencoder(input_dim=30, encoding_dim=32,activation="sigmoid"):
    """
    Sigmoid aktivasyonlu autoencoder ve encoder modeli.
    run_autoencoder scripti için merkezi model tanımı.
    Expects float32 input from TensorFlow Keras.
    """
    input_layer = Input(shape=(input_dim,), dtype="float32", name="input_layer")

    encoded_hidden = Dense(128, activation, name="enc_dense_1")(input_layer)
    encoded = Dense(encoding_dim, activation, name="enc_dense_2")(encoded_hidden)

    decoded_hidden = Dense(128, activation, name="dec_dense_1")(encoded)
    decoded = Dense(input_dim, activation, name="dec_output")(decoded_hidden)

    autoencoder = Model(inputs=input_layer, outputs=decoded, name="autoencoder")
    encoder = Model(inputs=input_layer, outputs=encoded, name="encoder")

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        jit_compile=False
    )

    return autoencoder, encoder


def build_latent_classifier(
    input_dim,
    num_classes=2,
    hidden_units=(32, 16),
    dropout_rates=None,
    learning_rate=0.001,
):
    """
    Encoder çıktısı üzerinde çalışan classifier modeli.
    - num_classes == 2: sigmoid + binary_crossentropy
    - num_classes > 2 : softmax + sparse_categorical_crossentropy
    Expects float32 input from encoder.
    """
    classifier_input = Input(shape=(input_dim,), dtype="float32", name="classifier_input")
    x = classifier_input

    if not hidden_units:
        raise ValueError("hidden_units en az bir katman icermeli.")

    if dropout_rates is not None and len(dropout_rates) != len(hidden_units):
        raise ValueError("dropout_rates uzunlugu hidden_units ile ayni olmali.")

    for i, units in enumerate(hidden_units, start=1):
        x = Dense(int(units), activation="relu", name=f"classifier_dense_{i}")(x)
        if dropout_rates is not None and float(dropout_rates[i - 1]) > 0:
            x = Dropout(float(dropout_rates[i - 1]), name=f"classifier_dropout_{i}")(x)

    if num_classes == 2:
        classifier_output = Dense(1, activation="sigmoid", name="classifier_output")(x)
        loss = "binary_crossentropy"
    else:
        classifier_output = Dense(num_classes, activation="softmax", name="classifier_output")(x)
        loss = "sparse_categorical_crossentropy"

    classifier = Model(inputs=classifier_input, outputs=classifier_output, name="latent_classifier")
    classifier.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
        jit_compile=False
    )

    return classifier


def build_latent_regressor(
    input_dim,
    hidden_units=(32, 16),
    dropout_rates=None,
    learning_rate=0.001,
):
    """
    Encoder çıktısı üzerinde çalışan regression modeli.
    Sürekli hedef değerler için linear çıkış + MSE loss kullanır.
    """
    regressor_input = Input(shape=(input_dim,), dtype="float32", name="regressor_input")
    x = regressor_input

    if not hidden_units:
        raise ValueError("hidden_units en az bir katman icermeli.")

    if dropout_rates is not None and len(dropout_rates) != len(hidden_units):
        raise ValueError("dropout_rates uzunlugu hidden_units ile ayni olmali.")

    for i, units in enumerate(hidden_units, start=1):
        x = Dense(int(units), activation="relu", name=f"regressor_dense_{i}")(x)
        if dropout_rates is not None and float(dropout_rates[i - 1]) > 0:
            x = Dropout(float(dropout_rates[i - 1]), name=f"regressor_dropout_{i}")(x)

    regressor_output = Dense(1, activation="linear", name="regressor_output")(x)
    regressor = Model(inputs=regressor_input, outputs=regressor_output, name="latent_regressor")
    regressor.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
        jit_compile=False
    )

    return regressor


__all__ = [
    "build_baseline_model",
    "build_cnn",
    "build_sigmoid_autoencoder",
    "build_latent_classifier",
    "build_latent_regressor",
]
