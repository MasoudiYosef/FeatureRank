"""Neural-network model definitions used by FeatureRank experiments."""

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.config import (
    CLASSIFIER_DROPOUT_RATES,
    CLASSIFIER_HIDDEN_UNITS,
    CLASSIFIER_LEARNING_RATE,
)


def _add_hidden_layers(input_tensor, layer_prefix: str):
    """Add the configured dense/dropout layers and return the last tensor."""
    if not CLASSIFIER_HIDDEN_UNITS:
        raise ValueError("hidden_units en az bir katman icermeli.")
    if CLASSIFIER_DROPOUT_RATES is not None and len(CLASSIFIER_DROPOUT_RATES) != len(
        CLASSIFIER_HIDDEN_UNITS
    ):
        raise ValueError("dropout_rates uzunlugu hidden_units ile ayni olmali.")

    output = input_tensor
    for index, units in enumerate(CLASSIFIER_HIDDEN_UNITS, start=1):
        output = Dense(
            int(units),
            activation="relu",
            name=f"{layer_prefix}_dense_{index}",
        )(output)
        if CLASSIFIER_DROPOUT_RATES is not None:
            rate = float(CLASSIFIER_DROPOUT_RATES[index - 1])
            if rate > 0:
                output = Dropout(rate, name=f"{layer_prefix}_dropout_{index}")(output)
    return output


def build_sigmoid_autoencoder(
    input_dim: int = 30,
    encoding_dim: int = 32,
    activation: str = "sigmoid",
) -> tuple[Model, Model]:
    """Build the autoencoder used for FeatureRank and its encoder view."""
    input_layer = Input(shape=(input_dim,), dtype="float32", name="input_layer")

    encoded_hidden = Dense(128, activation, name="enc_dense_1")(input_layer)
    encoded = Dense(encoding_dim, activation, name="enc_dense_2")(encoded_hidden)

    decoded_hidden = Dense(128, activation, name="dec_dense_1")(encoded)
    decoded = Dense(input_dim, activation, name="dec_output")(decoded_hidden)

    autoencoder = Model(inputs=input_layer, outputs=decoded, name="autoencoder")
    encoder = Model(inputs=input_layer, outputs=encoded, name="encoder")

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse", jit_compile=False)

    return autoencoder, encoder


def build_latent_classifier(
    input_dim: int,
    num_classes: int = 2,
) -> Model:
    """Build a classifier that consumes encoded feature vectors."""
    classifier_input = Input(shape=(input_dim,), dtype="float32", name="classifier_input")
    x = _add_hidden_layers(classifier_input, "classifier")

    if num_classes == 2:
        classifier_output = Dense(1, activation="sigmoid", name="classifier_output")(x)
        loss = "binary_crossentropy"
    else:
        classifier_output = Dense(num_classes, activation="softmax", name="classifier_output")(x)
        loss = "sparse_categorical_crossentropy"

    classifier = Model(inputs=classifier_input, outputs=classifier_output, name="latent_classifier")
    classifier.compile(
        optimizer=Adam(learning_rate=CLASSIFIER_LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
        jit_compile=False,
    )

    return classifier


def build_latent_regressor(
    input_dim: int,
) -> Model:
    """Build a regressor that consumes encoded feature vectors."""
    regressor_input = Input(shape=(input_dim,), dtype="float32", name="regressor_input")
    x = _add_hidden_layers(regressor_input, "regressor")

    regressor_output = Dense(1, activation="linear", name="regressor_output")(x)
    regressor = Model(inputs=regressor_input, outputs=regressor_output, name="latent_regressor")
    regressor.compile(
        optimizer=Adam(learning_rate=CLASSIFIER_LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
        jit_compile=False,
    )

    return regressor


__all__ = [
    "build_sigmoid_autoencoder",
    "build_latent_classifier",
    "build_latent_regressor",
]
