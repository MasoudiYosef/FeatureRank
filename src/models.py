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


#CNN modeli oluşturma fonksiyonu
def build_cnn(input_shape, num_classes=2):
    """
    Basit ve açıklanabilir CNN modeli.
    İlk Conv1D katmanı kernel_size=1 olduğu için feature ranking daha yorumlanabilir hale gelir.
    
    64 filtre → 64 farklı pattern öğrenilecek (F1, F2, F3... birlikte nasıl etkili  1.(F1-F2-F3) 2. (F1-F2-F4) gibi)
    kernel_size=3 → her seferinde 3 feature’a birlikte bakıyor. F1-F2-F3 gibi.
    padding="same" → output boyutu input ile aynı kalır
    Activation("relu")(x) => negatifleri sıfırlar,pozitifleri bırakır.Gürültüyü azaltır. [-2, -1, 0, 3, 5] → [0, 0, 0, 3, 5]
    1.Pattern çıkarır (Conv1D)
    2. Stabil hale getirir (BatchNorm)
    3. Önemli sinyali bırakır (ReLU)

    GlobalAveragePooling1D()(x)
        Conv katmanlarından gelen çıktıyı özetler.
        Her filtrenin genel aktivasyonunu tek sayıya indirger.
    Dropout(0.2)(x)
        Eğitim sırasında bazı nöronları overfitting azaltmak için rastgele kapatır.

    Activation("relu")(x):
        - Gereksiz zayıf/negatif sinyalleri temizler
        - Modeli doğrusal olmaktan çıkarır
        - Daha karmaşık örüntüler öğrenmesini sağlar
    """
    inputs = Input(shape=input_shape, name="input_layer")

    # Feature ranking için en önemli katman:
    # İlk conv katmanı -> her feature için öğrenilen ağırlıkları buradan çıkaracağız
    x = Conv1D(
        filters=32,
        kernel_size=input_shape[0],  # Her seferinde tüm feature’lara bakacak şekilde kernel_size=feature sayısı (Her bakışta tüm feature’ları birlikte değerlendirir ve bir şeyler öğrenir.Her filtre, tüm feature seti üzerinde bir ağırlık dizisi öğreniyor.)
        padding="same",
        name="feature_conv"
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(filters=64, kernel_size=input_shape[0], padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)   

    x = Conv1D(filters=64, kernel_size=input_shape[0], padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)

    if num_classes == 2:
        outputs = Dense(1, activation="sigmoid", name="output_layer")(x)
    else:
        outputs = Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="feature_ranking_cnn")
    return model


def build_autoencoder(input_dim=30, encoding_dim=8):
    """
    Input→Encoder→Bottleneck→Decoder→Reconstruction
    Tablosal veri için dense autoencoder modeli oluşturur.
    """
    input_layer = Input(shape=(input_dim,), name="input_layer")

    # Encoder
    encoded = Dense(16, activation="relu", name="encoder_dense_1")(input_layer)
    bottleneck = Dense(encoding_dim, activation="relu", name="bottleneck")(encoded)

    # Decoder
    decoded = Dense(16, activation="relu", name="decoder_dense_1")(bottleneck)
    output_layer = Dense(input_dim, activation="linear", name="output_layer")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
    encoder = Model(inputs=input_layer, outputs=bottleneck, name="encoder")

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    return autoencoder, encoder


def build_sigmoid_autoencoder(input_dim=30, encoding_dim=8,activation="sigmoid"):
    """
    Sigmoid aktivasyonlu autoencoder ve encoder modeli.
    run_autoencoder scripti için merkezi model tanımı.
    """
    input_layer = Input(shape=(input_dim,), name="input_layer")

    encoded_hidden = Dense(16, activation, name="enc_dense_1")(input_layer)
    encoded = Dense(encoding_dim, activation, name="enc_dense_2")(encoded_hidden)

    decoded_hidden = Dense(16, activation, name="dec_dense_1")(encoded)
    decoded = Dense(input_dim, activation, name="dec_output")(decoded_hidden)

    autoencoder = Model(inputs=input_layer, outputs=decoded, name="autoencoder")
    encoder = Model(inputs=input_layer, outputs=encoded, name="encoder")

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
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
    """
    classifier_input = Input(shape=(input_dim,), name="classifier_input")
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
        metrics=["accuracy"]
    )

    return classifier


__all__ = [
    "build_baseline_model",
    "build_cnn",
    "build_autoencoder",
    "build_sigmoid_autoencoder",
    "build_latent_classifier",
]