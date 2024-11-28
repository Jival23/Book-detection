import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Directories
data_directory = "Consolidated_data"
model_file = "models/dummy.keras"

# Image size and batch size
image_size = (224, 224)
batch_size = 32

# Split the data into 90% training and 10% validation
train_data = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=0.1,  # 10% for validation
    subset="training",     # Load the training set
    seed=123,              # Seed for reproducibility
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=0.1,  # 10% for validation
    subset="validation",   # Load the validation set
    seed=123,              # Seed for reproducibility
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Add rescaling layer separately
rescale_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1.0 / 255)
])

train_data = train_data.map(lambda x, y: (data_augmentation(x), y))
validation_data = validation_data.map(lambda x, y: (rescale_layer(x), y))

# Pre-trained model (MobileNet)
pre_trained_model = MobileNetV2(input_shape=[224, 224] + [3], weights="imagenet", include_top=False)

# Freeze pre-trained layers
for layer in pre_trained_model.layers:
    layer.trainable = False

# Model architecture
inputs = tf.keras.Input(shape=(224, 224, 3))

# Apply rescaling and data augmentation directly to the inputs
# w = rescale_layer(inputs)

# Pass through pre-trained MobileNet
w = pre_trained_model(inputs, training=False)

# Compression and flattening
w = GlobalAveragePooling2D()(w)
w = Flatten()(w)

# Add dropout
# x = keras.layers.Dropout(0.2)(x)

# Output layer
outputs = Dense(2, activation='softmax')(w)

# Create the model
new_model = Model(inputs, outputs)

# Compile the model
new_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ModelCheckpoint(model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
]

# Train the model
r = new_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=50,
    callbacks=callbacks
)


