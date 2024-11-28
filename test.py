import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np

test_individual_images = False
validation_file = "validation"
model_file = "models/book_detection_v5.keras"
model = tf.keras.models.load_model(model_file)

def prepareImage(pathForImage):
    """Prepares an image to be used with the keras model"""
    image = load_img(pathForImage, target_size=(224, 224))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.
    return imgResult

if not test_individual_images:
    rescale_data = tf.keras.layers.Rescaling(1.0 / 255)

    validation_data = tf.keras.utils.image_dataset_from_directory(
        validation_file,
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical'
    )

    validation_data = validation_data.map(lambda x, y: (rescale_data(x), y))

    test_loss, test_acc = model.evaluate(validation_data)
else:
    classes = ['Book', 'Not_a_book']
    testImagePath = "validation/Book/20240927_201739.jpg"

    imgForModel = prepareImage(testImagePath)

    resultsArray = model(imgForModel)
    print(resultsArray)

    answer = np.argmax(resultsArray, axis=1)
    print(answer)

    index = answer[0]
    print(classes[index])
