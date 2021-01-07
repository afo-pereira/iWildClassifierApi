import numpy as np
import os
import shutil

from flask import Flask, render_template, request, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_bootstrap import Bootstrap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as k
from werkzeug.utils import secure_filename, redirect
from wtforms import SubmitField

base_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(base_dir, 'static/uploads')

bootstrap = Bootstrap(app)

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


# custom metrics
def recall_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + k.epsilon()))


# Load the model:
cnn_model = load_model('NoEmpty_LowRes_OverSampling_DataAugmentation.h5',
                       custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})
CLASS_INDICES = {0: 'deer',
                 1: 'fox',
                 2: 'coyote',
                 3: 'raccoon',
                 4: 'skunk',
                 5: 'bobcat',
                 6: 'cat',
                 7: 'dog',
                 8: 'opossum',
                 9: 'mountain_lion',
                 10: 'squirrel',
                 11: 'rodent',
                 12: 'rabbit'}


# Form where image will be uploaded:
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file to upload!')])
    submit = SubmitField('Get Prediction')


@app.route('/', methods=['GET'])
def index():
    _delete_image()
    return render_template('home.html', form=UploadForm())


@app.route('/prediction/', methods=['POST'])
def prediction():
    # Saving file to folder
    file = request.files['photo']
    try:
        filename = secure_filename(file.filename)
        file.save(os.path.join('static/uploads', filename))
        results = return_prediction(filename=filename)
    except:
        return render_template('404.html')
    return render_template('prediction.html', results=results, filename=filename)


@app.route('/context', methods=['GET'])
def project_context():
    return render_template('projectContext.html')


@app.route("/display_image/<filename>", methods=['GET'])
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def return_prediction(filename):
    input_image_matrix = _image_process(filename)
    score = cnn_model.predict(input_image_matrix)
    class_index = np.argmax(score, axis=-1)

    return CLASS_INDICES[class_index[0]], score


def _image_process(filename):
    img = image.load_img('static/uploads/' + filename, target_size=(96, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    input_matrix = np.vstack([x])
    input_matrix /= 255.
    return input_matrix


def _delete_image():
    folder = 'static/uploads/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    app.run(debug=True)
