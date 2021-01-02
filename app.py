import numpy as np
import os

from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from wtforms import SubmitField


base_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(base_dir, 'uploads')

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

# Load the model:
cnn_model = load_model('cats_and_dogs_model9.h5')
CLASS_INDICES = {0: 'cat', 1: 'dog'}


# Form where image will be uploaded:
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file to upload!')])
    submit = SubmitField('Get Prediction')


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html', form=UploadForm())


@app.route('/prediction/', methods=['POST'])
def prediction():
    # Saving file to folder
    file = request.files['photo']
    filename = secure_filename(file.filename)
    file.save(os.path.join('uploads', filename))

    results = return_prediction(filename=filename)
    return render_template('prediction.html', results=results)


def return_prediction(filename):
    input_image_matrix = _image_process(filename)
    score = cnn_model.predict(input_image_matrix)
    class_index = cnn_model.predict_classes(input_image_matrix, batch_size=1)
    
    return CLASS_INDICES[class_index[0]], score


def _image_process(filename):
    img = image.load_img('uploads/' + filename, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    input_matrix = np.vstack([x])
    input_matrix /= 255.
    return input_matrix


if __name__ == '__main__':
    app.run(debug=True)
