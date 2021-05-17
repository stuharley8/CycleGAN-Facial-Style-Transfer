import os
import datetime
from argparse import ArgumentParser

from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from packages.cyclegan import CycleGAN
from packages.cyclegan.util.util import tensor2im, save_image
from packages.face_detection import FaceDetector
from packages.image_util import image_util

# Configure the flask app
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads')

# Configure the photo uploads
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

# Configure the CycleGAN models
monet_model = CycleGAN.CycleGAN('style_monet_pretrained')
vangogh_model = CycleGAN.CycleGAN('style_vangogh_pretrained')
ukiyoe_model = CycleGAN.CycleGAN('style_ukiyoe_pretrained')
cezanne_model = CycleGAN.CycleGAN('style_cezanne_pretrained')

# Define the form for uploading content to the application
class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')

# Define the base page to serve
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        date = datetime.datetime.now()
        date_string = date.strftime('%Y%m%d-%H%M%S') + '.jpg'
        filename = photos.save(form.photo.data, name=date_string)
        file_url = photos.url(filename)
    
        # Get output file
        fname = filename.split(".")
        out_fname = fname[0] + "_updated"
        out_fname_full = out_fname + "." + fname[1]
        out_url = str(file_url).replace(filename, out_fname_full)

        # Save the file path respective to the package location
        pkg_file_path = './uploads/' + filename
        pkg_file_out_fname = './uploads/' + out_fname + '.jpg'

        # Detect face
        faceDetector = FaceDetector.FaceDetector('./packages/face_detection/model/model.txt', './packages/face_detection/model/weights.caffemodel')
        face_selection = faceDetector.detect_face_from_image(pkg_file_path)

        # Produce the output images from the models
        ukiyoe_model.set_model_input(face_selection)
        produced_visuals = ukiyoe_model.run_inference()

        # Save the output image
        np_img = tensor2im(produced_visuals['fake'])
        save_image(np_img, pkg_file_out_fname)
    else:
        file_url = None
        out_url = None
    return render_template('index.html', form=form, file_url=file_url, out_url=out_url)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--containerize_build', required=False, type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    args = parser.parse_args()
    containerize_build = args.containerize_build
    
    if containerize_build == 1:
        app.run(host='0.0.0.0')
    else:
        app.run()