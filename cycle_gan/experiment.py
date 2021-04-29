import FaceDetector as fd
from options.test_options import TestOptions
from models import create_model
import torch
import numpy as np
from PIL import Image
from util.util import tensor2im, save_image


def cycle_gan_model(model):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.name = model
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()            # regular setup: load and print networks; create schedulers
    
    return model


def image_to_cycleGAN_data(image):
    data = {"A": None, "A_paths": None}
    image = np.array([image])
    image = image.transpose([0,3,1,2])
    data['A'] = torch.Tensor(image)
    return data


if __name__ == '__main__':

    face_detect = fd.FaceDetector('faceDetectionModel/deploy.prototxt.txt', 'faceDetectionModel/opencv_face_detector.caffemodel')
    face = face_detect.detect_face_from_image('me.jpg')
    face_data = image_to_cycleGAN_data(face)
    
    image = np.asarray(Image.open('yosemite.jpg'))
    image_data = image_to_cycleGAN_data(image)

    style_models_names = ['style_monet_pretrained',
                    'style_vangogh_pretrained',
                    'style_ukiyoe_pretrained',
                    'style_cezanne_pretrained']

    style_models = []

    for smn in style_models_names:
        style_models.append(cycle_gan_model(smn))

    for (model, name) in zip(style_models, style_models_names):

        model.set_input(face_data)
        model.test()
        result_image = model.get_current_visuals()['fake']
        save_image(tensor2im(result_image), f'me_{name}.jpg')

        model.set_input(image_data)
        model.test()
        result_image = model.get_current_visuals()['fake']
        save_image(tensor2im(result_image), f'yosemite_{name}.jpg')

    print('experiment complete')
