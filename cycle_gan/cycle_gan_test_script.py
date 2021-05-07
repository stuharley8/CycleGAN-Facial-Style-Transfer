from options.test_options import TestOptions
from models import create_model
import torch
import numpy as np
from PIL import Image
from util.util import tensor2im, save_image
import glob


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
    monet = cycle_gan_model('style_monet_pretrained')
    monet_style = []
    vangogh = cycle_gan_model('style_vangogh_pretrained')
    vangogh_style = []
    ukiyoe = cycle_gan_model('style_ukiyoe_pretrained')
    ukiyoe_style = []
    cezanne = cycle_gan_model('style_cezanne_pretrained')
    cezanne_style = []

    for image in glob.glob("test_faces/*.jpg"):
        cycle_gan_data = image_to_cycleGAN_data(np.asarray(Image.open(image)))

        monet.set_input(cycle_gan_data)
        monet.test()
        monet_style.append(monet.get_current_visuals()['fake'])

        vangogh.set_input(cycle_gan_data)
        vangogh.test()
        vangogh_style.append(vangogh.get_current_visuals()['fake'])

        ukiyoe.set_input(cycle_gan_data)
        ukiyoe.test()
        ukiyoe_style.append(ukiyoe.get_current_visuals()['fake'])

        cezanne.set_input(cycle_gan_data)
        cezanne.test()
        cezanne_style.append(cezanne.get_current_visuals()['fake'])

    for i, image in enumerate(monet_style):
        save_image(tensor2im(image), f'monet/monet{i}.jpg')

    for i, image in enumerate(vangogh_style):
        save_image(tensor2im(image), f'vangogh/vangogh{i}.jpg')

    for i, image in enumerate(ukiyoe_style):
        save_image(tensor2im(image), f'ukiyoe/ukiyoe{i}.jpg')

    for i, image in enumerate(cezanne_style):
        save_image(tensor2im(image), f'cezanne/cezanne{i}.jpg')
