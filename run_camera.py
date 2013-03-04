import yaml
import time

import scipy.misc

from lib.camera import capture_image
from lib.image import preprocess, split


def main():
    img = capture_image()
    with(open('settings.yml')) as f:
        settings = yaml.load(f)

    #img = img[settings['box_y'] : settings['box_y'] + settings['box_h'],
    #          settings['box_x'] : settings['box_x'] + settings['box_w'], :]

    #img = preprocess(img)

    name = 'images2/%d.png' % time.time()
    scipy.misc.imsave(name, img)

    print 'Image Saved to %s' % name


if __name__ == '__main__':
    while True:
        main()
        time.sleep(30)
        

