import glob
import numpy as np
from skimage import io, filter, transform
import matplotlib.pyplot as pp

images = []
fns = set(glob.glob('images/*.jpg')).union(
        set(glob.glob('images/*.png')))
#fns = glob.glob('images/*.png')

for fn in fns:
    img = io.imread(fn)
    img = transform.resize(img, (50, 165))
    img = filter.threshold_adaptive(img, block_size=30)
    images.append(np.asarray(img, dtype=int))
    


sumimage = np.zeros(images[0].shape, dtype=int)
for img in images:
    sumimage += img

pp.gray()
pp.imshow(sumimage)
pp.plot(np.sum(sumimage, axis=0) / len(images))

pp.show()