from lib.camera import capture_image
from lib.image import preprocess

from matplotlib.patches import Rectangle
from lib.rectangle import DraggableRectangle

from skimage import io
import matplotlib.pyplot as pp

#img = io.imread('snapshot.jpg')
img = capture_image()


fig, axes = pp.subplots(nrows=2, ncols=1, figsize=(8, 5))
pp.gray()

zoom_horizontal = (340, 525)
zoom_vertial = (390, 310)
box_xy = (370, 342)
box_width = 122
box_height = 37


axes[0].imshow(img)
axes[0].set_xlim(*zoom_horizontal)
axes[0].set_ylim(*zoom_vertial)

zoomrect = DraggableRectangle(Rectangle(box_xy, height=box_height,
                                    width=box_width, alpha=0.25), ax=axes[0])
zoomrect.connect()

def onpress(event):
    if event.inaxes == axes[0]:
        x0, y0 = zoomrect.rect.xy
        w0, h0 = zoomrect.rect.get_width(), zoomrect.rect.get_height()
        img2 = preprocess(img[y0:y0+h0, x0:x0+w0, :])
        axes[1].imshow(img2)
        axes[1].xaxis.grid(color='gray', which='minor', linestyle='dashed')
        
        print {'xy': (x0, y0), 'height':h0, 'width': w0}

axes[0].figure.canvas.mpl_connect('button_press_event', onpress)


pp.show()


