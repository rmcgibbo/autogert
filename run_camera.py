import os
import pipes
import shutil
import tempfile
import subprocess
import numpy as np
import cPickle as pickle

import matplotlib.pyplot as pp
from skimage import io, transform, filter

from process_camera import preprocess, separate_chars

def capture_image(wait=2.0, camera_name='USB 2.0 PC Camera', verbose=False,
         imagesnap_path='~/local/bin/imagesnap', tmp_fn=None):
    """Capture an image from an attached webcam, (Mac OSX)
    
    This function calls an external command line program, ImageSnap,
    which is available for MacOSX from the following website
    
    http://iharder.sourceforge.net/current/macosx/imagesnap/
    
    Parameters
    ----------
    wait : float
        How to long to wait for the camera to warm up
    camera_name : str
        The name of the camera. You can check the avaialble names of the
        camera on your system using the imagesnap executable
    verbose : bool
        Print verbose output to stdout from the imagesnap executable
    imagesnap_path : str
        Path to the imagesnap executable command line program
    tmp_fn : str, optional
        The filename in which to save the image. If left unspecified, the image
        will not be saved to disk.
        
        
    Returns
    -------
    img : np.ndarray
        The image
    """
    tempdir = None

    if tmp_fn is None:
        tempdir = tempfile.mkdtemp()
        tmp_fn = os.path.join(tempdir, 'snapshot.jpg')
        
    try:
        v_flag = '-v' if verbose else ''
        command = '%s -w %f %s -d %s %s' % (imagesnap_path, wait, v_flag,
            pipes.quote(camera_name), tmp_fn)
        subprocess.check_output(command , shell=True)
        image = io.imread(tmp_fn)
    finally:
        if tempdir is not None:
            shutil.rmtree(tempdir)
            
    return image

def main():
    img = capture_image(tmp_fn='snapshot.jpg')
    background = io.imread('camera_background.jpg')
    img = background - img
    #img = io.imread('snapshot.jpg')
    img = img[0:200, :, :]
    
    img = preprocess(img)
    chars = separate_chars(img)
    
    with open('classifier.pickl') as f:
        classifier = pickle.load(f)
    
    fig, axes = pp.subplots(nrows=1, ncols=6, figsize=(8, 5))
    pp.gray()
    for char, ax in zip(chars, axes):
        ax.matshow(char)                
        print classifier.predict(char.reshape(-1))

    pp.show()

if __name__ == '__main__':
    main()