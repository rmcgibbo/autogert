import yaml
import os
import re
import time
import numpy as np
import matplotlib.pyplot as pp
import glob


from skimage import io
from sklearn import cross_validation, svm

from lib.image import split, align_fft, preprocess

def collect_dataset():
    with(open('training/training_set.yml')) as f:
        labels = yaml.load(f)

    X, y = [], []
    
    for fn, answer in labels.iteritems():
        path = os.path.join('training', fn)
        chars = split(io.imread(path))

        assert len(chars) == len(answer)
        for char, label in zip(chars, answer):
            X.append(char.reshape(-1)) # flatten
            y.append(int(label))

    X = np.array(X)
    y = np.array(y)
    
    print 'COLLECTED DATASET'
    
    return X, y


def train(X, y):
    clf = svm.LinearSVC()
    # run 5-fold cross validaton to assess the accuracy of the classfier
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)        
    print 'CLASSIFIER SCORES -- 5 FOLD CROSS VALIDATION'
    print '%s, mean=%s' % (scores, np.mean(scores))

    clf.fit(X, y)
    
    return clf


def crop(img, settings):
    h = settings['box_h']
    w = settings['box_w']
    x = settings['box_x']
    y = settings['box_y']
    
    return img[y:y+h, x:x+w]

def predict(clf, settings, img_fn=None, align_template=None):
    if img_fn is None:
        raise NotImplementedError('Camera!')
    else:
        raw_img = io.imread(img_fn)

    if align_template is None:
        aligned_img = raw_img
    else:
        aligned_img = align_fft(raw_img, align_template)

    cropped_img = crop(aligned_img, settings)
    binarized_img = preprocess(cropped_img)
    chars = split(binarized_img)
    
    # this gets an array of ints
    predictions = clf.predict(np.array([c.reshape(-1) for c in chars]))
    # make into a string
    prediction = ''.join([str(p) for p in predictions])
    return prediction, binarized_img, chars

def interactive_benchmark(clf, raw_images_glob, save_to):
    with(open('settings.yml')) as f:
        settings = yaml.load(f)
        template = io.imread(settings['template_img'])

    # interactive plotting
    pp.ion()
    pp.figure()
    pp.gray()
    
    if not os.path.exists(save_to):
        print "CREATING DIRECTORY", save_to
        os.makedirs(save_to)
    
    instructions = '[Enter] to accept, [q] to quit, or key the correct answer'
    print instructions
    
    lines = []
    
    raw_images_fn = glob.glob(raw_images_glob)
    training_images = set(yaml.load(open('training/training_set.yml')).keys())
    
    for fn in raw_images_fn:
        if os.path.basename(fn) in training_images:
            print 'Already trained on %s. Skipping' % fn
            continue
        
        prediction, binarized_img, chars = predict(clf, img_fn=fn,
            settings=settings)
        
        # plot the binarize_img across the top panel
        pp.clf()
        pp.subplot(2,1,1)
        pp.imshow(binarized_img)
        
        # plot the 6 chars (separated) across the bottom panel
        for i in range(6):
            pp.subplot(2,6,7+i)
            pp.imshow(chars[i])
            
        title = ' '.join(prediction[:3]) + '  ' + ' '.join(prediction[3:])
        pp.suptitle(title, fontsize='xx-large')
        pp.draw()
        
        ask_again = True
        while ask_again:
            response = raw_input('[Enter/q/key]: ')
            ask_again = False  # default
            
            if response == '':
                line = '%s: "%s"' % (os.path.basename(fn), prediction)
                print line
                lines.append(line)
                io.imsave(os.path.join(save_to, os.path.basename(fn)),
                    1.0*binarized_img)

            elif re.match('\d{6}', response):
                line = '%s: "%s"' % (os.path.basename(fn), response)
                print 'CORRECTED', line
                lines.append(line)
                io.imsave(os.path.join(save_to, os.path.basename(fn)),
                    1.0*binarized_img)

            elif response == 'q':
                print 'Quitting...\n'
                print ('Instructions: Move the files from %s into the training '
                       'directory, and add the lines below to the file'
                       'training_set.yml:\n' % save_to)
                print os.linesep.join(lines)
                return

            else:
                print instructions
                ask_again = True
    
def main():
    X, y = collect_dataset()
    clf = train(X, y)
    interactive_benchmark(clf, raw_images_glob='raw_images/*.png',
        save_to='interactive_training')
        
    
if __name__ == '__main__':
    main()