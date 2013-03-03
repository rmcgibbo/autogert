import yaml
import os
import numpy as np
import matplotlib.pyplot as pp
import glob

from skimage import io
from sklearn import cross_validation, svm

from lib.image import split, split2


def collect_dataset():
    with(open('settings.yml')) as f:
        settings = yaml.load(f)
    with(open('trainingset.yml')) as f:
        labels = yaml.load(f)

    X, y = [], []
    
    for fn, answer in labels.iteritems():
        path = os.path.join('images', fn)
        #chars = split(io.imread(path), settings['charseps'])
        chars = split2(io.imread(path))
        
        # for ax, char in zip(axes, chars):
        #     ax.imshow(char)    
        # pp.show()

        assert len(chars) == len(answer)
        for char, label in zip(chars, answer):
            X.append(char.reshape(-1)) # flatten
            y.append(int(label))

    X = np.array(X)
    y = np.array(y)
    
    return X, y


def train(X, y):
    clf = svm.LinearSVC()
    # run 5-fold cross validaton to assess the accuracy of the classfier
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)        
    print 'CLASSIFIER SCORES'
    print '%s, mean=%s' % (scores, np.mean(scores))

    clf.fit(X, y)
    
    return clf

def predict(clf):
    for fn in glob.glob('images/*.png'):
        chars = split2(io.imread(fn))
        predictions = clf.predict(np.array([c.reshape(-1) for c in chars]))

        prediction = ''.join([str(p) for p in predictions])
        
        print '%s: %s' % (os.path.basename(fn), prediction)
    
def main():
    X, y = collect_dataset()
    clf = train(X, y)
    
    predict(clf)
        
    
if __name__ == '__main__':
    main()