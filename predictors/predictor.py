from abc import ABCMeta, abstractmethod
import cv2 as cv
import numpy as np


class Predictor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.hog_output_shape = 441
        self.number_featuers = []
        self.winSize = (28, 28)
        self.blockSize = (4, 4)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.n_bins = 9
        self.hog = cv.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.n_bins)

    def hog_compute(self, image):
        return np.array(self.hog.compute(image, None, None)).reshape((-1, self.hog_output_shape))

    def pre_process(self, image): raise NotImplementedError

    @abstractmethod
    def predict(self, image, show_image_before_model_feed=False): raise NotImplementedError
