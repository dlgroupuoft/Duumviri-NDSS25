import glob
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import progressbar
import cv2
import utils
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import code
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

deprecation._PRINT_DEPRECATION_WARNINGS = False


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    aspect = w/h

    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(
            int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(
            int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor]*3

    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


class ScreenshotPairs():
    def __init__(self, second, shape=(600, 600)):
        self.complete = False
        self.idx = int(os.path.basename(second).split('_')[0])
        self.second_path = second
        self.first_path = second.replace('_2.png', '_1.png')
        self.complete = os.path.isfile(self.first_path)
        self.shape = shape

        self.load_both()

    def load_one(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img.shape[0] > 5000:
            return None
        return resizeAndPad(img, self.shape)

    def load_both(self):
        self.first = self.load_one(self.first_path)
        self.second = self.load_one(self.second_path)
        if self.first is not None and self.second is not None:
            self.complete = True
        else:
            self.complete = False


class ImageData():
    def __init__(self):
        self.true_dirs = ['true_dataset_req_exceptions']
        self.false_dirs = ['false_dataset_0-100.parsed',
                           'false_dataset_100-1k.parsed', 'false_dataset_4k-5k.parsed']
        self.cache_path = 'ml_data/images_data'
        self.dir_cache = 'ml_data/images_data/dir_cache'

    def dump(self, compressed, out_path):
        pairs = utils.load(compressed)
        print("%s has %d" % (compressed, len(pairs)))

        for pair in pairs:
            cv2.imwrite(os.path.join(out_path, '%d_1.png' %
                        pair.idx), pair.first)
            cv2.imwrite(os.path.join(out_path, '%d_2.png' %
                        pair.idx), pair.second)

    def check_inputs(self):
        t_path = os.path.join(self.cache_path, 'true')
        f_path = os.path.join(self.cache_path, 'false')
        assert (os.path.isfile(t_path))
        assert (os.path.isfile(f_path))

        os.makedirs(os.path.join(self.cache_path,
                    'uncompressed', 'true'), exist_ok=True)
        os.makedirs(os.path.join(self.cache_path,
                    'uncompressed', 'false'), exist_ok=True)

        self.dump(t_path, os.path.join(
            self.cache_path, 'uncompressed', 'true'))
        self.dump(f_path, os.path.join(
            self.cache_path, 'uncompressed', 'false'))

    def load_images(self, dir_name):
        screenshots = glob.glob(os.path.join(
            dir_name, 'screenshots', '*_2.png'))
        ret = []

        for idx in progressbar.progressbar(range(len(screenshots))):
            screenshot_path = screenshots[idx]
            pair = ScreenshotPairs(screenshot_path)

            if not pair.complete:
                continue

            ret.append(pair)
        return ret

    def para_load_dir(self, dir_name, rebuild):
        dir_cache = os.path.join(self.dir_cache,  dir_name)

        if not rebuild and dir_cache and os.path.isfile(dir_cache) and not self.disable_caching:
            one_dir = utils.load(dir_cache)
        else:
            one_dir = self.load_images(dir_name)

        if one_dir is None:
            print('%s no data' % dir_name)
        else:
            print("%s %d datapoints" % (dir_name, len(one_dir)))

        if not self.disable_caching:
            utils.dump(one_dir, dir_cache)
        return one_dir

    def _load_data(self, data_dirs, cache_fname, rebuild=False, req_only=False, is_true=False, disable_caching=False):
        self.is_true = is_true
        self.disable_caching = disable_caching
        cache_path = os.path.join(self.cache_path, cache_fname)

        if not rebuild and cache_path and os.path.isfile(cache_path):
            out = utils.load(cache_path)
            print("From cache %s %s" % (len(out), cache_path))
            return out

        if True or multiprocessing.current_process().daemon:
            total = map(self.para_load_dir, data_dirs,
                        [rebuild] * len(data_dirs))
        else:
            args = [(data_dir, rebuild) for data_dir in data_dirs]
            num_instances = len(data_dirs)
            with Pool(processes=num_instances) as pool:
                total = pool.starmap(self.para_load_dir, args)

        out = []
        for dir_data in total:
            out += dir_data

        if self.cache_path:
            print("Caching %s to %s" % (len(out), cache_path))
            utils.dump(out, cache_path)
        return out


def load_model(feature_vector=False):

    img1 = tf.keras.Sequential([hub.KerasLayer(
        "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1", trainable=False)])
    img1.build([None, 600, 600, 3])
    if feature_vector:
        return img1

    img2 = tf.keras.Sequential([hub.KerasLayer(
        "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1", trainable=False)])
    img2.build([None, 600, 600, 3])

    combined = tf.keras.layers.Concatenate()([img1.output, img2.output])
    out = tf.keras.layers.Dense(32, activation="relu")(combined)
    out = tf.keras.layers.Dense(1, activation="linear")(out)

    model = tf.keras.Model(inputs=[img1.input, img2.input], outputs=out)
    return model


def load_data():
    data_cls = ImageData()
    trues = data_cls._load_data(data_cls.true_dirs, 'true', rebuild=False)
    falses = data_cls._load_data(data_cls.false_dirs, 'false', rebuild=False)

    xs = trues+falses
    ys = ([1] * len(trues)) + ([0] * len(falses))
    return xs, ys


def get_feature_vectors(m, imgs):
    for i in range(len(imgs)):
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)

        imgs[i] = resizeAndPad(imgs[i], (600, 600))

    imgs = np.stack((imgs,)*3, axis=-1)
    imgs = np.array(imgs)/255.0

    return m(imgs)

