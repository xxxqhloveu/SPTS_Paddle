import sys
import cv2
import six
import json
import random
import bezier
import numbers
import numpy as np
from copy import deepcopy
from misc import bezier2bbox
from PIL import Image, ImageEnhance

class RandomCrop(object):
    def __init__(self, min_size_ratio, max_size_ratio, prob):
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.prob = prob 

    def __call__(self, data):
        image, target = data['image'], data['target']
        if random.random() > self.prob or len(target['bboxes']) == 0:
            return data

        # h, w = image.shape
        for _ in range(100):
            crop_w = int(image.width * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_h = int(image.height * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_region = self.get_params(image, [crop_h, crop_w])
            cropped_image, cropped_target = self.crop(deepcopy(image), deepcopy(target), crop_region)
            if not cropped_image is None:
                data['image'], data['target'] = cropped_image, cropped_target
                return data

        print('Can not be cropped with texts')
        data['image'], data['target'] = image, target
        return data
    
    def crop(self, image, target, crop_region):
        bboxes = target['bboxes']
        crop_region, keep_instance = self.adjust_crop_region(bboxes, crop_region)
        
        if crop_region is None:
            return None, None

        top, left, height, width = crop_region
        cropped_image = image.crop((left, top, left + width, top + height))

        rg_ymin, rg_xmin, rg_h, rg_w = crop_region
        target['size'] = np.array([rg_h, rg_w])
        if bboxes.shape[0] > 0:
            target['bboxes'] = target['bboxes'] - np.array([rg_xmin, rg_ymin] * 2).astype('float32')
            target['bezier_pts'] = target['bezier_pts'] - np.array([rg_xmin, rg_ymin] * 8).astype('float32')
            for k in ['labels', 'area', 'iscrowd', 'recog', 'bboxes', 'bezier_pts']:
                target[k] = target[k][keep_instance]

        return cropped_image, target

    def adjust_crop_region(self, bboxes, crop_region):
        rg_ymin, rg_xmin, rg_h, rg_w = crop_region 
        rg_xmax = rg_xmin + rg_w 
        rg_ymax = rg_ymin + rg_h 

        pre_keep = np.zeros(bboxes.shape[0]).astype(bool)
        while True:
            ov_xmin = np.clip(bboxes[:, 0], a_min=rg_xmin, a_max=None)
            ov_ymin = np.clip(bboxes[:, 1], a_min=rg_ymin, a_max=None)
            ov_xmax = np.clip(bboxes[:, 2], a_min=None, a_max=rg_xmax)
            ov_ymax = np.clip(bboxes[:, 3], a_min=None, a_max=rg_ymax)
            ov_h = ov_ymax - ov_ymin 
            ov_w = ov_xmax - ov_xmin 
            keep = np.bitwise_and(ov_w > 0, ov_h > 0)

            if (keep == False).all():
                return None, None
            
            # if keep.equal(pre_keep):
            if (keep == pre_keep).all():
                break 

            keep_bboxes = bboxes[keep]
            keep_bboxes_xmin = int(min(keep_bboxes[:, 0]).item())
            keep_bboxes_ymin = int(min(keep_bboxes[:, 1]).item())
            keep_bboxes_xmax = int(max(keep_bboxes[:, 2]).item())
            keep_bboxes_ymax = int(max(keep_bboxes[:, 3]).item())
            rg_xmin = min(rg_xmin, keep_bboxes_xmin)
            rg_ymin = min(rg_ymin, keep_bboxes_ymin)
            rg_xmax = max(rg_xmax, keep_bboxes_xmax)
            rg_ymax = max(rg_ymax, keep_bboxes_ymax)

            pre_keep = keep
        
        crop_region = (rg_ymin, rg_xmin, rg_ymax - rg_ymin, rg_xmax - rg_xmin)
        return crop_region, keep

    def get_params(self, image, output_size):
        # def _get_image_size(img):
        #     if isinstance(img, np.ndarray):
        #         if img.ndim >= 2:
        #             return [img.shape[-1], img.shape[-2]]
        #         raise TypeError("Unexpected input type")
        #     if isinstance(img, Image.Image):
        #         return img.size
        #     raise TypeError("Unexpected type {}".format(type(img)))        
        # w, h = _get_image_size(img)
        h, w = image.height, image.width
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th + 1)
        j = random.randint(0, w - tw + 1)
        return i, j, th, tw


class RandomRotate(object):
    def __init__(self, max_angle, prob):
        self.max_angle = max_angle 
        self.prob = prob 

    def __call__(self, data):
        image, target = data['image'], data['target']
        if random.random() > self.prob:
            data['image'], data['target'] = image, target
            return data
        
        angle = random.uniform(-self.max_angle, self.max_angle)
        # image_h, image_w = img.shape[:2]
        image_w, image_h = image.size
        rotation_matrix = cv2.getRotationMatrix2D((image_w//2, image_h//2), angle, 1)
        image = image.rotate(angle, expand=True)

        new_w, new_h = image.size 
        target['size'] = np.array([new_h, new_w])
        pad_w = (new_w - image_w) / 2
        pad_h = (new_h - image_h) / 2

        bezier_pts = target['bezier_pts']
        bezier_pts = bezier_pts.reshape(-1, 8, 2)
        bezier_pts = self.rotate_points(bezier_pts, rotation_matrix, (pad_w, pad_h))
        bezier_pts = bezier_pts.reshape(-1, 16)
        target['bezier_pts'] = np.array(bezier_pts).astype('float32')

        bboxes = [bezier2bbox(ele) for ele in bezier_pts]
        target['bboxes'] = np.array(bboxes).astype('float32').reshape([-1, 4]) if len(target['bboxes']) != 0 else np.array([])
        
        data['image'], data['target'] = image, target
        return data

    def rotate_points(self, coords, rotation_matrix, paddings):
        coords = np.pad(coords, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
        coords = np.dot(coords, rotation_matrix.transpose())
        coords[:, :, 0] += paddings[0]
        coords[:, :, 1] += paddings[1]
        return coords


class RandomResize(object):
    def __init__(self, min_size, max_size):
        self.min_sizes = min_size
        self.max_size = max_size
    
    def __call__(self, data):
        image, target = data['image'], data['target']
        min_size = random.choice(self.min_sizes)
        size = self.get_size_with_aspect_ratio(image.size, min_size, self.max_size)
        rescaled_image = image.resize(size[::-1])

        ratio_width = rescaled_image.size[0] / image.size[0]
        ratio_height = rescaled_image.size[1] / image.size[1]

        target['size'] = np.array(size)
        target['area'] = target['area'] * (ratio_width * ratio_height) if len(target['area']) != 0 else np.array([])
        target['bboxes'] = target['bboxes'] * np.array([ratio_width, ratio_height] * 2) if len(target['bboxes']) != 0 else np.array([])
        target['bezier_pts'] = target['bezier_pts'] * np.array([ratio_width, ratio_height] * 8) if len(target['bezier_pts']) != 0 else np.array([])

        data['image'], data['target'] = rescaled_image, target
        return data

    def get_size_with_aspect_ratio(self, image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)


class RandomDistortion(object):
    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.prob = prob
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, data):
        image, target = data['image'], data['target']
        if random.random() > self.prob:
            data['image'], data['target'] = image, target
            return data
        data['image'], data['target'] = self.tfm(image), target
        return data
    
    def tfm(self, image):
        fn_idx = [0, 1, 2, 3]
        random.shuffle(fn_idx)
        # fn_idx = np.random.choice(4, 3)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = np.random.uniform(brightness[0], brightness[1])
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = np.random.uniform(contrast[0], contrast[1])
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = np.random.uniform(saturation[0], saturation[1])
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = np.random.uniform(hue[0], hue[1])
                image = self.adjust_hue(image, hue_factor)
        
        return image

    def adjust_brightness(self, img, brightness_factor):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return img

    def adjust_hue(self, img, hue_factor):
        if not(-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

        input_mode = img.mode
        if input_mode in {'L', '1', 'I', 'F'}:
            return img

        h, s, v = img.convert('HSV').split()

        np_h = np.array(h, dtype=np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over='ignore'):
            np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, 'L')

        img = Image.merge('HSV', (h, s, v)).convert(input_mode)
        return img


def sample_bezier_curve(bezier_pts, num_points=10, mid_point=False):
    curve = bezier.Curve.from_nodes(bezier_pts.transpose())
    if mid_point:
        x_vals = np.array([0.5])
    else:
        x_vals = np.linspace(0, 1, num_points)
    points = curve.evaluate_multi(x_vals).transpose()
    return points 


class MakeSequence(object):
    def __init__(self, num_bins, max_num_text_ins):
        self.num_bins = num_bins
        self.max_num_text_ins = max_num_text_ins
        self.chars = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
        num_char_classes = len(self.chars) + 1 # unknown
        recog_pad_index = num_bins + num_char_classes
        self.eos_index = recog_pad_index + 1
        self.sos_index = self.eos_index + 1
        self.padding_index = self.sos_index + 1
        self.num_classes = self.padding_index + 1

    def __call__(self, data):
        target = data['target']
        max_target = min(self.max_num_text_ins, len(target['labels']))
        # if max_target == 0: center_pts = np.ones(0).reshape(-1, 2).astype('float32')
        center_pts = []
        recog_labels = []
        for i in range(max_target):
            bezier_pt = target['bezier_pts'][i].reshape(8, 2)
            mid_pt1 = sample_bezier_curve(bezier_pt[:4], mid_point=True)
            mid_pt2 = sample_bezier_curve(bezier_pt[4:], mid_point=True)
            center_pt = ((mid_pt1 + mid_pt2) / 2).reshape(-1)
            center_pts.append(center_pt)
            recog_label = target['recog'][i] + self.num_bins
            recog_labels.append(recog_label)
        center_pts = np.array(center_pts).astype('float32')
        center_pts = np.floor(center_pts * self.num_bins).astype("int64")
        np.clip(center_pts, 0, self.num_bins - 1)
        recog_labels = np.array(recog_labels).astype('float32')
        pt_label = np.concatenate((center_pts, recog_labels), axis=1)
        pt_label = pt_label.flatten()
        input_seq = np.concatenate((np.array([self.sos_index]), pt_label))
        output_seq = np.concatenate((pt_label, np.array([self.eos_index])))
        sequence = np.concatenate((input_seq, output_seq)).reshape(2, -1)
        data['sequence'] = sequence
        data['val_sequence'] = np.array([1098])
        return data
        # pad_input_seq = np.ones((1621)) * 1099
        # pad_output_seq = np.ones((1621)) * 1099
        # pad_input_seq[:input_seq.shape[0]] = input_seq
        # pad_output_seq[:output_seq.shape[0]] = output_seq
        # sequence = np.concatenate((pad_input_seq, pad_output_seq)).reshape(2, -1)
        # data['sequence'] = sequence
        # pad_val_seq = np.ones((1621)) * 1099
        # pad_val_seq[0] = 1098.
        # data['val_sequence'] = pad_val_seq
        # return data


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img, target = data['image'], data['target']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        h, w = img.shape[-2:]
        target['bboxes'] = target['bboxes'] / np.array([w, h] * 2).astype("float32")
        target['bezier_pts'] = target['bezier_pts'] / np.array([w, h] * 8).astype("float32")
        data['image'], data['target'] = img, target
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        image = data['image']
        from PIL import Image
        if isinstance(image, Image.Image):
            img = np.array(image)
        data['image'] = img.transpose((2, 0, 1))
        return data
