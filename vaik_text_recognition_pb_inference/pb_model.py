from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np
import json


class PbModel:
    def __init__(self, input_saved_model_dir_path: str = None, classes: Tuple = None, feature_divide_num=16, blank_index=0):
        self.model = tf.saved_model.load(input_saved_model_dir_path)
        self.model_input_shape = self.model.signatures["serving_default"].inputs[0].shape
        self.model_input_dtype = self.model.signatures["serving_default"].inputs[0].dtype
        self.model_output_shape = self.model.signatures["serving_default"].outputs[0].shape
        self.model_output_dtype = self.model.signatures["serving_default"].outputs[0].dtype
        self.classes = classes
        self.feature_divide_num = feature_divide_num
        self.blank_index = blank_index

    def inference(self, input_image_list: List[np.ndarray], batch_size: int = 8) -> Tuple[List[Dict], Tuple[np.ndarray, np.ndarray, List[str]]]:
        resized_image_array = self.__preprocess_image_list(input_image_list, self.model_input_shape[1:3])
        raw_pred = self.__inference(resized_image_array, batch_size)
        output = self.__output_parse(raw_pred)
        return output, raw_pred

    def __inference(self, resize_input_tensor: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resize_input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resize_input_tensor.dtype}')

        decode_list = []
        prob_list = []
        labels_list = []
        for index in range(0, resize_input_tensor.shape[0], batch_size):
            batch = resize_input_tensor[index:index + batch_size, :, :, :]
            raw_pred = self.model(tf.cast(batch, self.model_input_dtype))
            decode, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(raw_pred, (1, 0, 2)),
                                                         tf.ones((raw_pred.shape[0], ), dtype=tf.int32) * raw_pred.shape[1],
                                                         top_paths=10)
            decode = tf.sparse.to_dense(decode[0]).numpy()
            prob = tf.exp(log_prob).numpy()
            labels = self.__decode2labels(decode)
            decode_list.append(decode)
            prob_list.append(prob)
            labels_list.extend(labels)
        return np.vstack(decode_list), np.vstack(prob_list), labels_list

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> np.ndarray:
        resized_image_list = []
        for input_image in input_image_list:
            resized_image = self.__preprocess_image(input_image, resize_input_shape)
            if resize_input_shape[1] is None:
                resized_image = tf.image.pad_to_bounding_box(resized_image, 0, 0, max(1, resize_input_shape[0]),
                                                            max(1, resized_image.shape[1] + (
                                                                    self.feature_divide_num - resized_image.shape[
                                                                1] % self.feature_divide_num))).numpy().astype(np.uint8)
            resized_image_list.append(resized_image)
        max_height = max([image.shape[0] for image in resized_image_list])
        max_width = max([image.shape[1] for image in resized_image_list])
        max_ch = max([image.shape[2] for image in resized_image_list])
        canvas_array = np.zeros((len(input_image_list), max_height, max_width, max_ch), dtype=resized_image_list[0].dtype)
        for index, image in enumerate(resized_image_list):
            canvas_array[index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
        return canvas_array

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')
        input_image = self.__resize_base_height(input_image, resize_input_shape[0])
        if resize_input_shape[1] is not None:
            input_image = self.__resize_upper_width_limit(input_image, resize_input_shape[1])
        return input_image

    def __output_parse(self, pred) -> List[Dict]:
        output_dict_list = []
        for index in range(pred[0].shape[0]):
            output_dict = {'text': pred[2][index],
                           'labels': pred[0][index],
                           'scores': pred[1][index]}
            output_dict_list.append(output_dict)
        return output_dict_list

    @classmethod
    def char_json_read(cls, char_json_path):
        with open(char_json_path, 'r') as f:
            json_dict = json.load(f)
        classes = []
        for character_dict in json_dict['character']:
            classes.extend(character_dict['classes'])
        return classes

    def __resize_base_height(self, np_image, resize_base_height):
        resize_width = max(1, int((np_image.shape[0]/resize_base_height) * np_image.shape[1]))
        np_image = tf.image.resize(np_image, (resize_base_height, resize_width)).numpy()
        return np_image

    def __resize_upper_width_limit(self, np_image, resize_width):
        if (np_image.shape[1] > resize_width):
            resize_height = max(1, int((np_image.shape[1]/resize_width) * np_image.shape[0]))
            np_image = tf.image.resize(np_image, (resize_height, resize_width)).numpy()
        return np_image

    def __decode2labels(self, decode):
        labels_list = []
        for decode_index in range(decode.shape[0]):
            labels = ""
            for label_index in decode[decode_index]:
                if label_index == self.blank_index:
                    continue
                labels += self.classes[label_index]
            labels_list.append(labels)
        return labels_list