from typing import List, Dict, Tuple
import tensorflow as tf
import numpy as np
import json


class PbModel:
    def __init__(self, input_saved_model_dir_path: str = None, classes: Tuple = None,
                 feature_divide_num=16, top_paths=10, beam_width=10, softmax_threshold=0.05):
        self.model = tf.saved_model.load(input_saved_model_dir_path)
        self.model_input_shape = self.model.signatures["serving_default"].inputs[0].shape
        self.model_input_dtype = self.model.signatures["serving_default"].inputs[0].dtype
        self.model_output_shape = self.model.signatures["serving_default"].outputs[0].shape
        self.model_output_dtype = self.model.signatures["serving_default"].outputs[0].dtype
        self.classes = classes
        self.blank_index = len(classes)-1
        self.feature_divide_num = feature_divide_num
        self.top_paths = top_paths
        self.beam_width = beam_width
        self.softmax_threshold = softmax_threshold

    def inference(self, input_image_list: List[np.ndarray], batch_size: int = 8) -> Tuple[List, List]:
        resized_image_array, sequence_length_list = self.__preprocess_image_list(input_image_list, self.model_input_shape[1:3])
        raw_decode_list, raw_prob_list = self.__inference(resized_image_array, sequence_length_list, batch_size)
        output = self.__output_parse(raw_decode_list, raw_prob_list)
        return output, (raw_decode_list, raw_prob_list)

    def __inference(self, resize_input_tensor: np.ndarray, sequence_length_list: List[int], batch_size: int) -> Tuple[List, List]:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resize_input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resize_input_tensor.dtype}')
        decode_list = []
        prob_list = []
        for index in range(0, resize_input_tensor.shape[0], batch_size):
            batch = resize_input_tensor[index:index + batch_size, :, :, :]
            batch_sequence_length_list = sequence_length_list[index:index + batch_size]
            raw_pred = self.model(tf.cast(batch, self.model_input_dtype)).numpy()
            decode, log_prob = self.__decode(raw_pred, batch_sequence_length_list)
            decode_list.append(decode)
            prob_list.append(tf.exp(log_prob).numpy())
        return decode_list, prob_list

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> np.ndarray:
        resized_image_list = []
        sequence_length_list = []
        for input_image in input_image_list:
            resized_image = self.__preprocess_image(input_image, resize_input_shape)
            if resize_input_shape[1] is None:
                resized_image = tf.image.pad_to_bounding_box(resized_image, 0, 0, max(1, resize_input_shape[0]),
                                                             max(1, resized_image.shape[1] + (
                                                                     self.feature_divide_num - resized_image.shape[
                                                                 1] % self.feature_divide_num))).numpy().astype(np.uint8)
            resized_image_list.append(resized_image)
            sequence_length_list.append(resized_image.shape[1]//self.feature_divide_num)
        max_height = max([image.shape[0] for image in resized_image_list])
        max_width = max([image.shape[1] for image in resized_image_list])
        max_ch = max([image.shape[2] for image in resized_image_list])
        canvas_array = np.zeros((len(input_image_list), max_height, max_width, max_ch), dtype=resized_image_list[0].dtype)
        for index, image in enumerate(resized_image_list):
            canvas_array[index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
        return canvas_array, sequence_length_list

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

    def __output_parse(self, decode_list, prob_list) -> List[Dict]:
        output_dict_list = []
        for batch_index in range(len(decode_list)):
            for data_index in range(prob_list[batch_index].shape[0]):
                text_list = []
                classes_list = []
                scores_list = []
                for candidate_index in range(len(decode_list[batch_index])):
                    decode = tf.sparse.to_dense(decode_list[batch_index][candidate_index], default_value=self.blank_index).numpy()[data_index].tolist()
                    text = self.__decode2labels(decode)
                    prob = prob_list[batch_index][data_index][candidate_index]
                    text_list.append(text)
                    classes_list.append(decode)
                    scores_list.append(float(prob))
                output_dict = {'text': text_list,
                               'classes': classes_list,
                               'scores': scores_list}
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

    def __decode(self, raw_pred, sequence_length_list):
        threshold_pred = np.copy(raw_pred)
        threshold_pred_min, threshold_pred_max = np.min(threshold_pred), np.max(threshold_pred)
        threshold_blank_max = np.max(threshold_pred[:, :, -1])

        threshold_pred_softmax = tf.nn.softmax((raw_pred - threshold_pred_min + threshold_pred_max)).numpy()
        a = np.argmax(threshold_pred_softmax, axis=-1)
        threshold_pred[threshold_pred_softmax < self.softmax_threshold] = threshold_pred_min
        for batch_index in range(raw_pred.shape[0]):
            for width_index in range(raw_pred.shape[1]):
                if np.max(threshold_pred[batch_index, width_index, :]) == threshold_pred_min:
                    threshold_pred[batch_index, width_index, self.blank_index] = threshold_blank_max

        decode, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(threshold_pred, (1, 0, 2)),
                                                         tf.convert_to_tensor(sequence_length_list, dtype=tf.int32),
                                                         top_paths=self.top_paths, beam_width=self.beam_width)
        return decode, log_prob
    def __decode2labels(self, decode):
        labels = ""
        for label_index in decode:
            if label_index == self.blank_index:
                continue
            labels += self.classes[label_index]
        return labels