import json
import os

import numpy as np

from downstream import save_imagegrid, save_oscillating_video


class ResultsMixin:
    def __init__(self, load_from_disk=True):
        if load_from_disk and self._results_exist():
            self.results = self._load_results()
        else:
            self.results = {}

    def __getitem__(self, key):
        return self.results[key]

    def __setitem__(self, key, value):
        self.results[key] = value

    def __contains__(self, item):
        return item in self.results.keys()

    def save_image_result(self, model_type, tag, image):
        image_path = self._get_image_path(model_type, tag)
        save_imagegrid(image, image_path)
        self.safe_add(model_type, tag, image_path)
        self.save()

    def _get_image_path(self, model_type, tag):
        return self._get_file_path(model_type, tag, 'jpeg')

    def save_video_result(self, model_type, tag, video):
        video_path = self._get_video_path(model_type, tag)
        save_oscillating_video(video, video_path)
        self.safe_add(model_type, tag, video_path)
        self.save()

    def _get_video_path(self, model_type, tag):
        return self._get_file_path(model_type, tag, 'gif')

    def save_array_result(self, model_type, tag, *arrays):
        array_path = self._get_array_path(model_type, tag)
        np.savez(array_path, *arrays)
        self.safe_add(model_type, tag, array_path)
        self.save()

    def _get_array_path(self, model_type, tag):
        return self._get_file_path(model_type, tag, 'npz')

    def _get_file_path(self, model_type, tag, extension):
        log_path = self._get_log_path()
        samples_path = os.path.join(log_path, tag)
        os.makedirs(samples_path, exist_ok=True)
        samples_path = os.path.join(samples_path, f'{model_type}.{extension}')

        return samples_path

    def safe_add(self, model_type, key, value):
        if model_type in self.keys():
            self[model_type][key] = value
        else:
            self[model_type] = {key: value}

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def empty(self):
        return not self.results

    def missing_model_types(self, model_types):
        return list(set(model_types).difference(self.keys()))

    def save(self):
        checkpoint_path = self._get_results_path()
        with open(checkpoint_path, mode='wt') as f:
            json.dump(self.results, f, indent=4)

    def _results_exist(self):
        checkpoint_path = self._get_results_path()
        exists = os.path.exists(checkpoint_path)

        return exists

    def _load_results(self):
        checkpoint_path = self._get_results_path()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, mode='rt') as f:
                checkpoints = json.load(f)
        else:
            raise FileNotFoundError(f'No checkpoint file found at {checkpoint_path}')

        return checkpoints

    def render(self):
        raise NotImplementedError

    @staticmethod
    def _get_log_path():
        script_path = os.path.dirname(__file__)
        log_path = os.path.join(script_path, '..', '..', 'logs')
        log_path = os.path.normpath(log_path)

        return log_path
