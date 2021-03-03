import datetime
import os

from tensorboard.backend.event_processing import event_accumulator


class TrainingTime:
    _SIZE_GUIDANCE = {
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    }
    _TRAIN_TAG = 'train/loss'
    _VAL_TAG = 'val/loss'

    def __init__(self, event_file_path):
        if not os.path.exists(event_file_path):
            raise ValueError(f'Event file "{event_file_path}" not found.')

        self.event_file_path = event_file_path
        self.event_acc = self._get_event_accumulator()

    def _get_event_accumulator(self):
        ea = event_accumulator.EventAccumulator(self.event_file_path, size_guidance=self._SIZE_GUIDANCE)
        ea.Reload()
        if self._TRAIN_TAG not in ea.Tags()['scalars']:
            raise RuntimeError(f'Could not find scalar "{self._TRAIN_TAG}" in event file to extract start time.')
        if self._VAL_TAG not in ea.Tags()['scalars']:
            raise RuntimeError(f'Could not find scalar "{self._VAL_TAG}" in event file to extract end time.')

        return ea

    def get_training_time(self):
        train_events = self.event_acc.Scalars(self._TRAIN_TAG)
        val_events = self.event_acc.Scalars(self._VAL_TAG)
        first_event = train_events[0]
        last_event = val_events[-1]
        start_time = datetime.datetime.fromtimestamp(first_event.wall_time)
        end_time = datetime.datetime.fromtimestamp(last_event.wall_time)
        duration = end_time - start_time

        return duration
