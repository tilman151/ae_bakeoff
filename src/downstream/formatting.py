import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def save_imagegrid(image_grid, file_name):
    image_grid *= 255
    image_grid = image_grid.astype(np.uint8)
    image_grid = image_grid.transpose(1, 2, 0)
    img = Image.fromarray(image_grid)
    img.save(file_name)


def save_oscillating_video(video, file_name, duration=None):
    oscillation = _build_oscillating_video(video)
    save_video(oscillation, file_name, duration, loop=True)


def _build_oscillating_video(video):
    start_frame = video[0][None]
    end_frame = video[-1][None]
    num_frames = video.shape[0]
    num_still_fames = (5 * num_frames) // 4 - num_frames
    oscillation = _build_oscillation(video, start_frame, end_frame, num_still_fames)

    return oscillation


def _build_oscillation(video, start_frame, end_frame, num_still_fames):
    start_still = np.tile(start_frame, (num_still_fames // 2, 1, 1, 1))
    end_still = np.tile(end_frame, (num_still_fames - 2, 1, 1, 1))
    reverse_video = video[::-1]
    oscillation = [start_still, video, end_still, reverse_video, start_still]
    oscillation = np.concatenate(oscillation)

    return oscillation


def save_video(video, file_name, duration=None, loop=False):
    video *= 255
    video = video.transpose(0, 2, 3, 1)
    video = video.astype(np.uint8)
    duration = duration or video.shape[0] / 25
    _save_gif(video, file_name, duration, loop)


def _save_gif(video, file_name, duration, loop):
    loop = 0 if loop else 1
    duration = 1000 * duration / video.shape[0]
    img, *imgs = [Image.fromarray(frame) for frame in video]
    img.save(fp=file_name, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=loop)


def save_roc_plot(tpr, fpr, file_name):
    fig = _plot_roc(fpr, tpr)
    plt.savefig(file_name)
    plt.close(fig)


def _plot_roc(fpr, tpr):
    fig = plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    return fig
