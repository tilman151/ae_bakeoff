import matplotlib.pyplot as plt
import numpy as np
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


def save_roc_plot(tpr, fpr, auc, file_name):
    fig = plt.figure(figsize=(5, 5))
    plot_roc(plt.gca(), fpr, tpr, auc)
    plt.savefig(file_name)
    plt.close(fig)


def plot_roc(ax, fpr, tpr, auc, title=None):
    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    if title is not None:
        ax.set_title(title)


def plot_risk_coverage(ax: plt.Axes, coverage, risk, title=None):
    coverage, mean_risk, std_risk = _coverage_wise_risk_stats(coverage, risk)
    if title is None:
        ax.plot(coverage, mean_risk)
    else:
        ax.plot(coverage, mean_risk, label=title)
    ax.fill_between(coverage, mean_risk - std_risk, mean_risk + std_risk, alpha=0.5)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)


def _coverage_wise_risk_stats(coverages, risks):
    bins = np.linspace(0, 1, num=100)
    bins[-1] += 1e-6
    resampled_risks = np.zeros((len(bins), len(coverages)))
    for i, (coverage, risk) in enumerate(zip(coverages, risks)):
        sorting_idx = np.argsort(coverage)
        coverage = np.array(coverage)[sorting_idx]
        risk = np.array(risk)[sorting_idx]
        resampled_risks[:, i] = np.interp(bins, coverage, risk)
    mean_risks = resampled_risks.mean(axis=1)
    std_risks = resampled_risks.std(axis=1)

    return bins, mean_risks, std_risks


def plot_perfect_risk_coverage(ax):
    coverage = np.linspace(0, 1, num=11)
    perfect_risk = np.maximum(0, np.linspace(-0.5, 0.5, num=11))
    ax.plot(coverage, perfect_risk, c="gray", linestyle=":", label="perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)


def plot_reduction(ax, features, labels, title=None):
    classes = np.unique(labels)
    for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
        class_features = features[labels == cls]
        ax.scatter(class_features[:, 0], class_features[:, 1], c=[color], label=cls, s=[2], alpha=0.5)
    if title is not None:
        ax.set_title(title)
