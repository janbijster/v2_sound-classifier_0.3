from matplotlib import pyplot as plt
import numpy as np


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def audio_volume (recording, smoothing_n=1000):
    volumes = 20 * np.log10(np.abs(recording) + 0.01)
    if smoothing_n > 0:
        volumes = running_mean(volumes, smoothing_n)
    return volumes.flatten()

def initialize_visualizations():
    fig, axes = plt.subplots(2, 2)
    plt.ion()
    plt.show()
    return (fig, axes)

def update_visualizations(fig, axes, recording, spectrogram, probabilities, class_names, volume_threshold, sample_length):
    # Top 5
    top_n = 5
    probabilities_classes_sorted = [(name, proba) for proba, name in sorted(zip(probabilities, class_names), reverse=True)]
    classes_sorted = [x[0] for x in probabilities_classes_sorted[:top_n]]
    probabilities_sorted = [x[1] for x in probabilities_classes_sorted[:top_n]]

    xrange_top_n = list(range(top_n))
    axes[0, 0].clear()
    axes[0, 0].set_title('Top {}'.format(top_n))
    axes[0, 0].barh(xrange_top_n, probabilities_sorted, align='center')
    axes[0, 0].set_yticks(xrange_top_n)
    axes[0, 0].set_yticklabels(classes_sorted)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel('Estimated probabilities')

    # All classes
    xrange = list(range(len(probabilities)))
    axes[0, 1].clear()
    axes[0, 1].set_title('All classes')
    axes[0, 1].barh(xrange, probabilities, align='center')
    axes[0, 1].set_yticks(xrange)
    axes[0, 1].set_yticklabels(class_names)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('Estimated probabilities')
    axes[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Volume
    volumes = audio_volume(recording.flatten())
    xrange = np.linspace(0, sample_length, volumes.shape[0])
    axes[1, 0].clear()
    axes[1, 0].set_title('Volume')
    axes[1, 0].plot(xrange, volumes)

    # Spectrogram
    axes[1, 1].clear()
    axes[1, 1].set_title('Spectrogram')
    axes[1, 1].imshow(spectrogram.reshape((128, 128)))

    plt.draw()
    plt.pause(0.9 * sample_length)

