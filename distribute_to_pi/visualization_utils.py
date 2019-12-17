from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from platform import system
import time


def initialize_visualizations():
    fig, axes = plt.subplots(2, 3)
    plt.ion()
    maximize_window()
    plt.show()
    return (fig, axes)

def update_visualizations(fig, axes, volumes, spectrogram, probabilities, class_names, volume_threshold, sample_length, stop_fun, record_fun, is_recording):
    tic = time.time()

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
    xrange = np.linspace(0, sample_length, volumes.shape[0])
    axes[1, 0].clear()
    axes[1, 0].set_title('Volume', color='#0033ee')
    axes[1, 0].plot(xrange, volumes, label='Volume over time')
    axes[1, 0].axhline(y=volume_threshold, linewidth=2, color='#ee0033', label='Volume threshold')
    axes[1, 0].legend()

    # Spectrogram
    axes[1, 1].clear()
    axes[1, 1].set_title('Spectrogram')
    axes[1, 1].imshow(spectrogram.reshape((128, 128)))

    # Buttons
    axes[0, 2].clear()
    axes[1, 2].clear()
    
    btn_stop = Button(axes[0, 2], 'Stop')
    btn_stop.on_clicked(stop_fun)
    if is_recording:
        btn_record = Button(axes[1, 2], 'Recording...')
    else:
        btn_record = Button(axes[1, 2], 'Record')
        btn_record.on_clicked(record_fun)

    plt.draw()
    dt = 0.25
    t = time.time() - tic
    while t < 0.9 * sample_length:
        fig.suptitle('Perdicted class: {}. New sample in {:.0f}...'.format(classes_sorted[0], sample_length - t))
        plt.pause(dt)
        t = time.time() - tic


def maximize_window():
    # See discussion: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    backend = plt.get_backend()
    cfm = plt.get_current_fig_manager()
    if backend == "wxAgg":
        cfm.frame.Maximize(True)
    elif backend == "TkAgg":
        if system() == "win32":
            cfm.window.state('zoomed')  # This is windows only
        else:
            cfm.resize(*cfm.window.maxsize())
    elif backend == 'QT4Agg':
        cfm.window.showMaximized()
    elif callable(getattr(cfm, "full_screen_toggle", None)):
        if not getattr(cfm, "flag_is_max", None):
            cfm.full_screen_toggle()
            cfm.flag_is_max = True
    else:
        print('Couldn\'t maximize window, unknown pyplot backend {}'.format(backend))