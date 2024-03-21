# Evaluation metrics for streaming data
import numpy as np


# Gama et al.: 1000, 2000, 3000, 4000, 5000 (PAW: w=1000)
def prequential_error(losses, window_size=1000):
    l_slided = []
    l_window = losses[:window_size]
    for i_l in losses[window_size:]:
        l_slided.append(np.mean(l_window))
        l_window.pop(0)
        l_window.append(i_l)
    return l_slided


# Prequential error computed at time i (=len(losses)), with fading factor alpha
def fading_prequential_error(losses, fading_factor=0.995):
    time_elapsed = len(losses)
    dividend, divisor = 0, 0
    for k in range(1, time_elapsed+1):
        dividend += (fading_factor**(time_elapsed-k)) * losses[k-1]
        divisor += fading_factor ** (time_elapsed - k)
    return dividend/divisor


def fading_prequential_over_time(losses, fading_factor=0.995):
    fading_losses = []
    for i in range(len(losses)):
        fading_losses.append(fading_prequential_error(losses[:i+1], fading_factor=fading_factor))
    return fading_losses
