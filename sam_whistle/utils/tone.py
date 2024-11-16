
import numpy as np

def find_peaks_simple(signal):
    """Simple peak detection by looking at the change in sign of the first derivative."""
    diff_signal = np.diff(np.sign(np.diff(signal)))
    peaks = np.where(diff_signal < 0)[0] + 1
    valleys = np.where(diff_signal > 0)[0] + 1
    return peaks, valleys

def find_peaks_simpleN(signal, order = 1):
    """Detect peaks within N nearest neighbors.
    
    Args:
        signal: (H,) signal to analyze
        order: number of nearest neighbors to consider, default 1 followed Marie Roch
    """
    peaks, valleys = [], []
    for i in range(order, len(signal) - order):
        if all(signal[i] > signal[i - order:i]) and all(signal[i] > signal[i + 1:i + order + 1]):
            peaks.append(i)
        if all(signal[i] < signal[i - order:i]) and all(signal[i] < signal[i + 1:i + order + 1]):
            valleys.append(i)
    return np.array(peaks), np.array(valleys)

def consolidate_peaks(peaks, spectrum, min_gap = 2):
    """
    Consolidates peaks that are closer than a specified minimum gap.


    """
    peaks = np.sort(peaks)
    peak_dist = np.diff(peaks)
    too_close_idx = np.where(peak_dist < min_gap)[0]

    while too_close_idx.size > 0:
        to_close_vals = spectrum[peaks[too_close_idx]]
        maxidx = np.argmax(to_close_vals) 
        max_close_idx = too_close_idx[maxidx] 
        next_close_idx = max_close_idx + 1
        if next_close_idx < len(peaks):
            if spectrum[peaks[max_close_idx]] >= spectrum[peaks[next_close_idx]]:
                peaks = np.delete(peaks, next_close_idx)
            else:
                peaks = np.delete(peaks, max_close_idx)
        else:
            raise ValueError("next_close_idx out of range of peaks")

        # Recalculate distances and find any remaining close peaks
        peak_dist = np.diff(peaks)
        too_close_idx = np.where(peak_dist < min_gap)[0]

    return peaks


def get_overlapped_tonals(gt_range, dt_ranges):
    """Get overlapped tonals between ground truth and detected tonals."""
    gt_start, gt_end = gt_range
    dt_start_times, dt_end_times = dt_ranges[:, 0], dt_ranges[:, 1]
    overlaped_cond = (dt_start_times <= gt_start) & (dt_end_times >= gt_start) | \
                    (dt_start_times >= gt_start) & (dt_start_times<= gt_end)
    overlaped_idx = np.nonzero(overlaped_cond)[0]
    return overlaped_idx

if __name__ == "__main__":
    peaks = find_peaks_simple(np.ones(10))
    print(len(peaks[0]))