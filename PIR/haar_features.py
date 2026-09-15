import numpy as np

# Haar-transform based PIR feature extraction (8 frequency-band
# energies), following the approach used for PIR-based clutter
# rejection against wind-blown vegetation: a fixed-length block of the
# signal is passed through a Haar transform and the transformed
# coefficients are grouped into a small number of frequency bins, whose
# energies form the feature vector.
#
# That paper analyses a 128-sample block and reports 8 frequency bins,
# but does not publish the exact coefficient-to-bin grouping. This
# implementation groups coefficients by dyadic scale using a 3-level
# full Haar wavelet-packet decomposition (2**3 = 8 equal-width leaves),
# which is the natural way to get 8 bins out of a Haar transform.
# Applying a fixed 128-sample block to our variable-length completed
# movement passages (rather than the paper's continuously-updating
# sliding window) is an engineering adaptation, not a reproduction of
# the paper -- flag this in any report text.

BLOCK_LENGTH = 128
N_LEVELS = 3
N_BINS = 2 ** N_LEVELS  # 8


def _haarStep(signal):
    """One level of the Haar transform: pairwise sum/difference.

    Splits `signal` (even length) into an approximation half (local
    averages) and a detail half (local differences), each half the
    length of the input.
    """
    evens = signal[0::2]
    odds = signal[1::2]
    approx = (evens + odds) / np.sqrt(2.0)
    detail = (evens - odds) / np.sqrt(2.0)
    return approx, detail


def _fullTreeEnergies(signal, levels):
    """Recursively splits `signal` into 2**levels leaves via repeated
    Haar steps (a full wavelet-packet tree, not just the approximation
    branch), returning the energy (sum of squares) of each leaf.
    """
    if levels == 0:
        return [float(np.sum(signal ** 2))]
    approx, detail = _haarStep(signal)
    return _fullTreeEnergies(approx, levels - 1) + _fullTreeEnergies(detail, levels - 1)


def _prepareBlock(samples, blockLength=BLOCK_LENGTH):
    """Extracts the voltage values from a completed passage and pads or
    truncates them to a fixed `blockLength` so the wavelet-packet tree
    always splits evenly into N_BINS leaves.
    """
    voltages = np.array([v for _, v in samples], dtype=np.float64)
    if voltages.size >= blockLength:
        return voltages[:blockLength]
    padded = np.zeros(blockLength, dtype=np.float64)
    padded[: voltages.size] = voltages
    return padded


def getHaarFeatureVector(samples, blockLength=BLOCK_LENGTH, levels=N_LEVELS):
    """Returns the 8-element Haar frequency-bin energy vector for a
    completed PIR movement passage.
    """
    block = _prepareBlock(samples, blockLength)
    energies = _fullTreeEnergies(block, levels)
    return np.array(energies, dtype=np.float64)
