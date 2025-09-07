# correlation_helpers.py
# Uses the C++ binary correlator; keeps Python-side convenience functions.

import numpy as np
import binxcorr  # the pybind11 module we build

def binary_sync_correlate_roi(
    received_pm: np.ndarray,
    headers_pm: np.ndarray,
    data_pm: np.ndarray,
    end_pm: np.ndarray
):
    """
    Compute 3 correlations (headers, data, end) using the C++ binary correlator.
    Returns (corr_headers, corr_data, corr_end_roi, data_start_est).
    """
    # Headers & data over the full range
    ch = binxcorr.correlate_sliding_bin(received_pm, headers_pm, debug=True)  # float32
    cd = binxcorr.correlate_sliding_bin(received_pm, data_pm, debug=True)

    data_start_est = int(np.argmax(cd)) + data_pm.size

    # END only over ROI starting at the estimated data start
    ce_roi = binxcorr.correlate_sliding_bin(
        received_pm, end_pm, start_bit=data_start_est, end_bit=received_pm.size, debug=True
    )
    return ch, cd, ce_roi, data_start_est
