# correlation_helpers.py
# Uses the C++ binary correlator; keeps Python-side convenience functions.

import numpy as np
import binxcorr  # the pybind11 module we build

def binary_sync_correlate_roi(
    received_pm: np.ndarray,
    headers_pm: np.ndarray,
    data_pm: np.ndarray,
    end_pm: np.ndarray,
    headers_search_frac: float = 0.10,
):
    """
    Compute 3 correlations (headers, data, end) using the C++ binary correlator.

    Performance tweak:
      - Headers sync is searched only in the first `headers_search_frac` of the
        bitstream (default: 10%), since headers are known to be near the start.
      - Data sync is searched over the full stream (unchanged).
      - End sync is searched only from the estimated data start onward (unchanged).

    Returns:
        (corr_headers, corr_data, corr_end_roi, data_start_est)
    """
    N = int(received_pm.size)
    Lh = int(headers_pm.size)

    # ROI for headers: [0, end_hdr)
    if headers_search_frac is None:
        end_hdr = N
    else:
        end_hdr = max(Lh, min(N, int(N * headers_search_frac)))

    # 1) Headers over first X% only
    ch = binxcorr.correlate_sliding_bin(received_pm, headers_pm, start_bit=0, end_bit=end_hdr)

    # 2) Data over full range
    cd = binxcorr.correlate_sliding_bin(received_pm, data_pm)

    # Estimate data start -> used as ROI start for END
    data_start_est = int(np.argmax(cd)) + data_pm.size

    # 3) END only over ROI starting at the estimated data start
    ce_roi = binxcorr.correlate_sliding_bin(
        received_pm, end_pm, start_bit=data_start_est, end_bit=N
    )
    return ch, cd, ce_roi, data_start_est

