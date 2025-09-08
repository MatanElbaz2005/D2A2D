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

def binary_sync_correlate_roi_argmax(
    received_pm: np.ndarray,
    headers_pm: np.ndarray,
    data_pm: np.ndarray,
    end_pm: np.ndarray,
    headers_search_frac: float = 0.10,
    debug: bool = False,
):
    """
    Same logic as binary_sync_correlate_roi, but returns only (corr, index) per pattern,
    using the C++ argmax function which includes per-position early-skip.

    Returns:
        ((ch_corr, ch_idx), (cd_corr, cd_idx), (ce_corr, ce_idx), data_start_est)
    """
    N = int(received_pm.size)
    Lh = int(headers_pm.size)
    end_hdr = N if headers_search_frac is None else max(Lh, min(N, int(N * headers_search_frac)))

    # 1) Headers in first X%
    ch_corr, ch_idx = binxcorr.correlate_sliding_bin_argmax(
        received_pm, headers_pm, 0, end_hdr, debug=debug
    )
    # 2) Data on full stream
    cd_corr, cd_idx = binxcorr.correlate_sliding_bin_argmax(
        received_pm, data_pm, 0, N, debug=debug
    )
    data_start_est = int(cd_idx) + int(data_pm.size)

    # 3) End from data_start_est to end
    ce_corr, ce_idx = binxcorr.correlate_sliding_bin_argmax(
        received_pm, end_pm, data_start_est, N, debug=debug
    )

    return (ch_corr, int(ch_idx)), (cd_corr, int(cd_idx)), (ce_corr, int(ce_idx)), int(data_start_est)
