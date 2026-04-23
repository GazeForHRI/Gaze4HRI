import numpy as np

def find_stable_start(
    data: np.ndarray,
    nominal_period_ms: float,
    stability_tolerance_ms: float = 2.0,
    stability_window_size: int = 5
) -> int:
    """
    Identify where a timestamped data stream begins to exhibit stable sampling intervals.
    Returns the index immediately after the first run of
    `stability_window_size` intervals all within ±`stability_tolerance_ms`
    of `nominal_period_ms`.
    """
    n = data.shape[0]
    if n < stability_window_size + 1:
        return 0

    ts = data[:, 0]
    deltas = np.diff(ts)
    lower = nominal_period_ms - stability_tolerance_ms
    upper = nominal_period_ms + stability_tolerance_ms

    for start in range(len(deltas) - stability_window_size + 1):
        window = deltas[start : start + stability_window_size]
        if np.all((window >= lower) & (window <= upper)):
            return start + stability_window_size

    return 0


def match_regular_to_regular(
    tuple1,
    tuple2,
    max_match_diff_ms: float,
    stability_tolerance_ms: float = 2.0,
    stability_window_size: int = 5
):
    arr1, period1_ms = tuple1
    arr2, period2_ms = tuple2

    ts1 = arr1[:, 0]
    ts2 = arr2[:, 0]

    idx1 = []
    idx2 = []

    for i, t in enumerate(ts1):
        j = np.searchsorted(ts2, t)
        candidates = []
        if j < len(ts2): candidates.append(j)
        if j > 0:         candidates.append(j - 1)

        best_diff = float('inf')
        best_j = None
        for c in candidates:
            diff = abs(ts2[c] - t)
            if diff < best_diff:
                best_diff = diff
                best_j = c

        if best_diff <= max_match_diff_ms:
            idx1.append(i)
            idx2.append(best_j)

    return np.array(idx1, dtype=int), np.array(idx2, dtype=int)

def match_irregular_to_regular(
    irregular_data: np.ndarray,
    regular_data: np.ndarray,
    regular_period_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align an irregular stream to a (nominally) regular stream by timestamp.

    Returns TWO arrays with matching row counts:
      - irregular_matched: filtered irregular rows (same columns as input irregular_data;
                           if irregular_data was 1D (timestamps-only), this will be (K, 1))
      - regular_matched_vals: matched regular VALUES (i.e., regular_data[:, 1:]) for each irregular row
                              If regular_data was timestamps-only (1D or (N,1)), this will be (K, 0).

    Notes:
      * The unstable head of the regular stream is excluded (using find_stable_start).
      * Any irregular rows earlier than the first stable regular timestamp are dropped.
      * Core interpolation/endpoint logic is unchanged.
      * 1D inputs are treated as timestamps-only and are internally reshaped to 2D.
    """

    # --- Normalize dimensionality: accept 1D arrays as timestamps-only ---
    if irregular_data.ndim == 1:
        irregular_data = irregular_data.reshape(-1, 1)  # [ts] -> (N,1)
    if regular_data.ndim == 1:
        regular_data = regular_data.reshape(-1, 1)      # [ts] -> (N,1)

    # Keep track of column counts to build empty outputs correctly
    irregular_cols = irregular_data.shape[1]  # 1 if timestamps-only
    regular_cols_after_ts = max(0, regular_data.shape[1] - 1)  # 0 if timestamps-only

    # Find stable start (works with our 2D normalization)
    start = find_stable_start(regular_data, regular_period_ms)
    if not isinstance(start, (int, np.integer)) or start < 0:
        start = 0

    ts_reg_full = regular_data[:, 0]
    vals_reg_full = regular_data[:, 1:]  # shape (N, M), M can be 0

    ts_reg = ts_reg_full[start:]
    vals_reg = vals_reg_full[start:]

    # Edge case: no stable regular data
    if ts_reg.size == 0:
        empty_ir = np.empty((0, irregular_cols), dtype=irregular_data.dtype)
        empty_reg = np.empty((0, regular_cols_after_ts), dtype=regular_data.dtype)
        return empty_ir, empty_reg

    # Drop irregular rows earlier than the stable start timestamp
    stable_start_ts = ts_reg[0]
    keep_mask = (irregular_data[:, 0] >= stable_start_ts)
    irregular_kept = irregular_data[keep_mask]

    if irregular_kept.shape[0] == 0:
        empty_ir = np.empty((0, irregular_cols), dtype=irregular_data.dtype)
        empty_reg = np.empty((0, vals_reg.shape[1]), dtype=vals_reg.dtype)
        return empty_ir, empty_reg

    # --- original matching logic (unchanged), applied to filtered inputs ---
    n_ir = irregular_kept.shape[0]
    m = vals_reg.shape[1]  # can be 0
    matched_vals = np.zeros((n_ir, m), dtype=vals_reg.dtype)

    for i in range(n_ir):
        t = irregular_kept[i, 0]
        j = np.searchsorted(ts_reg, t)

        if j < len(ts_reg) and ts_reg[j] == t:
            if m: matched_vals[i] = vals_reg[j]
        elif j == 0:
            if m: matched_vals[i] = vals_reg[0]
        elif j >= len(ts_reg):
            if m: matched_vals[i] = vals_reg[-1]
        else:
            if m: matched_vals[i] = 0.5 * (vals_reg[j - 1] + vals_reg[j])

    return irregular_kept, matched_vals

def arrays_equal_verbose(a, b, name="array"):
    if np.array_equal(a, b):
        print(f"{name}: PASS")
    else:
        print(f"{name}: FAIL")
        print(f"Expected: {b}")
        print(f"Got     : {a}")

if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility

    # ---------- Test 1 ----------
    print("Running Test 1")
    t1 = np.arange(0, 1000, 10)
    gt = np.column_stack((t1, np.random.rand(len(t1), 3)))
    t2 = np.arange(0, 1000, 33.33)
    est = np.column_stack((t2, np.random.rand(len(t2), 3)))

    id1, id2 = match_regular_to_regular((gt, 10), (est, 1000.0 / 30.0), max_match_diff_ms=15)

    expected_id1 = np.array([
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,
        53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99
    ])
    expected_id2 = np.array([ 0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,
                              5,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10, 10, 10,
                             11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16,
                             16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21,
                             21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26,
                             27, 27, 27, 28, 28, 28, 29, 29, 29, 30])

    arrays_equal_verbose(id1, expected_id1, "Test 1 - indices1")
    arrays_equal_verbose(id2, expected_id2, "Test 1 - indices2")

    # ---------- Test 2 ----------
    print("\nRunning Test 2")
    t1 = np.arange(0, 1000, 10)
    gt = np.column_stack((t1, np.random.rand(len(t1), 3)))
    t2 = np.arange(7, 1007, 10)
    est = np.column_stack((t2, np.random.rand(len(t2), 3)))

    id1, id2 = match_regular_to_regular((gt, 10), (est, 10), max_match_diff_ms=5)

    expected_len = len(gt) - 1  # Only the first item won't match
    assert len(id1) == expected_len and len(id2) == expected_len, "Test 2: incorrect number of matches"
    print("Test 2: PASS (matched all except first offset sample)")

    # ---------- Test 3 ----------
    print("\nRunning Test 3")
    t1 = np.arange(0, 1000, 10)
    gt = np.column_stack((t1, np.random.rand(len(t1), 3)))
    t2 = np.arange(7, 1007, 5)
    est = np.column_stack((t2, np.random.rand(len(t2), 3)))

    id1, id2 = match_regular_to_regular((gt, 10), (est, 5), max_match_diff_ms=5)

    assert len(id1) == len(gt) - 1, "Test 3: unexpected match count"
    assert all(est[id2, 0] - gt[id1, 0] <= 5), "Test 3: timestamps mismatch too large"
    print("Test 3: PASS")

    # ---------- Irregular Matching ----------
    print("\nRunning match_irregular_to_regular test")
    t1 = np.arange(0, 1000, 10)
    cam = np.column_stack((t1, np.random.rand(len(t1), 16)))
    t2 = np.arange(0, 1000, 33.33)
    est = np.column_stack((t2, np.random.rand(len(t2), 3)))
    irr = est.copy()
    irr[:, 0] += np.random.uniform(-5, 5, size=irr.shape[0])

    matched_vals = match_irregular_to_regular(irr, cam, regular_period_ms=10.0)

    assert matched_vals.shape == (irr.shape[0], 16), "Irregular match: output shape mismatch"
    assert np.isfinite(matched_vals).all(), "Irregular match: contains NaNs or infs"
    print("match_irregular_to_regular: PASS")
