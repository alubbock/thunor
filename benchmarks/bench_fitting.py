"""
Benchmark: DIP rate and dose-response curve fitting.

Dataset: thunor/testdata/hts007.h5

Run with:
    python benchmarks/bench_fitting.py
"""

import sys
import importlib
import time
from pathlib import Path

# Ensure this project's thunor is imported, not any editable-install sibling.
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import numpy as np  # noqa: E402
import thunor.io  # noqa: E402
import thunor.dip  # noqa: E402
import thunor.viability  # noqa: E402
import thunor.curve_fit  # noqa: E402

REPEATS = 10


def time_fn(fn, repeats=REPEATS):
    """Run fn() `repeats` times; return (mean_ms, std_ms, min_ms)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return times.mean(), times.std(), times.min()


def bench_dip_rates(dataset):
    thunor.dip.dip_rates(dataset)


def bench_fit_dip(ctrl_dip_data, expt_dip_data):
    thunor.curve_fit.fit_params_minimal(ctrl_dip_data, expt_dip_data)


def bench_fit_viability(viability_data):
    thunor.curve_fit.fit_params_minimal(
        None,
        viability_data,
        fit_cls=thunor.curve_fit.HillCurveLL3u,
    )


def main():
    print('=' * 60)
    print('Thunor curve fitting benchmark')
    print('=' * 60)
    print()

    # ── Load dataset ──────────────────────────────────────────────
    print('Loading dataset...', flush=True)
    ref = importlib.resources.files('thunor') / 'testdata/hts007.h5'
    with importlib.resources.as_file(ref) as filename:
        dataset = thunor.io.read_hdf(filename)

    ctrl_dip_data, expt_dip_data = thunor.dip.dip_rates(dataset)
    viability_data, _ = thunor.viability.viability(dataset, include_controls=False)

    n_wells = len(expt_dip_data.index.get_level_values('well_id').unique())
    n_groups = len(expt_dip_data.groupby(level=['cell_line', 'drug']).groups)
    n_via_groups = len(viability_data.groupby(level=['cell_line', 'drug']).groups)
    print(
        f'Wells: {n_wells}  |  DIP curve groups: {n_groups}'
        f'  |  Viability curve groups: {n_via_groups}\n'
    )

    print(f'── Timing  ({REPEATS} runs) ──────────────────────────────')

    mean, std, mn = time_fn(lambda: bench_dip_rates(dataset))
    print(
        f'  DIP rate calculation:      {mean:7.1f} ± {std:4.1f} ms  (min {mn:.1f} ms)'
    )

    mean, std, mn = time_fn(lambda: bench_fit_dip(ctrl_dip_data, expt_dip_data))
    print(
        f'  DIP curve fitting:         {mean:7.1f} ± {std:4.1f} ms  (min {mn:.1f} ms)'
    )

    mean, std, mn = time_fn(lambda: bench_fit_viability(viability_data))
    print(
        f'  Viability curve fitting:   {mean:7.1f} ± {std:4.1f} ms  (min {mn:.1f} ms)'
    )

    print()


if __name__ == '__main__':
    main()
