# adaptive-filtering-lms-demo

LMS-based FIR system identification demo for adaptive audio signal processing.

This repository demonstrates how the LMS algorithm can estimate an unknown FIR system from an input signal and a desired output signal.

## Overview

The script:

- generates a white noise input signal,
- creates an artificial unknown FIR system,
- generates the desired signal by convolution,
- estimates the FIR filter using LMS,
- saves the estimated filter and error curve,
- displays a 3D animation of LMS movement on a quadratic cost surface.

## Run

```bash
python examples/01_lms_system_identification.py
```

## Outputs

- `figures/01_lms_filter_estimation.png`
- `figures/01_lms_error_curve.png`
- 3D LMS trajectory animation

## Requirements

```bash
pip install -r requirements.txt
```

## Status

- [x] LMS system identification
- [ ] NLMS comparison
- [ ] RLS comparison
- [ ] Filtered-x LMS demo
