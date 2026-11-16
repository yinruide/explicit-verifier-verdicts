"""pi-hat calibration (post-pilot extension, 2026-10-31).

For each verifier i, ``g_i(s) = sigma(a_i * logit(s) + b_i)`` estimates P(task failure | s_i = s)
(logit-scale Platt, fitted with Platt's smoothed targets and no other regularization), and

    pi-hat_i = 1 - g_i(s_i)  if y_i = 1,     pi-hat_i = g_i(s_i)  if y_i = 0,

the estimated chance that verifier i's verdict ``y_i = 1[s_i >= tau_i]`` is wrong. pi-hat reads
s_obs only (Exp 1: the post-channel score; Exp 2: the raw score).

* :mod:`rafr.calibration.platt` -- the numbers: clipped logit, the fit, ECE / Brier / reliability
  bins and their bootstrap intervals. No I/O, no labels.
* :mod:`rafr.calibration.apply` -- reads the frozen calibrator file (checking its sha256) and maps
  s_obs to pi-hat. No labels: this is what the experiment runners import.
* :mod:`rafr.calibration.fit` -- fits on the unlabeled pool and diagnoses on D_gate, reading y*
  through the label gate (purposes CALIBRATION_FIT and CALIBRATION_DIAG). Only
  ``pipelines/ext_calibrate.py`` calls it.
"""
