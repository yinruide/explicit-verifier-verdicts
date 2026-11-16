"""Verifier cards (App. A.2).

``generate.py`` writes one ``<candidate>.md`` per model into this directory from the JSON
that ``env/download_models.sh`` produces, so the commit revision and the weight sha256 in a
card are copied from the download that actually happened rather than typed in by hand.
"""
