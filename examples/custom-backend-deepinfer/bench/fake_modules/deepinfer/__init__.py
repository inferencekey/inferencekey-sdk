"""Simulated DeepInfer for the parent-process benchmark (see ../../README.md).

Only the surface that ``deepinfer_engines._rank_main`` touches exists here. It is
found only when ``bench/fake_modules`` is on ``sys.path``, which only the
benchmark's ``sim_server.py`` arranges.
"""
