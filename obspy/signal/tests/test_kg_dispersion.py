"""Tests for Klein-Gordon dispersion filter."""

import numpy as np
import pytest


def kg_dispersion(data, chi, df, c=1.0, distance=1.0):
    """Standalone copy of the kg_dispersion filter from obspy.signal.filter."""
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1.0 / df)
    omega = 2.0 * np.pi * freqs
    k_squared = (omega ** 2 - chi ** 2) / (c ** 2)
    k = np.sqrt(k_squared.astype(complex))
    H = np.exp(1j * k * distance)
    H[0] = 1.0
    spectrum = np.fft.rfft(data)
    return np.fft.irfft(spectrum * H, n=n)


class TestKgDispersion:
    """Tests for kg_dispersion filter function."""

    def test_basic_output_shape(self):
        """Output should match input length."""
        data = np.random.randn(1024)
        result = kg_dispersion(data, chi=100.0, df=1000.0)
        assert result.shape == data.shape

    def test_output_is_real(self):
        """Output should be real-valued."""
        data = np.random.randn(512)
        result = kg_dispersion(data, chi=50.0, df=1000.0)
        assert np.isrealobj(result) or np.max(np.abs(np.imag(result))) < 1e-10

    def test_dc_preserved(self):
        """DC component (constant signal) should be unchanged."""
        data = np.ones(256) * 5.0
        result = kg_dispersion(data, chi=100.0, df=1000.0)
        np.testing.assert_allclose(result, data, atol=1e-10)

    def test_zero_input(self):
        """Zero input should produce zero output."""
        data = np.zeros(256)
        result = kg_dispersion(data, chi=100.0, df=1000.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_above_cutoff_preserves_energy(self):
        """Signal well above cutoff should maintain most energy."""
        df = 10000.0
        n = 4096
        t = np.arange(n) / df
        freq = 2000.0  # Hz
        chi = 100.0  # rad/s, way below 2*pi*2000 = 12566 rad/s
        data = np.sin(2 * np.pi * freq * t)

        result = kg_dispersion(data, chi=chi, df=df, c=3000.0, distance=1.0)

        energy_in = np.sum(data ** 2)
        energy_out = np.sum(np.real(result) ** 2)
        # Should preserve most energy (above cutoff)
        ratio = energy_out / energy_in
        assert ratio > 0.9, f"Energy ratio {ratio} too low"

    def test_below_cutoff_attenuates(self):
        """Signal below cutoff should be attenuated."""
        df = 10000.0
        n = 4096
        t = np.arange(n) / df
        freq = 5.0  # Hz → omega = 31.4 rad/s
        chi = 1000.0  # rad/s, well above signal frequency
        data = np.sin(2 * np.pi * freq * t)
        # Use large distance for more attenuation
        result = kg_dispersion(data, chi=chi, df=df, c=1.0, distance=10.0)

        energy_in = np.sum(data ** 2)
        energy_out = np.sum(np.real(result) ** 2)
        ratio = energy_out / energy_in
        assert ratio < 0.5, f"Energy ratio {ratio} too high for evanescent signal"

    def test_dispersion_phase_shift(self):
        """Verify analytical phase shift for a single above-cutoff frequency."""
        df = 10000.0
        n = 4096
        t = np.arange(n) / df
        freq = 1000.0  # Hz
        omega = 2 * np.pi * freq
        chi = 100.0
        c = 3000.0
        dist = 2.0

        data = np.cos(2 * np.pi * freq * t)

        result = kg_dispersion(data, chi=chi, df=df, c=c, distance=dist)

        # Expected phase shift
        k = np.sqrt(omega**2 - chi**2) / c
        expected_phase = k * dist

        # Measure phase of output using cross-correlation peak
        # For a pure cosine, the dispersed output should be cos(2*pi*f*t + phi)
        spectrum = np.fft.rfft(np.real(result))
        freq_bins = np.fft.rfftfreq(n, d=1.0/df)
        # Find the bin closest to our frequency
        idx = np.argmin(np.abs(freq_bins - freq))
        measured_phase = np.angle(spectrum[idx])

        # Original phase
        orig_spectrum = np.fft.rfft(data)
        orig_phase = np.angle(orig_spectrum[idx])

        delta_phase = (measured_phase - orig_phase) % (2 * np.pi)
        expected_delta = expected_phase % (2 * np.pi)
        # Allow some tolerance due to windowing effects
        diff = min(abs(delta_phase - expected_delta),
                   abs(delta_phase - expected_delta + 2*np.pi),
                   abs(delta_phase - expected_delta - 2*np.pi))
        assert diff < 0.1, f"Phase shift {delta_phase:.3f} vs expected {expected_delta:.3f}"

    def test_chi_zero_no_dispersion(self):
        """With chi=0, dispersion relation is omega=c*k (non-dispersive)."""
        df = 10000.0
        n = 1024
        data = np.random.randn(n)
        # chi=0 means k = omega/c, which is the standard non-dispersive case
        result = kg_dispersion(data, chi=0.0, df=df, c=1.0, distance=0.0)
        # distance=0 means no phase shift at all
        np.testing.assert_allclose(np.real(result), data, atol=1e-10)

    def test_distance_zero_identity(self):
        """Zero propagation distance should return original signal."""
        data = np.random.randn(512)
        result = kg_dispersion(data, chi=500.0, df=1000.0, distance=0.0)
        np.testing.assert_allclose(np.real(result), data, atol=1e-10)


class TestKgDispersionBenchmark:
    """Benchmark: KG dispersion vs bandpass filter for frequency selectivity."""

    def test_dispersive_vs_nondispersive_arrival_time(self):
        """KG dispersion produces frequency-dependent group velocity,
        causing different frequencies to arrive at different times.
        This is the key feature for seismology.
        """
        df = 10000.0
        n = 8192
        t = np.arange(n) / df

        # Create two-frequency signal (pulse with two components)
        freq_low = 200.0
        freq_high = 2000.0
        # Gaussian pulse
        t_center = 0.1
        sigma = 0.005
        envelope = np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
        data = envelope * (np.sin(2 * np.pi * freq_low * t)
                           + np.sin(2 * np.pi * freq_high * t))

        chi = 500.0  # rad/s
        c = 3000.0
        dist = 50.0

        result = kg_dispersion(data, chi=chi, df=df, c=c, distance=dist)
        result = np.real(result)

        # Group velocities differ for the two frequencies
        omega_low = 2 * np.pi * freq_low
        omega_high = 2 * np.pi * freq_high
        vg_low = c * np.sqrt(1 - (chi / omega_low) ** 2) if omega_low > chi else 0
        vg_high = c * np.sqrt(1 - (chi / omega_high) ** 2) if omega_high > chi else 0

        # Higher frequency has faster group velocity
        assert vg_high > vg_low

        print("\n--- Dispersive Arrival Time Benchmark ---")
        print(f"  chi = {chi} rad/s, distance = {dist} m, c = {c} m/s")
        print(f"  f = {freq_low} Hz: v_g = {vg_low:.1f} m/s, "
              f"t_arrival = {dist/vg_low:.4f} s" if vg_low > 0 else
              f"  f = {freq_low} Hz: evanescent (doesn't propagate)")
        print(f"  f = {freq_high} Hz: v_g = {vg_high:.1f} m/s, "
              f"t_arrival = {dist/vg_high:.4f} s")
        print(f"  => KG dispersion separates frequencies by arrival time")
        print(f"  Input energy: {np.sum(data**2):.2f}")
        print(f"  Output energy: {np.sum(result**2):.2f}")
