from pathlib import Path

import numpy as np

from spacecraft_acs import config as cfg_mod
from spacecraft_acs import quaternion as qt
from spacecraft_acs import simulate
from spacecraft_acs.config import EstimatorConfig, SensorConfig
from spacecraft_acs.estimator import Mekf
from spacecraft_acs.sensors import ARCSEC, SensorSuite

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def run_filter(
    duration=600.0,
    dt=0.25,
    st_every=4,
    bias=(2.0e-7, -1.0e-7, 1.5e-7),
    st_noise=10.0,
    gyro_noise=1.0e-6,
    seed=7,
):
    """Standalone MEKF run against a synthetic constant-rate truth."""
    sensors_cfg = SensorConfig(
        gyro_rate_noise=gyro_noise,
        gyro_bias=np.array(bias),
        star_tracker_noise_arcsec=st_noise,
        seed=seed,
    )
    est_cfg = EstimatorConfig(enabled=True)
    sensors = SensorSuite(sensors_cfg, dt)
    omega_true = np.array([0.0, -7.2921159e-5, 0.0])
    q_true = qt.IDENTITY.copy()
    mekf = Mekf(est_cfg, sensors_cfg, dt, q_true)

    n = int(duration / dt)
    att_err = np.zeros((n, 3))
    bias_err = np.zeros((n, 3))
    sigma = np.zeros((n, 6))
    for k in range(n):
        q_true = qt.normalize(
            qt.multiply(q_true, qt.from_rotation_vector(omega_true * dt))
        )
        q_meas, omega_meas = sensors.measure(q_true, omega_true)
        mekf.propagate(omega_meas)
        if k % st_every == 0:
            mekf.update_star_tracker(q_meas)
        att_err[k] = 2.0 * qt.error(q_true, mekf.q)[1:]
        bias_err[k] = mekf.bias - sensors.bias
        sigma[k] = mekf.sigma
    return att_err, bias_err, sigma


def test_attitude_error_beats_raw_star_tracker():
    """Converged MEKF attitude error must be well below the raw ST noise."""
    att_err, _, _ = run_filter()
    tail = att_err[len(att_err) // 2 :]
    rms = np.sqrt(np.mean(tail**2))
    assert rms < 0.4 * 10.0 * ARCSEC  # at least ~2.5x better than raw


def test_bias_estimate_converges():
    att_err, bias_err, sigma = run_filter(duration=1200.0)
    tail = bias_err[-100:]
    # True bias magnitude is ~2e-7 rad/s; estimate should recover most of it
    assert np.all(np.abs(np.mean(tail, axis=0)) < 1e-7)


def test_covariance_consistency():
    """Estimation errors should respect the filter's own 3-sigma bound."""
    att_err, bias_err, sigma = run_filter(duration=1200.0)
    half = len(att_err) // 2
    inside = np.abs(att_err[half:]) < 3.0 * sigma[half:, :3]
    assert np.mean(inside) > 0.95


def test_perfect_bias_free_tracking():
    """With zero noise and zero bias the filter tracks essentially exactly."""
    att_err, bias_err, _ = run_filter(
        bias=(0.0, 0.0, 0.0), st_noise=1e-9, gyro_noise=1e-15
    )
    assert np.max(np.abs(att_err[10:])) < 1e-9
    assert np.max(np.abs(bias_err[10:])) < 1e-12


def test_closed_loop_with_estimator():
    """Default config (estimator on): sim runs, pointing is maintained, and
    the wheel torque command is quieter than with raw sensor feedthrough."""
    cfg = cfg_mod.load(DEFAULT_YAML)
    cfg.simulation.duration_s = 400.0
    cfg.guidance.step.angle_deg = 0.0  # quiet nadir hold
    assert cfg.estimator.enabled
    res_est = simulate.run(cfg)

    cfg_raw = cfg_mod.load(DEFAULT_YAML)
    cfg_raw.simulation.duration_s = 400.0
    cfg_raw.guidance.step.angle_deg = 0.0
    cfg_raw.estimator.enabled = False
    res_raw = simulate.run(cfg_raw)

    # Estimator output is populated and converged
    tail = slice(len(res_est.t) // 2, None)
    est_rms = np.sqrt(np.mean(res_est.est_att_err[tail] ** 2))
    assert 0.0 < est_rms < 10.0 * ARCSEC
    # Pointing comparable or better, torque activity reduced
    def torque_rms(res):
        return np.sqrt(np.mean(res.torque_cmd[tail] ** 2))

    assert torque_rms(res_est) < torque_rms(res_raw)
    assert np.max(np.abs(res_est.att_err_deg[tail])) < 0.01
