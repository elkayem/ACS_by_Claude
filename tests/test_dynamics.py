import numpy as np

from spacecraft_acs import quaternion as qt
from spacecraft_acs.actuators import ReactionWheelSet
from spacecraft_acs.config import ModeConfig, SpacecraftConfig, WheelConfig
from spacecraft_acs.dynamics import FlexibleSpacecraft

J = np.diag([8000.0, 4500.0, 6500.0])


def make_flex(damping=0.005):
    modes = [
        ModeConfig(freq_hz=0.10, damping=damping, participation=[45.0, 0.0, 0.0]),
        ModeConfig(freq_hz=0.35, damping=damping, participation=[0.0, 25.0, 0.0]),
    ]
    return FlexibleSpacecraft(SpacecraftConfig(inertia=J, modes=modes))


def simulate_free(sc, x0, dt, n_steps):
    x = x0.copy()
    zero = np.zeros(3)
    history = [x.copy()]
    for _ in range(n_steps):
        x = sc.rk4_step(x, dt, zero, zero)
        history.append(x.copy())
    return np.array(history)


def test_rigid_body_limit():
    """With no modes, constant torque about a principal axis gives ω = T t / J."""
    sc = FlexibleSpacecraft(SpacecraftConfig(inertia=J, modes=[]))
    x = sc.initial_state()
    torque = np.array([0.1, 0.0, 0.0])
    dt, t_end = 0.1, 50.0
    for _ in range(int(t_end / dt)):
        x = sc.rk4_step(x, dt, torque, np.zeros(3))
    _, omega, _, _, _ = sc.unpack(x)
    assert np.isclose(omega[0], torque[0] * t_end / J[0, 0], rtol=1e-6)


def test_modal_free_response_uncoupled():
    """A mode with zero participation rings at exactly its configured ωd, ζ."""
    modes = [ModeConfig(freq_hz=0.10, damping=0.02, participation=[0.0, 0.0, 0.0])]
    sc = FlexibleSpacecraft(SpacecraftConfig(inertia=J, modes=modes))
    x0 = sc.initial_state()
    x0[7] = 1.0  # initial modal displacement
    dt = 0.05
    t_end = 40.0
    hist = simulate_free(sc, x0, dt, int(t_end / dt))
    t = np.arange(len(hist)) * dt
    wn = 2 * np.pi * 0.10
    zeta = 0.02
    wd = wn * np.sqrt(1 - zeta**2)
    eta_analytic = np.exp(-zeta * wn * t) * (
        np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t)
    )
    assert np.allclose(hist[:, 7], eta_analytic, atol=1e-6)


def test_energy_conservation_undamped():
    """Unforced, undamped coupled system conserves total mechanical energy."""
    sc = make_flex(damping=0.0)
    x0 = sc.initial_state(omega=np.array([1e-3, -2e-3, 5e-4]))
    x0[7] = 0.5
    x0[8] = -0.2
    e0 = sc.kinetic_energy(x0) + sc.potential_energy(x0)
    hist = simulate_free(sc, x0, 0.01, 10000)  # 100 s
    e_end = sc.kinetic_energy(hist[-1]) + sc.potential_energy(hist[-1])
    # Exactly conservative model; residual is RK4 truncation error
    assert np.isclose(e_end, e0, rtol=1e-5)


def test_momentum_conservation_with_wheels():
    """Internal wheel torque must not change total inertial angular momentum."""
    sc = make_flex()
    x = sc.initial_state(omega=np.array([1e-3, 2e-3, -1e-3]), h_wheel=np.array([5.0, -3.0, 1.0]))
    x[7] = 0.3

    def h_inertial(x):
        q, omega, _, eta_dot, h_w = sc.unpack(x)
        h_body = sc.J @ omega + sc.L @ eta_dot + h_w
        return qt.dcm(q).T @ h_body  # body -> inertial

    h0 = h_inertial(x)
    torque = np.array([0.05, -0.02, 0.03])  # wheel (internal) torque only
    for _ in range(2000):
        x = sc.rk4_step(x, 0.05, torque, np.zeros(3))
    assert np.allclose(h_inertial(x), h0, atol=1e-9)


def test_coupled_mode_frequency_shift():
    """Free-free coupled frequency is ωn·sqrt(J/(J−l²)) for one mode, one axis."""
    l, wn_hz = 45.0, 0.10
    modes = [ModeConfig(freq_hz=wn_hz, damping=0.0, participation=[l, 0.0, 0.0])]
    sc = FlexibleSpacecraft(SpacecraftConfig(inertia=J, modes=modes))
    x0 = sc.initial_state()
    x0[7] = 1.0
    dt = 0.02
    hist = simulate_free(sc, x0, dt, 20000)  # 400 s
    eta = hist[:, 7]
    # Measure the ringing frequency from a zero-padded FFT peak
    n_fft = 8 * len(eta)
    freqs = np.fft.rfftfreq(n_fft, dt)
    spectrum = np.abs(np.fft.rfft(eta * np.hanning(len(eta)), n=n_fft))
    peak = freqs[np.argmax(spectrum)]
    expected = wn_hz * np.sqrt(J[0, 0] / (J[0, 0] - l**2))
    assert np.isclose(peak, expected, rtol=0.005)


def test_wheel_torque_and_momentum_saturation():
    wheels = ReactionWheelSet(WheelConfig(max_torque=0.2, max_momentum=10.0))
    # Torque clamp
    t = wheels.apply(np.array([1.0, -0.5, 0.1]), np.zeros(3))
    assert np.allclose(t, [0.2, -0.2, 0.1])
    # Momentum saturation: wheel momentum rate is −T, so h_w at +limit blocks
    # further negative body torque on that axis but allows positive
    h = np.array([10.0, 0.0, 0.0])
    t = wheels.apply(np.array([-0.1, 0.0, 0.0]), h)
    assert t[0] == 0.0
    t = wheels.apply(np.array([0.1, 0.0, 0.0]), h)
    assert t[0] == 0.1
    # Ideal mode bypasses limits
    ideal = ReactionWheelSet(WheelConfig(ideal=True, max_torque=0.2, max_momentum=10.0))
    assert np.allclose(ideal.apply(np.array([5.0, 0, 0]), h), [5.0, 0, 0])
