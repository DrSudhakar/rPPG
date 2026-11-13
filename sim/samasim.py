"""SAMA-Sim: a physics-based simulator at skin-patch level (SYNTHETIC tier).

Chain: physiology (beats, pulse wave, HRV, ectopics) -> skin optics (ITA -> diffuse
reflectance, melanin-independent relative pulse amplitude [Chari et al. 2020 premise],
specular term) -> scene light and motion -> camera (exposure controller, photon shot noise,
read noise, gain, ADC noise, tone curve, exact expectation of 8-bit rounding) -> optional codec
surrogate (World A / World B) -> patch traces (T, 30, 3) and face-mean traces (T, 3).

Every free parameter lives in SimParams with its source in PARAM_SOURCES; free parameters are
varied in a sensitivity grid rather than trusted (prompt s.6 Tier 2).
Everything produced here is SYNTHETIC; anything that follows from a premise written into this
file is ASSUMPTION-DRIVEN.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import zlib

import numpy as np
from scipy import signal
from scipy.special import erf

# ----------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class SimParams:
    fps: float = 30.0
    fine_hz: float = 600.0                 # internal time grid for pulse integration
    # camera / photon budget
    k_e: float = 1200.0                    # electrons per pixel per (lux * s) at reflectance 1, green
    full_well: float = 6000.0              # electrons at DN 255 and unit gain
    read_noise_e: float = 3.0              # pre-gain read noise (electrons, per pixel)
    adc_noise_dn: float = 0.6              # post-gain noise (DN, per pixel)
    gamma: float = 2.2                     # tone curve exponent (output = lin^(1/gamma))
    max_exposure_s: float = 1 / 30         # spec step1
    max_gain: float = 16.0
    qe: tuple = (0.75, 1.0, 0.65)          # relative channel sensitivity x illuminant
    illum: tuple = (1.0, 1.0, 0.85)
    n_patches: int = 30
    pix_per_patch: float = 160.0           # effective independent pixels per patch (free; calibrated, configs/sim_calibrated.json)
    # skin
    b_star_mean: float = 17.0
    a_star: float = 12.0
    rel_amp_green: float = 0.0015          # pulsatile AC/DC in green, diffuse light (free; calibrated)
    rel_amp_sd_log: float = 0.45           # between-participant perfusion spread (log)
    pbv: tuple = (0.33, 0.77, 0.53)        # blood-volume pulse signature (de Haan & van Leest 2014)
    specular: float = 0.02                 # white specular reflectance (free)
    texture_sd: float = 0.06               # within-patch reflectance spread (for clipping)
    melanin_rel_amp_slope: float = 0.0     # counter-world: relative amplitude loss per unit melanin
    # scene / auto-exposure
    bg_reflectance: float = 0.30
    ae_face_weight: float = 0.5
    ae_target: float = 0.18                # linear mid-grey
    adaptive_target: float = 0.50          # SAMA step 1 target (from pilot; free)
    # motion (relative intensity SD by block kind)
    motion: tuple = (("rest", 0.0015), ("talk", 0.010), ("clip", 0.003), ("task", 0.003), ("breath", 0.002),
                     ("recovery", 0.006))
    colour_motion_frac: float = 0.05       # share of motion that is not colour-neutral
    roi_noise_rel: float = 0.0020          # colour-neutral ROI jitter / illumination flicker (relative SD)
    chan_noise_rel: float = 0.00005        # channel-independent residual (relative SD per patch)
    # codec surrogate v1 (per level), fitted to real ffmpeg on rendered tiles (sim/codec_real.py,
    # configs/codec_surrogate_v1.json): kept amplitude = floor + (top - floor) A^2/(A^2 + tau^2), A in DN
    # (peak), plus noise sigma (DN). v0 (pre-validation) had tau 0.03-0.24 and sigma 0.02-0.14 (D-23).
    codec_floor: tuple = (("2000", 0.073), ("1000", 0.099), ("500", 0.071), ("250", 0.304), ("vp8_500", 0.266), ("vp8_250", 0.208))
    codec_top: tuple = (("2000", 1.0), ("1000", 1.0), ("500", 1.0), ("250", 1.0), ("vp8_500", 0.828), ("vp8_250", 0.82))
    codec_tau: tuple = (("2000", 1.507), ("1000", 2.057), ("500", 2.494), ("250", 3.227), ("vp8_500", 0.01), ("vp8_250", 0.171))
    codec_sigma: tuple = (("2000", 0.123), ("1000", 0.17), ("500", 0.234), ("250", 0.451), ("vp8_500", 0.979), ("vp8_250", 1.082))
    world: str = "A"                       # A, B, null, melanin

    def get(self, table: str, key: str, default=0.0):
        return dict(getattr(self, table)).get(key, default)


PARAM_SOURCES = {
    "fps": "spec.yaml streams (30 fps consumer, ladder 30 fps)",
    "k_e/full_well/read_noise_e": "free; typical small-pixel CMOS values (EMVA 1288 style datasheets); sensitivity grid",
    "max_exposure_s": "spec.yaml sama_method.step1 (1/30 s cap)",
    "gamma": "sRGB-like tone curve; free",
    "pix_per_patch": "free; calibrated so lossless POS MAE matches the proposal's H1 anchor (LITERATURE: Chari et al. 2020 via proposal s.6)",
    "rel_amp_green": "free; calibrated jointly with pix_per_patch; plausible range 0.1-0.5 % (LITERATURE, rPPG reviews)",
    "pbv": "de Haan & van Leest 2014 signature",
    "specular": "free; sensitivity grid {0, 0.02, 0.05}",
    "melanin_rel_amp_slope": "counter-world for I-02 (0 = Chari premise)",
    "ae_*": "free; centre-weighted mid-grey metering typical of consumer AE",
    "adaptive_target": "free; proposal says 'fixed from the pilot'",
    "motion": "free; ordered rest < clip/task < recovery < talk",
    "codec_tau/codec_sigma": "calibrated against real ffmpeg encodes of rendered SAMA-Sim video (sim/codec_real.py)",
}

# ----------------------------------------------------------------------------- skin optics


def lab_to_linear_rgb(L: float, a: float, b: float) -> np.ndarray:
    """CIELAB (D65) -> linear sRGB reflectance."""
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29

    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))

    X, Y, Z = 0.95047 * finv(fx), 1.0 * finv(fy), 1.08883 * finv(fz)
    M = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    return np.clip(M @ np.array([X, Y, Z]), 0.005, 1.0)


def skin_reflectance(ita_deg: float, b_star: float = 17.0, a_star: float = 12.0) -> np.ndarray:
    """Diffuse skin reflectance (linear RGB) for a given ITA° at fixed b*: L* = 50 + b* tan(ITA)."""
    L = 50 + b_star * np.tan(np.radians(ita_deg))
    return lab_to_linear_rgb(float(np.clip(L, 20, 90)), a_star, b_star)


def melanin_index(ita_deg: float) -> float:
    """0 (ITA 60) .. 1 (ITA -40), a monotone proxy used only by the counter-world."""
    return float(np.clip((60 - ita_deg) / 100, 0, 1))


# ----------------------------------------------------------------------------- physiology


BLOCKS = {  # kind, minutes per light half, HR shift (bpm), ln RMSSD shift, resp Hz
    "paced_breathing": ("breath", 2.0, 0.0, +0.45, 0.1),
    "rest": ("rest", 3.0, 0.0, 0.0, 0.25),
    "mock_teleconsultation": ("talk", 4.0, +4.0, -0.10, 0.30),
    "neutral_clips": ("clip", 3.0, -1.0, +0.02, 0.25),
    "high_arousal_clips": ("clip", 3.0, +3.0, -0.12, 0.28),
    "mental_arithmetic": ("task", 3.0, +8.0, -0.30, 0.30),
    "recovery": ("rest", 2.0, +1.0, -0.05, 0.25),
    "post_step": ("recovery", 3.0, +25.0, -0.40, 0.35),
}
SEATED_BLOCKS = list(BLOCKS)  # step itself is not recorded seated; post_step is the 3 seated minutes


def ibi_series(dur_s: float, hr: float, ln_rmssd: float, resp_hz: float, rng, ectopic_rate: float = 0.003,
               hr_decay: float = 0.0) -> np.ndarray:
    """Beat times (s) of a block. IBI = mean + RSA + LF (0.1 Hz) + white; scaled to hit the target
    RMSSD; optional exponential HR decay (post-exercise); occasional premature beats."""
    target = np.exp(ln_rmssd) / 1000.0
    mu = 60.0 / hr
    n = int(dur_s / mu * 1.3) + 10
    k = np.arange(n)
    t_approx = k * mu
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    base = (np.sin(2 * np.pi * resp_hz * t_approx + ph1) + 0.5 * np.sin(2 * np.pi * 0.1 * t_approx + ph2)
            + 0.25 * rng.standard_normal(n))
    d = np.sqrt(np.mean(np.diff(base) ** 2))
    var = base / d * target
    if hr_decay > 0:  # HR elevated then decaying with tau = 60 s
        mu_t = 60.0 / (hr - hr_decay * (1 - np.exp(-t_approx / 60.0)))
    else:
        mu_t = np.full(n, mu)
    ibi = np.clip(mu_t + var, 0.33, 1.8)
    ect = rng.random(n) < ectopic_rate
    for i in np.where(ect)[0]:
        if 0 < i < n - 1:
            s = 0.3 * ibi[i]
            ibi[i] -= s
            ibi[i + 1] += s
    t = np.concatenate([[rng.uniform(0, ibi[0])], ibi[1:]]).cumsum()
    return t[t < dur_s]


def pulse_wave(beat_t: np.ndarray, t: np.ndarray, ptt=0.22) -> np.ndarray:
    """Blood-volume waveform at times t: systolic + diastolic Gaussians on the beat phase.
    ptt: scalar or per-beat pulse transit times (s)."""
    bt = np.sort(beat_t + ptt)
    idx = np.searchsorted(bt, t) - 1
    valid = (idx >= 0) & (idx < len(bt) - 1)
    out = np.zeros_like(t)
    i = idx[valid]
    ph = (t[valid] - bt[i]) / (bt[i + 1] - bt[i])
    out[valid] = np.exp(-((ph - 0.18) / 0.075) ** 2) + 0.35 * np.exp(-((ph - 0.48) / 0.11) ** 2)
    out[~valid] = 0.3
    return out - 0.3


def motion_series(n: int, fs: float, sd: float, rng, band=(0.3, 4.0)) -> np.ndarray:
    if sd <= 0:
        return np.zeros(n)
    x = rng.standard_normal(n + 200)
    sos = signal.butter(2, band, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)[100:-100]
    return y / (y.std() + 1e-12) * sd


# ----------------------------------------------------------------------------- participants


@dataclass
class Participant:
    pid: str
    ita: float
    stratum: str
    sex: str
    age: int
    hr_rest: float
    ln_rmssd_rest: float
    reactivity_ln: float        # individual arithmetic-minus-rest change in ln RMSSD
    rel_amp: float              # participant perfusion multiplier
    beard: bool
    glasses: bool
    motion_scale: float
    b_star: float
    seed: int
    ptt: float = 0.22


def stratum_of(ita: float) -> str:
    return "lighter" if ita > 28 else ("middle" if ita > 10 else "darker")


ITA_RANGES = {"lighter": (28.0, 50.0), "middle": (10.0, 28.0), "darker": (-25.0, 10.0)}


def make_participant(pid: str, stratum: str, rng, sex: str | None = None, ita: float | None = None) -> Participant:
    lo, hi = ITA_RANGES[stratum]
    ita = float(rng.uniform(lo + 1e-6, hi)) if ita is None else ita
    sex = sex or ("F" if rng.random() < 0.5 else "M")
    age = int(rng.integers(18, 61))
    hr = float(np.clip(rng.normal(74 if sex == "F" else 70, 9), 50, 105))
    ln_rmssd = float(np.clip(rng.normal(np.log(42) - 0.012 * (age - 35), 0.45), np.log(8), np.log(150)))
    return Participant(pid, ita, stratum_of(ita), sex, age, hr, ln_rmssd,
                       reactivity_ln=float(rng.normal(-0.30, 0.22)),
                       rel_amp=float(np.exp(rng.normal(0, 0.45))),
                       beard=bool(sex == "M" and rng.random() < 0.45), glasses=bool(rng.random() < 0.3),
                       motion_scale=float(np.exp(rng.normal(0, 0.35))), b_star=float(rng.normal(17, 2.0)),
                       seed=int(rng.integers(0, 2 ** 31)), ptt=float(np.clip(rng.normal(0.22, 0.03), 0.12, 0.35)))


def make_cohort(n_per_stratum: int, rng, women_frac: float = 0.5) -> list[Participant]:
    out = []
    for s in ("lighter", "middle", "darker"):
        nw = int(round(n_per_stratum * women_frac))
        for j in range(n_per_stratum):
            out.append(make_participant(f"{s[0].upper()}{j:03d}", s, rng, sex="F" if j < nw else "M"))
    return out


# ----------------------------------------------------------------------------- camera


@dataclass(frozen=True)
class Device:
    name: str
    fps: float
    pix_scale: float      # relative effective pixels per patch
    read_noise_scale: float
    gamma: float
    exposure: str         # "auto" | "adaptive" | "fixed"
    bits: int = 8
    internal_codec: str = ""   # phone encoder (I-15)


DEVICES = {
    "reference": Device("reference", 60.0, 2.0, 0.7, 1.0, "fixed", bits=10),
    "reference30": Device("reference30", 30.0, 1.0, 0.7, 1.0, "fixed", bits=8),   # ladder source: 720p 30 fps
    "reference30hq": Device("reference30hq", 30.0, 2.0, 0.7, 1.0, "fixed", bits=10),  # fps-only contrast to "reference"
    "webcam": Device("webcam", 30.0, 0.8, 1.3, 2.2, "auto"),
    "phoneA": Device("phoneA", 30.0, 1.0, 1.0, 2.2, "auto", internal_codec="phone"),
    "phoneB": Device("phoneB", 30.0, 1.0, 1.0, 2.2, "adaptive", internal_codec="phone"),
}


def choose_exposure(p: SimParams, dev: Device, lux: float, face_refl_g: float) -> tuple[float, float]:
    """Return (exposure_s, gain). Auto: centre-weighted metering to mid-grey. Adaptive (SAMA
    step 1): face to target share of full scale, exposure first (capped), then gain, keeping
    under 1 % of skin pixels clipped (texture spread). Fixed: reference camera set once for the
    lab at 500 lux on a mid-grey card (no per-person adaptation)."""
    flux = p.k_e * lux / p.full_well  # full-scale fraction per second per unit reflectance
    if dev.exposure == "fixed":
        need = 0.5 / (flux * 0.3) if lux > 0 else p.max_exposure_s
        need = min(need, 0.5 / (p.k_e * 500 / p.full_well * 0.3))  # set at 500 lux, not changed
        e = min(need, p.max_exposure_s)
        return e, 1.0
    if dev.exposure == "auto":
        metered = p.ae_face_weight * face_refl_g + (1 - p.ae_face_weight) * p.bg_reflectance
        want = p.ae_target / (flux * metered)
    else:
        # largest target with <1 % of skin pixels above full scale: mean*(1+2.33*texture_sd) <= 1
        tgt = min(p.adaptive_target, 1.0 / (1 + 2.33 * p.texture_sd + p.specular / max(face_refl_g, 1e-3)))
        want = tgt / (flux * face_refl_g)
    e = min(want, p.max_exposure_s)
    g = float(np.clip(want / e, 1.0, p.max_gain))
    return e, g


def expected_quantised(mu: np.ndarray, sd: np.ndarray, levels: int = 255) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of round(mu + sd*Z) clipped to [0, levels] for Gaussian pixel noise.
    For sd >= 1 DN rounding is fully dithered (mean = mu, variance = sd^2 + 1/12, error < 1e-3 DN);
    below that the exact sum over levels within +-7 of mu is used (covers > 7 sd)."""
    sd = np.maximum(np.asarray(sd, float), 1e-6)
    mu = np.asarray(mu, float)
    m_d = np.clip(mu, 0, levels)
    v_d = sd ** 2 + 1 / 12
    small = sd < 1.0
    if not small.any():
        return m_d, v_d
    mus, sds = mu[small], sd[small]
    base = np.floor(mus)
    m1 = np.zeros_like(mus)
    m2 = np.zeros_like(mus)
    for k in range(-7, 9):
        lv = np.clip(base + k, 0, levels)
        lo = (base + k - 0.5 - mus) / (sds * np.sqrt(2))
        hi = (base + k + 0.5 - mus) / (sds * np.sqrt(2))
        pr = 0.5 * (erf(hi) - erf(lo))
        m1 += pr * lv
        m2 += pr * lv * lv
    m_d = m_d.copy(); v_d = v_d.copy()
    m_d[small] = m1
    v_d[small] = np.maximum(m2 - m1 ** 2, 0.0)
    return m_d, v_d


# ----------------------------------------------------------------------------- recording


@dataclass
class Recording:
    rgb_patches: np.ndarray     # (T, P, 3) DN, after tone curve (what a camera delivers)
    rgb_face: np.ndarray        # (T, 3) pixel-weighted face mean
    fps: float
    beats: np.ndarray           # ECG R-peak times (s), the reference
    beats_valid: np.ndarray     # bool per beat (ECG artefact flags)
    meta: dict = field(default_factory=dict)


def _patch_layout(p: SimParams, part: Participant, rng):
    P = p.n_patches
    refl_mult = np.clip(rng.normal(1.0, 0.05, P), 0.8, 1.2)
    perf = np.exp(rng.normal(0, 0.35, P))
    pix = np.full(P, p.pix_per_patch) * np.exp(rng.normal(0, 0.2, P))
    if part.beard:  # lower-cheek patches partly masked by beard
        pix[20:] *= 0.4
    if part.glasses:
        pix[:4] *= 0.5
    return refl_mult, perf, pix


def simulate_block(part: Participant, block: str, light: str, device: str, p: SimParams,
                   codec: str = "lossless", seed_offset: int = 0, dur_s: float | None = None) -> Recording:
    """Simulate one recorded block for one participant under one light and device."""
    kind, minutes, dhr, dln, resp = BLOCKS[block]
    dev = DEVICES[device]
    rng = np.random.default_rng([part.seed, zlib.crc32(f'{block}|{light}'.encode()), seed_offset])   # physiology + motion
    rng_cam = np.random.default_rng([part.seed, zlib.crc32(f'{block}|{light}|{device}|{codec}'.encode()), seed_offset])
    dur = dur_s or minutes * 60.0
    lux = 500.0 if light == "standard" else 100.0
    # physiology (same beats for every device/codec of this block: seed excludes device)
    ln_r = part.ln_rmssd_rest + dln + (part.reactivity_ln + 0.30 if block == "mental_arithmetic" else 0.0)
    hr = part.hr_rest + dhr
    decay = 22.0 if block == "post_step" else 0.0
    beats = ibi_series(dur + 2.0, hr, ln_r, resp, rng, hr_decay=decay)
    beats_valid = np.ones(beats.size, bool)
    if rng.random() < 0.15:  # ECG artefact burst: 3-8 s of unusable ECG
        a = rng.uniform(0, dur)
        beats_valid[(beats > a) & (beats < a + rng.uniform(3, 8))] = False
    # optics
    ita_eff = 15.0 if p.world == "null" else part.ita
    R = skin_reflectance(ita_eff, part.b_star, p.a_star)
    refl_mult, perf, pix = _patch_layout(p, part, np.random.default_rng(part.seed))
    pix = pix * dev.pix_scale
    amp = p.rel_amp_green * part.rel_amp * perf
    if p.world == "melanin" or p.melanin_rel_amp_slope > 0:
        slope = p.melanin_rel_amp_slope or 0.5
        amp = amp * (1 - slope * melanin_index(part.ita))
    sig_vec = np.array(p.pbv) / p.pbv[1]
    # exposure
    e_s, gain = choose_exposure(p, dev, lux, float(R[1]))
    fps = dev.fps
    T = int(dur * fps)
    tf = np.arange(int((dur + 0.5) * p.fine_hz)) / p.fine_hz
    nb = beats.size
    ptt_b = part.ptt + 0.010 * np.sin(2 * np.pi * resp * beats + rng.uniform(0, 6.3)) + 0.004 * rng.standard_normal(nb)
    wave = pulse_wave(beats, tf, ptt_b)
    cs = np.concatenate([[0], np.cumsum(wave)]) / p.fine_hz
    t0 = np.arange(T) / fps
    i0 = np.round(t0 * p.fine_hz).astype(int)
    i1 = np.round((t0 + e_s) * p.fine_hz).astype(int)
    integ = (cs[i1] - cs[i0]) / np.maximum(e_s, 1e-9)       # exposure-averaged pulse, (T,)
    integ = integ / (wave.std() + 1e-12)
    msd = dict(p.motion).get(kind, 0.003) * part.motion_scale
    m_common = motion_series(T, fps, msd, rng)
    m_patch = m_common[:, None] * np.exp(rng.normal(0, 0.3, p.n_patches))[None, :]
    m_col = motion_series(T, fps, msd * p.colour_motion_frac, rng)[:, None, None] * rng.normal(0, 1, (1, p.n_patches, 3))
    # ROI jitter / illumination flicker: mostly colour-neutral (shared by channels), small independent part
    roi = (p.roi_noise_rel * motion_series(T, fps, 1.0, rng_cam, band=(0.2, 6.0))[:, None, None] * np.exp(rng_cam.normal(0, 0.3, (1, p.n_patches, 1)))
           + p.chan_noise_rel * rng_cam.standard_normal((T, p.n_patches, 3)))
    # signal in electrons per pixel (mean), per patch and channel
    qe = np.array(p.qe) * np.array(p.illum if light == "standard" else (1.05, 1.0, 0.75))
    diffuse = (R[None, :] * refl_mult[:, None])                              # (P,3)
    spec = p.specular * np.ones(3)
    pulse = 1 + amp[None, :, None] * sig_vec[None, None, :] * integ[:, None, None]  # (T,P,3)
    lin_refl = (diffuse[None] * pulse + spec[None, None, :] * (1 + 3 * m_patch[:, :, None])) * (1 + m_patch[:, :, None] + m_col + roi)
    electrons = p.k_e * lux * e_s * qe[None, None, :] * lin_refl              # mean e- per pixel
    electrons = np.clip(electrons, 0, None)
    # per-pixel noise (e-) -> DN after gain, plus ADC noise
    dn_per_e = 255.0 / p.full_well * gain
    rn = p.read_noise_e * dev.read_noise_scale
    lin_dn = electrons * dn_per_e
    pix_sd_dn = np.sqrt(electrons + rn ** 2) * dn_per_e
    pix_sd_dn = np.sqrt(pix_sd_dn ** 2 + p.adc_noise_dn ** 2)
    levels = 255 if dev.bits == 8 else 1023
    scale = levels / 255.0
    if dev.gamma != 1.0:  # tone curve applied per pixel: linearise noise through the slope
        x = np.clip(lin_dn / 255.0, 1e-6, 1.0)
        out_mu = levels * x ** (1 / dev.gamma)
        slope = levels / dev.gamma * x ** (1 / dev.gamma - 1) / 255.0
        out_sd = pix_sd_dn * slope
    else:
        out_mu = np.clip(lin_dn * scale, 0, levels)
        out_sd = pix_sd_dn * scale
    q_mu, q_var = expected_quantised(out_mu, out_sd, levels)
    patch = q_mu + np.sqrt(q_var / pix[None, :, None]) * rng_cam.standard_normal(q_mu.shape)
    patch = patch / scale  # report on a 0-255 scale for every device
    # clipped share of skin pixels (texture spread) for the exposure check
    clip_frac = float(np.mean(0.5 * (1 - erf((levels - out_mu.mean(0)) / (np.sqrt(2) * (out_mu.mean(0) * p.texture_sd + 1e-6))))))
    # internal phone encoder, then the transmission codec
    if dev.internal_codec:
        patch = codec_surrogate(patch, "phone", p, fps, rng=rng_cam)
    if codec != "lossless":
        patch = codec_surrogate(patch, codec, p, fps, rng=rng_cam)
    w = pix / pix.sum()
    face = (patch * w[None, :, None]).sum(1)
    meta = {"pid": part.pid, "ita": part.ita, "stratum": part.stratum, "block": block, "kind": kind, "light": light,
            "device": device, "codec": codec, "lux": lux, "exposure_s": e_s, "gain": gain,
            "face_dn_g": float(face[:, 1].mean()), "electrons_g": float(electrons[..., 1].mean()),
            "clip_frac": clip_frac, "pix": pix, "world": p.world, "gamma": dev.gamma,
            "dn_per_e": dn_per_e, "read_noise_e": rn, "adc_noise_dn": p.adc_noise_dn,
            "ppg": integ.astype(np.float32),
            "ppg_ecg": pulse_wave(beats, t0 + e_s / 2, 0.22).astype(np.float32)}
    return Recording(patch.astype(np.float32), face, fps, beats, beats_valid, meta)


# ----------------------------------------------------------------------------- codec surrogate


def codec_surrogate(patch: np.ndarray, level: str, p: SimParams, fps: float, rng=None) -> np.ndarray:
    """Analytic stand-in for a video codec acting on patch-mean traces.

    World A: temporal variations are shrunk by a Wiener-like factor A^2/(A^2+tau^2) where A is
    the local (2 s) RMS of the variation in DN, so small absolute pulses (darker skin, dim light)
    are removed first; plus noise sigma. World B: no amplitude dependence, noise sigma only.
    'phone' is the phone's own encoder (light). Calibrated against real ffmpeg in codec_real.py.
    """
    if level == "phone":   # phone's own high-bitrate encoder: mild
        floor, top, tau, sig = 0.5, 1.0, 0.3, 0.05
    else:
        floor, top = p.get("codec_floor", level), p.get("codec_top", level, 1.0)
        tau, sig = p.get("codec_tau", level), p.get("codec_sigma", level)
    rng = rng or np.random.default_rng(zlib.crc32(f'{level}|{patch.shape}|{float(patch[0,0,0]):.6f}'.encode()))
    dc = signal.savgol_filter(patch, int(fps * 2) | 1, 1, axis=0)
    ac = patch - dc
    if p.world == "B":  # relative noise, no amplitude-dependent loss: same relative SNR cost for every tone
        return dc + ac + (sig / 120.0) * np.abs(dc) * rng.standard_normal(patch.shape)
    if tau > 0:
        k = int(fps * 2)
        amp = np.sqrt(2.0) * np.sqrt(signal.convolve(ac ** 2, np.ones((k, 1, 1)) / k, mode="same"))
        ac = ac * (floor + (top - floor) * amp ** 2 / (amp ** 2 + tau ** 2))
    return dc + ac + sig * rng.standard_normal(patch.shape)


def with_params(**kw) -> SimParams:
    return replace(SimParams(), **kw)


CHART_REFLECTANCE = np.linspace(0.03, 0.90, 24)  # 24-patch chart, green-channel reflectances (grey-ramp stand-in)


def chart_for(part: "Participant", device: str, light: str, p: SimParams, seed: int = 0):
    ita_eff = 15.0 if p.world == "null" else part.ita
    return simulate_chart(device, light, p, float(skin_reflectance(ita_eff, part.b_star, p.a_star)[1]),
                          seed=zlib.crc32(f"{part.pid}|{device}|{light}|{seed}".encode()))


def simulate_chart(device: str, light: str, p: SimParams, face_refl_g: float, n_frames: int = 150,
                   n_pix: float = 400.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Static chart frames through the same camera model and exposure as the block: returns
    (mean DN per patch (24,), static linearised traces (n_frames, 24)) for step 2."""
    dev = DEVICES[device]
    lux = 500.0 if light == "standard" else 100.0
    e_s, gain = choose_exposure(p, dev, lux, face_refl_g)
    rng = np.random.default_rng(seed)
    el = p.k_e * lux * e_s * CHART_REFLECTANCE[None, :] * np.ones((n_frames, 1))
    dn_per_e = 255.0 / p.full_well * gain
    rn = p.read_noise_e * dev.read_noise_scale
    lin = el * dn_per_e
    sd = np.sqrt(np.sqrt(el + rn ** 2) ** 2 * dn_per_e ** 2 + p.adc_noise_dn ** 2)
    lin_obs = lin + sd / np.sqrt(n_pix) * rng.standard_normal(lin.shape)
    x = np.clip(lin.mean(0) / 255.0, 1e-6, 1.0)
    out = 255.0 * x ** (1 / dev.gamma)
    return out, lin_obs
