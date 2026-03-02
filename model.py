"""
model.py -- Bayesian Attention Model for Value-Based Decision Making (v4)
=========================================================================

A computational model of gaze allocation during sequential value-based
decision tasks, built on Bayesian belief updating and information-theoretic
attention control. This implementation reproduces the key behavioral
phenomena reported in Ferro et al. (2024, Nature Communications).

Architecture overview:
    1. Gaussian belief states N(mu, sigma^2) for each option
    2. Kalman filter updates with distinct observation noise for external
       stimuli vs. internal memory replay
    3. Memory leak (sigma^2 growth) for unattended options
    4. Value-of-Information (VOI) driven attention switching in delay2
    5. Preference-biased gaze stickiness (pref_bias parameter)

Author: Puzhi YU
"""

import numpy as np
from scipy.stats import norm as _norm

_phi = _norm.pdf   # standard normal PDF
_Phi = _norm.cdf   # standard normal CDF


# ---------------------------------------------------------------------------
# 1. Value of Information (VOI) computation
# ---------------------------------------------------------------------------
def compute_voi_batch(mu_L, mu_R, sigma2_unattended, sigma2_min):
    """Compute the Expected Improvement-based Value of Information.

    Uses the analytical formula from Bayesian optimization:
        VOI = delta_sigma * [phi(z) + z * Phi(z)]
    where z = -|mu_L - mu_R| / delta_sigma.

    Parameters
    ----------
    mu_L, mu_R : np.ndarray
        Current belief means for left and right options.
    sigma2_unattended : np.ndarray
        Variance of the unattended option (candidate for switching to).
    sigma2_min : float
        Minimum achievable variance after Kalman convergence.

    Returns
    -------
    np.ndarray
        Non-negative VOI values for each trial.
    """
    sigma_un = np.sqrt(np.maximum(sigma2_unattended, 1e-10))
    sigma_min = np.sqrt(np.maximum(sigma2_min, 1e-10))
    delta_sigma = np.maximum(sigma_un - sigma_min, 1e-10)
    mu_diff = np.abs(mu_L - mu_R)
    z = -mu_diff / delta_sigma
    voi = delta_sigma * (_phi(z) + z * _Phi(z))
    return np.maximum(voi, 0.0)


# ---------------------------------------------------------------------------
# 2. Main model class
# ---------------------------------------------------------------------------
class BayesianAttentionModel:
    """Bayesian attention model with VOI-driven gaze allocation.

    Simulates the sequential offer paradigm from Ferro et al. (2024):
        offer1 (400 ms) -> delay1 (600 ms) -> offer2 (400 ms) -> delay2 (600 ms)

    During delay2, attention switches between options based on the
    Value of Information criterion, producing the "look-at-nothing"
    behavior observed in primate data.

    Parameters
    ----------
    alpha : float
        Memory leak rate -- per-step increase in sigma^2 for unattended option.
    c_shift : float
        Fixed cost of attention switching (VOI must exceed this threshold).
    sigma2_obs : float
        Observation noise variance for external stimuli.
    sigma2_init : float
        Initial belief uncertainty for each option at trial onset.
    sigma2_decision : float
        Additional noise variance injected at the final choice stage.
    dt : float
        Duration of each simulation time step in milliseconds.
    min_fixation_steps : int
        Minimum number of steps before attention can switch again.
    switch_noise : float
        Standard deviation of stochastic noise added to switching decision.
    pref_bias : float
        Preference-driven gaze stickiness -- increases the effective switching
        threshold when currently attending the preferred (higher-mu) option.
    """

    # 2.1 Initialization and epoch timing
    # .....................................................................
    def __init__(self, alpha=0.05, c_shift=0.3, sigma2_obs=0.5,
                 sigma2_init=4.0, sigma2_decision=0.2, dt=10.0,
                 min_fixation_steps=5, switch_noise=0.05, pref_bias=0.0):
        self.alpha = alpha
        self.c_shift = c_shift
        self.sigma2_obs = sigma2_obs
        self.sigma2_init = sigma2_init
        self.sigma2_decision = sigma2_decision
        self.dt = dt
        self.min_fixation_steps = min_fixation_steps
        self.switch_noise = switch_noise
        self.pref_bias = pref_bias

        # Epoch lengths (number of time steps)
        self.n_offer1 = int(400 / dt)
        self.n_delay1 = int(600 / dt)
        self.n_offer2 = int(400 / dt)
        self.n_delay2 = int(600 / dt)
        self.n_total = self.n_offer1 + self.n_delay1 + self.n_offer2 + self.n_delay2

        # Cumulative epoch boundaries
        self.t_offer1_end = self.n_offer1
        self.t_delay1_end = self.t_offer1_end + self.n_delay1
        self.t_offer2_end = self.t_delay1_end + self.n_offer2
        self.t_delay2_end = self.t_offer2_end + self.n_delay2

        # Theoretical Kalman lower bound on variance
        self.sigma2_min = sigma2_obs * 0.5

    # 2.2 Batch simulation
    # .....................................................................
    def run_batch(self, ev_L_arr, ev_R_arr, rng=None, record_dynamics=False):
        """Run the model on a batch of trials in parallel.

        Parameters
        ----------
        ev_L_arr, ev_R_arr : array-like
            True expected values for left and right options (length N).
        rng : np.random.Generator, optional
            Random number generator (default: seed 42).
        record_dynamics : bool
            If True, store full time-series of beliefs and VOI.

        Returns
        -------
        dict
            Trial-level results including choices, final beliefs, attention
            history, and (optionally) full dynamics.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        N = len(ev_L_arr)
        ev_L = np.asarray(ev_L_arr, dtype=np.float64)
        ev_R = np.asarray(ev_R_arr, dtype=np.float64)

        # Initialize belief states
        mu_L = np.zeros(N)
        mu_R = np.zeros(N)
        sigma2_L = np.full(N, self.sigma2_init)
        sigma2_R = np.full(N, self.sigma2_init)

        # Attention state: True = looking Right, False = looking Left
        att_R = np.zeros(N, dtype=bool)
        att_history = np.zeros((self.n_total, N), dtype=bool)

        # Fixation duration counter (for minimum fixation constraint)
        steps_since_switch = np.zeros(N, dtype=int)

        # Switch counter for delay2
        n_switches_d2 = np.zeros(N, dtype=int)

        # Optional: full dynamics recording
        if record_dynamics:
            mu_L_hist = np.zeros((self.n_total, N))
            mu_R_hist = np.zeros((self.n_total, N))
            s2_L_hist = np.zeros((self.n_total, N))
            s2_R_hist = np.zeros((self.n_total, N))
            voi_hist = np.zeros((self.n_total, N))

        # Observation noise levels
        obs_noise_ext = self.sigma2_obs          # external stimulus
        obs_noise_int = self.sigma2_obs * 3.0    # internal replay (noisier)

        # -------------------------------------------------------------------
        # 2.3 Main simulation loop
        # -------------------------------------------------------------------
        for t in range(self.n_total):

            # Determine current epoch
            if t < self.t_offer1_end:
                epoch = 'offer1'
            elif t < self.t_delay1_end:
                epoch = 'delay1'
            elif t < self.t_offer2_end:
                epoch = 'offer2'
            else:
                epoch = 'delay2'

            # ---------------------------------------------------------------
            # 2.3.1 Attention control
            # ---------------------------------------------------------------
            if epoch == 'offer1':
                # Forced fixation on L (stimulus-driven)
                att_R[:] = False
                steps_since_switch[:] = 0

            elif epoch == 'delay1':
                # Stay on L -- offer R has not been presented yet
                att_R[:] = False

            elif epoch == 'offer2':
                if t == self.t_delay1_end:
                    # Forced switch to R at offer2 onset
                    att_R[:] = True
                    steps_since_switch[:] = 0
                else:
                    att_R[:] = True
                    steps_since_switch += 1

            elif epoch == 'delay2':
                # VOI-driven free switching
                steps_since_switch += 1
                can_switch = steps_since_switch >= self.min_fixation_steps

                if can_switch.any():
                    voi_switch = np.zeros(N)
                    pref_stay = np.zeros(N)

                    # Currently looking L -> evaluate switching to R
                    m_L = can_switch & (~att_R)
                    if m_L.any():
                        voi_switch[m_L] = compute_voi_batch(
                            mu_L[m_L], mu_R[m_L], sigma2_R[m_L], self.sigma2_min)
                        pref_stay[m_L] = self.pref_bias * (mu_L[m_L] - mu_R[m_L])

                    # Currently looking R -> evaluate switching to L
                    m_R = can_switch & att_R
                    if m_R.any():
                        voi_switch[m_R] = compute_voi_batch(
                            mu_L[m_R], mu_R[m_R], sigma2_L[m_R], self.sigma2_min)
                        pref_stay[m_R] = self.pref_bias * (mu_R[m_R] - mu_L[m_R])

                    # Switch when VOI exceeds cost + preference inertia
                    noise = rng.normal(0, self.switch_noise, N) if self.switch_noise > 0 else 0
                    should_switch = can_switch & (voi_switch + noise > self.c_shift + pref_stay)
                    att_R[should_switch] = ~att_R[should_switch]
                    steps_since_switch[should_switch] = 0
                    n_switches_d2[should_switch] += 1

            att_history[t] = att_R.copy()

            # ---------------------------------------------------------------
            # 2.3.2 Belief updates (Kalman filter)
            # ---------------------------------------------------------------
            looking_L = ~att_R
            looking_R = att_R

            if epoch == 'offer1':
                # External observation of L
                obs = ev_L + rng.normal(0, np.sqrt(obs_noise_ext), N)
                kg = sigma2_L / (sigma2_L + obs_noise_ext)
                mu_L += kg * (obs - mu_L)
                sigma2_L *= (1 - kg)

            elif epoch == 'delay1':
                # Internal replay of L (noisier observation)
                obs = ev_L + rng.normal(0, np.sqrt(obs_noise_int), N)
                kg = sigma2_L / (sigma2_L + obs_noise_int)
                mu_L += kg * (obs - mu_L)
                sigma2_L *= (1 - kg)

            elif epoch == 'offer2':
                # External observation of R + memory leak on L
                obs_R = ev_R + rng.normal(0, np.sqrt(obs_noise_ext), N)
                kg_R = sigma2_R / (sigma2_R + obs_noise_ext)
                mu_R += kg_R * (obs_R - mu_R)
                sigma2_R *= (1 - kg_R)
                sigma2_L += self.alpha

            elif epoch == 'delay2':
                # Attended option: internal replay; unattended: memory leak
                if looking_L.any():
                    obs_L = ev_L[looking_L] + rng.normal(0, np.sqrt(obs_noise_int), looking_L.sum())
                    kg_L = sigma2_L[looking_L] / (sigma2_L[looking_L] + obs_noise_int)
                    mu_L[looking_L] += kg_L * (obs_L - mu_L[looking_L])
                    sigma2_L[looking_L] *= (1 - kg_L)
                    sigma2_R[looking_L] += self.alpha

                if looking_R.any():
                    obs_R = ev_R[looking_R] + rng.normal(0, np.sqrt(obs_noise_int), looking_R.sum())
                    kg_R = sigma2_R[looking_R] / (sigma2_R[looking_R] + obs_noise_int)
                    mu_R[looking_R] += kg_R * (obs_R - mu_R[looking_R])
                    sigma2_R[looking_R] *= (1 - kg_R)
                    sigma2_L[looking_R] += self.alpha

            # Record dynamics if requested
            if record_dynamics:
                mu_L_hist[t] = mu_L.copy()
                mu_R_hist[t] = mu_R.copy()
                s2_L_hist[t] = sigma2_L.copy()
                s2_R_hist[t] = sigma2_R.copy()
                vv = np.zeros(N)
                mL = ~att_R
                mR = att_R
                if mL.any():
                    vv[mL] = compute_voi_batch(mu_L[mL], mu_R[mL], sigma2_R[mL], self.sigma2_min)
                if mR.any():
                    vv[mR] = compute_voi_batch(mu_L[mR], mu_R[mR], sigma2_L[mR], self.sigma2_min)
                voi_hist[t] = vv

        # -------------------------------------------------------------------
        # 2.4 Final choice (noisy argmax)
        # -------------------------------------------------------------------
        noise_L = np.sqrt(sigma2_L + self.sigma2_decision)
        noise_R = np.sqrt(sigma2_R + self.sigma2_decision)
        v_L = rng.normal(mu_L, noise_L)
        v_R = rng.normal(mu_R, noise_R)
        choose_R = (v_R > v_L).astype(int)

        result = {
            'choose_R': choose_R,
            'ev_L': ev_L, 'ev_R': ev_R,
            'ev_diff': ev_R - ev_L,
            'mu_L_final': mu_L, 'mu_R_final': mu_R,
            'sigma2_L_final': sigma2_L, 'sigma2_R_final': sigma2_R,
            'att_history': att_history,
            'n_switches_d2': n_switches_d2,
        }
        if record_dynamics:
            result['mu_L_hist'] = mu_L_hist
            result['mu_R_hist'] = mu_R_hist
            result['s2_L_hist'] = s2_L_hist
            result['s2_R_hist'] = s2_R_hist
            result['voi_hist'] = voi_hist
        return result


# ---------------------------------------------------------------------------
# 3. Stimulus generation
# ---------------------------------------------------------------------------
def generate_ev_pairs_ruben(n_trials, rng=None):
    """Generate expected-value pairs matching the reward structure in
    Ferro et al. (2024).

    Distribution per option:
        12.5%  probability -> EV = 1.0
        43.75% probability -> EV ~ Uniform(0, 2)
        43.75% probability -> EV ~ Uniform(0, 3)

    Parameters
    ----------
    n_trials : int
        Number of trial pairs to generate.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    ev_L, ev_R : np.ndarray
        Expected values for left and right options.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    ev_L = np.zeros(n_trials)
    ev_R = np.zeros(n_trials)
    for ev_arr in [ev_L, ev_R]:
        r = rng.random(n_trials)
        for i in range(n_trials):
            if r[i] < 0.125:
                ev_arr[i] = 1.0
            elif r[i] < 0.5625:
                ev_arr[i] = 2.0 * rng.random()
            else:
                ev_arr[i] = 3.0 * rng.random()
    return ev_L, ev_R
