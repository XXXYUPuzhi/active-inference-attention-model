"""
run_experiments.py -- Simulation experiments for the Bayesian Attention Model
=============================================================================

Runs seven experiments and generates ten figures that benchmark the model
against the primate behavioral data from Ferro et al. (2024, Nat. Commun.).

Experiments:
    1. Psychometric curve (sigmoid fit)
    2. Gaze allocation pattern over time
    3. Gaze-modulated psychometric curves by epoch
    4-5. Attention switches and look-back frequency vs. EV difference
    5b. Symmetry verification (full model vs. no-reallocation control)
    6. Parameter sensitivity analysis (alpha, C_shift)
    7. Looking-time difference vs. choice probability

Author: Puzhi YU
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
import os
import time

sys.stdout.reconfigure(encoding='utf-8')

from model import BayesianAttentionModel, generate_ev_pairs_ruben

# ---------------------------------------------------------------------------
# 0. Global configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Tuned parameter set (v4: includes pref_bias for gaze-choice correlation)
DEFAULT_PARAMS = dict(
    alpha=0.08,
    c_shift=0.05,
    sigma2_obs=0.2,
    sigma2_init=4.0,
    sigma2_decision=0.02,
    min_fixation_steps=5,
    switch_noise=0.06,
    pref_bias=0.05,
)

N_TRIALS = 6000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def logistic(x, beta0, beta1):
    """Standard logistic (sigmoid) function."""
    return 1.0 / (1.0 + np.exp(-(beta0 + beta1 * x)))


def fit_logistic(x, y):
    """Fit a two-parameter logistic curve to binned choice data."""
    try:
        popt, _ = curve_fit(logistic, x, y, p0=[0, 1.5], maxfev=5000)
        return popt
    except:
        return np.array([0.0, 1.5])


def gaze_frac_R(att_history, t_start, t_end):
    """Compute per-trial fraction of time spent looking Right in [t_start, t_end)."""
    return att_history[t_start:t_end, :].mean(axis=0)


def bin_and_average(x, y, bins, min_count=10):
    """Bin x-values and compute mean of y within each bin."""
    bc = (bins[:-1] + bins[1:]) / 2
    probs = np.full(len(bc), np.nan)
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() >= min_count:
            probs[i] = y[mask].mean()
    return bc, probs


# ===========================================================================
# 1. Experiment 1 -- Psychometric curve
# ===========================================================================
def experiment1(res, model):
    """Fig 1: Sigmoid psychometric curve (P(choose R) vs. EV difference).

    Reproduces Ferro et al. (2024) Fig. 1B.
    """
    print("  [Exp 1] Fitting psychometric curve...")
    ev_diff = res['ev_diff']
    choose_R = res['choose_R'].astype(float)

    bins = np.linspace(-3, 3, 25)
    bc, probs = bin_and_average(ev_diff, choose_R, bins)
    v = ~np.isnan(probs)
    b0, b1 = fit_logistic(bc[v], probs[v])

    fig, ax = plt.subplots(figsize=(7, 5))
    xf = np.linspace(-3, 3, 200)
    ax.plot(xf, logistic(xf, b0, b1), 'b-', lw=2.5,
            label=f'Model: $\\beta_0$={b0:.2f}, $\\beta_1$={b1:.2f}')
    ax.fill_between(xf, logistic(xf, b0 - 0.08, b1), logistic(xf, b0 + 0.08, b1),
                     alpha=0.12, color='blue')
    ax.scatter(bc[v], probs[v], c='k', s=35, zorder=5, label='Binned data')
    ax.plot(xf, logistic(xf, 0, 1.74), 'r--', lw=1.5, alpha=0.6,
            label='Ferro et al.: $\\beta_1$=1.74')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.4)
    ax.axvline(0, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('$EV_R - EV_L$', fontsize=13)
    ax.set_ylabel('P(choose R)', fontsize=13)
    ax.set_title('Exp 1: Psychometric Curve', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-3.2, 3.2)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'Fig01_sigmoid_psychometric.png'), dpi=150)
    plt.close(fig)
    print(f"    beta0={b0:.3f}, beta1={b1:.3f} (target: ~1.74)")


# ===========================================================================
# 2. Experiment 2 -- Gaze pattern over time
# ===========================================================================
def experiment2(res, model):
    """Fig 2: Fraction of trials looking Right across the full trial timeline.

    Reproduces Ferro et al. (2024) Fig. 1C.
    """
    print("  [Exp 2] Plotting gaze allocation over time...")
    frac_R = res['att_history'].mean(axis=1)
    time_ms = np.arange(model.n_total) * model.dt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    # Upper panel: gaze fraction
    ax = axes[0]
    ax.fill_between(time_ms, frac_R, 0.5, where=frac_R >= 0.5,
                     color='steelblue', alpha=0.5, label='Look R')
    ax.fill_between(time_ms, frac_R, 0.5, where=frac_R < 0.5,
                     color='forestgreen', alpha=0.5, label='Look L')
    ax.plot(time_ms, frac_R, 'k-', lw=1.2)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    for b in [model.t_offer1_end, model.t_delay1_end, model.t_offer2_end]:
        ax.axvline(b * model.dt, color='red', ls=':', alpha=0.7)
    ax.set_ylabel('Fraction looking Right', fontsize=12)
    ax.set_title('Exp 2: Gaze Pattern Over Time (cf. Ferro et al. Fig. 1C)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1)

    # Lower panel: epoch timeline
    ax2 = axes[1]
    labels = ['offer1\n(400ms)', 'delay1\n(600ms)', 'offer2\n(400ms)', 'delay2\n(600ms)']
    starts = [0, model.t_offer1_end * model.dt, model.t_delay1_end * model.dt,
              model.t_offer2_end * model.dt]
    ends = [model.t_offer1_end * model.dt, model.t_delay1_end * model.dt,
            model.t_offer2_end * model.dt, model.t_delay2_end * model.dt]
    cols = ['#90EE90', '#D3D3D3', '#87CEEB', '#D3D3D3']
    for s, e, lb, c in zip(starts, ends, labels, cols):
        ax2.barh(0, e - s, left=s, height=0.5, color=c, edgecolor='k')
        ax2.text((s + e) / 2, 0, lb, ha='center', va='center', fontsize=10)
    ax2.set_xlim(0, model.n_total * model.dt)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'Fig02_gaze_pattern.png'), dpi=150)
    plt.close(fig)


# ===========================================================================
# 3. Experiment 3 -- Gaze-modulated psychometric curves
# ===========================================================================
def experiment3(res, model):
    """Fig 3: Psychometric curves split by gaze direction within each epoch.

    Reproduces Ferro et al. (2024) Fig. 2A.
    """
    print("  [Exp 3] Computing gaze-modulated psychometric curves...")
    epoch_ranges = {
        'offer1': (0, model.t_offer1_end),
        'delay1': (model.t_offer1_end, model.t_delay1_end),
        'offer2': (model.t_delay1_end, model.t_offer2_end),
        'delay2': (model.t_offer2_end, model.t_delay2_end),
    }
    ev_diff = res['ev_diff']
    choose_R = res['choose_R'].astype(float)
    bins = np.linspace(-3, 3, 20)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for idx, (ename, (ts, te)) in enumerate(epoch_ranges.items()):
        ax = axes[idx]
        fR = gaze_frac_R(res['att_history'], ts, te)

        # Split trials by gaze fraction percentiles
        fR_hi = np.percentile(fR, 70)
        fR_lo = np.percentile(fR, 30)
        for cond, color, label in [
            (fR > fR_hi, 'red', f'$f_R > {fR_hi:.2f}$ (top 30%)'),
            (np.ones(len(fR), dtype=bool), 'black', 'All trials'),
            (fR < fR_lo, 'blue', f'$f_R < {fR_lo:.2f}$ (bottom 30%)'),
        ]:
            sub_diff = ev_diff[cond]
            sub_ch = choose_R[cond]
            if len(sub_diff) < 30:
                continue
            bc, probs = bin_and_average(sub_diff, sub_ch, bins)
            v = ~np.isnan(probs)
            if v.sum() > 3:
                popt = fit_logistic(bc[v], probs[v])
                xf = np.linspace(-3, 3, 200)
                ax.plot(xf, logistic(xf, *popt), color=color, lw=2, label=label)
            ax.scatter(bc[v], probs[v], c=color, s=15, alpha=0.6)

        ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
        ax.axvline(0, color='gray', ls='--', alpha=0.3)
        ax.set_xlabel('$EV_R - EV_L$', fontsize=11)
        if idx == 0:
            ax.set_ylabel('P(choose R)', fontsize=11)
        ax.set_title(ename, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle('Exp 3: Gaze-Modulated Psychometric Curves (cf. Ferro et al. Fig. 2A)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'Fig03_gaze_modulated_sigmoid.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# 4. Experiments 4-5 -- Switches and look-back vs. EV difference
# ===========================================================================
def experiment4_5(res, model):
    """Fig 4-5: Attention switches and look-back to offer1 as a function of
    absolute EV difference during delay2."""
    print("  [Exp 4-5] Analyzing EV difference effects on gaze dynamics...")
    abs_diff = np.abs(res['ev_diff'])

    # Count switches in delay2
    att_d2 = res['att_history'][model.t_offer2_end:model.t_delay2_end, :]
    switches = np.sum(np.diff(att_d2.astype(int), axis=0) != 0, axis=0)

    # Fraction of time looking Left during delay2 (proxy for look-at-nothing)
    fR_d2 = att_d2.mean(axis=0)
    fL_d2 = 1 - fR_d2

    bins = np.linspace(0, 3, 10)
    bc = (bins[:-1] + bins[1:]) / 2
    mean_sw, se_sw, mean_fL, se_fL = [], [], [], []

    for i in range(len(bins) - 1):
        mask = (abs_diff >= bins[i]) & (abs_diff < bins[i + 1])
        n = mask.sum()
        if n > 10:
            mean_sw.append(switches[mask].mean())
            se_sw.append(switches[mask].std() / np.sqrt(n))
            mean_fL.append(fL_d2[mask].mean())
            se_fL.append(fL_d2[mask].std() / np.sqrt(n))
        else:
            mean_sw.append(np.nan); se_sw.append(np.nan)
            mean_fL.append(np.nan); se_fL.append(np.nan)

    mean_sw, se_sw = np.array(mean_sw), np.array(se_sw)
    mean_fL, se_fL = np.array(mean_fL), np.array(se_fL)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    v = ~np.isnan(mean_sw)
    ax1.errorbar(bc[v], mean_sw[v], yerr=se_sw[v], fmt='o-', color='purple', capsize=3, lw=2, ms=6)
    ax1.set_xlabel('$|EV_R - EV_L|$', fontsize=12)
    ax1.set_ylabel('Mean attention switches (delay2)', fontsize=12)
    ax1.set_title('Exp 4: Switches vs. EV Difference', fontsize=12)
    ax1.grid(alpha=0.3)

    v = ~np.isnan(mean_fL)
    ax2.errorbar(bc[v], mean_fL[v], yerr=se_fL[v], fmt='s-', color='teal', capsize=3, lw=2, ms=6)
    ax2.set_xlabel('$|EV_R - EV_L|$', fontsize=12)
    ax2.set_ylabel('Fraction looking Left\n("look-at-nothing" for offer1)', fontsize=12)
    ax2.set_title('Exp 5: Look-back to Offer1 vs. EV Difference', fontsize=12)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'Fig04_05_switches_and_lookback.png'), dpi=150)
    plt.close(fig)


# ===========================================================================
# 5. Experiment 5b -- Symmetry verification
# ===========================================================================
def experiment5_symmetry(params):
    """Fig 6: Compare full model (with VOI-driven reallocation) against a
    no-reallocation control to verify that look-at-nothing preserves
    psychometric symmetry and decision accuracy."""
    print("  [Exp 5b] Running symmetry verification...")
    rng = np.random.default_rng(123)
    ev_L, ev_R = generate_ev_pairs_ruben(N_TRIALS, rng=rng)
    bins = np.linspace(-3, 3, 25)

    conditions = [
        (params['c_shift'], params.get('pref_bias', 0),
         f'Full model (VOI + $C_{{shift}}$)'),
        (100.0, 0.0,
         'No reallocation (stays on R in delay2)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (cs, pb, title) in enumerate(conditions):
        p = dict(params)
        p['c_shift'] = cs
        p['pref_bias'] = pb
        m = BayesianAttentionModel(**p)
        r = m.run_batch(ev_L, ev_R, rng=np.random.default_rng(456))

        bc, probs = bin_and_average(r['ev_diff'], r['choose_R'].astype(float), bins)
        v = ~np.isnan(probs)
        b0, b1 = fit_logistic(bc[v], probs[v])

        fR_d2 = gaze_frac_R(r['att_history'], m.t_offer2_end, m.t_delay2_end)

        ax = axes[idx]
        xf = np.linspace(-3, 3, 200)
        ax.plot(xf, logistic(xf, b0, b1), 'b-', lw=2.5,
                label=f'$\\beta_0$={b0:.3f}, $\\beta_1$={b1:.2f}')
        ax.scatter(bc[v], probs[v], c='k', s=25)
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('$EV_R - EV_L$', fontsize=12)
        ax.set_ylabel('P(choose R)', fontsize=12)
        ax.set_title(f'{title}\n$\\beta_0$={b0:.3f}, delay2 mean $f_R$={fR_d2.mean():.2f}',
                     fontsize=11)
        ax.legend(fontsize=10, loc='lower right')
        ax.set_ylim(-0.05, 1.05)
        sym = 'Symmetric' if abs(b0) < 0.15 else 'Asymmetric'
        ax.text(0.05, 0.92, sym, transform=ax.transAxes, fontsize=12,
                fontweight='bold', color='green' if abs(b0) < 0.15 else 'red')
        print(f"    {title}: beta0={b0:.3f}, beta1={b1:.2f}, fR_d2={fR_d2.mean():.2f} ({sym})")

    fig.suptitle('Exp 5b: $C_{shift}$ enables look-at-nothing to maintain symmetry',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'Fig06_symmetry_verification.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# 6. Experiment 6 -- Parameter sensitivity analysis
# ===========================================================================
def experiment6_sensitivity(params):
    """Fig 7-8: Sensitivity of the psychometric curve to the memory leak rate
    (alpha) and the attention switch cost (C_shift)."""
    print("  [Exp 6] Running sensitivity analysis...")
    N_SENS = 3000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fig 7: sensitivity to alpha (memory leak rate)
    ax = axes[0]
    alphas = [0.0, 0.02, 0.05, 0.08, 0.15]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))
    for a_val, c in zip(alphas, cmap):
        p = dict(params); p['alpha'] = a_val
        m = BayesianAttentionModel(**p)
        rng = np.random.default_rng(789)
        eL, eR = generate_ev_pairs_ruben(N_SENS, rng=rng)
        r = m.run_batch(eL, eR, rng=np.random.default_rng(101))
        bins = np.linspace(-3, 3, 20)
        bc, probs = bin_and_average(r['ev_diff'], r['choose_R'].astype(float), bins)
        v = ~np.isnan(probs)
        if v.sum() > 3:
            popt = fit_logistic(bc[v], probs[v])
            xf = np.linspace(-3, 3, 200)
            ax.plot(xf, logistic(xf, *popt), color=c, lw=2, label=f'$\\alpha$={a_val}')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('$EV_R - EV_L$', fontsize=12)
    ax.set_ylabel('P(choose R)', fontsize=12)
    ax.set_title('Fig 7: Sensitivity to $\\alpha$ (memory leak rate)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    # Fig 8: sensitivity to C_shift (switch cost)
    ax = axes[1]
    c_shifts = [0.01, 0.03, 0.05, 0.1, 0.3]
    cmap2 = plt.cm.plasma(np.linspace(0.1, 0.9, len(c_shifts)))
    for cs_val, c in zip(c_shifts, cmap2):
        p = dict(params); p['c_shift'] = cs_val
        m = BayesianAttentionModel(**p)
        rng = np.random.default_rng(789)
        eL, eR = generate_ev_pairs_ruben(N_SENS, rng=rng)
        r = m.run_batch(eL, eR, rng=np.random.default_rng(101))
        bins = np.linspace(-3, 3, 20)
        bc, probs = bin_and_average(r['ev_diff'], r['choose_R'].astype(float), bins)
        v = ~np.isnan(probs)
        if v.sum() > 3:
            popt = fit_logistic(bc[v], probs[v])
            xf = np.linspace(-3, 3, 200)
            ax.plot(xf, logistic(xf, *popt), color=c, lw=2, label=f'$C_{{shift}}$={cs_val}')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('$EV_R - EV_L$', fontsize=12)
    ax.set_ylabel('P(choose R)', fontsize=12)
    ax.set_title('Fig 8: Sensitivity to $C_{shift}$ (switch cost)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'Fig07_08_parameter_sensitivity.png'), dpi=150)
    plt.close(fig)


# ===========================================================================
# 7. Experiment 7 -- Looking time vs. choice
# ===========================================================================
def experiment7(res, model):
    """Fig 9: P(choose R) as a function of normalized gaze-time difference
    during delay2, split by EV condition.

    Reproduces Ferro et al. (2024) Fig. 2B.
    """
    print("  [Exp 7] Analyzing looking time vs. choice relationship...")
    ts, te = model.t_offer2_end, model.t_delay2_end
    fR = gaze_frac_R(res['att_history'], ts, te)
    norm_diff = 2 * fR - 1  # (t_R - t_L) / (t_R + t_L)

    ev_diff = res['ev_diff']
    choose_R = res['choose_R'].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 15)

    for cond, color, label in [
        (ev_diff > 0, 'darkred', '$EV_R > EV_L$'),
        (np.ones(len(ev_diff), dtype=bool), 'black', 'All trials'),
        (ev_diff < 0, 'darkblue', '$EV_L > EV_R$'),
    ]:
        sub_nd = norm_diff[cond]
        sub_ch = choose_R[cond]
        if len(sub_nd) < 30:
            continue
        bc, probs = bin_and_average(sub_nd, sub_ch, bins)
        v = ~np.isnan(probs)
        if v.sum() > 3:
            popt = fit_logistic(bc[v], probs[v])
            xf = np.linspace(-1, 1, 200)
            ax.plot(xf, logistic(xf, *popt), color=color, lw=2, label=label)
        ax.scatter(bc[v], probs[v], c=color, s=25, alpha=0.7)

    ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
    ax.axvline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('$(t_R - t_L) / (t_R + t_L)$ during delay2', fontsize=12)
    ax.set_ylabel('P(choose R)', fontsize=12)
    ax.set_title('Exp 7: Choice vs. Looking Time Difference (cf. Ferro et al. Fig. 2B)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'Fig09_looking_time_vs_choice.png'), dpi=150)
    plt.close(fig)


# ===========================================================================
# 8. Figure 10 -- Single-trial dynamics visualization
# ===========================================================================
def figure10_dynamics(params):
    """Fig 10: Detailed view of belief, uncertainty, VOI, and attention
    trajectories within a single representative trial."""
    print("  [Fig 10] Generating single-trial dynamics plot...")
    m = BayesianAttentionModel(**params)

    # Search for a trial with at least 2 switches in delay2
    best_trial = None
    for seed in range(50, 200):
        rng = np.random.default_rng(seed)
        ev_L = np.array([1.4])
        ev_R = np.array([1.6])
        r = m.run_batch(ev_L, ev_R, rng=rng, record_dynamics=True)
        att_d2 = r['att_history'][m.t_offer2_end:m.t_delay2_end, 0]
        n_sw = np.sum(np.diff(att_d2.astype(int)) != 0)
        if n_sw >= 2:
            best_trial = (seed, r, ev_L, ev_R)
            break

    if best_trial is None:
        rng = np.random.default_rng(77)
        ev_L = np.array([1.4])
        ev_R = np.array([1.6])
        r = m.run_batch(ev_L, ev_R, rng=rng, record_dynamics=True)
        best_trial = (77, r, ev_L, ev_R)

    seed, r, ev_L, ev_R = best_trial
    time_ms = np.arange(m.n_total) * m.dt
    i = 0  # single trial index

    fig = plt.figure(figsize=(13, 11))
    gs = GridSpec(4, 1, figure=fig, hspace=0.35)

    def shade_epochs(ax):
        """Add background shading to mark task epochs."""
        ax.axvspan(0, m.t_offer1_end * m.dt, alpha=0.06, color='green')
        ax.axvspan(m.t_offer1_end * m.dt, m.t_delay1_end * m.dt, alpha=0.06, color='gray')
        ax.axvspan(m.t_delay1_end * m.dt, m.t_offer2_end * m.dt, alpha=0.06, color='blue')
        ax.axvspan(m.t_offer2_end * m.dt, m.t_delay2_end * m.dt, alpha=0.06, color='orange')
        for b in [m.t_offer1_end, m.t_delay1_end, m.t_offer2_end]:
            ax.axvline(b * m.dt, color='red', ls=':', alpha=0.5)

    # Panel A: Belief means (mu)
    ax1 = fig.add_subplot(gs[0])
    shade_epochs(ax1)
    ax1.plot(time_ms, r['mu_L_hist'][:, i], 'g-', lw=2, label=f'$\\mu_L$ (true={ev_L[i]:.1f})')
    ax1.plot(time_ms, r['mu_R_hist'][:, i], 'b-', lw=2, label=f'$\\mu_R$ (true={ev_R[i]:.1f})')
    ax1.axhline(ev_L[i], color='green', ls='--', alpha=0.4)
    ax1.axhline(ev_R[i], color='blue', ls='--', alpha=0.4)
    ax1.set_ylabel('$\\mu$ (belief mean)', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left')
    ch = 'R' if r['choose_R'][i] else 'L'
    ax1.set_title(f'Single Trial Dynamics (EV_L={ev_L[i]}, EV_R={ev_R[i]}, Choice={ch})',
                  fontsize=13, fontweight='bold')

    # Panel B: Belief uncertainties (sigma^2)
    ax2 = fig.add_subplot(gs[1])
    shade_epochs(ax2)
    ax2.plot(time_ms, r['s2_L_hist'][:, i], 'g-', lw=2, label='$\\sigma^2_L$')
    ax2.plot(time_ms, r['s2_R_hist'][:, i], 'b-', lw=2, label='$\\sigma^2_R$')
    ax2.set_ylabel('$\\sigma^2$ (uncertainty)', fontsize=11)
    ax2.legend(fontsize=9)

    # Panel C: Value of Information
    ax3 = fig.add_subplot(gs[2])
    shade_epochs(ax3)
    ax3.plot(time_ms, r['voi_hist'][:, i], 'k-', lw=1.5, label='VOI (to unattended)')
    ax3.axhline(m.c_shift, color='red', ls='--', lw=2, label=f'$C_{{shift}}$={m.c_shift}')
    ax3.set_ylabel('VOI', fontsize=11)
    ax3.legend(fontsize=9)

    # Panel D: Attention state
    ax4 = fig.add_subplot(gs[3])
    shade_epochs(ax4)
    att_num = r['att_history'][:, i].astype(float)
    ax4.fill_between(time_ms, att_num, step='mid', alpha=0.5, color='steelblue')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Look L', 'Look R'])
    ax4.set_xlabel('Time (ms)', fontsize=11)
    ax4.set_ylabel('Attention', fontsize=11)

    # Epoch labels at bottom
    epoch_names = ['offer1', 'delay1', 'offer2', 'delay2']
    epoch_bounds = [0, m.t_offer1_end * m.dt, m.t_delay1_end * m.dt,
                    m.t_offer2_end * m.dt, m.t_delay2_end * m.dt]
    for j in range(4):
        mid = (epoch_bounds[j] + epoch_bounds[j + 1]) / 2
        ax4.text(mid, -0.3, epoch_names[j], ha='center', va='top', fontsize=10,
                 transform=ax4.get_xaxis_transform())

    fig.savefig(os.path.join(RESULTS_DIR, 'Fig10_single_trial_dynamics.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# 9. Preference-bias tuning utility
# ===========================================================================
def quick_tune_pref_bias(base_params):
    """Grid search over pref_bias values to find the setting that best
    balances sigmoid slope (target: beta1 ~ 1.74) and a positive
    gaze-choice correlation (Fig 9 slope)."""
    print("\n--- Preference-bias tuning ---")
    N_TUNE = 4000
    rng_ev = np.random.default_rng(42)
    ev_L, ev_R = generate_ev_pairs_ruben(N_TUNE, rng=rng_ev)

    best = None
    for pb in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        p = dict(base_params)
        p['pref_bias'] = pb
        m = BayesianAttentionModel(**p)
        r = m.run_batch(ev_L, ev_R, rng=np.random.default_rng(42))

        # Psychometric slope
        bins = np.linspace(-3, 3, 25)
        bc, probs = bin_and_average(r['ev_diff'], r['choose_R'].astype(float), bins)
        v = ~np.isnan(probs)
        b0, b1 = fit_logistic(bc[v], probs[v]) if v.sum() > 3 else (0, 1)

        # Gaze-choice correlation metric
        fR_d2 = gaze_frac_R(r['att_history'], m.t_offer2_end, m.t_delay2_end)
        nd = 2 * fR_d2 - 1
        bc9, pr9 = bin_and_average(nd, r['choose_R'].astype(float), np.linspace(-1, 1, 12))
        v9 = ~np.isnan(pr9)
        if v9.sum() > 3:
            b0_9, b1_9 = fit_logistic(bc9[v9], pr9[v9])
        else:
            b0_9, b1_9 = 0, 0

        fR_std = fR_d2.std()
        n_sw_mean = r['n_switches_d2'].mean()

        print(f"  pref_bias={pb:.2f}  beta1={b1:.2f}  fig9_slope={b1_9:.2f}  "
              f"fR_std={fR_std:.3f}  switches={n_sw_mean:.1f}")

        # Composite score: match target slope + reward positive gaze-choice correlation
        score = -abs(b1 - 1.74) + 0.3 * b1_9 + fR_std
        if best is None or score > best[0]:
            best = (score, pb, b1, b1_9)

    print(f"  => Best: pref_bias={best[1]:.2f} (beta1={best[2]:.2f}, fig9_slope={best[3]:.2f})")
    return best[1]


# ===========================================================================
# 10. Main entry point
# ===========================================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("Bayesian Attention Model - Experiment Suite v4")
    print("=" * 60)

    params = DEFAULT_PARAMS.copy()

    # Tune the preference-bias parameter
    best_pb = quick_tune_pref_bias(params)
    params['pref_bias'] = best_pb
    print(f"\nFinal parameters: {params}")

    # Run main simulation
    model = BayesianAttentionModel(**params)
    rng = np.random.default_rng(42)
    ev_L, ev_R = generate_ev_pairs_ruben(N_TRIALS, rng=rng)
    print(f"\nRunning main simulation ({N_TRIALS} trials)...")
    res = model.run_batch(ev_L, ev_R, rng=np.random.default_rng(42))
    pct = res['choose_R'].mean() * 100
    print(f"  Done in {time.time() - t0:.1f}s. P(choose R)={pct:.1f}%")

    # Summary statistics for delay2 gaze
    fR_d2 = gaze_frac_R(res['att_history'], model.t_offer2_end, model.t_delay2_end)
    print(f"  Delay2 gaze: mean fR={fR_d2.mean():.3f}, std={fR_d2.std():.3f}, "
          f"fR>0.75: {(fR_d2 > 0.75).mean():.1%}, fR<0.25: {(fR_d2 < 0.25).mean():.1%}")

    # Run all experiments
    print("\n--- Running Experiments ---")
    experiment1(res, model)
    experiment2(res, model)
    experiment3(res, model)
    experiment4_5(res, model)
    experiment5_symmetry(params)
    experiment6_sensitivity(params)
    experiment7(res, model)
    figure10_dynamics(params)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All done in {elapsed:.1f}s. Figures saved to: {RESULTS_DIR}")
    print("=" * 60)
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
