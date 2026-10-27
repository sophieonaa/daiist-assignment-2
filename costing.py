"""costing.py: the cost model, shared by train.ipynb and app.ipynb.

It lives in one file so that the threshold the training notebook picks and the number the
dashboard shows are computed by exactly the same code. There is no model code in here,
only per-image scores and money.

Vocabulary (positive = "this photo contains fire or smoke"):

    score   the highest confidence the detector gave to a fire or smoke box in the photo,
            or 0.0 if it found none.
    low     below this score the photo is PASSED: no alarm.
    high    at or above this score the photo is FAILED: the emergency service is called
            straight away, with nobody looking at it first.
    between low and high the photo goes to a person (REVIEW).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SCORE_MAX = 1.0
NEVER = 1.01  # a "high" above any possible score means: never auto-dispatch


@dataclass
class CostParams:
    """Every assumption that turns errors into euros. All of them are decisions."""

    # What an error costs once it happens (EUR per photo).
    c_fn: float = 10_000.0  # a fire the system missed, found late
    c_tp: float = 2_000.0  # a fire caught in time (some damage is unavoidable)
    c_tn: float = 0.0  # a normal photo correctly passed
    c_dispatch_fp: float = 300.0  # PLACEHOLDER: an engine sent out for nothing. Decide this.

    # What one human review costs.
    review_minutes: float = 3.0
    operator_eur_per_hour: float = 12.0
    overhead_per_review: float = 0.40

    # How good the person is. 1.0 = the operator always judges correctly.
    human_recall: float = 1.0
    human_specificity: float = 1.0

    # The world FindFire operates in.
    home_fire_per_year: float = 0.001  # chance that one home has a fire in a year
    photos_per_day: float = 4.0  # per camera
    cameras: int = 20_000
    operator_productive_hours: float = 7.0  # review hours per operator per day
    max_operators: float = 5.0  # how many operators FindFire can staff

    @property
    def c_review(self) -> float:
        """EUR for one person to look at one photo."""
        return self.review_minutes / 60.0 * self.operator_eur_per_hour + self.overhead_per_review

    @property
    def base_rate(self) -> float:
        """Share of photos that really contain fire (one fire shows up in one photo)."""
        return self.home_fire_per_year / (self.photos_per_day * 365.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["c_review"] = self.c_review
        d["base_rate"] = self.base_rate
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CostParams":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


DEFAULT_GRID = np.unique(np.round(np.concatenate([np.linspace(0.0, 1.0, 201), [NEVER]]), 4))


# ---------------------------------------------------------------- counting photos

def band_counts(sorted_scores, low, high):
    """(passed, reviewed, failed) counts for one population of photos."""
    n = len(sorted_scores)
    high = max(high, low)
    below_low = int(np.searchsorted(sorted_scores, low, side="left"))
    below_high = int(np.searchsorted(sorted_scores, high, side="left"))
    return below_low, below_high - below_low, n - below_high


def band_fractions(sorted_scores, low, high, smooth=True):
    """(pass, review, fail) fractions for one population.

    With few photos a zero count is not a zero rate: 0 false alarms in 58 normal photos
    does not mean the false alarm rate is 0. `smooth` adds half an observation to every
    cell that could happen (Jeffreys), so a missing event costs a little instead of nothing.
    Cells that cannot happen (a pass band when low = 0, a fail band when high > 1) stay 0.
    """
    high = max(high, low)
    n = len(sorted_scores)
    counts = band_counts(sorted_scores, low, high)
    possible = (low > 0.0, high > low and low <= SCORE_MAX, high <= SCORE_MAX)
    a = 0.5 if smooth else 0.0
    cells = sum(possible)
    return tuple(
        (k + a) / (n + a * cells) if ok else 0.0 for k, ok in zip(counts, possible)
    )


# ---------------------------------------------------------------- money

def expected_cost(f_pos, f_neg, p: CostParams, base_rate=None) -> float:
    """Expected EUR per photo, given where positives and negatives land."""
    br = p.base_rate if base_rate is None else base_rate
    hr, hs = p.human_recall, p.human_specificity
    pos = (
        f_pos[0] * p.c_fn  # missed: nobody ever sees it
        + f_pos[1] * (p.c_review + hr * p.c_tp + (1 - hr) * p.c_fn)  # a person decides
        + f_pos[2] * p.c_tp  # dispatched on the model's word alone
    )
    neg = (
        f_neg[0] * p.c_tn
        + f_neg[1] * (p.c_review + hs * p.c_tn + (1 - hs) * p.c_dispatch_fp)
        + f_neg[2] * p.c_dispatch_fp
    )
    return br * pos + (1 - br) * neg


def review_load(review_share: float, p: CostParams) -> dict:
    """What a review share means in people."""
    photos_per_day = p.cameras * p.photos_per_day
    reviews = photos_per_day * review_share
    hours = reviews * p.review_minutes / 60.0
    return {
        "photos_per_day": photos_per_day,
        "reviews_per_day": reviews,
        "review_hours_per_day": hours,
        "operators_needed": hours / p.operator_productive_hours,
    }


def evaluate(pos_s, neg_s, low, high, p: CostParams, base_rate=None, smooth=True, fp_mult=1.0):
    """Everything about one (low, high) choice. pos_s and neg_s must be sorted arrays.

    fp_mult stresses the negatives: 3.0 means "in the field, three times as many normal
    photos trip the system as in our test set". It is how we test the domain gap.
    """
    high = max(high, low)
    f_pos = band_fractions(pos_s, low, high, smooth)
    f_neg = band_fractions(neg_s, low, high, smooth)
    if fp_mult != 1.0:
        old = f_neg[1] + f_neg[2]
        new = min(1.0, old * fp_mult)
        scale = new / old if old > 0 else 0.0
        f_neg = (1.0 - new, f_neg[1] * scale, f_neg[2] * scale)

    br = p.base_rate if base_rate is None else base_rate
    cost = expected_cost(f_pos, f_neg, p, br)
    review_share = br * f_pos[1] + (1 - br) * f_neg[1]
    fail_share = br * f_pos[2] + (1 - br) * f_neg[2]
    load = review_load(review_share, p)
    photos_per_year = load["photos_per_day"] * 365.0
    out = {
        "low": float(low),
        "high": float(high),
        "f_pos": f_pos,
        "f_neg": f_neg,
        "cost_per_case": cost,
        "cost_per_1000": cost * 1000.0,
        "cost_per_year": cost * photos_per_year,
        "review_share": review_share,
        "fail_share": fail_share,
        "false_dispatches_per_day": load["photos_per_day"] * (1 - br) * f_neg[2],
        "fires_missed_per_year": photos_per_year * br * (f_pos[0] + f_pos[1] * (1 - p.human_recall)),
    }
    out.update(load)
    return out


# ---------------------------------------------------------------- sweeps

def sweep(pos_s, neg_s, p: CostParams, grid=None, mode="operator", **kw):
    """Cost at every threshold t.

    mode "operator": score >= t sends the photo to a person (low = t, no auto-dispatch).
    mode "alone":    score >= t dispatches straight away (low = high = t, no person).
    """
    grid = DEFAULT_GRID if grid is None else grid
    rows = []
    for t in grid:
        low, high = (t, NEVER) if mode == "operator" else (t, t)
        rows.append(evaluate(pos_s, neg_s, low, high, p, **kw))
    return rows


def best_threshold(rows):
    """The cheapest row of a sweep (the lowest threshold wins an exact tie)."""
    return min(rows, key=lambda r: (round(r["cost_per_case"], 12), r["low"]))


def best_band(pos_s, neg_s, p: CostParams, grid=None, max_operators=None, **kw):
    """Cheapest (low, high). With max_operators, only bands that many people can staff."""
    grid = DEFAULT_GRID if grid is None else grid
    best = None
    for i, low in enumerate(grid):
        for high in grid[i:]:
            r = evaluate(pos_s, neg_s, low, high, p, **kw)
            if max_operators is not None and r["operators_needed"] > max_operators:
                continue
            if best is None or r["cost_per_case"] < best["cost_per_case"] - 1e-15:
                best = r
    return best


# ---------------------------------------------------------------- plain metrics

def confusion_at(pos_s, neg_s, cut) -> dict:
    """The 2x2 table when everything at or above `cut` is called positive."""
    tp = int((np.asarray(pos_s) >= cut).sum())
    fn = len(pos_s) - tp
    fp = int((np.asarray(neg_s) >= cut).sum())
    tn = len(neg_s) - fp
    n = tp + fn + fp + tn
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "accuracy": (tp + tn) / n if n else float("nan"),
    }


def auc(pos_s, neg_s) -> float:
    """Chance that a random positive outscores a random negative (ties count half)."""
    pos = np.asarray(pos_s, float)[:, None]
    neg = np.asarray(neg_s, float)[None, :]
    return float(((pos > neg).mean() + 0.5 * (pos == neg).mean()))


def three_way_table(pos_s, neg_s, low, high):
    """Plain counts: rows = truth (fire, no fire), columns = (pass, review, fail)."""
    return (
        band_counts(np.sort(np.asarray(pos_s, float)), low, high),
        band_counts(np.sort(np.asarray(neg_s, float)), low, high),
    )