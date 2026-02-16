import numpy as np


def probability_to_score(pd_value: float, factor: float = 50.0, offset: float = 650.0) -> int:
    pd_value = float(np.clip(pd_value, 1e-6, 1 - 1e-6))
    odds = (1 - pd_value) / pd_value
    score = offset + factor * np.log(odds)
    return int(np.clip(score, 300, 850))


def segment_and_decision(score: int) -> tuple[str, str]:
    # Business-facing risk buckets aligned with presentation output.
    if score >= 750:
        return "Low", "Instant Approval"
    if score >= 620:
        return "Medium", "Normal Loan / Manual Review"
    return "High", "Reject / Collateral Required"
