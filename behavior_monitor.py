def early_default_alert(
    emi_delay_trend: int,
    balance_drop_pct: float,
    spending_spike_pct: float,
) -> dict:
    flags = []

    if emi_delay_trend >= 2:
        flags.append("EMI delay trend worsening")
    if balance_drop_pct >= 25:
        flags.append("Sharp balance deterioration")
    if spending_spike_pct >= 40:
        flags.append("Sudden spending spike")

    risk_level = "Low"
    if len(flags) == 1:
        risk_level = "Medium"
    elif len(flags) >= 2:
        risk_level = "High"

    return {
        "alert": len(flags) > 0,
        "risk_level": risk_level,
        "reasons": flags,
    }
