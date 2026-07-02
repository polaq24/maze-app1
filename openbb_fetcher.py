"""OpenBB fetcher: output SPX option chains JSON to stdout.
Usage: conda_env_python openbb_fetcher.py [--min-dte 3] [--max-dte 120] [--max-chains 10]
Output: JSON con tutte le catene richieste."""

import json, sys, argparse
from datetime import datetime, date

from openbb import obb


def fetch(min_dte=3, max_dte=120, max_chains=10):
    today = date.today()
    chain = obb.derivatives.options.chains("SPX", provider="cboe")
    df = chain.to_dataframe()
    if df is None or df.empty:
        return {"error": "No data from CBOE"}

    underlying = float(df["underlying_price"].iloc[0])

    # Prepara expiry list sorted by DTE
    exps = sorted(df["expiration"].dropna().unique())
    exp_dates = []
    for e in exps:
        e_date = e.date() if hasattr(e, "date") else datetime.strptime(str(e)[:10], "%Y-%m-%d").date()
        dte = (e_date - today).days
        if min_dte <= dte <= max_dte:
            exp_dates.append((dte, str(e_date)))

    # Max N expirations equidistanti
    if len(exp_dates) > max_chains:
        import numpy as np
        indices = np.linspace(0, len(exp_dates) - 1, max_chains).astype(int)
        exp_dates = [exp_dates[i] for i in indices]

    result = {
        "underlying_price": underlying,
        "date": str(today),
        "expirations": [],
    }

    for dte, e_str in exp_dates:
        mask = df["expiration"].apply(
            lambda x: (x.date() if hasattr(x, "date") else
                       datetime.strptime(str(x)[:10], "%Y-%m-%d").date()).isoformat() == e_str
        )
        sub = df[mask].copy()
        if sub.empty:
            continue

        options = []
        for _, row in sub.iterrows():
            opt = {
                "strike": float(row["strike"]),
                "option_type": str(row["option_type"]).upper(),
                "last_price": float(row["last_price"]) if row.get("last_price") not in (None, 0, "") else None,
                "bid": float(row["bid"]) if row.get("bid") not in (None, 0, "") else None,
                "ask": float(row["ask"]) if row.get("ask") not in (None, 0, "") else None,
                "implied_volatility": float(row["implied_volatility"]) if row.get("implied_volatility") not in (None, 0, "") else None,
            }
            if opt["last_price"] is not None or opt["bid"] is not None or opt["ask"] is not None:
                options.append(opt)

        result["expirations"].append({
            "dte": dte,
            "date": e_str,
            "options": options,
        })

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-dte", type=int, default=3)
    p.add_argument("--max-dte", type=int, default=120)
    p.add_argument("--max-chains", type=int, default=10)
    args = p.parse_args()

    data = fetch(args.min_dte, args.max_dte, args.max_chains)
    json.dump(data, sys.stdout, indent=2)
