
---

### File 2: `attribution_simulation.py`

This is the full script, ready to paste:

```python
import numpy as np
import pandas as pd

"""
Ad Attribution & Pricing Elasticity Playground
----------------------------------------------
This script has two conceptual parts:

1. Multi-channel attribution:
   - Simulates synthetic customer journeys across marketing channels.
   - Compares Last-Touch vs Linear attribution at the channel level.

2. Pricing elasticity (conjoint-style, simplified):
   - Uses a simple discrete-choice / logit-inspired model to show
     how demand and revenue change as you move price up and down.
   - Computes arc elasticity between price points.

The goal is not production-grade modeling. The goal is to make the
mental models behind attribution and pricing elasticity concrete
and inspectable in a single file.
"""

# ======================================================
# Part 1 — Multi-channel journeys & attribution models
# ======================================================

np.random.seed(42)

n_users = 1500
channels = ["Search", "Social", "Display", "Email", "Affiliate"]

rows = []

for user_id in range(1, n_users + 1):
    # Number of touchpoints for this user
    n_touches = np.random.randint(1, 6)  # between 1 and 5 touches

    # Sample channels with replacement
    user_channels = np.random.choice(channels, size=n_touches, replace=True)

    # Simple assumption: more touches -> higher probability to convert
    base_p = 0.04
    incremental_p = 0.03  # added per touch after the first
    p_convert = min(base_p + incremental_p * (n_touches - 1), 0.7)

    converted = np.random.rand() < p_convert

    # If converted, assign a revenue amount
    revenue = 0.0
    if converted:
        revenue = float(np.round(np.random.uniform(60, 350), 2))

    for touch_order, channel in enumerate(user_channels, start=1):
        rows.append(
            {
                "user_id": user_id,
                "touch_order": touch_order,
                "channel": channel,
                "converted": converted,
                "revenue": revenue if converted else 0.0,
            }
        )

events = pd.DataFrame(rows)

print("\n=== Sample of simulated events (first 10 rows) ===")
print(events.head(10))


def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """
    Last-touch attribution:
    For each converting user, assign 100% of revenue to the last channel
    they touched before converting.
    """
    conv = events[events["converted"]].copy()
    if conv.empty:
        return pd.DataFrame(columns=["channel", "revenue_last_touch"])

    last_touches = (
        conv.sort_values(["user_id", "touch_order"])
        .groupby("user_id")
        .tail(1)
    )

    channel_rev = last_touches.groupby("channel")["revenue"].sum().reset_index()
    channel_rev.rename(columns={"revenue": "revenue_last_touch"}, inplace=True)

    return channel_rev


def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """
    Linear attribution:
    For each converting user, split revenue equally across all channels
    in the journey.
    """
    conv = events[events["converted"]].copy()
    if conv.empty:
        return pd.DataFrame(columns=["channel", "revenue_linear"])

    touch_counts = conv.groupby("user_id")["touch_order"].max().rename("n_touches")
    conv = conv.merge(touch_counts, on="user_id", how="left")

    conv["attributed_revenue"] = conv["revenue"] / conv["n_touches"]

    channel_rev = conv.groupby("channel")["attributed_revenue"].sum().reset_index()
    channel_rev.rename(columns={"attributed_revenue": "revenue_linear"}, inplace=True)

    return channel_rev


last_touch = last_touch_attribution(events)
linear = linear_attribution(events)

print("\n=== Revenue by channel (Last-Touch) ===")
print(last_touch)

print("\n=== Revenue by channel (Linear) ===")
print(linear)

comparison = pd.merge(last_touch, linear, on="channel", how="outer").fillna(0.0)

total_last = comparison["revenue_last_touch"].sum()
total_linear = comparison["revenue_linear"].sum()

if total_last > 0:
    comparison["share_last_touch"] = comparison["revenue_last_touch"] / total_last
else:
    comparison["share_last_touch"] = 0.0

if total_linear > 0:
    comparison["share_linear"] = comparison["revenue_linear"] / total_linear
else:
    comparison["share_linear"] = 0.0

comparison = comparison.sort_values("channel")

print("\n=== Comparison: channel revenue & share by attribution model ===")
print(comparison.to_string(index=False))


# ======================================================
# Part 2 — Pricing elasticity (discrete-choice style)
# ======================================================

"""
Here we switch from media attribution to pricing.

We use a simple logit-style demand model that is structurally similar
to what underpins many conjoint / discrete-choice frameworks:

- There is a focal product with utility U(price) = beta0 + beta_price * price.
- There is an "outside option" (do nothing, choose competitor, etc.) with
  fixed utility.

We then convert utilities into choice probabilities via a softmax over
{product, outside option}. Multiplying by an assumed market size gives
expected demand, from which we derive revenue and arc elasticities.
"""

prices = np.array([60, 80, 100, 120, 140])  # example price ladder

# Utility parameters (for illustration only)
beta0 = 3.0          # intercept (baseline attractiveness)
beta_price = -0.04   # price sensitivity (negative)
u_out = 0.0          # outside option utility (normalized)
market_size = 100000  # notional potential customers

records = []
for p in prices:
    U = beta0 + beta_price * p
    exp_u = np.exp(U)
    exp_out = np.exp(u_out)

    share = exp_u / (exp_u + exp_out)  # choice probability for the product
    demand = share * market_size
    revenue = demand * p

    records.append(
       {
            "price": p,
            "utility": U,
            "choice_share": share,
            "expected_demand": demand,
            "expected_revenue": revenue,
        }
    )

pricing_df = pd.DataFrame(records)

print("\n=== Pricing ladder: demand & revenue by price point ===")
print(pricing_df.round(2).to_string(index=False))

# Arc elasticity between adjacent price points
elasticity_rows = []
for i in range(len(prices) - 1):
    p1, p2 = prices[i], prices[i + 1]
    q1, q2 = pricing_df.loc[i, "expected_demand"], pricing_df.loc[i + 1, "expected_demand"]

    pct_change_q = (q2 - q1) / q1
    pct_change_p = (p2 - p1) / p1
    arc_elasticity = pct_change_q / pct_change_p

    elasticity_rows.append(
        {
            "price_from": p1,
            "price_to": p2,
            "demand_from": q1,
            "demand_to": q2,
            "pct_change_price": pct_change_p,
            "pct_change_demand": pct_change_q,
            "arc_elasticity": arc_elasticity,
        }
    )

elasticity_df = pd.DataFrame(elasticity_rows)

print("\n=== Arc price elasticity between adjacent price points ===")
print(elasticity_df.round(3).to_string(index=False))

print(
    "\nInterpretation (qualitative):"
    "\n- Utilities decrease as price increases, reducing choice share and demand."
    "\n- Revenue typically increases with price up to a point, then declines."
    "\n- Arc elasticities become more negative as you move up the price ladder,"
    "\n  reflecting higher sensitivity when you are already charging more."
    "\n"
    "\nIn a full conjoint / dynamic pricing system you would:"
    "\n- Estimate these parameters (beta0, beta_price, and others) from real choice data."
    "\n- Extend the utility function to include multiple attributes (brand, features, etc.)."
    "\n- Segment by audience, channel, or cohort."
    "\n- Feed elasticity estimates into pricing, promotion, and bidding strategies."
)
