# Ad Attribution & Pricing Elasticity Playground

A small but opinionated sandbox for thinking about **attribution**, **pricing elasticity**, and **decision-making** in performance marketing.

The code simulates multi-channel customer journeys, compares simple attribution models, and then applies a **conjoint-style pricing elasticity model** (discrete-choice / logit-inspired) to show how demand and revenue change along a price ladder.

This is not meant to be production ad tech. It is a **thinking tool**: something a senior product / marketing / data leader could use to align stakeholders on how attribution and pricing interact.

---

## Why this exists

If you’re managing large marketing and pricing systems, you keep circling around three questions:

1. **Attribution**  
   Given multi-touch journeys, *which channels or touchpoints get credit* for a conversion?

2. **Pricing elasticity (conjoint-style)**  
   Given a set of possible prices and an outside option, *how does demand and revenue respond* as you move price up or down?

3. **Systemic decisions**  
   How do these two lenses combine when you decide:
   - which channels to fund,
   - which price points to test,
   - and how to trade off short-term ROAS vs long-term value?

This repository builds a deliberately small model you can read in one sitting and use as a concrete artefact in those discussions.

---

## What’s in this repo

- `attribution_simulation.py`

  One script, two conceptual parts:

  1. **Multi-channel attribution**
     - Simulates synthetic customer journeys across channels (Search, Social, Display, Email, Affiliate).
     - Computes revenue per channel under:
       - **Last-Touch Attribution**
       - **Linear Attribution**
     - Compares channel revenue and share under each model.

  2. **Pricing elasticity (conjoint-style, simplified)**
     - Defines a **price ladder** (e.g. 60, 80, 100, 120, 140).
     - Uses a **discrete-choice / logit-style utility**:
       - Product utility: `U(price) = β₀ + β_price * price` (β_price < 0).
       - Outside-option utility: fixed baseline (e.g. “no purchase”).
     - Converts utilities into **choice probabilities**, **expected demand**, and **revenue** for each price.
     - Computes **arc price elasticity** between adjacent price points.

No external datasets are needed; everything is synthetic and transparent.

---

## Mental models encoded here

### 1. Attribution is a lens, not the truth

Different attribution models are just different ways of assigning credit:

- Last-touch over-rewards “closers”.
- Linear smears credit across the whole path.

The underlying journeys do not change, but your perception of which channels “work” can change dramatically. For a leader owning hundreds of millions in spend, understanding this *lens* property is critical before tying attribution naively to budget or bonuses.

### 2. Pricing elasticity as a discrete-choice / conjoint problem

The pricing section uses a **logit-style demand curve**:

- Each price point has a utility `U(price) = β₀ + β_price * price`.
- There is an outside option with its own (fixed) utility.
- We turn utilities into **choice shares** via a softmax-style function.
- Multiplying by an assumed market size gives demand and revenue.

This structure mirrors the core of many **conjoint / discrete-choice** models:

- Utility is linear in parameters.
- Price enters with a negative coefficient.
- Elasticity can be derived from demand changes along the price ladder.

It’s a simplified, single-attribute illustration—deliberately so—but the mental model is aligned with how formal conjoint and dynamic pricing systems think.

### 3. Pricing + marketing are one system

In reality:

- Media drives **who shows up and with what intent**.
- Pricing and packaging drive **whether and how they convert**.
- Measurement (attribution, incrementality, elasticity) sits across both.

By putting attribution and pricing elasticity in a single script, the goal is to reinforce that these are not separate conversations—you are designing **one system** where acquisition and monetization are tightly coupled.

---

## How to run the script

You need **Python 3** and two libraries:

- `pandas`
- `numpy`

From a terminal:

```bash
pip install pandas numpy
