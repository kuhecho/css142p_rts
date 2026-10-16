# RTS: Roundabout Traffic Simulation

Modeling and simulation of peak-period traffic at the Quezon Memorial Circle (Elliptical Road), Quezon City.
Course project for CSS142P (FOPM01), Mapúa University, by Frank Charles Agliam.

## Description

RTS is a discrete-event simulation built with SimPy models the six approaches to the roundabout (Commonwealth, Visayas, North, Quezon, East and Kalayaan) using gap acceptance and first-in-first-out entry queues. The model is checked against M/M/1 and M/G/1 queuing formulas and the HCM 7th Edition capacity equation, then used to compare scenarios: relocated pedestrian crossings, an added entry lane on Commonwealth Avenue, metering at Kalayaan, and demand increases of 10% and 20%. Each run uses 30 replications, a 15-minute warm-up and a 2-hour peak. A sensitivity analysis is included.

Demand, origin-destination shares and the reference sample are planning-level assumptions, not field data.

## Files

| File | Purpose |
|------|---------|
| `roundabout_sim.py` | Full simulation, statistics and figures as a script |
| `roundabout_sim.ipynb` | Notebook version with explanations and saved outputs |
| `results/` | CSV tables and PNG figures produced by the study |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install simpy numpy pandas scipy matplotlib jupyter
```

## Usage

```bash
python roundabout_sim.py                                  # full study (30 replications)
python roundabout_sim.py --quick                          # fast check (5 replications)
python roundabout_sim.py --scenario baseline --reps 10    # one scenario only
python roundabout_sim.py --verify                         # verification tests only
jupyter notebook roundabout_sim.ipynb                     # notebook version
```

Outputs are written to the `results/` folder. In the notebook, set `REPS = 5` for a quick check.
