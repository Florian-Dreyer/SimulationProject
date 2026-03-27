# Simulation-Based Inference for an Adaptive-Network Epidemic Model

Group project for the Simulation course at NUS. We implement and compare
simulation-based inference (SBI) methods to recover the parameters of a
stochastic SIR epidemic model on an adaptive contact network, where the
likelihood is intractable.

Full project specification:
https://alexxthiery.github.io/teaching/SBI_infection/SBI-infection.html

## Project outline

The model simulates a population of N = 200 agents on a dynamic graph. Three
parameters govern the epidemic dynamics:

| Parameter | Meaning |
|-----------|---------|
| `β` | Infection probability per S–I edge per time step |
| `γ` | Recovery probability per infected agent per time step |
| `ρ` | Rewiring probability (behavioural avoidance) per S–I edge per time step |

We are given observed data from 40 independent realisations with unknown
parameters and must infer (β, γ, ρ) using:

1. **Rejection ABC** — baseline approximate Bayesian computation
2. **Summary statistics design** — investigating which summaries are informative
3. **Advanced methods** — regression adjustment, ABC-MCMC, SMC-ABC, or
   synthetic likelihood

## Repository structure

```
.
├── data/                   # Observed data
├── notebooks/              # Exploratory analysis and results
├── scripts/                # Standalone scripts
├── requirements.txt        # Python dependencies
├── .pre-commit-config.yaml
└── README.md
```

## Setup

### 1. Prerequisites

Make sure you have Python 3.12 installed:

```bash
python3.12 --version
```

If not, install it via Homebrew (macOS):

```bash
brew install python@3.12
```

### 2. Clone the repository

```bash
git clone https://github.com/Florian-Dreyer/SimulationProject
cd SimulationProject
```

### 3. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up pre-commit hooks

```bash
pre-commit install
```

Hooks will now run automatically on every `git commit`. To run them manually:

```bash
pre-commit run --all-files
```
