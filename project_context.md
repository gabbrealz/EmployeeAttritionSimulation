# Project Context: Employee Attrition Risk Simulation

> **Purpose of this file:** paste it at the start of a new AI chat so the assistant has the full project context.
> **Last updated:** 2026-09-28
> **Provenance tags used below:** **[Spec]** = stated in the user's design documents · **[Decision]** = made during implementation · **[Assumption]** = my interpretation, not confirmed · **[Open]** = unresolved question.

---

## 1. Summary

A **single-agent, replacement-style simulation** of employee attrition risk in a corporate operations department. One employee has five state variables (workload, burnout, engagement, manager support, time since promotion). At each iteration:

1. Four of the variables are updated by a **Gaussian-copula Monte Carlo** step (which preserves dependencies such as burnout ↔ overtime).
2. `years_since_promotion` increases by 1.
3. A **predictive regression model** converts the updated state into a probability of attrition.
4. A **random experiment** decides whether the employee resigns. If so, the attrition counter increments and a new employee with the **initial** state values takes their place.

Scenarios (e.g. heavier overtime, weaker management) are created **before** the run by shifting the **location parameters** of the state variables. Both the Monte Carlo component and the regression model are built from **one existing dataset** whose columns match the state variables.

## 2. System representation [Spec]

| Element | Description |
|---|---|
| System | Corporate Operations Department |
| Agents | Employee |
| Environment | Closed corporate workplace |
| State variables | `overtime_hours`, `burnout_score`, `engagement_score`, `manager_support_score`, `years_since_promotion`, `attrition` |
| Events / interactions | Employee resignation |
| Dataset inputs | Provides the state variables' initial values (listed below). Otherwise used only to train the AI model and to compute the **fixed manager-support-score probability** the simulation needs. |
| Outputs | `employee_attrition`, `avg_employee_burnout`, `avg_employee_engagement`, `avg_hours_overtime`, `avg_years_since_promotion`, `avg_manager_support`, `time_elapsed` |

## 3. State variables [Spec]

| Variable | Description | Type | Initial value | Updated by |
|---|---|---|---|---|
| `overtime_hours` | Dynamic additional overtime hours from task redistribution | Numeric | 6.83 | Copula Monte Carlo |
| `burnout_score` | Cumulative burnout metric, updated by overtime | Numeric | 4.03 | Copula Monte Carlo |
| `engagement_score` | Baseline engagement and workplace motivation | Numeric | 6.51 | Copula Monte Carlo |
| `manager_support_score` | Perceived leadership support | Numeric | 6.61 | Copula Monte Carlo |
| `years_since_promotion` | Time since last career advancement | Numeric | 2.00 | Deterministic: +1 per iteration (exempt from Monte Carlo) |
| `attrition` | Employment state (0 = active, 1 = left) | Integer | 0 | Random experiment |

Note: the source table labels `attrition = 1` as "Retired"; the event is described everywhere else as **resignation**. [Open] — treated here as "resigned / left".

## 4. Simulation workflow [Spec]

1. **Before the run:** adjust the location parameters of chosen state variables to deviate from the baseline scenario.
2. **Run start:** the employee has the initial state values.
3. **Update:** copula-based Monte Carlo updates `overtime_hours`, `burnout_score`, `manager_support_score`, `engagement_score`; `years_since_promotion += 1`. Monte Carlo is used to capture random workplace-stress variability while staying within possible outcomes; the copula preserves multi-variable dependencies (e.g. burnout correlating with overtime).
4. **Predict:** the updated state goes through the predictive model, which outputs P(attrition).
5. **Random experiment:** draw resignation ~ Bernoulli(P). If resigned → `employee_attrition += 1`, replace the employee with a new one at the **initial** state values. Return to step 3.

```text
state = INITIAL_STATE
for t in 1..T:
    state[MC_VARS] = copula_sample(location_shifts)      # step 3
    state.years_since_promotion += 1                     # step 3
    p = attrition_model.predict_proba(state)             # step 4
    if random() < p:                                     # step 5
        employee_attrition += 1
        state = INITIAL_STATE                            # new employee
    record(state); time_elapsed += 1
```

## 5. Modeling components

### 5.1 Gaussian-copula Monte Carlo [Spec + Decision]

- **Copula variables [Spec]:** `overtime_hours`, `burnout_score`, `engagement_score`, `manager_support_score`. `years_since_promotion` and `attrition` are excluded.
- **Fitting from the dataset [Decision]:**
  - *Marginals:* per variable, best parametric fit by AIC (gamma, lognormal, skew-normal, Weibull, normal, logistic), with automatic fallback to an empirical distribution if the fit is poor (KS > 0.05). Variables with ≤ 15 distinct values use a **discrete empirical PMF**, which is how the "fixed manager support score probability" is modelled. [Assumption]
  - *Dependence:* correlation matrix R of normal scores of the ranks (marginal-free); repaired to be positive definite if needed.
- **Sampling:** `z ~ N(0, R)` → `u = Φ(z)` → `x = F⁻¹(u) + location_shift` → clipped to bounds.
- **Location parameters / scenarios [Decision]:** an additive shift per variable, in that variable's own units (`0` = baseline). The copula is unchanged by shifts.
  - **Shifts do not propagate.** Raising `overtime_hours` does *not* raise `burnout_score` by itself. To model that causal story, shift both explicitly.
- **Bounds [Decision]:** applied after the shift. Default is the observed min/max of the training data; set explicit domain bounds (e.g. `(0, 10)` for a score scale) if scenarios may legitimately exceed the observed range.
- **Optional temporal persistence [Decision, off by default]:** AR(1) on the latent Gaussian (`z_t = ρ·z_{t-1} + √(1−ρ²)·ε`, `ε ~ N(0,R)`). This keeps marginals and copula intact while letting burnout/overtime carry over between iterations. With `persistence = 0` (default) draws are i.i.d., exactly as the documented workflow reads.

### 5.2 Predictive attrition model [Spec + Assumption]

- Regression model trained on the same dataset; **must output a probability** of attrition (a logistic-type model is the natural fit). [Assumption]
- Input features: the updated state variables: `overtime_hours`, `burnout_score`, `engagement_score`, `manager_support_score`, `years_since_promotion`. Target: `attrition` (0/1). [Assumption]
- **Status: not yet implemented.**

## 6. Outputs [Spec]

`employee_attrition` (count), `avg_employee_burnout`, `avg_employee_engagement`, `avg_hours_overtime`, `avg_years_since_promotion`, `avg_manager_support`, `time_elapsed`.
[Open] Exact definition of the `avg_*` outputs (assumed: mean over all iterations of the run).

## 7. Code artifacts

**`gaussian_copula_sampler.py`**: implemented and tested (numpy, scipy, pandas).

```python
from gaussian_copula_sampler import GaussianCopulaSampler, INITIAL_STATE

sampler = GaussianCopulaSampler(
    marginals={"manager_support_score": "discrete"},           # optional override
    bounds={"burnout_score": (0, 10), "overtime_hours": (0, None)},
    random_state=42,
).fit(df)                                                       # df has the dataset columns

draws = sampler.sample(n_steps, location_shifts={"overtime_hours": 3.0, "burnout_score": 1.0})

sampler.marginal_summary(); sampler.correlation_; sampler.diagnostics()   # validation

# Optional persistence:
z = sampler.initial_latent(INITIAL_STATE)
row, z = sampler.step(z, location_shifts=..., persistence=0.6)  # reset z with initial_latent() on replacement
```

Notes: for `persistence = 0`, pre-draw the entire trajectory in one vectorised `sample(n_steps)` call, because draws do not depend on attrition outcomes. Still to build: the regression model, the simulation loop and the output aggregation.

## 8. Open questions / known gaps

- **Missing section 3** in the design document (numbering jumps from 2 to 4). Contents unknown, possibly parameters or model details.
- **Time unit:** `years_since_promotion` increments by 1 per iteration, implying **1 iteration = 1 year**. [Assumption]
- **`attrition` label:** "Retired" vs "resigned" (see section 3).
- **Initial values:** they look like dataset averages (e.g. 6.83 overtime hours). [Assumption]. Verify against the dataset.
- **Manager support "fixed probability":** interpreted as its dataset-derived (discrete) marginal, still shiftable by scenarios. Confirm whether it should be shiftable at all.
- **Temporal realism:** the dataset is presumably cross-sectional, so temporal dynamics (persistence, cumulative burnout) are **not learned from data**; they are modelling choices.
- **Discrete values under shifts:** a non-integer shift on a discrete-scale variable yields non-integer scores.

## 9. Guidance for the AI assistant continuing this project

- Treat **[Spec]** items as fixed; flag any proposal that conflicts with them.
- Keep Monte Carlo and the regression model consistent with the same dataset columns and units.
- Prefer vectorised numpy/pandas code, explicit input validation, and reproducible seeding (`numpy.random.Generator`).
- When adding features, state whether they are **[Spec]-conformant** or an **extension** (default off).
