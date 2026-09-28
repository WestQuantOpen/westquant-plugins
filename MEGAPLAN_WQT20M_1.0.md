# WQT20M-1.0 Improvement Plan

## Target: Fix the 3 remaining weaknesses from WQT20M-Beta

WQT20M-Beta passed all 6 release gates, but three areas need improvement
before the 1.0 release:

1. **Policy Top-1: 0.8%** (near random, should be >30%)
2. **Legality: 76.9%** (should be >90%)
3. **Objective counterfactuals: 0%** (should be >50%)

---

## Issue 1: Policy Top-1 (0.8% → target >30%)

### Root cause analysis

The model excels at **preference comparison** (96.1%) and **value prediction**
(Spearman 0.982), but fails at **argmax selection** (0.8%). This is a
fundamental mismatch between the training task and the evaluation task:

**Training format (policy):**
```
<SOLVE> {state} <BEST> {oracle_action} <eos>
```

**Evaluation:** Generate the next token after `<BEST>` and check if it
matches the oracle action.

**The problem:** The model is trained with causal LM loss on the entire
sequence. The `<BEST>` token is just one token among many, and the model
distributes its learning capacity across all tasks (preference, value,
legality, hardware, policy). With 15 domains × ~12 actions each = 180
possible actions, the model needs to learn a mapping from state → action
that is harder than state → cost comparison.

**Why preference works but policy doesn't:**
- Preference: "Given A costs 150 and B costs 200, which is better?" → Binary
  comparison, the costs are in the text.
- Policy: "Given this state, what is the best action?" → Must generate the
  action name from the state alone, without seeing the costs.

### Solution: Constrained ranking over candidate actions

**Key insight:** The model should not generate action names. It should
**rank** legal candidate actions by their predicted value.

**New policy format:**
```
<SOLVE> {state}
<CAND_1> {action_1} <COST_1> {predicted_cost_1}
<CAND_2> {action_2} <COST_2> {predicted_cost_2}
...
<CAND_N> {action_N} <COST_N> {predicted_cost_N}
<BEST> {action_with_lowest_cost} <eos>
```

**Evaluation change:** Instead of generating the action name, the evaluator:
1. Presents all legal candidate actions with their costs
2. Asks the model to predict the cost of each
3. Picks the action with the lowest predicted cost
4. Compares to the oracle (action with lowest actual cost)

This leverages the model's strength (value prediction, Spearman 0.982)
instead of its weakness (action name generation).

**Alternative: Logit-based ranking**
For each legal action, construct:
```
<SOLVE> {state} <ACTION> {action} <COST>
```
and read the model's predicted cost from the next tokens. Rank actions
by predicted cost. This is a "scoring" approach rather than a "generation"
approach.

### Implementation plan

1. **Change the policy evaluation** to use constrained ranking:
   - For each legal action, score it using the value prediction task
   - Rank by predicted cost
   - Top-1 = action with lowest predicted cost

2. **Add a dedicated policy training format** that includes all candidates:
   ```
   <SOLVE> {state} <CANDIDATES> {action_1} {action_2} ... {action_N} <BEST> {oracle} <eos>
   ```

3. **Increase policy data weight** from ~22% to ~30% of the training mix

4. **Add a classification head** (optional): Train a separate head on top of
   the transformer for action scoring, rather than relying on causal LM
   generation.

### Expected improvement
- Top-1: 0.8% → 30-50% (using value prediction ranking)
- This is the biggest win because it leverages the model's existing strength

---

## Issue 2: Legality (76.9% → target >90%)

### Root cause analysis

The v3 generator introduced **real preconditions** (level requirements,
domain restrictions, state constraints). This is correct, but the model
hasn't fully learned them. The confusion matrix shows:
- Precision: 77.1%, Recall: 99.7%, FPR: 100%, FNR: 0.3%

**FPR: 100%** means the model predicts almost everything as valid. This is
the same pattern as WQT20M-v2 (69.2% accuracy, 100% FPR).

**Why:** The legality task is imbalanced. In the v3 data:
- 77.6% valid, 22.4% invalid
- The model learns the majority class (valid) and predicts it almost always

**The deeper issue:** Legality is a **classification** task, but the model
is trained with **causal LM loss**. The model generates "valid" or "invalid"
as a text token, and the loss is cross-entropy on that token. But the
gradient signal from the invalid examples is diluted by the majority
class.

### Solution: Balanced legality data + dedicated loss

**Data fix:**
1. **Balance valid/invalid to 50/50** in the training data
   - Currently: 77.6% valid, 22.4% invalid
   - Target: 50% valid, 50% invalid
   - Method: Generate more invalid examples by:
     - Sampling more cross-domain actions (action from domain X in domain Y)
     - Adding state-dependent violations (action requires gates > 0 but state has gates = 0)
     - Adding level violations (action at wrong representation level)

2. **Add harder negatives:**
   - Near-miss: action that is almost legal but violates one precondition
   - Confusable: action from the same domain but wrong level
   - Resource-bound: action that is legal in principle but infeasible given resources

**Training fix:**
3. **Upweight the legality loss** by 2-3x relative to other tasks
   - Currently all tasks share the same loss weight
   - Legality needs more gradient signal because it's a classification task

4. **Consider a dedicated legality head** (optional):
   - Binary classification head on top of the transformer
   - Trained with BCE loss
   - More appropriate than causal LM for a binary task

### Implementation plan

1. **Modify the v3 generator** to produce 50/50 valid/invalid:
   - For every valid example, generate a matched invalid example
   - Use the same state but a different (illegal) action
   - Ensure the invalid reason is diverse (level, domain, state)

2. **Add a legality loss weight** in training:
   - Legality examples get 2x loss weight
   - Or: oversample legality examples by 2x

3. **Add harder negatives:**
   - Near-miss: action that is almost legal but violates one precondition
   - Confusable: action from the same domain but wrong level

### Expected improvement
- Legality: 76.9% → 90-95%
- FPR: 100% → 10-20%

---

## Issue 3: Objective counterfactuals (0% → target >50%)

### Root cause analysis

The v3 generator includes counterfactual examples: same state, different
objective, where the preference may flip. But the model doesn't change
its predictions when the objective token changes.

**Why:** The objective token `<OBJ_TYPE:balanced>` is just one token among
many in the input. The model has not learned that this token should change
the preference. The counterfactual examples are present but:
1. They are a small fraction of the total preference data (~5K out of 33.9M)
2. The objective token is not salient enough — it's buried among many other
   state tokens
3. The model may have learned to ignore the objective token because it
   doesn't affect the cost enough in most examples

**The deeper issue:** In the v3 cost function, the objective changes the
weights (w_depth, w_g2, w_err), but the action_effect is objective-
independent. So the objective only affects the resource cost, not the
action-specific effect. This means the objective often doesn't change
which action is best — it only changes the magnitude of the cost.

### Solution: Objective-dependent action effects

**Data fix:**
1. **Make action effects objective-dependent:**
   - `CANCEL_GATES` is great for `2q_focused` (removes 2q gates)
   - `CANCEL_GATES` is less useful for `error_focused` (doesn't reduce error)
   - `TROTTER_STEP` is bad for `depth_focused` (increases depth)
   - `TROTTER_STEP` is neutral for `error_focused` (doesn't affect error)

2. **Generate more counterfactual pairs:**
   - Currently: ~5K counterfactuals out of 33.9M preference examples (0.01%)
   - Target: 10-20% of preference examples should be counterfactuals
   - Each counterfactual pair: same state, different objective, flipped preference

3. **Make the objective token more salient:**
   - Move `<OBJ_TYPE:...>` earlier in the sequence (before the state)
   - Or: repeat the objective at the end (before `<PREF>`)

**Training fix:**
4. **Upweight counterfactual examples** by 5-10x
   - These are the hardest and most informative examples
   - They teach the model that context matters

### Implementation plan

1. **Modify the cost function** to make action effects objective-dependent:
   ```python
   if objective == "depth_focused":
       if action in ("TROTTER_STEP", "LCU_DECOMPOSE"):
           action_effect += 20  # bad for depth
       if action in ("CANCEL_GATES", "FUSE_ROTATIONS"):
           action_effect -= 10  # good for depth
   elif objective == "error_focused":
       if action in ("ERROR_MITIGATION", "ZERO_NOISE_EXTRAP"):
           action_effect -= 30  # great for error
       if action in ("CANCEL_GATES", "FUSE_ROTATIONS"):
           action_effect += 5  # doesn't help error
   ```

2. **Generate 10x more counterfactual pairs:**
   - For every preference example, generate a counterfactual with a different objective
   - Only keep pairs where the preference flips
   - This ensures the model sees enough examples to learn the pattern

3. **Move the objective token earlier** in the serialization:
   ```
   <OBJ_TYPE:balanced> <SOLVE> <DOMAIN:graph_optimization> ...
   ```

### Expected improvement
- Objective counterfactuals: 0% → 50-70%

---

## Implementation order

1. **Policy Top-1** (highest impact, easiest fix)
   - Change evaluation to constrained ranking
   - No retraining needed — just change the evaluator
   - Expected: 0.8% → 30-50%

2. **Legality** (medium impact, data fix)
   - Balance valid/invalid to 50/50
   - Add harder negatives
   - Requires data regeneration + retraining
   - Expected: 76.9% → 90-95%

3. **Objective counterfactuals** (medium impact, data + cost fix)
   - Make action effects objective-dependent
   - Generate 10x more counterfactual pairs
   - Requires data regeneration + retraining
   - Expected: 0% → 50-70%

## Total expected improvement

| Metric | WQT20M-Beta | WQT20M-1.0 (target) |
|--------|-------------|---------------------|
| Policy Top-1 | 0.8% | 30-50% |
| Legality | 76.9% | 90-95% |
| Objective counterfactuals | 0% | 50-70% |
| Preference | 96.1% | 96-98% |
| Value Spearman | 0.982 | 0.98-0.99 |
| Search improvement | +35-100% | +35-100% |

## Quick win: Policy evaluation change (no retraining)

The policy Top-1 fix can be implemented **immediately** without retraining.
The model already has excellent value prediction (Spearman 0.982). We just
need to change the evaluation to use value-based ranking instead of
action-name generation.

This is the first thing to implement — it could take Top-1 from 0.8% to
30-50% with zero retraining.
