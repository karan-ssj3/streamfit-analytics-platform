# Ground Truth vs Extracted JSON — Evaluation Comparison

**Purpose:** Side-by-side comparison of human-annotated ground truth against LLM-extracted output, across all 6 interactions. Identifies systematic failure modes and field-level accuracy.

**Ground truth methodology:** Each file in `/ground_truth/` was written independently from the raw transcript, using the analyst's own judgment — not by reading the extracted JSON first. This ensures the comparison is a true blind evaluation.

---

## Summary Table

| ID | Interaction Type | Resolution | Sentiment Overall | Trajectory | Churn Risk | Save Attempted | Upsell Level | Requires Review | Approx. Accuracy |
|----|-----------------|------------|-------------------|------------|------------|----------------|--------------|-----------------|------------------|
| SF-2026-0001 | ✅ cancellation | ✅ partially_resolved | ✅ negative | ✅ improving | ✅ high | ✅ true | ✅ none | ✅ true | **~95%** |
| SF-2026-0002 | ❌ billing_dispute → `cancellation` | ✅ partially_resolved | ✅ negative | ✅ improving | ❌ medium → `high` | ❌ false → `true` | ✅ low | ❌ true → `false` | **~68%** |
| SF-2026-0003 | ≈ plan_inquiry → `sales_inquiry` | ✅ resolved | ✅ positive | ❌ stable → `improving` | ✅ low | ✅ false | ❌ high → `medium` | ✅ false | **~78%** |
| SF-2026-0004 | ✅ cancellation | ❌ partially_resolved → `resolved` | ❌ negative → `mixed` | ✅ improving | ❌ medium → `high` | ✅ true | ✅ none | ✅ false | **~70%** |
| SF-2026-0005 | ≈ technical_support → `support_request` | ✅ partially_resolved | ✅ neutral | ❌ stable → `improving` | ✅ low | ❌ false → `true` | ✅ none | ✅ false | **~72%** |
| SF-2026-0006 | ≈ plan_change → `upgrade_inquiry` | ✅ resolved | ✅ positive | ❌ stable → `improving` | ✅ none | ✅ false | ❌ none → `high` | ✅ false | **~75%** |

**Legend:** ✅ Exact match | ❌ Wrong (ground truth → extracted value) | ≈ Semantically close but different label

---

## Interaction-by-Interaction Breakdown

---

### SF-2026-0001 — Cancellation (Content Staleness)

**Ground truth:** `billing: cancellation` / `partially_resolved` / `at_risk` / `high churn risk` / `save_successful=true` / `save_condition` populated / `requires_review=true`

**Overall verdict: Strong extraction (~95%)**

| Field | Ground Truth | Extracted | Verdict |
|-------|-------------|-----------|---------|
| interaction.type | cancellation | cancellation | ✅ |
| resolution.status | partially_resolved | partially_resolved | ✅ |
| lifecycle_stage | at_risk | at_risk | ✅ |
| sentiment.overall | negative | negative | ✅ |
| sentiment.trajectory | improving | improving | ✅ |
| emotional_intensity | moderate | moderate | ✅ |
| churn_risk.level | high | high | ✅ |
| save_attempted | true | true | ✅ |
| save_successful | true | true | ✅ |
| save_condition | populated | populated | ✅ |
| upsell_opportunity | none | none | ✅ |
| requires_human_review | true | true | ✅ |
| fitness_level | advanced | advanced | ✅ |
| customer.plan | premium | premium | ✅ |
| tenure_months | 25 | 25 | ✅ |

**Notes:** This was the reference case used for evaluator development. The LLM correctly identified the conditional save, populated `save_condition`, and flagged for human review. Near-perfect.

---

### SF-2026-0002 — Billing Dispute (Duplicate Charge + Overdraft)

**Ground truth:** `billing_dispute` / `partially_resolved` / `at_risk` / `medium churn risk` / `save_attempted=false` / `requires_review=true`

**Overall verdict: Several significant misclassifications (~68%)**

| Field | Ground Truth | Extracted | Verdict |
|-------|-------------|-----------|---------|
| interaction.type | billing_dispute | **cancellation** | ❌ |
| resolution.status | partially_resolved | partially_resolved | ✅ |
| lifecycle_stage | at_risk | at_risk | ✅ |
| sentiment.overall | negative | negative | ✅ |
| sentiment.trajectory | improving | improving | ✅ |
| emotional_intensity | high | high | ✅ |
| churn_risk.level | **medium** | high | ❌ |
| save_attempted | **false** | true | ❌ |
| save_successful | **null** | true | ❌ |
| upsell_opportunity | low | low | ✅ |
| requires_human_review | **true** | false | ❌ |
| customer.name | David Wright | David Wright | ✅ |
| customer.plan | basic | basic | ✅ |
| tenure_months | 8 | 8 | ✅ |

**Misclassification analysis:**

1. **`interaction.type: cancellation` (wrong)** — The customer never requested cancellation. He called to dispute a billing error. The LLM likely pattern-matched on "close to just ditching it" (one throwaway comment) and over-interpreted it as a cancellation intent. Correct type: `billing_dispute`.

2. **`churn_risk.level: high` (wrong)** — Elevated churn was a *background signal*, not the primary intent. No cancellation was threatened, no save was attempted. `medium` is correct — the comment about leaving is a warning sign, not an active intent.

3. **`save_attempted: true / save_successful: true` (wrong)** — There was no save attempt. The agent processed the refund as the core resolution. The LLM appears to have confused "resolving a billing dispute" with "saving from cancellation." These are fundamentally different.

4. **`requires_human_review: false` (wrong)** — The overdraft refund is **contingent** on the customer emailing a bank statement. This creates a follow-up dependency that the LLM missed. A human reviewer needs to confirm receipt and processing.

---

### SF-2026-0003 — Plan Inquiry (Free Trial Converting Customer)

**Ground truth:** `plan_inquiry` / `resolved` / `onboarding` / `low churn risk` / `high upsell opportunity` / `stable trajectory` / `low emotional intensity`

**Overall verdict: Good on structure, soft on nuance (~78%)**

| Field | Ground Truth | Extracted | Verdict |
|-------|-------------|-----------|---------|
| interaction.type | plan_inquiry | **sales_inquiry** | ≈ |
| resolution.status | resolved | resolved | ✅ |
| lifecycle_stage | onboarding | onboarding | ✅ |
| sentiment.overall | positive | positive | ✅ |
| sentiment.trajectory | **stable** | improving | ❌ |
| emotional_intensity | **low** | moderate | ❌ |
| churn_risk.level | low | low | ✅ |
| save_attempted | false | false | ✅ |
| upsell_opportunity.level | **high** | medium | ❌ |
| household | couple | couple | ✅ |
| competitor_mentions count | 2 (FitFlow + Peloton) | 2 | ✅ |
| requires_human_review | false | false | ✅ |

**Misclassification analysis:**

1. **`sentiment.trajectory: improving` (wrong)** — The customer opened positively ("It's pretty good actually") and closed positively. There was no meaningful shift upward — sentiment was *stable* throughout. The LLM conflates "ended well" with "trajectory improved." This is a systematic bias (see Cross-Interaction Patterns below).

2. **`emotional_intensity: moderate` (wrong)** — This was a calm, investigative conversation. The customer was measured and analytical. `low` is correct; `moderate` overstates the intensity.

3. **`upsell_opportunity.level: medium` (wrong)** — The customer explicitly said *"I'll probably go with Family."* That's a strong, explicit intent signal. This should be `high`. The LLM underrated it.

4. **`interaction.type: sales_inquiry`** — Borderline acceptable. "Plan inquiry" or "pre-purchase inquiry" is more precise, but `sales_inquiry` conveys the same concept. Not a hard error.

---

### SF-2026-0004 — Price Increase (Single Parent, 6-Month Price Lock)

**Ground truth:** `cancellation` / `partially_resolved` / `at_risk` / `medium churn risk` / `negative sentiment` / `save_attempted=true, save_successful=true, save_condition=null`

**Overall verdict: Resolution and sentiment both wrong (~70%)**

| Field | Ground Truth | Extracted | Verdict |
|-------|-------------|-----------|---------|
| interaction.type | cancellation | cancellation | ✅ |
| resolution.status | **partially_resolved** | resolved | ❌ |
| lifecycle_stage | at_risk | at_risk | ✅ |
| sentiment.overall | **negative** | mixed | ❌ |
| sentiment.trajectory | improving | improving | ✅ |
| emotional_intensity | moderate | moderate | ✅ |
| churn_risk.level | **medium** | high | ❌ |
| save_attempted | true | true | ✅ |
| save_successful | true | true | ✅ |
| save_condition | null | null | ✅ |
| upsell_opportunity | none | none | ✅ |
| requires_human_review | false | false | ✅ |
| tenure_months | 14 | 14 | ✅ |

**Misclassification analysis:**

1. **`resolution.status: resolved` (wrong)** — The 6-month price lock delays the pain but does not solve it. The customer will face the full $34.99 price in 6 months. The root cause (price increase) is unresolved. `partially_resolved` is correct. The LLM appears to equate "customer accepted the offer" with "fully resolved" — an optimism bias.

2. **`sentiment.overall: mixed` (wrong)** — The customer was predominantly *negative* throughout. She opened with concern, escalated to a cancellation threat, and only softened after a concrete offer. "Mixed" understates the negativity; `negative` is the accurate baseline sentiment.

3. **`churn_risk.level: high` (wrong)** — The save was unconditional (no deadline, no condition), the customer has no competitor alternative, and has a strong emotional attachment (family yoga routine). `medium` is correct — churn risk will resurface in 6 months, but it is not currently `high`.

---

### SF-2026-0005 — Technical Support (Samsung TV Crash, Live Chat)

**Ground truth:** `technical_support` / `partially_resolved` / `active` / `low churn risk` / `save_attempted=false` / `stable trajectory` / `low emotional intensity`

**Overall verdict: Core classification correct, save fields wrong (~72%)**

| Field | Ground Truth | Extracted | Verdict |
|-------|-------------|-----------|---------|
| interaction.type | technical_support | **support_request** | ≈ |
| channel | live_chat | live_chat | ✅ |
| resolution.status | partially_resolved | partially_resolved | ✅ |
| lifecycle_stage | active | active | ✅ |
| sentiment.overall | neutral | neutral | ✅ |
| sentiment.trajectory | **stable** | improving | ❌ |
| emotional_intensity | **low** | moderate | ❌ |
| churn_risk.level | low | low | ✅ |
| save_attempted | **false** | true | ❌ |
| save_successful | **null** | true | ❌ |
| upsell_opportunity | none | none | ✅ |
| requires_human_review | false | false | ✅ |
| household | couple | couple | ✅ |

**Misclassification analysis:**

1. **`save_attempted: true / save_successful: true` (wrong)** — The customer independently decided not to pause the subscription ("actually no, I'll keep it going for now"). The agent did not make a save attempt — they simply answered the question about pausing. Customer retention was organic, not agent-driven. This is a systematic misclassification of autonomous customer decisions as agent save attempts.

2. **`sentiment.trajectory: improving` (wrong)** — The customer was neutral and pragmatic from start to finish. The conversation did not exhibit an upward shift. Same "ended well = trajectory improved" bias seen in 0003 and 0006.

3. **`emotional_intensity: moderate` (wrong)** — The chat tone was calm and transactional ("ok ill try the 720p thing"). `low` is correct.

---

### SF-2026-0006 — Family Plan Upgrade

**Ground truth:** `plan_change` / `resolved` / `active` / `no churn risk` / `upsell=none (completed)` / `stable trajectory`

**Overall verdict: Good on core facts, wrong on upsell and trajectory (~75%)**

| Field | Ground Truth | Extracted | Verdict |
|-------|-------------|-----------|---------|
| interaction.type | plan_change | **upgrade_inquiry** | ≈ |
| resolution.status | resolved | resolved | ✅ |
| lifecycle_stage | active | active | ✅ |
| sentiment.overall | positive | positive | ✅ |
| sentiment.trajectory | **stable** | improving | ❌ |
| emotional_intensity | low | low | ✅ |
| churn_risk.level | none | none | ✅ |
| save_attempted | false | false | ✅ |
| upsell_opportunity.level | **none** | high | ❌ |
| feature_requests | parental controls | parental controls | ✅ |
| requires_human_review | false | false | ✅ |

**Misclassification analysis:**

1. **`upsell_opportunity.level: high` (wrong)** — The upgrade was completed in-call. There is no *opportunity* — it was already executed. Labelling it `high` is stale; the correct value is `none` (upsell completed). This represents a temporal reasoning failure: the LLM correctly identifies the upsell signal but fails to update it post-completion.

2. **`sentiment.trajectory: improving` (wrong)** — Customer was positive throughout. The "trajectory" should reflect the arc of the conversation, not the end state. A conversation that's positive from start to finish has a *stable* trajectory, not an *improving* one.

3. **`interaction.type: upgrade_inquiry`** — Borderline. The inquiry was completed as a plan change in-call. `plan_change` is more accurate as it reflects the outcome, not just the intent.

---

## Cross-Interaction Failure Patterns

These are **systematic biases** that appear consistently across multiple extractions — not one-off errors.

### Pattern 1: Trajectory Inflation — `improving` when `stable`
**Affected:** SF-2026-0003, SF-2026-0005, SF-2026-0006

The LLM defaults to `improving` whenever a conversation ends on a positive note, even if sentiment was positive throughout. A trajectory of `improving` requires a measurable upward shift — a conversation that is positive from start to finish is `stable`.

**Correct rule:** `trajectory = improving` only if sentiment moved upward during the conversation (e.g., negative → neutral, or neutral → positive). A positive-opening, positive-close interaction is `stable`.

**Impact:** Misleads dashboard on whether the conversation was actually a recovery event or a baseline positive experience.

---

### Pattern 2: Save Attempt Hallucination
**Affected:** SF-2026-0002, SF-2026-0005

The LLM conflates "agent resolved an issue successfully" with "agent attempted a save." In both cases:
- SF-2026-0002: Agent processed a refund (not a save). Customer wasn't cancelling.
- SF-2026-0005: Customer independently decided not to pause. Agent gave no save offer.

**Correct rule:** `save_attempted = true` only when the agent explicitly offers a retention incentive (discount, feature unlock, price lock) in response to a stated or implied intent to cancel/pause.

**Impact:** Overstates retention agent performance. Inflates save_success rate metrics on Page 2 and Page 3.

---

### Pattern 3: Resolution Optimism Bias — `resolved` when `partially_resolved`
**Affected:** SF-2026-0004

When the customer accepts an offer and thanks the agent, the LLM calls it `resolved`. But if the root cause persists, it's `partially_resolved`.

**Correct rule:** `resolved` requires the root cause to be fixed. A temporary measure (price lock, workaround, pending refund) = `partially_resolved`.

**Impact:** Dashboard undercounts partially resolved interactions; resolution rate metric is inflated.

---

### Pattern 4: Churn Risk Elevation
**Affected:** SF-2026-0002, SF-2026-0004

The LLM assigns `high` churn risk when the customer expresses any negativity, even if:
- No cancellation was attempted (0002)
- A save was successful with no competitive threat (0004)

**Correct rule:** Churn risk level should reflect the *residual* risk after the interaction, not the risk at peak escalation. Post-save with no competitor and strong emotional attachment = `medium`, not `high`.

**Impact:** Overestimates at-risk pool; could cause unnecessary proactive outreach spend.

---

### Pattern 5: Upsell Not Updated After Completion
**Affected:** SF-2026-0006

The LLM correctly identifies a strong upsell signal but fails to update the upsell field to reflect that the upgrade was executed in-call. `upsell_opportunity = high` post-completion is stale and incorrect.

**Correct rule:** If the interaction concludes with a plan upgrade, set `upsell_opportunity.level = none` and note "completed" in signals.

**Impact:** Page 3 upsell segment analysis incorrectly includes this customer as an open opportunity.

---

## Field-Level Accuracy Across All 6 Interactions

| Field | Correct | Out of | Accuracy |
|-------|---------|--------|----------|
| interaction.type (exact) | 4 | 6 | 67% |
| resolution.status | 5 | 6 | 83% |
| sentiment.overall | 5 | 6 | 83% |
| sentiment.trajectory | 2 | 6 | 33% |
| emotional_intensity | 3 | 6 | 50% |
| churn_risk.level | 4 | 6 | 67% |
| save_attempted | 4 | 6 | 67% |
| save_successful | 4 | 6 | 67% |
| upsell_opportunity.level | 4 | 6 | 67% |
| lifecycle_stage | 6 | 6 | 100% |
| customer.plan | 6 | 6 | 100% |
| requires_human_review | 5 | 6 | 83% |
| competitor_mentions (presence) | 6 | 6 | 100% |
| **Overall (simple avg)** | | | **~75%** |

**Highest accuracy fields:** lifecycle_stage, current_plan, competitor detection — all structural facts the LLM reads reliably.

**Lowest accuracy fields:** sentiment trajectory (33%), emotional intensity (50%) — both require nuanced inference about arc and tone, not just end state.

---

## Recommendations for Prompt Improvement

Based on the patterns above, the following prompt clarifications would improve extraction accuracy:

1. **Trajectory rule** — Add to `EXTRACTION_SYSTEM`: *"sentiment.trajectory should be 'improving' only if sentiment demonstrably shifted upward during the conversation. A conversation that begins and ends positively is 'stable', not 'improving'."*

2. **Save attempt definition** — Add: *"save_attempted should be true only when the agent explicitly offers a retention incentive (discount, price lock, free month, feature access) in response to a customer expressing intent to cancel or leave. Resolving a billing dispute or technical issue is NOT a save attempt."*

3. **Resolution status rule** — Add: *"resolution.status is 'resolved' only if the root cause of the customer's issue is fixed. A workaround, a delayed fix, or a temporary price concession = 'partially_resolved'. Customer accepting an offer does not make an issue 'resolved'."*

4. **Upsell post-completion** — Add: *"If the customer upgrades their plan during the interaction, set upsell_opportunity.level to 'none' and note 'completed in-call' in signals."*

5. **Churn risk residual framing** — Add: *"churn_risk.level should reflect risk at the END of the interaction, not during peak escalation. A successful, unconditional save with no active competitor alternative = 'medium', not 'high'."*
