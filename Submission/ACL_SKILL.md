---
name: paper-review-flow
description: Use when running the human-perception stage of paper review, providing the C/B/A three-stage funnel methodology for stress-testing papers from the human reader perspective
---

# Paper Review Flow

Reference skill providing the C/B/A three-stage funnel methodology used by the human-perception agent.

## Three-Stage Funnel

### Stage C: Broad Sweep (One-Time)

**C1 -- Presentation Quality:** One-sentence contribution, red thread (causal chain), acronym discipline, figure/table self-containment, worked example, results discussion, intro-conclusion alignment.

**C2 -- Logical Consistency:** Text-table number match, cross-reference accuracy, citation relevance, contribution consistency across abstract/intro/conclusion, limitation honesty, evidence strength vs verb choice.

**C3 -- Adversarial Stress Test:** Weakest claim identification, missing comparison, abstract-only impression, overclaiming audit ("prove"/"demonstrate"/"show"), strongest counterargument, inconvenient results handling.

### Stage B: Role-Based Stress Test

**B1 -- The Skimmer:** Reads ONLY title, abstract, figures, tables, headers, conclusion. States contribution and main result in one sentence each. Rates interest 1-5. Pass: correct contribution + result, interest >= 3.

**B2 -- The Confused Reader:** Linear read flagging every confusion point (Blocker/Friction/Ambiguity). Pass: 0 Blockers, <= 3 Friction per section.

**B3 -- The Hostile Reviewer:** 5 specific objections categorized (Novelty/Experimental Design/Missing Comparison/Overclaiming/Presentation) with damage rating. Pass: no High-damage undefended objection.

**B4 -- The Copy Editor:** Grammar, formatting consistency, reference format, dangling refs, tense consistency, number formatting. Pass: <= 5 errors.

### Stage A: Reader-Journey (Fine-Grained)

**A1 -- First Impression:** 30-second skim assessment (1-5 score, positive/negative first notice, density).

**A2 -- Narrative Thread:** Linear read tracking section transitions, red thread construction, argument breaks, sag point identification. CRITICAL if red thread cannot be stated in one sentence.

**A3 -- Concrete Grounding:** Every key claim needs concrete example (quality >= 3). Flag math islands (equations without intuitive explanation, concrete instantiation, or variable definitions).

**A4 -- Internal Consistency Audit:** Numbers in text vs tables, cross-references, abstract claims vs results evidence.

**A5 -- Reviewer Pushback Simulation:** Two mock reviews (Overall 2 and Overall 4) with gap analysis identifying fixable weaknesses and undersold strengths.

## Flow

C (once) -> fix -> B -> fix -> A -> fix -> B (re-test) -> pass/loop

## Readiness Assessment

| Level | Criteria | Recommendation |
|-------|----------|----------------|
| **Green** | 0 Critical, <= 2 Important, B1 interest >= 4 | Submit with confidence |
| **Yellow** | 0 Critical, <= 5 Important, B1 interest >= 3 | Submit with known risks |
| **Red** | Any Critical OR > 5 Important OR B1 interest < 3 | Revise before submitting |
