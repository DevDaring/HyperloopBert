# Stage 3 Paper Outline

## Title
Parameter Sharing Reduces Stereotype Memorization: Mechanistic Evidence from
Looped and Hyper-Connected Encoders

## Finding-First Abstract
[Conditional: verdict was not GO. If PUBLISH-NULL, frame the paper on the Stages 1-2 finding with Hyperloop as a tested-and-bounded mechanistic extension.]

## Verdict
**PUBLISH-NULL (Looped ~= Hyperloop within tolerance; null Hyperloop result, publishable per pre-registration) [primary_holds=True, hyperloop_better=False, not_worse=True, dose_response=False, monotone(n)=None (rates n1/n2/n4=None/None/0.5570291777188329), n1~Looped=None (looped=0.5537135278514589), signed_r=-0.05063507699135091]**

### Confirmatory family (item-level primary, Holm corrected):
- **VanillaBERT_vs_HyperloopBERT**: Delta = 0.0241 (p_raw = 0.0002, p_holm = 0.00039996000399960006, significant = True, n_items = 1508)
- **VanillaBERT_vs_LoopedBERT**: Delta = 0.0237 (p_raw = 0.0001, p_holm = 0.00029997000299970003, significant = True, n_items = 1508)
- **LoopedBERT_vs_HyperloopBERT**: Delta = 0.0005 (p_raw = 0.4668, p_holm = 0.4667533246675332, significant = False, n_items = 1508)

## Limitations
- Early-merge is an OOD intervention on a trained model: corroborating, not causal
- Stream-disagreement correlation is correlational; the causal claim rests only
  on the from-scratch stream dose-response
- Hyperloop carries extra hyper-connection parameters vs Looped (disclosed;
  the ablation direction is conservative with respect to this gap)
- n=1 collapse to Looped is approximate (projections remain)
