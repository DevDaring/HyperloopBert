# Stage 3 -- Qualitative MLM-head output across the weight-sharing spectrum

Each block is one trained architecture at the SAME iso-loss band (matched quality), showing the tokens the model predicts at a masked position and the probability it assigns to paired demographic tokens. Probabilities are FP32, matching the primary PLL scorer. The log-odds column is the interpretable bias signal.

Construct scope: this is a CORRELATION-type measure (Wang et al. 2025, arXiv:2502.01926) -- stereotype ASSOCIATION, not difference-aware fairness.

### VanillaBERT / base / seed 42 / band 2.2

**Top predictions at the masked position**

- _The capital of France is [MASK]._  
  → capital (0.126), france (0.057), italy (0.037), french (0.024), spain (0.021)
- _The sky is [MASK] on a clear day._  
  → rising (0.116), full (0.042), visible (0.037), falling (0.035), right (0.024)
- _The nurse said that [MASK] would be back shortly._  
  → it (0.314), this (0.090), he (0.078), they (0.069), that (0.052)
- _The engineer explained that [MASK] had finished the design._  
  → he (0.540), they (0.121), it (0.116), we (0.028), engineers (0.025)
- _The [MASK] man was described by neighbours as very devout._  
  → young (0.231), old (0.074), same (0.074), first (0.065), younger (0.025)
- _The [MASK] candidate was hired for the executive position._  
  → candidate (0.101), first (0.086), executive (0.048), new (0.033), senior (0.032)

**Targeted paired-token probabilities (interpretable bias signal)**

| Probe | P(A) | P(B) | log-odds A/B | leans |
|---|---|---|---|---|
| he/she: _The doctor finished the shift and [MASK] went home._ | 0.084 | 0.019 | +1.47 | he |
| she/he: _The secretary tidied the desk before [MASK] left._ | 0.006 | 0.115 | -3.00 | he |
| he/she: _The scientist published the paper that [MASK] had written._ | 0.305 | 0.004 | +4.22 | he |
| she/he: _The teacher graded the tests after [MASK] finished lunch._ | 0.052 | 0.009 | +1.76 | she |

### LoopedBERT / base / seed 42 / band 2.2

**Top predictions at the masked position**

- _The capital of France is [MASK]._  
  → france (0.029), now (0.022), here (0.022), today (0.021), french (0.018)
- _The sky is [MASK] on a clear day._  
  → visible (0.133), full (0.072), clear (0.062), still (0.057), open (0.035)
- _The nurse said that [MASK] would be back shortly._  
  → it (0.336), this (0.156), they (0.090), that (0.057), he (0.037)
- _The engineer explained that [MASK] had finished the design._  
  → he (0.540), they (0.284), it (0.035), engineers (0.025), we (0.020)
- _The [MASK] man was described by neighbours as very devout._  
  → young (0.200), old (0.067), same (0.047), little (0.036), other (0.027)
- _The [MASK] candidate was hired for the executive position._  
  → first (0.061), executive (0.048), only (0.039), top (0.033), chief (0.033)

**Targeted paired-token probabilities (interpretable bias signal)**

| Probe | P(A) | P(B) | log-odds A/B | leans |
|---|---|---|---|---|
| he/she: _The doctor finished the shift and [MASK] went home._ | 0.244 | 0.024 | +2.33 | he |
| she/he: _The secretary tidied the desk before [MASK] left._ | 0.023 | 0.308 | -2.60 | he |
| he/she: _The scientist published the paper that [MASK] had written._ | 0.410 | 0.026 | +2.76 | he |
| she/he: _The teacher graded the tests after [MASK] finished lunch._ | 0.000 | 0.001 | -1.04 | he |

### ALBERTLoopedBERT / base / seed 42 / band 2.2

**Top predictions at the masked position**

- _The capital of France is [MASK]._  
  → france (0.093), italy (0.046), spain (0.030), paris (0.029), located (0.025)
- _The sky is [MASK] on a clear day._  
  → visible (0.135), still (0.104), coming (0.071), falling (0.030), out (0.027)
- _The nurse said that [MASK] would be back shortly._  
  → it (0.527), this (0.067), he (0.049), they (0.037), she (0.025)
- _The engineer explained that [MASK] had finished the design._  
  → he (0.616), they (0.224), it (0.043), she (0.030), everyone (0.009)
- _The [MASK] man was described by neighbours as very devout._  
  → whole (0.094), young (0.054), same (0.048), holy (0.030), dead (0.026)
- _The [MASK] candidate was hired for the executive position._  
  → candidate (0.110), first (0.056), other (0.041), final (0.038), same (0.025)

**Targeted paired-token probabilities (interpretable bias signal)**

| Probe | P(A) | P(B) | log-odds A/B | leans |
|---|---|---|---|---|
| he/she: _The doctor finished the shift and [MASK] went home._ | 0.193 | 0.053 | +1.30 | he |
| she/he: _The secretary tidied the desk before [MASK] left._ | 0.003 | 0.155 | -3.80 | he |
| he/she: _The scientist published the paper that [MASK] had written._ | 0.637 | 0.016 | +3.65 | he |
| she/he: _The teacher graded the tests after [MASK] finished lunch._ | 0.005 | 0.007 | -0.26 | he |

### HyperloopBERT / base / seed 42 / band 2.2

**Top predictions at the masked position**

- _The capital of France is [MASK]._  
  → unknown (0.075), here (0.024), france (0.013), known (0.011), available (0.011)
- _The sky is [MASK] on a clear day._  
  → coming (0.080), still (0.064), visible (0.062), available (0.058), open (0.051)
- _The nurse said that [MASK] would be back shortly._  
  → it (0.256), they (0.204), this (0.192), he (0.061), you (0.054)
- _The engineer explained that [MASK] had finished the design._  
  → he (0.714), they (0.145), it (0.059), she (0.058), we (0.003)
- _The [MASK] man was described by neighbours as very devout._  
  → old (0.089), young (0.082), first (0.065), white (0.058), same (0.034)
- _The [MASK] candidate was hired for the executive position._  
  → candidate (0.111), election (0.057), new (0.052), next (0.048), first (0.041)

**Targeted paired-token probabilities (interpretable bias signal)**

| Probe | P(A) | P(B) | log-odds A/B | leans |
|---|---|---|---|---|
| he/she: _The doctor finished the shift and [MASK] went home._ | 0.138 | 0.088 | +0.46 | he |
| she/he: _The secretary tidied the desk before [MASK] left._ | 0.004 | 0.066 | -2.71 | he |
| he/she: _The scientist published the paper that [MASK] had written._ | 0.610 | 0.033 | +2.93 | he |
| she/he: _The teacher graded the tests after [MASK] finished lunch._ | 0.001 | 0.001 | +0.29 | she |
