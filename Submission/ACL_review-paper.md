---
description: Comprehensive pre-submission academic paper review (multi-LLM + references + human perception)
allowed-tools: [Read, Write, Bash, Grep, Glob, AskUserQuestion, Task]
argument-hint: [paper-path] [--venue ACL] [--resume]
---

# Review Paper Pipeline

You are orchestrating a comprehensive academic paper review pipeline. The user provided a paper at: $ARGUMENTS

## Step 1: Setup

1. Read the paper at the provided path
2. Check if `.claude/academic-review-suite.local.md` exists
   - If not: run `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/providers.py` to generate default settings, save to `.claude/academic-review-suite.local.md`, then ask user to configure API keys
   - If exists: load settings, check which providers have API keys
3. If no providers have keys configured, use AskUserQuestion:
   - Show list of all available providers
   - Ask which ones they have API keys for (multi-select)
   - For each selected, ask for the API key
   - Update the .local.md file
4. Save a copy of the paper as baseline: `.claude/academic-review-suite-baseline.txt`
5. Extract paper text (if PDF, use available tools; if .tex, read directly)

## Step 2: LLM Review (External API Calls)

Dispatch the `academic-review-suite:llm-reviewer` agent with the paper text.

The agent will:
- Health-check all configured models
- Estimate cost and get confirmation
- Collect reviews from all healthy models
- Synthesize into a unified report
- Tag issues as [writing] or [experiment] with compute tiers

Present the synthesized report to the user and WAIT for their response.

## Step 3: Fix Loop (NO External API Calls)

For each [writing] issue from the LLM review:
1. Ask user which issues to address
2. Fix the writing issues in the paper (you are Claude Code, you can edit files)
3. After each round of fixes, run pivot detection:
   ```bash
   python3 ${CLAUDE_PLUGIN_ROOT}/scripts/diff_detector.py --baseline .claude/academic-review-suite-baseline.txt --current <paper_path_text>
   ```
4. If change > 20%:
   - Alert: "Major revision detected (X% changed). This triggers a full LLM re-review."
   - Save new baseline
   - Go back to Step 2
5. If change <= 20%: continue fixing

After all writing fixes, present remaining [experiment] issues:
- Group by compute tier: [cpu], [gpu-24], [gpu-80], [gpu-h100], [human], [lab]
- Ask user which experiments they can run
- WAIT for user to complete experiments and provide results

## Step 4: Reference Check

Dispatch the `academic-review-suite:reference-checker` agent.

The agent will:
- Extract all references from the paper
- Verify against OpenAlex, Semantic Scholar, CrossRef
- Classify each reference
- Present report

WAIT for user to fix any mismatches or verify "Not Found" references.

## Step 5: Human Perception Review (NO External API Calls)

Dispatch the `academic-review-suite:human-perception` agent.

The agent will run the full three-stage funnel:
- Stage C: Broad sweep (presentation, consistency, adversarial)
- Stage B: Role-based (skimmer, confused reader, hostile reviewer, copy editor)
- Stage A: Reader-journey (impression, narrative, grounding, consistency, mock reviews)
- Loop: A -> fix -> B until pass or user breaks

## Step 6: Final Gate

After all stages complete, assess readiness:

- **Green** (0 Critical, <=2 Important, B1 interest >= 4): "Paper is ready for submission."
- **Yellow** (0 Critical, <=5 Important, B1 interest >= 3): "Paper can be submitted with known risks: [list]"
- **Red** (Any Critical OR >5 Important OR B1 interest < 3): "Paper needs more work. Remaining issues: [list]"

If Red, offer to re-run Stage 5 after fixes.

## State Persistence

Save progress after each step to `.claude/academic-review-suite-state.json`:
```json
{
  "paper_path": "...",
  "current_step": 3,
  "baseline_path": "...",
  "issues": [],
  "model_health": {},
  "change_history": []
}
```

If the user runs `/review-paper --resume`, load state and continue from the saved step.
