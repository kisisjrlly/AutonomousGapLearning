# Paper outline (Science Robotics format)

**Working title**: Learning to try again: safely aborted attempts drive in-context
adaptation for autonomous flight through unknown narrow gaps

**One-sentence claim** (from README §17): A single end-to-end sensorimotor policy
learns, entirely from its own safely aborted attempts held in recurrent memory, to
probe an unknown gap, abort before contact, adjust, and traverse — without any
hand-coded task state machine.

## Narrative spine

1. Problem: aerial traversal of *unknown* tight openings requires trial; trials
   risk collision; classical pipelines separate detection/planning/retry logic;
   learning systems usually learn *by* crashing (in sim) and deploy frozen.
2. Idea: make the safe abort the unit of experience. Meta-train a recurrent
   policy over multi-attempt episodes so cross-attempt memory becomes the
   adaptation mechanism (Plan A of README §5: weights frozen, behavior adapts).
3. Result headline (fill from data):
   - conditional success rises attempt-over-attempt (k2 > k1, p<…)
   - wiping context at aborts erases the gain (causal evidence)
   - no-memory / reset-trained ablations confirm mechanism
   - abort behavior emerges with margin …, contact rate …, give-up on
     infeasible …
   - OOD splits retain …
4. Significance: in-context adaptation as deployable alternative to online
   weight updates for safety-critical aerial autonomy; abort-first data regime.

## Figures

- Fig 1 (schematic + task): system, episode structure, task distribution,
  onboard view filmstrip. (fig1_overview)
- Fig 2 (training dynamics): success/difficulty/collision/attempts vs steps,
  all variants. (fig_training)
- Fig 3 (core): conditional success by attempt × {full, wipe, reset-trained,
  no-memory}; success after abort; alignment-error paired change. (fig_adaptation + fig_bars)
- Fig 4 (behavior anatomy): example multi-attempt episode trajectories +
  clearance/speed traces; abort margins distribution. (fig_episode)
- Fig 5 (safety & generalization): contact rates vs geo margin; give-up on
  infeasible; ID vs OOD bars; stopping-envelope violations. (fig_bars / custom)
- Table 1: README §12 full metric table across variants (supplementary S1 full).

## Results subsections

R1 Curriculum RL produces gap traversal with emergent abort-retry (training
   dynamics + behavior stats)
R2 Second attempts are better than first attempts — and only with memory
   (core adaptation evidence, all controls)
R3 What the memory carries (alignment correction analysis; speed/roll changes
   after aborts; optional: aux-head calibration)
R4 Aborting keeps the drone safe without a safety veto (contact rates, abort
   margins, shadow stopping-envelope)
R5 Knowing when to quit (infeasible tasks: give-up rate/time, false abandons)
R6 Generalization beyond the training distribution (OOD splits)
R7 Deployment cost (latency table, params)

## Discussion points

- Plan A validated as first rung; constrained Plan B (adapters + validation
  + rollback) as roadmap (README §6-7).
- Limitations: procedural rendering ≠ photoreal; quasi-static feasibility
  label; simulation only — sim-to-real plan (README stage 4) with soft gaps,
  safety net, external kill switch; truncation-bootstrap approximation.
- Relation to abort/recovery literature and in-context RL (verified refs).

## Honesty checklist (verify before submission)

- [ ] every number traceable to results/paper/summary.json or logs
- [ ] simulation-only framing explicit in abstract + discussion
- [ ] no claim of guaranteed zero collision (README §9)
- [ ] aborts never labeled as would-be collisions anywhere in text
- [ ] all 26 verified refs match references_verified.json; no unverified cites
