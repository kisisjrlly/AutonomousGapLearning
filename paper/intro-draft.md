# Introduction draft (v1 — numbers/refs to be finalized against data + references_verified.json)

Flying through a narrow opening is among the most demanding tests of aerial
autonomy: the vehicle must perceive the free volume, commit to a dynamically
feasible trajectory, and pass with centimeter-level clearance. Prior systems
have achieved spectacular one-shot traversals, but under a telling division of
labor. Aggressive gap flights by Falanga et al. and Loianno et al. relied on
explicit state estimation of a *known* gap and hand-designed trajectory
generation; learned agile-flight systems — from high-speed forest flight to
champion-level drone racing — train a policy in simulation and deploy it with
frozen weights on tracks or terrain it was trained to master. In both regimes,
the robot is expected to succeed on the first pass. What is missing is the
behavior that biological fliers exhibit when confronted with a *novel* tight
passage: approach, probe, hesitate, back off, adjust, and try again — without
ever touching the obstacle.

This trial-and-retry regime poses a specific learning problem that neither
classical planning nor standard sim-to-real RL addresses. The information that
decides feasibility — the true gap size relative to the vehicle, the vehicle's
own control authority under unknown mass, latency, and wind — is only partially
observable before the first approach and is revealed *by acting*. A policy that
adapts must therefore carry information across attempts. The obvious mechanism,
online weight updates, is hazardous on a flying robot: a single mislabeled
experience can corrupt the policy mid-flight, and every update invalidates
prior safety validation (README §6 risks; cf. safe-learning surveys). The
alternative we pursue is *in-context adaptation*: meta-train a recurrent
end-to-end policy over episodes that contain multiple attempts at the same
gap, keeping its memory alive across attempts, so that adaptation happens in
activation space while weights stay frozen — reproducible, verifiable, and
immediately deployable (RL^2 / learning-to-reinforcement-learn lineage; RMA;
recurrent policies as strong POMDP baselines).

Crucially, we reject collision as a data-collection mechanism. The unit of
experience in our framework is the *safely aborted attempt*: the policy flies
toward the gap, continuously re-evaluates whether continuing remains safe, and
— if predicted risk rises — brakes before the point of no return, retreats to a
retry region, and re-attempts with adjusted lateral offset, attitude, or speed.
Aborts are never labeled as would-be collisions; they are simply attempts that
ended without success, and the only hard negative signal comes from actual
contacts, which the training regime drives toward zero. This "learning from
safely aborted attempts" stands in contrast to learning-by-crashing and
connects to, but differs from, work on learned recovery and reset policies
(Leave-No-Trace; Recovery RL): abort is not a separate safety controller here
but an expressed behavior of the same unified policy being studied.

We instantiate this framework end to end. A single CNN+GRU sensorimotor policy
maps low-resolution onboard RGB, IMU, and proprioception directly to collective
thrust and body rates at 40 Hz. It observes no gap pose, no global position, no
attempt counter; attempt segmentation exists only in the evaluation
instrumentation. Training uses recurrent PPO over a procedurally randomized
distribution of walls with rotated rectangular gaps — including geometrically
infeasible ones — under randomized dynamics, latency, wind, and appearance, on
a single consumer GPU via a purpose-built vectorized simulator.

[RESULTS PARAGRAPH — fill with final numbers:
- overall success within 5 attempts on held-out feasible gaps: XX% (CI)
- first-attempt vs second-attempt conditional success: XX% -> XX% (p<...)
- context wiping at aborts collapses the gain to XX% (causal control)
- high-energy collision rate XX%; abort margin XX cm; give-up on infeasible XX%
- OOD retention …]

Contributions:
1. Problem formalization: multi-attempt gap traversal as a meta-POMDP with
   abort-mediated information gathering, plus an evaluation protocol that
   measures adaptation causally (context wiping, attempt-conditional success).
2. A minimal end-to-end system demonstrating that observe–probe–abort–retreat–
   adjust–retry emerges in a single recurrent policy from constraint-first
   multi-objective RL, with no task state machine.
3. Controlled evidence that cross-attempt memory is the mechanism: attempt-
   indexed conditional success, eval-time context wiping, and reset-trained /
   memoryless ablations.
4. Safety analysis of the abort-first regime: contact rates, abort clearance
   margins, a shadow stopping-envelope monitor, and rational give-up behavior
   on infeasible instances.
