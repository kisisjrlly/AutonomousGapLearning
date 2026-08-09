# Abstract draft（v1 骨架 — {{PLACEHOLDER}} 待填充）

Autonomous vehicles flying through a *novel* narrow opening face a dilemma that
one-shot planning systems ignore: the feasibility of the passage — true gap
clearance against the vehicle, its control authority under hidden mass,
latency, and wind — is revealed only by acting, and acting risks contact.
Classical aerial autonomy resolves this by decomposing the problem into
detection, planning, and retry logic; learning-based agile flight trains a
frozen policy to master a known track. Neither regime learns *how to try
again*.

Here we show that a single recurrent end-to-end policy can learn to probe an
unknown gap, brake and retreat before contact when risk rises, adjust its
approach, and re-attempt — with no hand-coded task state machine and no
collision-driven data collection. The key mechanism is *in-context adaptation
through cross-attempt memory*: the policy is meta-trained over multi-attempt
episodes, and its recurrent state carries information across attempts while
weights remain frozen, making adaptation immediate, verifiable, and
deployable.

We train one CNN+GRU policy mapping low-resolution onboard RGB, IMU, and
proprioception directly to collective thrust and body rates at 40 Hz, in a
purpose-built GPU-vectorized simulator with randomized gap geometry (width
0.34–0.90 m, height 0.30–0.80 m, in-plane roll ±40°, thickness 0.05–0.30 m),
hidden dynamics (mass 0.60–0.95 kg, thrust-to-weight 2.2–3.4, latency
0–50 ms), wind, and appearance — including geometrically infeasible gaps.
Evaluation on held-out task banks shows: {{FINAL_SUCCESS}}% success within
five attempts on feasible gaps; second-attempt conditional success
{{K2}}% vs first-attempt {{K1}}% (p={{P}}); zeroing the recurrent state at
every abort erases this gain ({{K2_WIPE}}%, n.s.), isolating cross-attempt
memory as the mechanism; {{COLL_HIGH}}% high-energy collision rate with
aborts occurring {{ABORT_MARGIN}} m before contact on average; and rational
give-up on {{GIVEUP_INFEASIBLE}}% of infeasible instances. The policy
generalizes beyond its training distribution to extrapolated geometry and
dynamics (ID {{ID}}% / OOD {{OOD}}%), and deploys at {{LATENCY}} ms per step
on a single CPU core.

**One-sentence significance**: learning from *safely aborted attempts* — not
from crashes — turns trial-and-error into a safe, deployable adaptation
strategy for agile aerial autonomy.
