# Implementation Path Comparison Analysis

> 2026-09-18：本计划已暂停，不得据 mean_diff 阈值选择架构，不得采用文末 90%/95% 无碰撞率作为验收。
> 以下保留为历史草案；当前执行协议为 `docs/MINIMAL_SAFE_PROBE_PROTOCOL.md`。

**Document Purpose**: Detailed comparison of three potential implementation paths based on M0.2 experimental results  
**Status**: Prepared for decision-making after experiments complete  
**Date**: 2026-09-18

---

## Executive Summary

This document provides a comprehensive analysis of three possible implementation paths for achieving zero-collision real-flight online adaptation. The path selection is contingent on the GRU context verification and attempt analysis experimental results.

**Decision Trigger**: Experimental metrics (mean_diff, improvement_rate)  
**Impact**: Determines 3-4 months of implementation work and resource allocation  
**Stakeholders**: Research team, training pipeline, real-flight validation

---

## Path A: Fast Track (Optimize Training)

### Trigger Conditions
- **mean_diff > 0.1**: GRU shows strong influence on policy decisions
- **improvement_rate > 70%**: Policy demonstrates clear cross-attempt learning

### Interpretation
✅ **Architecture is working as intended**
- GRU effectively captures and utilizes historical trial information
- The model has learned to adapt based on previous attempts
- Problem space: Training recipe, not architecture design

### Implementation Plan

#### Tasks to SKIP ✅
- ❌ M2.1: Context-Aware Policy refactoring
- ❌ Evidence Extractor implementation
- ❌ Context Adapter module design
- ❌ Architecture unit tests for new modules

**Time Saved**: ~2 weeks

#### Tasks to EXECUTE ⚡
1. **M2.2: Training Recipe Optimization** (1 week)
   - Enhance cross-attempt improvement loss weighting
   - Add curriculum learning stages
   - Optimize PPO hyperparameters (learning rate, clip ratio)
   - Data augmentation: gap width/height/orientation diversity

2. **M2.3: Safety Layer Integration** (1 week)
   - Implement Control Barrier Functions (CBF)
   - Hard constraints on minimum clearance
   - Emergency abort logic
   - Safety layer unit tests

3. **M3: Phase 1 Training** (3 weeks)
   - Train with optimized recipe
   - Monitor cross-attempt improvement metrics
   - Early stopping based on zero-collision rate

4. **M4: Phase 2 Training** (3 weeks)
   - Context learning phase with enhanced loss
   - Validation on held-out scenarios

5. **M5: Phase 3 Training** (3 weeks)
   - Zero-collision fine-tuning
   - Safety layer active throughout

**Total Timeline**: ~11 weeks (vs 13 weeks for Path B)

---

### Advantages ✅
- **Fastest path to real-flight validation**
- Lower implementation risk (no major refactoring)
- Existing code is validated and functional
- Can iterate on training faster

### Risks ⚠️
- May hit performance ceiling if architecture has hidden limitations
- Less flexible if requirements change
- Assumes current GRU capacity is sufficient for all scenarios

### Resource Requirements
- **Engineering**: 1 person-month (training pipeline work)
- **Compute**: High (more training iterations expected)
- **Testing**: Standard validation suite

---

## Path B: Full Refactoring (Rebuild Architecture)

### Trigger Conditions
- **mean_diff < 0.05**: GRU shows minimal influence on decisions
- **improvement_rate < 30%**: Policy does not learn from attempts

### Interpretation
⚠️ **Architecture has fundamental issues**
- GRU exists in code but model does not utilize it effectively
- Training signal is insufficient to force context usage
- Need explicit architectural mechanisms to enforce evidence extraction

### Implementation Plan

#### Tasks to EXECUTE 🔧

1. **M2.1: Context-Aware Policy Architecture** (2 weeks)
   
   **Evidence Extractor Module**:
   ```python
   class EvidenceExtractor(nn.Module):
       """
       Extracts actionable evidence from historical attempts.
       
       Input: Sequence of (obs, action, collision, clearance)
       Output: Evidence vector highlighting:
         - Where collisions occurred
         - Clearance margins per side
         - Dangerous directions
       """
       def forward(self, history_sequence):
           # Multi-head attention over history
           # Focus on collision events and near-misses
           evidence = self.attention(history_sequence)
           return evidence
   ```
   
   **Context Adapter Module**:
   ```python
   class ContextAdapter(nn.Module):
       """
       Fuses current observation with historical evidence.
       
       Forces policy to use evidence vector in decision making.
       """
       def forward(self, obs, evidence):
           # Cross-attention: obs queries evidence
           context = self.cross_attention(obs, evidence)
           return torch.cat([obs, context], dim=-1)
   ```
   
   **Refactored Policy**:
   ```python
   class ContextAwarePolicy(nn.Module):
       def __init__(self):
           self.evidence_extractor = EvidenceExtractor()
           self.context_adapter = ContextAdapter()
           self.policy_head = PolicyHead()
       
       def forward(self, obs, history):
           evidence = self.evidence_extractor(history)
           context_obs = self.context_adapter(obs, evidence)
           action = self.policy_head(context_obs)
           return action
   ```

2. **M2.2: Architecture Unit Tests** (3 days)
   - Evidence extractor output shape validation
   - Context adapter fusion correctness
   - Gradient flow verification
   - Historical replacement test (should fail without history)

3. **M2.3: Training Recipe Design** (1 week)
   - **Evidence Extraction Loss**: Force attention to collision events
   - **Context Usage Regularization**: Penalize policies that ignore evidence
   - **Cross-Attempt Improvement Loss**: Require improvement on attempt N+1
   
   ```python
   loss = (
       ppo_loss +
       lambda_evidence * evidence_loss +
       lambda_context * context_usage_loss +
       lambda_improve * cross_attempt_loss
   )
   ```

4. **M2.4: Safety Layer** (1 week)
   - Same as Path A

5. **M3-M5: Three-Phase Training** (9 weeks)
   - Same structure as Path A but with new architecture
   - Expect longer convergence times initially

**Total Timeline**: ~13 weeks

---

### Advantages ✅
- **Most robust solution**: Explicit architectural guarantees
- Higher performance ceiling
- More interpretable (can visualize evidence extraction)
- Better for paper publication (novel architecture)

### Risks ⚠️
- Longest implementation time
- Higher engineering complexity
- New bugs in custom modules
- May need multiple architecture iterations

### Resource Requirements
- **Engineering**: 2 person-months (architecture + training)
- **Compute**: Very High (more training from scratch)
- **Testing**: Extensive unit tests + integration tests

---

## Path C: Enhancement (Augment Existing Architecture)

### Trigger Conditions
- **mean_diff 0.05-0.1**: GRU shows moderate influence
- **improvement_rate 30-70%**: Partial cross-attempt learning

### Interpretation
🟡 **Architecture is on the right track but insufficient**
- GRU captures some context but not task-specific enough
- Model shows improvement behavior but inconsistently
- Need targeted enhancements, not full rebuild

### Implementation Plan

#### Tasks to EXECUTE 🔄

1. **M2.1: Attention-Augmented GRU** (1 week)
   
   ```python
   class AttentionGRU(nn.Module):
       """
       GRU with attention mechanism over its own hidden states.
       
       Allows model to selectively focus on relevant history.
       """
       def __init__(self):
           self.gru = nn.GRU(input_size, hidden_size)
           self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
       
       def forward(self, x_seq):
           gru_out, hidden = self.gru(x_seq)
           
           # Self-attention: which parts of history matter most?
           attn_out, attn_weights = self.attention(
               query=gru_out[-1:],  # current state
               key=gru_out,          # all history
               value=gru_out
           )
           
           return attn_out, attn_weights  # can visualize attention
   ```

2. **M2.2: Enhanced Training Objectives** (1 week)
   
   **Add Explicit Cross-Attempt Loss**:
   ```python
   def cross_attempt_improvement_loss(episode_data):
       """
       Penalize if attempt N+1 is not better than attempt N.
       """
       attempts = split_by_attempts(episode_data)
       loss = 0
       for i in range(len(attempts) - 1):
           if attempts[i+1].clearance <= attempts[i].clearance:
               loss += penalty
       return loss
   ```
   
   **Attention Supervision** (optional):
   - If we have labels for "important" historical moments
   - Guide attention to collision events

3. **M2.3: Data Augmentation** (3 days)
   - Increase diversity of multi-attempt scenarios in training
   - Generate more "progressive difficulty" episodes
   - Ensure sufficient coverage of collision → recovery patterns

4. **M2.4: Safety Layer** (1 week)
   - Same as Path A

5. **M3-M5: Three-Phase Training** (9 weeks)
   - Monitor attention weights for interpretability
   - Validate that attention focuses on relevant history

**Total Timeline**: ~12 weeks (saves ~1 week vs Path B)

---

### Advantages ✅
- **Balanced approach**: Faster than full refactoring
- Lower risk than Path A (adds explicit mechanisms)
- Attention weights provide interpretability
- Incremental improvement on working baseline

### Risks ⚠️
- May still underperform if GRU capacity is fundamentally limited
- Attention may not be sufficient (might need full refactoring later)
- Middle-ground solution might be "worst of both worlds"

### Resource Requirements
- **Engineering**: 1.5 person-months
- **Compute**: High
- **Testing**: Moderate (new attention module + training tests)

---

## Decision Matrix

| Criteria | Path A (Fast) | Path B (Refactor) | Path C (Enhance) |
|----------|---------------|-------------------|------------------|
| **Timeline** | 11 weeks ⚡ | 13 weeks 🐢 | 12 weeks 🚶 |
| **Risk** | Medium 🟡 | High ⚠️ | Medium 🟡 |
| **Performance Ceiling** | Lower ⬇️ | Highest ⬆️⬆️ | Medium ➡️ |
| **Interpretability** | Low 🔒 | High 🔓 | Medium 🔍 |
| **Engineering Effort** | Low ✅ | High ⚠️ | Medium 🔧 |
| **Paper Novelty** | Low 📄 | High 📝✨ | Medium 📃 |
| **Fallback Options** | Can pivot to C/B | None | Can escalate to B |

---

## Experimental Result Interpretation Guide

### Scenario 1: Clear Winner
**Results**: mean_diff = 0.15, improvement_rate = 78%  
**Decision**: Path A (Fast Track)  
**Confidence**: High ✅  
**Rationale**: Strong signals on both metrics, architecture is proven effective

---

### Scenario 2: Clear Failure
**Results**: mean_diff = 0.02, improvement_rate = 18%  
**Decision**: Path B (Full Refactoring)  
**Confidence**: High ✅  
**Rationale**: Weak signals on both metrics, fundamental architecture issue confirmed

---

### Scenario 3: Mixed Signals (Strong Context, Weak Improvement)
**Results**: mean_diff = 0.12, improvement_rate = 35%  
**Decision**: Path C (Enhancement) → Focus on training objectives  
**Rationale**: Architecture captures context, but training doesn't encourage improvement

---

### Scenario 4: Mixed Signals (Weak Context, Strong Improvement)
**Results**: mean_diff = 0.06, improvement_rate = 72%  
**Decision**: Path C (Enhancement) → Focus on attention mechanism  
**Rationale**: Model is learning to improve but context signal is noisy

---

### Scenario 5: Borderline Both Metrics
**Results**: mean_diff = 0.08, improvement_rate = 55%  
**Decision**: Path C (Enhancement)  
**Confidence**: Medium 🟡  
**Rationale**: Default to incremental improvement; can escalate to Path B if needed

---

## Recommended Decision Process

```
Step 1: Run experiments
   ↓
Step 2: Extract mean_diff, improvement_rate
   ↓
Step 3: Plot on decision space
   
   improvement_rate
        ↑
    100%|           A  A  A
        |        A  A  A
     70%|-----C--C--C------
        |     C  C  C
     50%|  C  C  C
        |  B  C  C
     30%|--B--B--C---------
        |  B  B  B
      0%|  B  B  B
        └──────────────→ mean_diff
          0.0  0.05 0.1 0.15
   
Step 4: Confirm decision with team
   ↓
Step 5: Update project plan with chosen path
   ↓
Step 6: Begin M1 tasks under selected path
```

---

## Contingency Plans

### If Path A Fails Mid-Training
**Symptoms**: 
- Training converges but zero-collision rate plateaus at <50%
- Attention analysis shows GRU ignoring critical information

**Action**:
- Pivot to Path C within 1 week
- Add attention mechanism
- Resume training from last checkpoint

---

### If Path C Underperforms
**Symptoms**:
- Attention weights show uniform distribution (not selective)
- Cross-attempt improvement rate does not increase

**Action**:
- Escalate to Path B within 2 weeks
- Implement full Context-Aware architecture
- Accept timeline delay

---

### If Path B Takes Too Long
**Symptoms**:
- Architecture debugging exceeds 3 weeks
- Training convergence is slower than expected

**Action**:
- Reassess scope: Can we ship Path C as MVP?
- Consider: Path C for paper v1, Path B for follow-up work

---

## Success Metrics (All Paths)

Regardless of path chosen, success is measured by:

1. **Simulation Performance**:
   - Zero-collision rate > 95% on novel gaps
   - Average attempts to success < 2.5
   - Success within 3 attempts > 98%

2. **Real-Flight Performance**:
   - Zero-collision rate > 90% on real gaps
   - Safe abort rate < 5%
   - Human-level adaptation demonstrated

3. **Timeline**:
   - M0-M6 completed by 2027-02-17
   - Real-flight validation by 2027-03-17

---

## Next Steps After Decision

Once experimental results are in:

1. **Immediate** (Day 1):
   - Update `.project/decisions.md` with chosen path + rationale
   - Update `.project/milestones.json` with path-specific tasks
   - Create M1 task breakdown document

2. **Within 1 Week**:
   - Begin M1.1: Baseline validation with chosen metrics
   - Set up monitoring for path-specific KPIs
   - Create checkpoint for potential pivot

3. **Ongoing**:
   - Weekly progress reviews
   - Contingency plan activation if needed

---

**Document Status**: ✅ Ready for decision-making  
**Awaiting**: Experimental results from M0.2 verification scripts  
**Next Update**: After path selection (post-experiments)
