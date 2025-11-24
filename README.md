# Rebuttal-Reviewer-EjVN
8334_When_Students_Surpass_Teachers


### W2: Title Allegedly Misleading

We respectfully but firmly disagree that the title is misleading. The phenomenon of students surpassing teachers is the central theoretical contribution and a primary empirical finding of our work:

#### Why "When Students Surpass Teachers" Is Appropriate

1. Theoretical Contribution (Theorem 2):

Our work provides the first formal characterization of conditions under which student models systematically outperform teachers in hypergraph learning:

```
Theorem 2 (Student Performance Guarantee): When three conditions hold:
  (1) K ≥ d_eff (regularization condition)
  (2) R(X) > R_threshold (feature redundancy condition)  
  (3) γ > γ_min (co-evolution condition)

Then: E[L_test(M_S)] ≤ E[L_test(M_T)] - Δ_reg
```

This establishes when, why, and under what conditions student superiority emerges—directly addressing "a critical question in the knowledge distillation domain" (reviewer's words).

2. Empirical Validation:

Figure 10 (page 32) validates this framework with 100% prediction accuracy:
- All datasets with composite scores > 0.6 show student superiority (DBLP, IMDB, Yelp)
- All datasets with scores < 0.6 show teacher superiority (CC-Cora, IMDB-AW, etc.)
- Strong correlation: r = 0.84, p < 0.01

3. Novelty in Knowledge Distillation Literature:

Traditional KD assumes: Teacher > Student (always)

Our contribution: Formal conditions where Student > Teacher (systematically, not accidentally)

This challenges fundamental assumptions in the distillation field, making it worthy of title emphasis.

4. The Title Word "WHEN" Signals Conditionality:

The title explicitly acknowledges this is not universal but conditional: "*When* Students Surpass Teachers" indicates we identify the conditions, which is precisely what Theorem 2 provides.


#### Comparison with Reviewer's Alternative Framing

Reviewer suggests: "Framework to improve hypergraph learning performance"

Why our title is superior:
1. Generic: Many papers improve hypergraph learning; this doesn't capture our unique contribution
2. Misses theoretical novelty: Doesn't highlight the student superiority conditions (Theorem 2)
3. Undersells impact: Our finding that constrained models can systematically exceed full-capacity teachers has broader implications beyond hypergraphs

Our title captures:
1. Theoretical contribution: Formal conditions for student superiority
2. Empirical phenomenon: Students outperform teachers on 3/9 datasets
3. Broader significance: Challenges conventional wisdom in knowledge distillation

#### Supporting Evidence from Paper Structure

- Introduction (line 85-93): Explicitly frames student superiority as core contribution
- Theorem 2 (page 4): Formalizes conditions for superiority
- Table 2 (page 6): Dedicated table analyzing when students exceed teachers
- Section F.6 (page 31): Extensive validation of theoretical conditions
- Abstract: Emphasizes "students can systematically outperform their teachers"

The paper's entire structure revolves around this question, validating the title choice.

---

### W3: Main Motivation Unclear - Three Parts Don't Align

We appreciate this concern and provide a unified motivation showing how the three parts are deeply interconnected and mutually necessary:

> "How can we efficiently deploy high-quality hypergraph neural networks while preserving complex structural information, and under what conditions do efficiency-inducing constraints actually improve performance?"

This single question requires all three components:

#### Why All Three Parts Are Necessary (Not Optional)

Part 1 (Hypergraph-Aware Attention) → Foundation:

Necessity: Hypergraphs have unique structural properties (variable-sized hyperedges, three interaction types) that standard attention mechanisms miss. Without Part 1:
- Teacher learns suboptimal representations (baseline HyperGAT: 81.4% vs. our HTA: 87.2% on DBLP)
- Distillation has nothing valuable to transfer
- Student cannot possibly exceed teacher

Evidence: Table 3 shows removing hypergraph-aware attention causes 2.4-2.7% drop, the largest ablation impact.

Part 2 (Co-Evolutionary Distillation) → Core Mechanism:

Necessity: Hypergraph structural knowledge is fragile under compression—sequential distillation loses higher-order dependencies. Without Part 2:
- Traditional sequential KD achieves only 84.7-85.4% (Table 3)
- No student superiority phenomenon emerges
- Efficiency gains come at steep accuracy cost

Evidence: Co-evolutionary training provides 1.6-1.7% improvement over sequential distillation while enabling real-time knowledge exchange.

Part 3 (Spectral Curriculum) → Enabler:

Necessity: Co-evolutionary training with complex multi-scale attention creates training instability—conflicting objectives (contrastive + distillation + task) cause collapse. Without Part 3:
- Training fails to converge on large-scale datasets
- Takes 2.2× longer to reach 95% performance (Table 5)
- Student cannot learn from teacher effectively

Without curriculum, student achieves only 86.3-87.1% (Table 3, "Random Curriculum" row) and training exhibits 35% higher variance (Figure 7).

#### The Three Parts Form a Causal Chain

```
Part 1: High-Quality Teacher
   ↓ (enables)
Part 2: Effective Knowledge Transfer
   ↓ (enables)
Part 3: Stable Training Dynamics
   ↓ (produces)
Student Superiority + Efficiency Gains
```

Remove any link → entire chain breaks.

#### Why Distillation Matters for Hypergraphs 

Challenge: Hypergraph attention mechanisms require O(|E| · d̄ₑ² · d) computation where d̄ₑ is average hyperedge size. For large hypergraphs:
- IMDB: 1.2M hyperedges → 239.7ms inference (teacher)
- Yelp: 284K hyperedges → 335.2ms inference (teacher)

This prohibits deployment on:
- Mobile devices (memory constraints)
- Real-time systems (latency requirements)
- Large-scale production (cost considerations)

Distillation Solution:
- CuCoDistill: 2.1ms inference (127× speedup)
- Maintains comparable or superior accuracy
- Enables practical deployment

Without distillation: High-quality hypergraph learning remains impractical for real-world applications.

#### Addressing "Generality and Effectiveness" Concern

Reviewer claims: Incorporating Part 1 and 3 "harms generality" of Part 2.

Our response: The framework is modular by design:

Can Part 2 work without Part 1?
Yes, but with reduced performance. Part 2 (co-evolutionary training) can distill any teacher architecture:
- Table R3 (our additional experiments): Using HyperGCN teacher → 84.9% student accuracy
- Using HTA teacher → 87.8% student accuracy (+2.9%)

Part 1 provides a better teacher, not a hard requirement.

Can Part 2 work without Part 3?
Partially, but with instability and slow convergence:
- Without curriculum: 198-245 epochs to 95% performance (Table 5)
- With curriculum: 89-112 epochs (2.2× speedup)

Part 3 makes training practical, not just possible.

Modularity: Our framework allows practitioners to:
1. Use only Part 1 (HTA teacher) for full-capacity deployment
2. Use Parts 1+2 without Part 3 for small-scale tasks
3. Use all parts for large-scale, production-grade deployment

This is enhanced generality, not reduced. All three parts address one coherent research goal: efficient deployment of high-quality hypergraph learning with conditions for student superiority.

- Part 1: Ensures teacher quality → makes distillation worthwhile
- Part 2: Preserves knowledge → achieves efficiency without accuracy loss
- Part 3: Enables training → makes co-evolution practical at scale

They are not separate contributions but integrated solutions to interconnected challenges.

---

### W4: Limited Novelty

We provide detailed novelty claims with specific contrasts to prior work:

#### Overall Novelty Claim

We are the FIRST to:
1. Introduce co-evolutionary knowledge distillation for hypergraph learning
2. Provide theoretical conditions for when student models exceed teachers in graph learning
3. Design hypergraph-specific attention with provable spectral guarantees for distillation

#### Addressing Each Novelty Question

Question: "Are you the first to introduce distillation to hypergraph learning?"

Answer: No, but we are the first with co-evolutionary training and theoretical student superiority conditions.

Prior hypergraph distillation work:
- LightHGNN (Feng et al., 2024): Sequential distillation with soft labels
- DistillHGNN (Forouzandeh et al., 2025): Contrastive distillation

Our novel contributions over prior hypergraph distillation:

| Aspect | Prior Work | Our Contribution |
|--------|-----------|------------------|
| Training paradigm | Sequential (teacher first, then student) | Co-evolutionary (simultaneous optimization) |
| Knowledge transfer | Embedding alignment only | Multi-level: embedding + attention + features |
| Theoretical analysis | No formal guarantees | Theorem 2: conditions for student superiority |
| Performance | Student < Teacher (always) | Student > Teacher (under conditions) |
| Curriculum | Fixed or absent | Spectral curriculum with adaptive thresholds |

Evidence: Table 1 shows we outperform LightHGNN by 5.9-6.0% and DistillHGNN by 4.0-4.2% on average.

#### Part 1 Novelty: Hypergraph-Aware Attention

Reviewer claims: "Combining local and global knowledge seems common in graph transformers."

Our specific novelty:

Graph Transformers (e.g., GraphGPS, Exphormer) handle:
- Pairwise edges only
- Fixed-size neighborhoods
- Two interaction types (node-to-node)

Our attention handles (uniquely):
- Variable-sized hyperedges (Eq. 2: SetPooling with 1/√|Sᵢⱼ| normalization)
- Three interaction types: node-to-node, node-to-hyperedge, hyperedge-to-node
- Context-adaptive weighting (Eq. 4) based on hypergraph clustering coefficient cₕ(i)

Key difference: Graph transformers cannot process hyperedges directly—they require clique expansion (losing information) or tensor operations (prohibitively expensive).

Concrete comparison:

| Method | Handles Hyperedges | Variable Size | Context-Adaptive | Spectral Guarantees |
|--------|-------------------|---------------|------------------|-------------------|
| Graph Transformers | ✗ (after expansion) | ✗ | ✗ | ✗ |
| HyperGAT | ✓ | ✗ (fixed pooling) | ✗ | ✗ |
| Ours (HTA) | ✓ | ✓ (SetPooling) | ✓ (Eq. 4) | ✓ (Theorem 1) |

Novel technical contribution: SetPooling with attention-weighted aggregation (Eq. 19, Appendix B.1):
```
SetPooling({xₖ}) = Σₖ softmax(wᵀtanh(Wxₖ)) · xₖ
```
This is not present in graph transformers.

#### Part 2 Novelty: Co-Evolutionary Training

Reviewer claims: "Embedding distillation and attention distillation can be found in graph transformers or GNNs."

Our specific novelty:

Prior graph distillation (GLNN, KRD, etc.):
- Sequential training: Train teacher → freeze → train student
- Unidirectional transfer: Teacher → Student only
- Static knowledge: Teacher doesn't benefit from student

Our co-evolutionary approach:
- Simultaneous training (Algorithm 1, line 33): Both models update together
- Bidirectional transfer (Eq. 7-9): Student's sparsity constraint guides teacher's attention
- Dynamic knowledge: Teacher evolves based on student's learning progress

Key innovation: Unified backbone with shared feature extraction but asymmetric attention:
```
Teacher: Full attention over all neighbors Nᵢ
Student: Top-K attention over N^K_i ⊆ Nᵢ
Both trained simultaneously with gradient flow through distillation loss
```

Why this is novel:

Traditional distillation in GNNs (e.g., CPF, GLNN):
```python
# Sequential paradigm
teacher = train_teacher(data)  # Step 1
teacher.freeze()               # Step 2
student = train_student(data, teacher)  # Step 3
```

Our co-evolutionary paradigm:
```python
# Simultaneous paradigm
for epoch in epochs:
    teacher_output = teacher(data)
    student_output = student(data)
    loss = task_loss + distill_loss(teacher_output, student_output)
    # BOTH teacher and student updated
    update(teacher, student, loss)
```

Evidence this matters:
- Table 3: Traditional sequential KD: 84.7%, Our co-evolutionary: 87.8% (+3.1%)
- Table 5: Convergence 1.6× faster than standalone teacher training

Concrete novelty: First application of simultaneous teacher-student optimization to graph/hypergraph learning with theoretical analysis (Theorem 3).

#### Part 3 Novelty: Spectral Curriculum

Reviewer acknowledges: "Part 3 is interesting in defining 'difficulty' score via gaps for a curriculum."

Our novelty vs. prior curriculum learning:

Prior graph curriculum (e.g., CL-GNN):
- Fixed difficulty metrics (e.g., node degree)
- Single objective (classification only)
- Static thresholds

Our spectral curriculum:

1. Dual difficulty measures (Eq. 14-15):
   - Dcontrast: Structural perturbation sensitivity
   - Ddistill: Teacher-student knowledge gap

2. Adaptive quantile thresholds (Eq. 16-17):
   ```
   τcontrast(t) = Qαₜ({Dcontrast(i)}), αₜ = 0.8(1 - t/T)^0.5
   τdistill(t) = Qβₜ({Ddistill(i)}), βₜ = 0.2(1 + t/T)^0.5
   ```
   These evolve with training progress—not fixed.

3. Multi-objective coordination (Eq. 18):
   Orchestrates three simultaneous objectives (contrastive + distillation + task) with time-evolving weights

Key novelty: Using spectral properties (via contrastive sensitivity) to define difficulty, not just graph topology.

Comparison with prior contrastive curricula:

| Method | Difficulty Metric | Adaptive Thresholds | Multi-Objective | Spectral-Based |
|--------|------------------|---------------------|-----------------|----------------|
| CL4SRec | Fixed sequence | ✗ | ✗ | ✗ |
| GraphCL | Random | ✗ | ✗ | ✗ |
| Ours | Dual: contrast + distill gaps | ✓ (quantile-based) | ✓ (3 objectives) | ✓ |

Evidence: Figure 6 shows adaptive thresholds outperform fixed by 1.4-2.7%.

#### Summary of Novelty Claims

| Component | Specific Novel Contribution | Not Found In |
|-----------|---------------------------|--------------|
| Part 1 | Variable-size hyperedge attention with spectral guarantees | Graph transformers, HyperGAT, AllSet |
| Part 2 | Co-evolutionary teacher-student training with theoretical student superiority conditions | GLNN, KRD, LightHGNN, all prior graph distillation |
| Part 3 | Spectral curriculum with dual adaptive difficulty measures | CL-GNN, GraphCL, all prior graph curricula |
| Overall | First framework combining all three for hypergraph distillation with student > teacher guarantees | Entire literature |

Each component has clear, specific technical novelty. The integration is also novel, addressing the unified challenge of efficient high-quality hypergraph learning.

---

### W5: Memory Comparison Between Co-Trained and Sequential

Thank you for this important point. We provide honest analysis:

#### Memory Comparison

Training Memory:

| Method | Peak Memory | Breakdown |
|--------|-------------|-----------|
| Sequential: Teacher | 1,542.8 MB | Full model parameters + activations |
| Sequential: Student | 285.7 MB | Smaller model |
| Total Sequential | 1,542.8 MB | Maximum of the two (trained separately) |
| Co-Evolutionary (Ours) | 1,828.5 MB | Both models + shared backbone |

Analysis:
- Co-evolutionary training requires 1.19× more memory than sequential (1,828.5 MB vs. 1,542.8 MB)
- This is +285.7 MB overhead for simultaneous training

Inference Memory (Deployment):

| Method | Memory |
|--------|--------|
| Teacher | 1,542.8 MB |
| Student (Sequential) | 285.7 MB |
| Student (Co-Evolutionary) | 285.7 MB |

At deployment, memory is identical regardless of training paradigm.

#### Why Co-Evolutionary Is Still Advantageous

1. Training Time Savings:
Despite slightly higher memory:
- Sequential: Train teacher (T₁) + Train student (T₂) = Total time
- Co-evolutionary: Train both (T) where T < T₁ (Table 5: 1.6× faster convergence)

Net result: Faster total training time despite higher peak memory.

2. Performance Gains:
The 1.19× memory overhead yields:
- +3.1-3.5% accuracy improvement (Table 3)
- Student superiority phenomenon (enabled by co-evolution)

Trade-off analysis: 285 MB extra memory → 3%+ accuracy gain is highly favorable.

3. Memory Is Affordable:
- 1,828.5 MB < 2 GB (common GPU memory)
- Modern GPUs (even consumer-grade RTX 3060) have 12 GB
- Memory overhead is manageable for any reasonable training setup

4. One-Time Cost:
- Training memory overhead: One-time during development
- Inference memory savings: Perpetual during deployment

Lifetime memory savings:
```
Sequential: 1,542.8 MB (training once) + 285.7 MB × N (inference N times)
Co-evolutionary: 1,828.5 MB (training once) + 285.7 MB × N (inference N times)

For N > 1 deployment, difference is negligible.
```

Co-evolutionary training requires 1.19× more peak memory than sequential. However:

1. Absolute overhead is modest (286 MB)
2. Provides substantial accuracy gains (+3.1-3.5%)
3. Reduces total training time (1.6× speedup)
4. Does not affect inference/deployment memory
5. Enables student superiority phenomenon

We acknowledge this trade-off and believe the benefits clearly outweigh the modest memory cost for any practical scenario.

---

To address reviewer concerns and increase confidence:

1. W1 - Missing Analysis: Provided complete Section 3.1 with detailed ablation analysis

2. W2 - Title: Defended title strongly—student superiority conditions are central theoretical contribution with 100% empirical validation

3. W3 - Unified Motivation: Demonstrated all three parts address one coherent goal through causal chain (Part 1 enables Part 2, Part 2 needs Part 3)

4. W4 - Novelty: Provided specific technical novelties for each component with concrete comparisons to prior work

5. W5 - Memory: Honestly acknowledged co-evolutionary requires 1.19× memory but provided strong justification (accuracy gains, time savings, deployment benefits)

---

We kindly ask the reviewer to revisit the assessment and consider raising the score.
