import Std

namespace PerfRuler.Operational

structure Device where
  (peakFlops peakTC peakALU peakSFU bandwidth : Nat)
  peakFlops_pos : 0 < peakFlops
  peakTC_pos : 0 < peakTC
  peakALU_pos : 0 < peakALU
  peakSFU_pos : 0 < peakSFU
  bandwidth_pos : 0 < bandwidth

structure Op where
  (work tcWork aluWork sfuWork traffic stepFloor : Nat)

structure DAGNode extends Op where
  parents : List Nat

inductive DAG where
  | empty
  | snoc (prior : DAG) (node : DAGNode)

def DAG.size : DAG → Nat
  | .empty => 0
  | .snoc prior _ => prior.size + 1

def totalBy (amount : Op → Nat) : DAG → Nat
  | .empty => 0
  | .snoc prior v => totalBy amount prior + amount v.toOp

def totalWork : DAG → Nat := totalBy Op.work
def totalTC : DAG → Nat := totalBy Op.tcWork
def totalALU : DAG → Nat := totalBy Op.aluWork
def totalSFU : DAG → Nat := totalBy Op.sfuWork
def totalTraffic : DAG → Nat := totalBy Op.traffic

def ForNodes (P : Nat → DAGNode → Prop) : DAG → Prop
  | .empty => True
  | .snoc prior v => ForNodes P prior ∧ P prior.size v

def Topological (g : DAG) : Prop :=
  ForNodes (fun i v => ∀ p ∈ v.parents, p < i) g

def maxParent (value : Nat → Nat) : List Nat → Nat
  | [] => 0
  | p :: ps => max (value p) (maxParent value ps)

def clock : DAG → Nat → Nat
  | .empty => fun _ => 0
  | .snoc prior v => fun i =>
      if i = prior.size then maxParent (clock prior) v.parents + v.stepFloor
      else clock prior i

def critPath : DAG → Nat
  | .empty => 0
  | .snoc prior v => max (critPath prior) (clock (.snoc prior v) prior.size)

def sumTicks : Nat → (Nat → Nat) → Nat
  | 0, _ => 0
  | n + 1, f => sumTicks n f + f n

def sumNodes (g : DAG) (f : Nat → Nat) : Nat := sumTicks g.size f

structure OpTrace (g : DAG) (d : Device) where
  T : Nat
  (workAt tcAt aluAt sfuAt memAt : Nat → Nat → Nat)
  (start finish : Nat → Nat)
  tick_work_cap : ∀ t < T, sumNodes g (workAt t) ≤ d.peakFlops
  tick_tc_cap : ∀ t < T, sumNodes g (tcAt t) ≤ d.peakTC
  tick_alu_cap : ∀ t < T, sumNodes g (aluAt t) ≤ d.peakALU
  tick_sfu_cap : ∀ t < T, sumNodes g (sfuAt t) ≤ d.peakSFU
  tick_mem_cap : ∀ t < T, sumNodes g (memAt t) ≤ d.bandwidth
  node_complete : ForNodes (fun i v =>
    v.work ≤ sumTicks T (fun t => workAt t i) ∧
    v.tcWork ≤ sumTicks T (fun t => tcAt t i) ∧
    v.aluWork ≤ sumTicks T (fun t => aluAt t i) ∧
    v.sfuWork ≤ sumTicks T (fun t => sfuAt t i) ∧
    v.traffic ≤ sumTicks T (fun t => memAt t i)) g
  within : ForNodes (fun i _ => finish i ≤ T) g
  causal : ForNodes (fun i v => ∀ p ∈ v.parents, p < i ∧ finish p ≤ start i) g
  duration : ForNodes (fun i v => start i + v.stepFloor ≤ finish i) g

private theorem sumTicks_zero (n : Nat) : sumTicks n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih

private theorem sumTicks_add (n : Nat) (f h : Nat → Nat) :
    sumTicks n (fun i => f i + h i) = sumTicks n f + sumTicks n h := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [sumTicks, ih]
      ac_rfl

theorem sumNodes_sumTicks_eq (g : DAG) (T : Nat) (f : Nat → Nat → Nat) :
    sumNodes g (fun i => sumTicks T (fun t => f t i)) =
      sumTicks T (fun t => sumNodes g (fun i => f t i)) := by
  induction T with
  | zero => exact sumTicks_zero g.size
  | succ T ih =>
      calc
        _ = sumNodes g (fun i => sumTicks T (fun t => f t i)) + sumNodes g (f T) :=
          sumTicks_add g.size _ _
        _ = _ := congrArg (fun n => n + sumNodes g (f T)) ih

private theorem sumTicks_le_cap (T cap : Nat) (f : Nat → Nat)
    (h : ∀ t < T, f t ≤ cap) : sumTicks T f ≤ cap * T := by
  induction T with
  | zero => exact Nat.le_refl 0
  | succ T ih =>
      rw [sumTicks, Nat.mul_succ]
      exact Nat.add_le_add (ih (fun t ht => h t (Nat.lt_succ_of_lt ht)))
        (h T (Nat.lt_succ_self T))

private theorem ForNodes.imp {P Q : Nat → DAGNode → Prop} {g : DAG}
    (hp : ForNodes P g) (h : ∀ i v, P i v → Q i v) : ForNodes Q g := by
  induction g with
  | empty => trivial
  | snoc prior v ih => exact ⟨ih hp.1, h prior.size v hp.2⟩

theorem OpTrace.topological {g : DAG} {d : Device} (tr : OpTrace g d) : Topological g :=
  ForNodes.imp tr.causal (fun _ _ h p hp => (h p hp).1)

private theorem totalBy_le_sumNodes (amount : Op → Nat) (f : Nat → Nat) {g : DAG}
    (h : ForNodes (fun i v => amount v.toOp ≤ f i) g) :
    totalBy amount g ≤ sumNodes g f := by
  induction g with
  | empty => exact Nat.le_refl 0
  | snoc prior v ih => exact Nat.add_le_add (ih h.1) h.2

private theorem channel_bound {g : DAG} {T cap : Nat} (amount : Op → Nat)
    (issued : Nat → Nat → Nat) (capacity : ∀ t < T, sumNodes g (issued t) ≤ cap)
    (complete : ForNodes (fun i v => amount v.toOp ≤ sumTicks T (fun t => issued t i)) g) :
    totalBy amount g ≤ cap * T := calc
  _ ≤ sumNodes g (fun i => sumTicks T (fun t => issued t i)) :=
    totalBy_le_sumNodes amount _ complete
  _ = sumTicks T (fun t => sumNodes g (issued t)) := sumNodes_sumTicks_eq g T issued
  _ ≤ cap * T := sumTicks_le_cap T cap _ capacity

variable {g : DAG} {d : Device}

theorem operational_compute_lower_bound (tr : OpTrace g d) :
    totalWork g ≤ d.peakFlops * tr.T :=
  channel_bound Op.work tr.workAt tr.tick_work_cap
    (ForNodes.imp tr.node_complete (fun _ _ h => h.1))

theorem operational_memory_lower_bound (tr : OpTrace g d) :
    totalTraffic g ≤ d.bandwidth * tr.T :=
  channel_bound Op.traffic tr.memAt tr.tick_mem_cap
    (ForNodes.imp tr.node_complete (fun _ _ h => h.2.2.2.2))

theorem operational_three_unit_lower_bound (tr : OpTrace g d) :
    totalTC g ≤ d.peakTC * tr.T ∧ totalALU g ≤ d.peakALU * tr.T ∧
      totalSFU g ≤ d.peakSFU * tr.T :=
  ⟨channel_bound Op.tcWork tr.tcAt tr.tick_tc_cap
      (ForNodes.imp tr.node_complete (fun _ _ h => h.2.1)),
   channel_bound Op.aluWork tr.aluAt tr.tick_alu_cap
      (ForNodes.imp tr.node_complete (fun _ _ h => h.2.2.1)),
   channel_bound Op.sfuWork tr.sfuAt tr.tick_sfu_cap
      (ForNodes.imp tr.node_complete (fun _ _ h => h.2.2.2.1))⟩

private theorem maxParent_le (value : Nat → Nat) (ps : List Nat) (b : Nat)
    (h : ∀ p ∈ ps, value p ≤ b) : maxParent value ps ≤ b := by
  induction ps with
  | nil => exact Nat.zero_le _
  | cons p ps ih =>
      exact Nat.max_le.mpr ⟨h p List.mem_cons_self,
        ih (fun q hq => h q (List.mem_cons_of_mem p hq))⟩

private theorem schedule_bounds {start finish : Nat → Nat} {T : Nat}
    (within : ForNodes (fun i _ => finish i ≤ T) g)
    (causal : ForNodes (fun i v => ∀ p ∈ v.parents, p < i ∧ finish p ≤ start i) g)
    (duration : ForNodes (fun i v => start i + v.stepFloor ≤ finish i) g) :
    (∀ i, clock g i ≤ finish i) ∧ critPath g ≤ T := by
  induction g with
  | empty => exact ⟨fun _ => Nat.zero_le _, Nat.zero_le _⟩
  | snoc prior v ih =>
      obtain ⟨hclock, hpath⟩ := ih within.1 causal.1 duration.1
      have hParents : maxParent (clock prior) v.parents ≤ start prior.size :=
        maxParent_le _ _ _ (fun p hp => Nat.le_trans (hclock p) (causal.2 p hp).2)
      have hLast : clock (.snoc prior v) prior.size ≤ finish prior.size := by
        simpa [clock] using
          (Nat.le_trans (Nat.add_le_add_right hParents v.stepFloor) duration.2)
      refine ⟨?_, Nat.max_le.mpr ⟨hpath, Nat.le_trans hLast within.2⟩⟩
      intro i
      by_cases hi : i = prior.size
      · subst i
        exact hLast
      · simpa [clock, hi] using hclock i

theorem operational_critical_path_lower_bound (tr : OpTrace g d) : critPath g ≤ tr.T :=
  (schedule_bounds tr.within tr.causal tr.duration).2

theorem operational_roofline_makespan_bound (tr : OpTrace g d) :
    totalWork g ≤ d.peakFlops * tr.T ∧ totalTraffic g ≤ d.bandwidth * tr.T ∧
    totalTC g ≤ d.peakTC * tr.T ∧ totalALU g ≤ d.peakALU * tr.T ∧
    totalSFU g ≤ d.peakSFU * tr.T ∧ critPath g ≤ tr.T := by
  obtain ⟨htc, halu, hsfu⟩ := operational_three_unit_lower_bound tr
  exact ⟨operational_compute_lower_bound tr, operational_memory_lower_bound tr,
    htc, halu, hsfu, operational_critical_path_lower_bound tr⟩

namespace Example

def device : Device where
  peakFlops := 4
  peakTC := 2
  peakALU := 1
  peakSFU := 1
  bandwidth := 8
  peakFlops_pos := by decide
  peakTC_pos := by decide
  peakALU_pos := by decide
  peakSFU_pos := by decide
  bandwidth_pos := by decide

def node (parents : List Nat) : DAGNode where
  work := 4
  tcWork := 2
  aluWork := 1
  sfuWork := 1
  traffic := 8
  stepFloor := 1
  parents := parents

def graph : DAG := .snoc (.snoc .empty (node [])) (node [0])

private def issue (amount t i : Nat) : Nat := if t = i then amount else 0

private theorem issue_cap (amount : Nat) : ∀ t < 2, sumNodes graph (issue amount t) ≤ amount := by
  intro t ht
  have h : t = 0 ∨ t = 1 := by omega
  rcases h with rfl | rfl <;> simp [sumNodes, graph, DAG.size, sumTicks, issue]

def trace : OpTrace graph device where
  T := 2
  workAt := issue 4
  tcAt := issue 2
  aluAt := issue 1
  sfuAt := issue 1
  memAt := issue 8
  start := id
  finish := Nat.succ
  tick_work_cap := issue_cap 4
  tick_tc_cap := issue_cap 2
  tick_alu_cap := issue_cap 1
  tick_sfu_cap := issue_cap 1
  tick_mem_cap := issue_cap 8
  node_complete := by
    simp only [graph, ForNodes]
    decide
  within := by
    simp only [graph, ForNodes]
    decide
  causal := by simp [graph, ForNodes, node, DAG.size]
  duration := by
    simp only [graph, ForNodes]
    decide

example : critPath graph = trace.T := rfl

end Example

#print axioms operational_roofline_makespan_bound

end PerfRuler.Operational