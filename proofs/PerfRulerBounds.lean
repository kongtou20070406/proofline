import Std

namespace PerfRuler

structure Device where
  peakFlops : Nat
  bandwidth : Nat
  peak_pos : 0 < peakFlops
  bandwidth_pos : 0 < bandwidth

structure Op where
  work : Nat
  traffic : Nat
  stepFloor : Nat
  deriving DecidableEq

structure DAGNode extends Op where
  parents : List Nat
  deriving DecidableEq

/-- Appending a vertex assigns it the next zero-based index. -/
inductive DAG where
  | empty
  | snoc (prior : DAG) (node : DAGNode)

def DAG.size : DAG → Nat
  | .empty => 0
  | .snoc g _ => g.size + 1

def totalWork : DAG → Nat
  | .empty => 0
  | .snoc g v => totalWork g + v.work

def totalTraffic : DAG → Nat
  | .empty => 0
  | .snoc g v => totalTraffic g + v.traffic

def ForNodes (P : Nat → DAGNode → Prop) : DAG → Prop
  | .empty => True
  | .snoc g v => ForNodes P g ∧ P g.size v

private def decForNodes (P : Nat → DAGNode → Prop) [∀ i v, Decidable (P i v)] :
    (g : DAG) → Decidable (ForNodes P g)
  | .empty => isTrue trivial
  | .snoc g v => @instDecidableAnd _ _ (decForNodes P g) (inferInstanceAs (Decidable (P g.size v)))

instance (P : Nat → DAGNode → Prop) [∀ i v, Decidable (P i v)] (g : DAG) :
    Decidable (ForNodes P g) := decForNodes P g

abbrev Topological (g : DAG) := ForNodes (fun i v => ∀ p ∈ v.parents, p < i) g

def maxOn (time : Nat → Nat) : List Nat → Nat
  | [] => 0
  | p :: ps => max (time p) (maxOn time ps)

private theorem maxOn_le (time : Nat → Nat) (ps : List Nat) (t : Nat)
    (h : ∀ p ∈ ps, time p ≤ t) : maxOn time ps ≤ t := by
  induction ps with
  | nil => exact Nat.zero_le _
  | cons p ps ih =>
    exact Nat.max_le.mpr ⟨h p (by simp), ih (fun u hu => h u (by simp [hu]))⟩

example : maxOn (fun i => i + 1) [1, 4, 2] ≤ 5 :=
  maxOn_le _ _ _ (by decide)

structure Clock where
  ends : Nat → Nat
  span : Nat

/-- Longest path ending at each vertex, and maximum over all endpoints. -/
def clock : DAG → Clock
  | .empty => ⟨fun _ => 0, 0⟩
  | .snoc g v =>
    let c := clock g
    let e := maxOn c.ends v.parents + v.stepFloor
    ⟨fun i => if i = g.size then e else c.ends i, max c.span e⟩

def critPath (g : DAG) := (clock g).span

structure Schedule (g : DAG) (d : Device) where
  T : Nat
  start : Nat → Nat
  finish : Nat → Nat
  within : ForNodes (fun i _ => finish i ≤ T) g
  causal : ForNodes (fun i v => ∀ p ∈ v.parents, finish p ≤ start i) g
  duration : ForNodes (fun i v => start i + v.stepFloor ≤ finish i) g
  compute : totalWork g ≤ d.peakFlops * T
  memory : totalTraffic g ≤ d.bandwidth * T

def toyDevice : Device := ⟨2, 4, by decide, by decide⟩

def task (cost : Nat) (ps : List Nat) : DAGNode := ⟨⟨cost, 2 * cost, cost⟩, ps⟩

def forkJoin := (((DAG.empty.snoc (task 2 [])).snoc (task 5 [0])).snoc
  (task 3 [0])).snoc (task 2 [1, 2])
def linearFour := (((DAG.empty.snoc (task 2 [])).snoc (task 5 [0])).snoc
  (task 3 [1])).snoc (task 2 [2])

def forkSchedule : Schedule forkJoin toyDevice where
  T := 9
  start i := [0, 2, 2, 7].getD i 0
  finish i := [2, 7, 5, 9].getD i 0
  within := by decide
  causal := by decide
  duration := by decide
  compute := by decide
  memory := by decide

def linearSchedule : Schedule linearFour toyDevice where
  T := 12
  start i := [0, 2, 7, 10].getD i 0
  finish i := [2, 7, 10, 12].getD i 0
  within := by decide
  causal := by decide
  duration := by decide
  compute := by decide
  memory := by decide

theorem universal_dag_makespan_lower_bound {g : DAG} {d : Device}
    (order : Topological g) (s : Schedule g d) :
    totalWork g ≤ d.peakFlops * s.T ∧ totalTraffic g ≤ d.bandwidth * s.T ∧
      critPath g ≤ s.T := by
  have sound (h : DAG) (topo : Topological h)
      (window : ForNodes (fun i _ => s.finish i ≤ s.T) h)
      (edges : ForNodes (fun i v => ∀ p ∈ v.parents, s.finish p ≤ s.start i) h)
      (floor : ForNodes (fun i v => s.start i + v.stepFloor ≤ s.finish i) h) :
      (∀ i < h.size, (clock h).ends i ≤ s.finish i) ∧ (clock h).span ≤ s.T := by
    induction h with
    | empty => exact ⟨by simp [DAG.size], Nat.zero_le _⟩
    | snoc g v ih =>
      obtain ⟨ends, span⟩ := ih topo.1 window.1 edges.1 floor.1
      have ready := maxOn_le (clock g).ends v.parents (s.start g.size)
        (fun p hp => Nat.le_trans (ends p (topo.2 p hp)) (edges.2 p hp))
      have done := Nat.le_trans (Nat.add_le_add_right ready v.stepFloor) floor.2
      refine ⟨?_, Nat.max_le.mpr ⟨span, Nat.le_trans done window.2⟩⟩
      intro i hi
      by_cases e : i = g.size
      · subst i
        simpa [clock] using done
      · simp only [DAG.size] at hi
        simpa [clock, e] using ends i (by omega)
  exact ⟨s.compute, s.memory, (sound g order s.within s.causal s.duration).2⟩

theorem fork_and_chain_witness :
    totalWork forkJoin = 12 ∧ totalTraffic forkJoin = 24 ∧
    totalWork linearFour = 12 ∧ totalTraffic linearFour = 24 ∧
    critPath forkJoin = 9 ∧ critPath linearFour = 12 ∧
    critPath forkJoin ≤ forkSchedule.T ∧ critPath linearFour ≤ linearSchedule.T := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl,
    (universal_dag_makespan_lower_bound (by decide) forkSchedule).2.2,
    (universal_dag_makespan_lower_bound (by decide) linearSchedule).2.2⟩

#print axioms universal_dag_makespan_lower_bound
#print axioms fork_and_chain_witness

/-- The smallest integer number of ticks that supplies the requested units. -/
def ceilTicks (units rate : Nat) := (units + rate - 1) / rate

private theorem ceilTicks_le {units rate t : Nat} (pos : 0 < rate) :
    ceilTicks units rate ≤ t ↔ units ≤ rate * t := by
  rw [ceilTicks, Nat.div_le_iff_le_mul_add_pred pos]
  omega

example : ceilTicks 13 4 ≤ 4 ∧ ¬ ceilTicks 13 4 ≤ 3 := by
  exact ⟨(ceilTicks_le (by decide)).2 (by decide), by decide⟩

def dagTicks (g : DAG) (d : Device) :=
  max (max (ceilTicks (totalWork g) d.peakFlops) (ceilTicks (totalTraffic g) d.bandwidth)) (critPath g)

theorem dag_time_and_throughput {g : DAG} {d : Device} (order : Topological g) (s : Schedule g d) :
    dagTicks g d ≤ s.T ∧ ∀ num den : Nat, num * s.T ≤ den → num * dagTicks g d ≤ den := by
  obtain ⟨w, q, c⟩ := universal_dag_makespan_lower_bound order s
  have h : dagTicks g d ≤ s.T := Nat.max_le.mpr ⟨Nat.max_le.mpr
    ⟨(ceilTicks_le d.peak_pos).2 w, (ceilTicks_le d.bandwidth_pos).2 q⟩, c⟩
  exact ⟨h, fun num _ hrate => Nat.le_trans (Nat.mul_le_mul_left num h) hrate⟩

example : dagTicks forkJoin toyDevice = 9 ∧ 1 * dagTicks forkJoin toyDevice ≤ 9 ∧
    9 < 1 * dagTicks linearFour toyDevice := by
  exact ⟨by decide, (dag_time_and_throughput (by decide) forkSchedule).2 1 9 (by decide), by decide⟩

#print axioms dag_time_and_throughput

/-- A source-rooted path ending at i, carrying its accumulated node floors. -/
inductive PathTo : DAG → Nat → Nat → Prop where
  | keep {g v i w} : PathTo g i w → PathTo (g.snoc v) i w
  | source {g : DAG} {v : DAGNode} : v.parents = [] → PathTo (g.snoc v) g.size v.stepFloor
  | extend {g v u w} : PathTo g u w → u ∈ v.parents →
      PathTo (g.snoc v) g.size (w + v.stepFloor)

private structure PathFacts (g : DAG) : Prop where
  sound : ∀ {i w}, PathTo g i w → i < g.size ∧ w ≤ (clock g).ends i
  hit : ∀ i < g.size, PathTo g i ((clock g).ends i)
  dominated : ∀ i < g.size, (clock g).ends i ≤ critPath g
  peak : 0 < g.size → ∃ i, PathTo g i (critPath g)

theorem critical_path_characterization (g : DAG) (topo : Topological g) :
    (∀ i w, PathTo g i w → w ≤ critPath g) ∧
    (0 < g.size → ∃ i, PathTo g i (critPath g)) := by
  have argMax (f : Nat → Nat) (ps : List Nat) :
      (∀ u ∈ ps, f u ≤ maxOn f ps) ∧ (ps ≠ [] → ∃ u ∈ ps, f u = maxOn f ps) := by
    induction ps with
    | nil => simp
    | cons a ps ih =>
      constructor
      · intro u hu
        rcases List.mem_cons.mp hu with rfl | h
        · exact Nat.le_max_left ..
        · exact Nat.le_trans (ih.1 u h) (Nat.le_max_right ..)
      · intro _
        by_cases h : maxOn f ps ≤ f a
        · exact ⟨a, by simp, (Nat.max_eq_left h).symm⟩
        · have ne : ps ≠ [] := by intro emp; simp [emp, maxOn] at h
          obtain ⟨u, hu, eq⟩ := ih.2 ne
          exact ⟨u, by simp [hu], eq.trans (Nat.max_eq_right (by omega)).symm⟩
  have facts (h : DAG) (ordered : Topological h) : PathFacts h := by
    induction h with
    | empty =>
      constructor
      · intro i w hp
        cases hp
      · simp [DAG.size]
      · simp [DAG.size]
      · simp [DAG.size]
    | snoc g v ih =>
      have prior := ih ordered.1
      let e := maxOn (clock g).ends v.parents + v.stepFloor
      have last : (clock (g.snoc v)).ends g.size = e := by simp [clock, e]
      have old {i} (hi : i < g.size) : (clock (g.snoc v)).ends i = (clock g).ends i := by
        simp [clock, Nat.ne_of_lt hi]
      have endsAt : PathTo (g.snoc v) g.size e := by
        by_cases emp : v.parents = []
        · simpa [e, emp, maxOn] using (PathTo.source (g := g) emp)
        · obtain ⟨u, hu, huTime⟩ := (argMax (clock g).ends v.parents).2 emp
          simpa only [e, huTime] using (PathTo.extend (prior.hit u (ordered.2 u hu)) hu)
      refine ⟨?_, ?_, ?_, ?_⟩
      · intro i w p
        cases p with
        | keep pp =>
          obtain ⟨hi, hw⟩ := prior.sound pp
          exact ⟨Nat.lt_succ_of_lt hi, (old hi).symm ▸ hw⟩
        | source emp =>
          exact ⟨Nat.lt_succ_self _, by simp [last, e, emp, maxOn]⟩
        | extend pp hu =>
          have weight := Nat.le_trans (prior.sound pp).2 ((argMax _ _).1 _ hu)
          exact ⟨Nat.lt_succ_self _, last.symm ▸ Nat.add_le_add_right weight _⟩
      · intro i hi
        by_cases eq : i = g.size
        · subst i; rw [last]; exact endsAt
        · have lt : i < g.size := by simp only [DAG.size] at hi; omega
          rw [old lt]; exact PathTo.keep (prior.hit i lt)
      · intro i hi
        change _ ≤ max (critPath g) e
        by_cases eq : i = g.size
        · subst i; rw [last]; exact Nat.le_max_right ..
        · have lt : i < g.size := by simp only [DAG.size] at hi; omega
          rw [old lt]; exact Nat.le_trans (prior.dominated i lt) (Nat.le_max_left ..)
      · intro _
        change ∃ i, PathTo _ i (max (critPath g) e)
        by_cases hmax : critPath g ≤ e
        · exact ⟨g.size, (Nat.max_eq_right hmax).symm ▸ endsAt⟩
        · have nonempty : 0 < g.size := by
            cases g with
            | empty => simp [critPath, clock] at hmax
            | snoc p u => exact Nat.zero_lt_succ _
          obtain ⟨i, hi⟩ := prior.peak nonempty
          rw [Nat.max_eq_left (by omega)]; exact ⟨i, PathTo.keep hi⟩
  have result := facts g topo
  exact ⟨fun i _ h => Nat.le_trans (result.sound h).2 (result.dominated i (result.sound h).1),
    result.peak⟩

theorem fork_path_witness : PathTo forkJoin 3 9 ∧ PathTo forkJoin 3 7 ∧
    (∀ i w, PathTo forkJoin i w → w ≤ 9) := by
  have root : PathTo (DAG.empty.snoc (task 2 [])) 0 2 := PathTo.source rfl
  have long : PathTo ((DAG.empty.snoc (task 2 [])).snoc (task 5 [0])) 1 7 :=
    PathTo.extend root (by decide)
  have short : PathTo (((DAG.empty.snoc (task 2 [])).snoc (task 5 [0])).snoc (task 3 [0])) 2 5 :=
    PathTo.extend (PathTo.keep root) (by decide)
  exact ⟨PathTo.extend (PathTo.keep long) (by decide), PathTo.extend short (by decide),
    (critical_path_characterization forkJoin (by decide)).1⟩

#print axioms critical_path_characterization
#print axioms fork_path_witness

structure Layer where
  fwd : Op
  dX : Op
  dW : Op

structure Port where
  graph : DAG
  parents : List Nat

def Port.ready (s : Port) := maxOn (clock s.graph).ends s.parents
abbrev Port.Valid (s : Port) := ∀ p ∈ s.parents, p < s.graph.size
def origin : Port := ⟨.empty, []⟩

def Port.push (s : Port) (op : Op) : Port :=
  ⟨s.graph.snoc ⟨op, s.parents⟩, [s.graph.size]⟩

/-- After dW is appended, return the dX index to the shallower layer. -/
def closeLayer (fIndex : Nat) (inner : Port) (a : Layer) : Port :=
  let x := inner.push a.dX
  ⟨x.graph.snoc ⟨a.dW, [fIndex, inner.graph.size]⟩, [inner.graph.size]⟩

def buildAD : List Layer → Port → Port
  | [], s => s
  | a :: rest, s => closeLayer s.graph.size (buildAD rest (s.push a.fwd)) a

def adDAG (ls : List Layer) := (buildAD ls origin).graph
def adWork (ls : List Layer) :=
  (ls.map (fun a => a.fwd.work + a.dX.work + a.dW.work)).sum
def naiveSerial (ls : List Layer) :=
  (ls.map (fun a => a.fwd.stepFloor + a.dX.stepFloor + a.dW.stepFloor)).sum

def spine : List Layer → Nat
  | [] => 0
  | a :: rest => a.fwd.stepFloor + spine rest + a.dX.stepFloor

def exactAD : List Layer → Nat
  | [] => 0
  | a :: rest =>
    max (a.fwd.stepFloor + exactAD rest) (spine (a :: rest) + a.dW.stepFloor)

structure ADAssembly (s out : Port) (ls : List Layer) : Prop where
  grows : s.graph.size ≤ out.graph.size
  old : ∀ i < s.graph.size, (clock out.graph).ends i = (clock s.graph).ends i
  order : Topological out.graph
  valid : out.Valid
  work : totalWork out.graph = totalWork s.graph + adWork ls
  tip : out.ready = s.ready + spine ls
  span : critPath out.graph = max (critPath s.graph) (s.ready + exactAD ls)

def layer (f x w : Nat) : Layer := ⟨⟨4, 1, f⟩, ⟨6, 1, x⟩, ⟨4, 1, w⟩⟩
def fourLayers := [layer 1 2 1, layer 2 1 9, layer 1 3 2, layer 2 2 1]

private theorem assembleAD (ls : List Layer) (s : Port) (topo : Topological s.graph)
    (ports : s.Valid) (bounded : s.ready ≤ critPath s.graph) :
    ADAssembly s (buildAD ls s) ls := by
  induction ls generalizing s with
  | nil =>
    exact ⟨Nat.le_refl _, fun _ _ => rfl, topo, ports, by simp [adWork, buildAD],
      by simp [buildAD, spine], by simp [buildAD, exactAD, Nat.max_eq_left bounded]⟩
  | cons a rest ih =>
    let f := s.push a.fwd
    let m := buildAD rest f
    let x := m.push a.dX
    let out := closeLayer s.graph.size m a
    have fs : f.graph.size = s.graph.size + 1 := rfl
    have ff : (clock f.graph).ends s.graph.size = s.ready + a.fwd.stepFloor := by
      simp [f, Port.push, clock, Port.ready]
    have fr : f.ready = s.ready + a.fwd.stepFloor := by
      simpa [Port.ready, f, Port.push, maxOn] using ff
    have fc : critPath f.graph = max (critPath s.graph) (s.ready + a.fwd.stepFloor) := rfl
    have sub : ADAssembly f m rest := ih f ⟨topo, ports⟩
      (by simp [f, Port.push, DAG.size, Port.Valid])
      (by rw [fr, fc]; exact Nat.le_max_right ..)
    have fm : s.graph.size < m.graph.size := by have := sub.grows; omega
    have xs : x.graph.size = m.graph.size + 1 := rfl
    have xe : (clock x.graph).ends m.graph.size = m.ready + a.dX.stepFloor := by
      simp [x, Port.push, clock, Port.ready]
    have xold (i) (hi : i < m.graph.size) : (clock x.graph).ends i = (clock m.graph).ends i := by
      simp [x, Port.push, clock, Nat.ne_of_lt hi]
    have xeF : (clock x.graph).ends s.graph.size = s.ready + a.fwd.stepFloor := by
      rw [xold _ fm, sub.old _ (by omega), ff]
    have xr : m.ready = s.ready + a.fwd.stepFloor + spine rest := by rw [sub.tip, fr]
    have sideReady : maxOn (clock x.graph).ends [s.graph.size, m.graph.size] =
        m.ready + a.dX.stepFloor := by
      simp only [maxOn, Nat.max_zero, xeF, xe]
      omega
    change ADAssembly s out (a :: rest)
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · change s.graph.size ≤ x.graph.size + 1
      omega
    · intro i hi
      have him : i < m.graph.size := by omega
      have hix : i ≠ x.graph.size := by omega
      change (if i = x.graph.size then _ else (clock x.graph).ends i) = _
      simp only [hix, ↓reduceIte, xold _ him, sub.old _ (show i < f.graph.size by omega)]
      simp [f, Port.push, clock, Nat.ne_of_lt hi]
    · refine ⟨⟨sub.order, sub.valid⟩, ?_⟩
      change ∀ p ∈ [s.graph.size, m.graph.size], p < x.graph.size
      simp [xs]; omega
    · change ∀ p ∈ [m.graph.size], p < x.graph.size + 1
      simp [xs]; omega
    · change totalWork m.graph + a.dX.work + a.dW.work = _
      rw [sub.work]
      change totalWork s.graph + a.fwd.work + adWork rest + a.dX.work + a.dW.work = _
      simp only [adWork, List.map_cons, List.sum_cons]
      omega
    · have hn : m.graph.size ≠ x.graph.size := by omega
      change max (if m.graph.size = x.graph.size then _ else (clock x.graph).ends m.graph.size) 0 = _
      simp only [hn, ↓reduceIte, Nat.max_zero, xe, xr, spine]
      omega
    · change max (max (critPath m.graph) (m.ready + a.dX.stepFloor))
          (maxOn (clock x.graph).ends [s.graph.size, m.graph.size] + a.dW.stepFloor) = _
      rw [sideReady, sub.span, fr, fc, xr, exactAD, spine]
      omega

example : ADAssembly origin (buildAD fourLayers origin) fourLayers :=
  assembleAD _ _ trivial (by decide) (by decide)

def adDevice : Device := ⟨4, 2, by decide, by decide⟩
def fourLayerSchedule : Schedule (adDAG fourLayers) adDevice where
  T := 21
  start i := [0, 1, 3, 4, 6, 8, 8, 11, 11, 12, 12, 14].getD i 0
  finish i := [1, 3, 4, 6, 8, 9, 11, 13, 12, 21, 14, 15].getD i 0
  within := by decide
  causal := by decide
  duration := by decide
  compute := by decide
  memory := by decide

theorem autodiff_dag_critical_path_bound (ls : List Layer) {d : Device}
    (s : Schedule (adDAG ls) d) :
    totalWork (adDAG ls) = adWork ls ∧ critPath (adDAG ls) = exactAD ls ∧
      exactAD ls ≤ naiveSerial ls ∧ exactAD ls ≤ s.T := by
  have assembly := assembleAD ls origin trivial (by decide) (by decide)
  have eqWork : totalWork (adDAG ls) = adWork ls := by
    simpa [adDAG, origin, totalWork] using assembly.work
  have eqSpan : critPath (adDAG ls) = exactAD ls := by
    simpa [adDAG, origin, Port.ready, maxOn, critPath, clock] using assembly.span
  have serial (layers : List Layer) :
      spine layers ≤ naiveSerial layers ∧ exactAD layers ≤ naiveSerial layers := by
    induction layers with
    | nil => decide
    | cons a rest ih =>
      simp only [naiveSerial, List.map_cons, List.sum_cons, exactAD, spine] at *
      omega
  exact ⟨eqWork, eqSpan, (serial ls).2, eqSpan ▸
    (universal_dag_makespan_lower_bound assembly.order s).2.2⟩

theorem ad_four_layer_witness :
    totalWork (adDAG fourLayers) = 56 ∧
    2 * (fourLayers.map (fun a => a.dX.work + a.dW.work)).sum =
      5 * (fourLayers.map (fun a => a.fwd.work)).sum ∧
    spine fourLayers = 14 ∧ exactAD fourLayers = 21 ∧ naiveSerial fourLayers = 27 ∧
    exactAD fourLayers < naiveSerial fourLayers ∧ critPath (adDAG fourLayers) = fourLayerSchedule.T := by
  have h := autodiff_dag_critical_path_bound fourLayers fourLayerSchedule
  exact ⟨h.1.trans (by decide), by decide, by decide, by decide, by decide, by decide,
    h.2.1.trans (by decide)⟩

#print axioms autodiff_dag_critical_path_bound
#print axioms ad_four_layer_witness

/-- A parameter-free attention primitive can put its whole adjoint in dX. -/
def attentionLayer : Layer := ⟨⟨4, 2, 1⟩, ⟨10, 2, 2⟩, ⟨0, 0, 0⟩⟩
def attentionDevice : Device := ⟨6, 2, by decide, by decide⟩
def attentionSchedule : Schedule (adDAG [attentionLayer]) attentionDevice where
  T := 3
  start i := [0, 1, 3].getD i 0
  finish i := [1, 3, 3].getD i 0
  within := by decide
  causal := by decide
  duration := by decide
  compute := by decide
  memory := by decide

theorem attention_ratio_witness :
    2 * attentionLayer.dX.work = 5 * attentionLayer.fwd.work ∧ attentionLayer.dW.work = 0 ∧
    totalWork (adDAG [attentionLayer]) = 14 ∧ critPath (adDAG [attentionLayer]) = 3 := by
  have h := autodiff_dag_critical_path_bound [attentionLayer] attentionSchedule
  exact ⟨rfl, rfl, h.1.trans (by decide), h.2.1.trans (by decide)⟩

#print axioms attention_ratio_witness

inductive Recovery where
  | store | remat
  deriving DecidableEq

structure Activation where
  actBytes : Nat
  rematWork : Nat
  deriving DecidableEq

structure Decision where
  activation : Activation
  method : Recovery
  deriving DecidableEq

def Decision.bytes (e : Decision) := match e.method with
  | .store => 2 * e.activation.actBytes
  | .remat => 0
def Decision.work (e : Decision) := match e.method with
  | .store => 0
  | .remat => e.activation.rematWork

/-- All activations share the time window; the two resources may overlap. -/
abbrev RecoveryFits (d : Device) (t : Nat) (plan : List Decision) :=
  (plan.map Decision.bytes).sum ≤ d.bandwidth * t ∧
  (plan.map Decision.work).sum ≤ d.peakFlops * t

def checkpointCost (d : Device) (a : Activation) :=
  min (d.peakFlops * (2 * a.actBytes)) (d.bandwidth * a.rematWork)

def bandwidthLimited : Device := ⟨8, 2, by decide, by decide⟩
def computeLimited : Device := ⟨2, 8, by decide, by decide⟩
def balanced : Device := ⟨4, 4, by decide, by decide⟩
def saved : Decision := ⟨⟨12, 24⟩, .store⟩
def rebuilt : Decision := ⟨saved.activation, .remat⟩
def mixedPlan := [rebuilt, ⟨⟨3, 48⟩, .store⟩]

theorem checkpoint_vs_store_lower_bound {d : Device} {t : Nat} {plan : List Decision}
    (fits : RecoveryFits d t plan) {e : Decision} (member : e ∈ plan) :
    checkpointCost d e.activation ≤ d.peakFlops * d.bandwidth * t := by
  have itemLe (f : Decision → Nat) (xs : List Decision) (h : e ∈ xs) :
      f e ≤ (xs.map f).sum := by
    induction xs with
    | nil => simp at h
    | cons a xs ih =>
      rcases List.mem_cons.mp h with rfl | ht
      · simp
      · have := ih ht
        simp only [List.map_cons, List.sum_cons]; omega
  have bytes := Nat.le_trans (itemLe Decision.bytes plan member) fits.1
  have work := Nat.le_trans (itemLe Decision.work plan member) fits.2
  cases h : e.method with
  | store =>
    simp only [Decision.bytes, h] at bytes
    apply Nat.le_trans (Nat.min_le_left ..)
    simpa [Nat.mul_assoc] using Nat.mul_le_mul_left d.peakFlops bytes
  | remat =>
    simp only [Decision.work, h] at work
    apply Nat.le_trans (Nat.min_le_right ..)
    simpa [Nat.mul_left_comm, Nat.mul_assoc] using Nat.mul_le_mul_left d.bandwidth work

theorem checkpoint_switch_witness :
    RecoveryFits bandwidthLimited 3 mixedPlan ∧
    checkpointCost bandwidthLimited rebuilt.activation = 8 * 2 * 3 ∧
    RecoveryFits bandwidthLimited 12 [saved] ∧ ¬ RecoveryFits bandwidthLimited 11 [saved] ∧
    RecoveryFits computeLimited 3 [saved] ∧ ¬ RecoveryFits computeLimited 2 [saved] ∧
    checkpointCost computeLimited saved.activation = 2 * 8 * 3 ∧
    RecoveryFits computeLimited 12 [rebuilt] ∧ ¬ RecoveryFits computeLimited 11 [rebuilt] ∧
    RecoveryFits balanced 6 [saved] ∧ RecoveryFits balanced 6 [rebuilt] ∧
    checkpointCost balanced saved.activation = 4 * 4 * 6 ∧
    checkpointCost bandwidthLimited rebuilt.activation ≤ 8 * 2 * 3 := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  exact checkpoint_vs_store_lower_bound (d := bandwidthLimited) (t := 3) (plan := mixedPlan)
    (by decide) (e := rebuilt) (by decide)

/-- Splitting the same activation into independent halves changes the bound. -/
theorem split_activation_witness :
    RecoveryFits toyDevice 1 [⟨⟨2, 2⟩, .store⟩, ⟨⟨2, 2⟩, .remat⟩] ∧
    toyDevice.peakFlops * toyDevice.bandwidth * 1 < checkpointCost toyDevice ⟨4, 4⟩ := by decide

#print axioms checkpoint_vs_store_lower_bound
#print axioms checkpoint_switch_witness
#print axioms split_activation_witness

/-- A credit counts a disjoint unit of actual slow traffic. Multiple credits
    may be grouped into a single phase, and its work quota may be partly used. -/
structure TileRule where
  fastCap : Nat
  tileBytes : Nat
  opsPerTile : Nat
  cap_pos : 0 < fastCap
  tile_pos : 0 < tileBytes
  ops_pos : 0 < opsPerTile

structure IOPhase where
  macs : Nat
  slowBytes : Nat
  peakResident : Nat
  credits : Nat
  deriving DecidableEq

abbrev FitsTile (r : TileRule) (p : IOPhase) :=
  p.peakResident ≤ r.fastCap ∧ p.macs ≤ p.credits * r.opsPerTile ∧
    p.credits * r.tileBytes ≤ p.slowBytes
def ioWork (phases : List IOPhase) := (phases.map IOPhase.macs).sum
def qSlow (phases : List IOPhase) := (phases.map IOPhase.slowBytes).sum

def smallTiles : List IOPhase := [⟨16, 8, 24, 1⟩, ⟨27, 16, 32, 2⟩]
def smallRule : TileRule := ⟨32, 8, 16, by decide, by decide, by decide⟩

theorem hong_kung_io_lower_bound (r : TileRule) (ps : List IOPhase)
    (covered : ∀ p ∈ ps, FitsTile r p) (d : Device) (t : Nat)
    (band : qSlow ps ≤ d.bandwidth * t) :
    ioWork ps * r.tileBytes ≤ qSlow ps * r.opsPerTile ∧
    ioWork ps * r.tileBytes ≤ d.bandwidth * t * r.opsPerTile := by
  have transfer (xs : List IOPhase) (ok : ∀ p ∈ xs, FitsTile r p) :
      ioWork xs * r.tileBytes ≤ qSlow xs * r.opsPerTile := by
    induction xs with
    | nil => simp [ioWork, qSlow]
    | cons p xs ih =>
      obtain ⟨_, work, paid⟩ := ok p (by simp)
      have localBound : p.macs * r.tileBytes ≤ p.slowBytes * r.opsPerTile := calc
        _ ≤ p.credits * r.opsPerTile * r.tileBytes := Nat.mul_le_mul_right _ work
        _ = p.credits * r.tileBytes * r.opsPerTile := by ac_rfl
        _ ≤ _ := Nat.mul_le_mul_right _ paid
      have rest := ih (fun e he => ok e (by simp [he]))
      simpa only [ioWork, qSlow, List.map_cons, List.sum_cons, Nat.add_mul] using
        Nat.add_le_add localBound rest
  have q := transfer ps covered
  exact ⟨q, Nat.le_trans q (Nat.mul_le_mul_right _ band)⟩

theorem tile_transfer_witness : ioWork smallTiles = 43 ∧ qSlow smallTiles = 24 ∧
    43 * 8 ≤ 24 * 16 ∧ 43 * 8 ≤ toyDevice.bandwidth * 6 * 16 := by
  exact ⟨rfl, rfl, hong_kung_io_lower_bound smallRule smallTiles (by decide) toyDevice 6 (by decide)⟩

#print axioms hong_kung_io_lower_bound
#print axioms tile_transfer_witness

/-- Integer AM-GM for the three matrix projections. -/
private theorem three_projection_volume (a b c : Nat) : 27 * (a*b*c) ≤ (a+b+c)^3 := by
  have sorted (x y z : Nat) (hxy : x ≤ y) (hyz : y ≤ z) :
      27 * (x*y*z) ≤ (x+y+z)^3 := by
    obtain ⟨u, rfl⟩ := Nat.exists_eq_add_of_le hxy
    obtain ⟨v, rfl⟩ := Nat.exists_eq_add_of_le hyz
    let slack := 9*x*(u^2 + u*v + v^2) + (2*u+v)^3
    calc
      _ ≤ 27 * (x*(x+u)*(x+u+v)) + slack := Nat.le_add_right .. 
      _ = _ := by dsimp [slack]; grind
  rcases Nat.le_total a b with h | h <;> rcases Nat.le_total b c with k | k <;>
    rcases Nat.le_total a c with j | j
  all_goals first
    | exact sorted a b c (by omega) (by omega)
    | simpa only [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc,
        Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using sorted a c b (by omega) (by omega)
    | simpa only [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc,
        Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using sorted b a c (by omega) (by omega)
    | simpa only [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc,
        Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using sorted b c a (by omega) (by omega)
    | simpa only [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc,
        Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using sorted c a b (by omega) (by omega)
    | simpa only [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc,
        Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using sorted c b a (by omega) (by omega)

example : 27 * (3*4*5) ≤ (3+4+5)^3 := three_projection_volume 3 4 5

def matRule : TileRule := ⟨128*1024, 524288, 256^3, by decide, by decide, by decide⟩

/-- Counts distinct A(i,k), B(k,j), C(i,j) words touched by one work phase. -/
structure MatrixPhase where
  aWords : Nat
  bWords : Nat
  cWords : Nat
  atEntry : Nat
  atExit : Nat
  peakBytes : Nat
  slowBytes : Nat
  deriving DecidableEq

def MatrixPhase.io (p : MatrixPhase) : IOPhase := ⟨matRule.opsPerTile, p.slowBytes, p.peakBytes, 1⟩
abbrev MatrixPhase.Valid (p : MatrixPhase) :=
  p.atEntry ≤ p.peakBytes ∧ p.atExit ≤ p.peakBytes ∧ p.peakBytes ≤ matRule.fastCap ∧
  4 * (p.aWords + p.bWords + p.cWords) ≤ p.atEntry + p.slowBytes + p.atExit ∧
  matRule.opsPerTile^2 ≤ p.aWords * p.bWords * p.cWords

def slab64 (first : Bool) : MatrixPhase :=
  ⟨512*64, 64*512, 512*512, 0, 0, 3*64^2*4, (if first then 3 else 4)*64^2*4*64⟩
def matrixTrace := slab64 true :: List.replicate 7 (slab64 false)

private theorem matrix_phase_credit (p : MatrixPhase) (ok : p.Valid) : FitsTile matRule p.io := by
  obtain ⟨entry, leave, cap, account, lw⟩ := ok
  have volume := Nat.le_trans (Nat.mul_le_mul_left 27 lw) (three_projection_volume ..)
  have footprint : 196608 ≤ p.aWords + p.bWords + p.cWords := by
    apply Nat.le_of_not_gt
    intro small
    have upper := Nat.pow_le_pow_left (show p.aWords + p.bWords + p.cWords ≤ 196607 by omega) 3
    have gap : 196607^3 < 27 * matRule.opsPerTile^2 := by decide
    omega
  change p.peakBytes ≤ 131072 ∧ 256^3 ≤ 1 * 256^3 ∧ 1 * 524288 ≤ p.slowBytes
  change p.peakBytes ≤ 131072 at cap
  exact ⟨cap, by decide, by omega⟩

example : FitsTile matRule (slab64 true).io := matrix_phase_credit _ (by decide)

def compulsory512 := 3 * 512^2 * 4

theorem matrix512_sram_bound (phases : List MatrixPhase) (count : phases.length = 8)
    (valid : ∀ p ∈ phases, p.Valid) (d : Device) (t : Nat)
    (band : qSlow (phases.map MatrixPhase.io) ≤ d.bandwidth * t) :
    ioWork (phases.map MatrixPhase.io) = 512^3 ∧
    4194304 ≤ qSlow (phases.map MatrixPhase.io) ∧ 4194304 ≤ d.bandwidth * t ∧
    compulsory512 < 4194304 := by
  have works (xs : List MatrixPhase) : ioWork (xs.map MatrixPhase.io) = xs.length * matRule.opsPerTile := by
    induction xs with
    | nil => rfl
    | cons p xs ih =>
      change matRule.opsPerTile + ioWork (xs.map MatrixPhase.io) = (xs.length + 1) * matRule.opsPerTile
      rw [ih, Nat.add_mul, Nat.one_mul, Nat.add_comm]
  have w := works phases
  rw [count] at w
  have bound := (hong_kung_io_lower_bound matRule (phases.map MatrixPhase.io)
    (fun _ h => by obtain ⟨p, hp, rfl⟩ := List.mem_map.mp h; exact matrix_phase_credit p (valid p hp))
    d t band).1
  rw [w] at bound
  change 4194304 * 16777216 ≤ qSlow _ * 16777216 at bound
  have bytes := Nat.le_of_mul_le_mul_right bound (by decide)
  exact ⟨w, bytes, Nat.le_trans bytes band, by decide⟩

theorem matrix512_tiled_witness :
    ioWork (matrixTrace.map MatrixPhase.io) = 512^3 ∧
    qSlow (matrixTrace.map MatrixPhase.io) = 32505856 ∧
    4194304 ≤ qSlow (matrixTrace.map MatrixPhase.io) ∧ compulsory512 < 4194304 := by
  have h := matrix512_sram_bound matrixTrace (by decide) (by decide) toyDevice 8126464 (by decide)
  exact ⟨h.1, by decide, h.2.1, h.2.2.2⟩

#print axioms matrix512_sram_bound
#print axioms matrix512_tiled_witness

/-- FLOP/ns, bytes/ns. The graph launch floor is a separate execution assumption. -/
def rtx5090 : Device := ⟨104800, 1792, by decide, by decide⟩
def h100 : Device := ⟨67000, 3350, by decide, by decide⟩

theorem matrix512_rtx5090_bandwidth (phases : List MatrixPhase) (count : phases.length = 8)
    (valid : ∀ p ∈ phases, p.Valid) (t : Nat)
    (band : qSlow (phases.map MatrixPhase.io) ≤ rtx5090.bandwidth * t) :
    ceilTicks compulsory512 rtx5090.bandwidth = 1756 ∧ 2341 ≤ t := by
  have h := (matrix512_sram_bound phases count valid rtx5090 t band).2.2.1
  exact ⟨by decide, (ceilTicks_le rtx5090.bandwidth_pos (units := 4194304)).2 h⟩

example : ceilTicks compulsory512 rtx5090.bandwidth = 1756 ∧ 2341 ≤ 18140 :=
  matrix512_rtx5090_bandwidth matrixTrace (by decide) (by decide) 18140 (by decide)

#print axioms matrix512_rtx5090_bandwidth

structure Demand where
  flops : Nat
  bytes : Nat
  steps : Nat
  deriving DecidableEq

structure Floors where
  computeNs : Nat
  memoryNs : Nat
  serialNs : Nat
  deriving DecidableEq

def Demand.floors (p : Demand) (d : Device) (tau : Nat) : Floors :=
  ⟨ceilTicks p.flops d.peakFlops, ceilTicks p.bytes d.bandwidth, p.steps * tau⟩
def Floors.ns (r : Floors) := max (max r.computeNs r.memoryNs) r.serialNs
abbrev Demand.Covered (p : Demand) (tau : Nat) (g : DAG) :=
  p.flops ≤ totalWork g ∧ p.bytes ≤ totalTraffic g ∧ p.steps * tau ≤ critPath g

/-- A representative serial graph with n+1 physical stages. Aggregate resource
    counts are booked at the first stage. Slot lengths below cover that stage. -/
def stageChain (w q tau : Nat) : Nat → DAG
  | 0 => DAG.empty.snoc ⟨⟨w, q, tau⟩, []⟩
  | n+1 => (stageChain w q tau n).snoc ⟨⟨0, 0, tau⟩, [n]⟩

private theorem chain_facts (w q tau slot n : Nat) (long : tau ≤ slot) :
    let g := stageChain w q tau n
    g.size = n+1 ∧ Topological g ∧ totalWork g = w ∧ totalTraffic g = q ∧
    (clock g).ends n = (n+1)*tau ∧ critPath g = (n+1)*tau ∧
    ForNodes (fun i v => ∀ u ∈ v.parents, (u+1)*slot ≤ i*slot) g ∧
    ForNodes (fun i v => i*slot + v.stepFloor ≤ (i+1)*slot) g := by
  induction n with
  | zero =>
    simp [stageChain, Topological, DAG.size, ForNodes, totalWork, totalTraffic, clock,
      maxOn, critPath, long]
  | succ n ih =>
    let g := stageChain w q tau n
    obtain ⟨size, order, work, traffic, tip, span, edges, duration⟩ := ih
    have endTime : maxOn (clock g).ends [n] + tau = (n+1+1)*tau := by
      simp only [maxOn, Nat.max_zero, g, tip, Nat.add_mul, Nat.one_mul]
    refine ⟨by change g.size+1 = _; rw [size], ⟨order, ?_⟩, ?_, ?_, ?_, ?_, ⟨edges, ?_⟩, ⟨duration, ?_⟩⟩
    · simp [size]
    · change totalWork g + 0 = w
      simpa using work
    · change totalTraffic g + 0 = q
      simpa using traffic
    · change (if n+1 = g.size then _ else (clock g).ends _) = _
      split
      · exact endTime
      · rename_i ne
        exact (ne size.symm).elim
    · change max (critPath g) (maxOn (clock g).ends [n] + tau) = _
      rw [endTime, show critPath g = _ from span, Nat.max_eq_right]
      exact Nat.mul_le_mul_right _ (by omega)
    · simp [size]
    · change g.size * slot + tau ≤ (g.size+1)*slot
      rw [Nat.add_mul, Nat.one_mul]; omega

example : critPath (stageChain 40 20 3 2) = 9 :=
  (chain_facts 40 20 3 10 2 (by decide)).2.2.2.2.2.1

private theorem for_indices (g : DAG) (P : Nat → Prop) (all : ∀ i < g.size, P i) :
    ForNodes (fun i _ => P i) g := by
  induction g with
  | empty => trivial
  | snoc g v ih =>
    simp only [DAG.size] at all
    exact ⟨ih (fun i hi => all i (by omega)), all g.size (by omega)⟩

example : ForNodes (fun i _ => i < 4) forkJoin := for_indices _ _ (fun _ hi => hi)

structure Certified (p : Demand) (d : Device) (tau : Nat) (r : Floors) : Prop where
  values : p.floors d tau = r
  lower : ∀ {g : DAG}, Topological g → (s : Schedule g d) → p.Covered tau g → r.ns ≤ s.T
  onChain : ∀ s : Schedule (stageChain p.flops p.bytes tau (p.steps-1)) d,
    r.computeNs ≤ s.T ∧ r.memoryNs ≤ s.T ∧ r.serialNs ≤ s.T ∧ r.ns ≤ s.T
  witness : ∃ s : Schedule (stageChain p.flops p.bytes tau (p.steps-1)) d,
    Topological (stageChain p.flops p.bytes tau (p.steps-1)) ∧
    p.Covered tau (stageChain p.flops p.bytes tau (p.steps-1)) ∧ r.ns ≤ s.T

private theorem certify (p : Demand) (d : Device) (tau : Nat) (pos : 0 < p.steps) :
    Certified p d tau (p.floors d tau) := by
  have lower {g : DAG} (order : Topological g) (s : Schedule g d) (covers : p.Covered tau g) :
      (p.floors d tau).ns ≤ s.T := by
    obtain ⟨w, q, path⟩ := universal_dag_makespan_lower_bound order s
    exact Nat.max_le.mpr ⟨Nat.max_le.mpr ⟨(ceilTicks_le d.peak_pos).2 (Nat.le_trans covers.1 w),
      (ceilTicks_le d.bandwidth_pos).2 (Nat.le_trans covers.2.1 q)⟩, Nat.le_trans covers.2.2 path⟩
  let slot := max tau (max (ceilTicks p.flops d.peakFlops) (ceilTicks p.bytes d.bandwidth))
  let n := p.steps - 1
  let g := stageChain p.flops p.bytes tau n
  have steps : n+1 = p.steps := by dsimp [n]; omega
  obtain ⟨size, order, work, traffic, _, span, edges, duration⟩ :=
    chain_facts p.flops p.bytes tau slot n (Nat.le_max_left ..)
  have slotLe : slot ≤ (n+1)*slot := by
    simpa only [Nat.one_mul] using Nat.mul_le_mul_right slot (show 1 ≤ n+1 by omega)
  have comp : ceilTicks p.flops d.peakFlops ≤ slot := Nat.le_trans (Nat.le_max_left ..) (Nat.le_max_right ..)
  have mem : ceilTicks p.bytes d.bandwidth ≤ slot := Nat.le_trans (Nat.le_max_right ..) (Nat.le_max_right ..)
  let s : Schedule g d := {
    T := (n+1)*slot, start := fun i => i*slot, finish := fun i => (i+1)*slot
    within := for_indices g _ (fun i hi =>
      Nat.mul_le_mul_right slot (by rw [show g.size = n+1 from size] at hi; omega))
    causal := edges, duration := duration
    compute := work ▸ (ceilTicks_le d.peak_pos).1 (Nat.le_trans comp slotLe)
    memory := traffic ▸ (ceilTicks_le d.bandwidth_pos).1 (Nat.le_trans mem slotLe) }
  have covers : p.Covered tau g := ⟨Nat.le_of_eq work.symm, Nat.le_of_eq traffic.symm,
    Nat.le_of_eq (by rw [span, steps])⟩
  refine ⟨rfl, @lower, ?_, ⟨s, order, covers, lower order s covers⟩⟩
  intro run
  have all := lower order run covers
  obtain ⟨resources, path⟩ := Nat.max_le.mp all
  exact ⟨(Nat.max_le.mp resources).1, (Nat.max_le.mp resources).2, path, all⟩

example : Certified ⟨12, 24, 3⟩ toyDevice 2 ⟨6, 6, 6⟩ :=
  certify _ _ _ (by decide)

/-- Card 011 batch-one Push-T FLOP counter; four rollout predictions are
    included in every row. The traffic quota here is cold FP32 weights only. -/
structure WorldArm where
  params : Nat
  counterFlops : Nat
  rounds : Nat
  coreLayers : Nat

def rcwm (rounds : Nat) : WorldArm :=
  ⟨1038251, 16496640 + rounds * 40279040, rounds, 2⟩
def nonrecParam : WorldArm := ⟨1038251, 56775680, 1, 2⟩
def nonrecFlop : WorldArm := ⟨6937691, 410027520, 1, 10⟩
def WorldArm.demand (a : WorldArm) : Demand :=
  ⟨a.counterFlops, 4 * a.params, a.rounds * (4 + a.coreLayers) + 4 * 4⟩

theorem rcwm_rtx5090_bounds :
    Certified (rcwm 1).demand rtx5090 930 ⟨542, 2318, 20460⟩ ∧
    Certified (rcwm 8).demand rtx5090 930 ⟨3233, 2318, 59520⟩ ∧
    Certified (rcwm 64).demand rtx5090 930 ⟨24756, 2318, 372000⟩ ∧
    Certified nonrecParam.demand rtx5090 930 ⟨542, 2318, 20460⟩ ∧
    Certified nonrecFlop.demand rtx5090 930 ⟨3913, 15486, 27900⟩ := by
  exact ⟨certify _ _ _ (by decide), certify _ _ _ (by decide), certify _ _ _ (by decide),
    certify _ _ _ (by decide), certify _ _ _ (by decide)⟩

theorem rcwm_separation_witness :
    let row := fun (a : WorldArm) => a.demand.floors rtx5090 930
    (rcwm 1).demand = nonrecParam.demand ∧ (rcwm 64).params = nonrecParam.params ∧
    nonrecFlop.params = 6937691 ∧
    (rcwm 1).counterFlops = 56775680 ∧ (rcwm 8).counterFlops = 338728960 ∧
    (rcwm 64).counterFlops = 2594355200 ∧
    (row (rcwm 1)).computeNs < (row (rcwm 8)).computeNs ∧
    (row (rcwm 8)).computeNs < (row nonrecFlop).computeNs ∧
    (row nonrecFlop).computeNs < (row (rcwm 64)).computeNs ∧
    (row nonrecParam).ns < (row nonrecFlop).ns ∧
    (row nonrecFlop).ns < (row (rcwm 8)).ns ∧ (row (rcwm 8)).ns < (row (rcwm 64)).ns := by decide

#print axioms rcwm_rtx5090_bounds
#print axioms rcwm_separation_witness

/-- Scheme A from perf_ruler.py: batch one, FP32, one HBM pass per fine
    update, halo factor one, 8 by 8 coarse grid, separate final readout. -/
def fineElements := 32 * (256 * 256)
def coarseElements := 32 * (8 * 8)
def nsiPasses := 5
def coarseSteps := nsiPasses * ceilTicks (8 - 1) nsiPasses

def nsiDemand : Demand :=
  ⟨40 * fineElements * nsiPasses,
    (2 + 9) * fineElements * 4 * nsiPasses, nsiPasses + 1⟩
def mgNsiDemand : Demand :=
  let extraFine := (4 * nsiPasses + 1 + 8 * nsiPasses) * fineElements
  let extraCoarse := 40 * coarseElements * coarseSteps
  let crossBytes := 2 * fineElements * 4 * nsiPasses + nsiPasses * coarseElements * 4
  let coarseBytes := (2 * coarseSteps + 9) * coarseElements * 4
  ⟨nsiDemand.flops + extraFine + extraCoarse, nsiDemand.bytes + crossBytes + coarseBytes,
    nsiDemand.steps + coarseSteps⟩

theorem nsi_hbm_serial_bounds :
    Certified nsiDemand rtx5090 930 ⟨4003, 257463, 5580⟩ ∧
    Certified mgNsiDemand rtx5090 930 ⟨5231, 304430, 14880⟩ ∧
    Certified nsiDemand h100 1000 ⟨6261, 137724, 6000⟩ ∧
    Certified mgNsiDemand h100 1000 ⟨8182, 162848, 16000⟩ := by
  exact ⟨certify _ _ _ (by decide), certify _ _ _ (by decide),
    certify _ _ _ (by decide), certify _ _ _ (by decide)⟩

theorem nsi_unfused_witness :
    nsiDemand = ⟨419430400, 461373440, 6⟩ ∧ mgNsiDemand = ⟨548175872, 545538048, 16⟩ ∧
    coarseSteps = 10 ∧
    (nsiDemand.floors rtx5090 930).ns = 257463 ∧ (mgNsiDemand.floors rtx5090 930).ns = 304430 ∧
    (nsiDemand.floors h100 1000).ns = 137724 ∧ (mgNsiDemand.floors h100 1000).ns = 162848 ∧
    (nsiDemand.floors h100 1000).memoryNs < (nsiDemand.floors rtx5090 930).memoryNs ∧
    (mgNsiDemand.floors rtx5090 930).serialNs < (mgNsiDemand.floors h100 1000).serialNs := by decide

#print axioms nsi_hbm_serial_bounds
#print axioms nsi_unfused_witness

/-! ## Proofline v4: Three-Unit Execution Roofs & SM Wave Quantization Ceilings -/

structure ThreeUnitDevice where
  tcPeak : Nat
  aluPeak : Nat
  sfuPeak : Nat
  tc_pos : 0 < tcPeak
  alu_pos : 0 < aluPeak
  sfu_pos : 0 < sfuPeak

structure ThreeUnitWork where
  tcWork : Nat
  aluWork : Nat
  sfuWork : Nat

structure ThreeUnitSchedule (w : ThreeUnitWork) (d : ThreeUnitDevice) where
  T : Nat
  tcBound : w.tcWork ≤ d.tcPeak * T
  aluBound : w.aluWork ≤ d.aluPeak * T
  sfuBound : w.sfuWork ≤ d.sfuPeak * T

def threeUnitTicks (w : ThreeUnitWork) (d : ThreeUnitDevice) : Nat :=
  max (ceilTicks w.tcWork d.tcPeak)
      (max (ceilTicks w.aluWork d.aluPeak) (ceilTicks w.sfuWork d.sfuPeak))

theorem three_unit_lower_bound {w : ThreeUnitWork} {d : ThreeUnitDevice}
    (s : ThreeUnitSchedule w d) : threeUnitTicks w d ≤ s.T := by
  exact Nat.max_le.mpr
    ⟨(ceilTicks_le d.tc_pos).2 s.tcBound,
     Nat.max_le.mpr ⟨(ceilTicks_le d.alu_pos).2 s.aluBound, (ceilTicks_le d.sfu_pos).2 s.sfuBound⟩⟩

def waveCount (nCta smCount : Nat) : Nat := ceilTicks nCta smCount

theorem wave_quantization_lower_bound {nCta smCount T : Nat} (hsm : 0 < smCount)
    (hcov : nCta ≤ smCount * T) : waveCount nCta smCount ≤ T := by
  exact (ceilTicks_le hsm).2 hcov

theorem offload_pigeonhole_lower_bound {reqMem capMem hostBw T : Nat} (hbw : 0 < hostBw)
    (hspill : reqMem - capMem ≤ hostBw * T) :
    ceilTicks (reqMem - capMem) hostBw ≤ T := by
  exact (ceilTicks_le hbw).2 hspill

#print axioms three_unit_lower_bound
#print axioms wave_quantization_lower_bound
#print axioms offload_pigeonhole_lower_bound

end PerfRuler
