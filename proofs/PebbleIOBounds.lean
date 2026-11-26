import Std

namespace PerfRuler.PebbleIO

structure CacheState where
  fast : List Nat
  deriving DecidableEq, Repr

inductive Move where
  | load (x : Nat)
  | evict (x : Nat)
  | compute (x : Nat)
  | store (x : Nat)
  deriving DecidableEq, Repr

def step (cap : Nat) (s : CacheState) (m : Move) : Option CacheState :=
  match m with
  | .load x =>
    if x ∈ s.fast then some s
    else if s.fast.length < cap then some ⟨x :: s.fast⟩
    else none
  | .evict x => some ⟨s.fast.filter (· != x)⟩
  | .compute x | .store x => if x ∈ s.fast then some s else none

inductive ValidRun (cap : Nat) : CacheState → List Move → CacheState → Prop where
  | nil {s} : ValidRun cap s [] s
  | cons {s t u m ms} (head : step cap s m = some t) (tail : ValidRun cap t ms u) :
      ValidRun cap s (m :: ms) u

private theorem step_capacity {cap : Nat} {s t : CacheState} {m : Move}
    (hs : step cap s m = some t) (hcap : s.fast.length ≤ cap) : t.fast.length ≤ cap := by
  cases m <;> grind [step, List.length_filter_le]

theorem ValidRun.capacity {cap : Nat} {s t : CacheState} {ms : List Move}
    (run : ValidRun cap s ms t) (hcap : s.fast.length ≤ cap) : t.fast.length ≤ cap := by
  induction run with
  | nil => exact hcap
  | cons head _ ih => exact ih (step_capacity head hcap)

private def Move.loads : Move → List Nat
  | .load x => [x]
  | _ => []

private def Move.computes : Move → List Nat
  | .compute x => [x]
  | _ => []

private def loadedItems (ms : List Move) : List Nat := ms.flatMap Move.loads

def computedItems (ms : List Move) : List Nat := ms.flatMap Move.computes

def loadCount (ms : List Move) : Nat := (loadedItems ms).length

def ioCount (ms : List Move) : Nat :=
  match ms with
  | [] => 0
  | .load _ :: ms | .store _ :: ms => ioCount ms + 1
  | _ :: ms => ioCount ms

theorem loadCount_le_ioCount (ms : List Move) : loadCount ms ≤ ioCount ms := by
  induction ms with
  | nil => exact Nat.le_refl 0
  | cons m ms ih =>
    cases m <;> simp_all [loadCount, loadedItems, Move.loads, ioCount] <;> omega

private theorem step_items {cap : Nat} {s t : CacheState} {m : Move}
    (hs : step cap s m = some t) :
    m.computes ⊆ s.fast ∧ t.fast ⊆ m.loads ++ s.fast := by
  cases m with
  | load x =>
    refine ⟨List.nil_subset _, ?_⟩
    simp only [step] at hs
    split at hs
    · cases hs
      exact List.subset_cons_self _ _
    · split at hs
      · cases hs
        exact List.Subset.refl _
      · contradiction
  | evict x =>
    cases hs
    exact ⟨List.nil_subset _, List.filter_sublist.subset⟩
  | compute x | store x =>
    simp only [step] at hs
    split at hs
    · cases hs
      simp_all [Move.computes, Move.loads]
    · contradiction

private theorem computed_subset {cap : Nat} {s t : CacheState} {ms : List Move}
    (run : ValidRun cap s ms t) : computedItems ms ⊆ s.fast ++ loadedItems ms := by
  induction run with
  | nil => exact List.nil_subset _
  | @cons s t u m ms hs _ ih =>
    obtain ⟨current, next⟩ := step_items hs
    intro x hx
    change x ∈ m.computes ++ computedItems ms at hx
    change x ∈ s.fast ++ (m.loads ++ loadedItems ms)
    simp only [List.mem_append] at hx ⊢
    rcases hx with hnow | hlater
    · exact .inl (current hnow)
    · rcases List.mem_append.mp (ih hlater) with hresident | hloaded
      · rcases List.mem_append.mp (next hresident) with hload | hstart
        · exact .inr (.inl hload)
        · exact .inl hstart
      · exact .inr (.inr hloaded)

variable {cap V M : Nat} {s₀ s₁ : CacheState} {ms : List Move} {req : List Nat}

theorem distinct_requests_le (run : ValidRun cap s₀ ms s₁)
    (hd : req.Nodup) (hc : req ⊆ computedItems ms) :
    req.length ≤ s₀.fast.length + loadCount ms := by
  simpa only [List.length_append, loadCount] using
    hd.length_le_of_subset (hc.trans (computed_subset run))

theorem cold_miss_lower_bound (run : ValidRun cap ⟨[]⟩ ms s₁)
    (hd : req.Nodup) (hc : req ⊆ computedItems ms) : req.length ≤ loadCount ms := by
  simpa using distinct_requests_le run hd hc

theorem cold_miss_io_lower_bound (run : ValidRun cap ⟨[]⟩ ms s₁)
    (hd : req.Nodup) (hc : req ⊆ computedItems ms) : req.length ≤ ioCount ms :=
  Nat.le_trans (cold_miss_lower_bound run hd hc) (loadCount_le_ioCount ms)

theorem pigeonhole_capacity_spill_lower_bound (run : ValidRun cap s₀ ms s₁)
    (hV : s₀.fast.length ≤ V) (hd : req.Nodup) (hc : req ⊆ computedItems ms) :
    req.length - V ≤ loadCount ms := by
  have h := distinct_requests_le run hd hc
  omega

private theorem nodup_eraseDups (xs : List Nat) : xs.eraseDups.Nodup := by
  suffices h : ∀ acc : List Nat, acc.Nodup →
      (List.eraseDupsBy.loop (· == ·) xs acc).Nodup from h [] .nil
  induction xs with
  | nil =>
    intro acc ha
    exact (List.reverse_perm acc).nodup_iff.mpr ha
  | cons x xs ih =>
    intro acc ha
    unfold List.eraseDupsBy.loop
    split
    · exact ih acc ha
    · rename_i h
      exact ih (x :: acc) (List.nodup_cons.mpr
        ⟨fun hx => (List.any_eq_false.mp h x hx) (by simp), ha⟩)

theorem hong_kung_epoch_access_bound (run : ValidRun cap s₀ ms s₁)
    (hM : s₀.fast.length ≤ M) : (computedItems ms).eraseDups.length ≤ M + loadCount ms := by
  have covered : (computedItems ms).eraseDups ⊆ computedItems ms :=
    fun _ hx => List.mem_eraseDups.mp hx
  exact Nat.le_trans (distinct_requests_le run (nodup_eraseDups _) covered)
    (Nat.add_le_add_right hM _)

example : ValidRun 1 ⟨[]⟩
    [.load 2, .compute 2, .store 2, .evict 2, .load 5, .compute 5] ⟨[5]⟩ := by
  repeat apply ValidRun.cons (by rfl)
  exact .nil

example : ValidRun 1 ⟨[7]⟩
    [.compute 7, .compute 7, .evict 7, .load 3, .compute 3] ⟨[3]⟩ := by
  repeat apply ValidRun.cons (by rfl)
  exact .nil

example : ValidRun 1 ⟨[7]⟩ [.load 7, .store 7] ⟨[7]⟩ :=
  .cons rfl (.cons rfl .nil)

#print axioms pigeonhole_capacity_spill_lower_bound
#print axioms hong_kung_epoch_access_bound

end PerfRuler.PebbleIO