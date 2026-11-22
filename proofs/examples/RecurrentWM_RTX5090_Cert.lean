import Std

/-!
Auto-generated formal hardware latency lower-bound certificate by `perf-ruler v2`.
Model  : RecurrentWM_T64_kgrad2_H4
Device : RTX5090-measured (dtype=fp32, F=95000 GFLOP/s, BW=1508 GB/s, tau=930 ns)
Phase  : train_store
-/

namespace RecurrentWM_T64_kgrad2_H4_RTX5090_measured_train_store

def peakFlopsPerNs : Nat := 95000
def bwBytesPerNs   : Nat := 1508
def tauNs          : Nat := 930

def workFlops      : Nat := 2102000000
def trafficBytes   : Nat := 8307000
def critSteps      : Nat := 428
def certifiedNs    : Nat := 398040

structure ValidSchedule where
  makespanNs : Nat
  compute_ok : workFlops <= peakFlopsPerNs * makespanNs
  memory_ok  : trafficBytes <= bwBytesPerNs * makespanNs
  serial_ok  : critSteps * tauNs <= makespanNs

theorem schedule_meets_compute_floor (s : ValidSchedule) :
    22126 <= s.makespanNs := by
  have h1 : peakFlopsPerNs * 22126 <= workFlops := by decide
  have h2 : peakFlopsPerNs * 22126 <= peakFlopsPerNs * s.makespanNs :=
    Nat.le_trans h1 s.compute_ok
  exact Nat.le_of_mul_le_mul_left h2 (by decide : 0 < peakFlopsPerNs)

theorem schedule_meets_memory_floor (s : ValidSchedule) :
    5508 <= s.makespanNs := by
  have h1 : bwBytesPerNs * 5508 <= trafficBytes := by decide
  have h2 : bwBytesPerNs * 5508 <= bwBytesPerNs * s.makespanNs :=
    Nat.le_trans h1 s.memory_ok
  exact Nat.le_of_mul_le_mul_left h2 (by decide : 0 < bwBytesPerNs)

theorem schedule_meets_serial_floor (s : ValidSchedule) :
    398040 <= s.makespanNs := by
  have h1 : 398040 = critSteps * tauNs := by rfl
  rw [h1]
  exact s.serial_ok

theorem certified_speed_limit (s : ValidSchedule) :
    certifiedNs <= s.makespanNs := by
  have hc := schedule_meets_compute_floor s
  have hm := schedule_meets_memory_floor s
  have hs := schedule_meets_serial_floor s
  have hmax : certifiedNs <= 22126 \/ certifiedNs <= 5508 \/ certifiedNs <= 398040 := by
    decide
  rcases hmax with h | h | h
  · exact Nat.le_trans h hc
  · exact Nat.le_trans h hm
  · exact Nat.le_trans h hs

theorem witness_feasible : Exists (fun s : ValidSchedule => s.makespanNs = 398040) := by
  refine ⟨⟨398040, by decide, by decide, by decide⟩, rfl⟩

#print axioms certified_speed_limit
#print axioms witness_feasible

end RecurrentWM_T64_kgrad2_H4_RTX5090_measured_train_store
