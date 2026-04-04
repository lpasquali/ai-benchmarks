---- MODULE FailClosedEstimation ----
\* Fail-Closed Cost-Estimation specification.
\*
\* Models the safety invariant: if API reachability confidence drops below
\* the 0.95 threshold, the system MUST transition to (and stay in) the
\* Halted state before any cost-estimation or provisioning action is taken.
\*
\* Confidence is represented as an integer in [0, 100] (i.e. percentage × 100)
\* to avoid TLA+ floating-point limitations.  The threshold is 95 (= 0.95 × 100).

EXTENDS Naturals

CONSTANTS
    MinConfidence,   \* lowest possible confidence value  (typically 0)
    MaxConfidence,   \* highest possible confidence value (typically 100)
    Threshold        \* fail-closed threshold              (typically 95)

ASSUME MinConfidence \in Nat
ASSUME MaxConfidence \in Nat
ASSUME Threshold     \in Nat
ASSUME MinConfidence <= Threshold
ASSUME Threshold     <= MaxConfidence

VARIABLES
    confidence,   \* current API reachability confidence in [MinConfidence, MaxConfidence]
    systemState   \* "active" | "halted"

SystemStates == {"active", "halted"}

TypeInvariant ==
    /\ confidence \in MinConfidence..MaxConfidence
    /\ systemState \in SystemStates

\* Core safety property: below-threshold confidence MUST mean halted.
FailClosedInvariant ==
    confidence < Threshold => systemState = "halted"

\* Dual liveness property: at or above threshold the system MAY be active.
\* (We do not force it to be active — operators can keep it halted manually.)
ActiveOnlyWhenSafe ==
    systemState = "active" => confidence >= Threshold

Init ==
    /\ confidence  = MaxConfidence   \* start fully reachable
    /\ systemState = "active"

\* Confidence degrades (any step down in [MinConfidence, MaxConfidence-1])
DegradeConfidence ==
    /\ confidence > MinConfidence
    /\ confidence' \in MinConfidence..(confidence - 1)
    \* Enforce fail-closed: if new confidence is below threshold, halt atomically.
    /\ systemState' =
            IF confidence' < Threshold
            THEN "halted"
            ELSE systemState

\* Confidence improves (any step up in [MinConfidence+1, MaxConfidence])
ImproveConfidence ==
    /\ confidence < MaxConfidence
    /\ confidence' \in (confidence + 1)..MaxConfidence
    /\ UNCHANGED systemState   \* recovery requires explicit operator action

\* Operator explicitly re-enables the system after confidence has recovered.
ResumeSystem ==
    /\ systemState = "halted"
    /\ confidence >= Threshold
    /\ systemState' = "active"
    /\ UNCHANGED confidence

\* Perform a cost-estimation (or provisioning) action — only allowed when active.
ExecuteAction ==
    /\ systemState = "active"
    /\ confidence >= Threshold   \* redundant guard, belt-and-suspenders
    /\ UNCHANGED <<confidence, systemState>>

Next ==
    \/ DegradeConfidence
    \/ ImproveConfidence
    \/ ResumeSystem
    \/ ExecuteAction

Spec == Init /\ [][Next]_<<confidence, systemState>>

\* TLC will verify both invariants hold in every reachable state.
THEOREM Spec => []TypeInvariant
THEOREM Spec => []FailClosedInvariant
THEOREM Spec => []ActiveOnlyWhenSafe

====
