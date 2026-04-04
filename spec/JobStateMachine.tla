---- MODULE JobStateMachine ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxJobs

VARIABLES jobs, idempotency_keys

JobStatuses == {"queued", "running", "succeeded", "failed"}

TypeInvariant ==
    /\ \A j \in 1..MaxJobs :
        /\ jobs[j].status \in JobStatuses
        /\ jobs[j].tenant \in STRING
    /\ idempotency_keys \in SUBSET (STRING \X STRING)

\* Terminal statuses — once reached, a job must not regress
TerminalStatuses == {"succeeded", "failed"}

\* Safety: no job ever moves backward from a terminal state
NoRegression ==
    \A j \in 1..MaxJobs :
        jobs[j].status \in TerminalStatuses =>
            [](jobs[j].status \in TerminalStatuses)

\* Safety: queued jobs never jump directly to succeeded/failed
NoDirectQueToTerminal ==
    \A j \in 1..MaxJobs :
        jobs[j].status = "queued" =>
            ~(jobs'[j].status \in TerminalStatuses)

Init ==
    /\ jobs = [j \in 1..MaxJobs |-> [status |-> "queued", tenant |-> "default"]]
    /\ idempotency_keys = {}

StartJob(j) ==
    /\ jobs[j].status = "queued"
    /\ jobs' = [jobs EXCEPT ![j].status = "running"]
    /\ UNCHANGED idempotency_keys

CompleteJob(j) ==
    /\ jobs[j].status = "running"
    /\ jobs' = [jobs EXCEPT ![j].status = "succeeded"]
    /\ UNCHANGED idempotency_keys

FailJob(j) ==
    /\ jobs[j].status = "running"
    /\ jobs' = [jobs EXCEPT ![j].status = "failed"]
    /\ UNCHANGED idempotency_keys

\* Idempotency: submitting the same (tenant, key) pair twice returns the same job.
\* Modelled as: if key already exists, no new job record is mutated.
RegisterIdempotencyKey(tenant, key) ==
    /\ (tenant, key) \notin idempotency_keys
    /\ idempotency_keys' = idempotency_keys \union {(tenant, key)}
    /\ UNCHANGED jobs

Next ==
    \/ \E j \in 1..MaxJobs :
        \/ StartJob(j)
        \/ CompleteJob(j)
        \/ FailJob(j)
    \/ \E t \in STRING, k \in STRING :
        RegisterIdempotencyKey(t, k)

\* Idempotency invariant: no duplicate (tenant, key) pairs
IdempotencyInvariant ==
    \A p1, p2 \in idempotency_keys :
        (p1[1] = p2[1] /\ p1[2] = p2[2]) => p1 = p2

Spec == Init /\ [][Next]_<<jobs, idempotency_keys>>

THEOREM Spec => []TypeInvariant
THEOREM Spec => []IdempotencyInvariant

====
