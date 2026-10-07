# Current stored-byte manifest correction

Exact evaluated head c42c68eb7e55fa216515a2f5c8e2cce5af4480f2 had two stale parent-manifest entries: README.md and replay.py. Independent audit checked 113 stored members and 100 decoded members across the parent and new revalidation manifests. Only those two stored hashes differed. The prior parent manifest and README are preserved verbatim here. Parent entries now bind current bytes; the initial source/test metadata is explicitly revision-bound.

Independent actual execution on this exact source: focused20PASS; all six named production mutations RED1, byte-identical restoration GREEN0. Raw console and JUnit phases, controls, original mismatch packet and current source/test/replay hash identities are retained. No production behavior or test code is edited by this correction. Earlier full-suite and coverage remain historical; no fresh full-suite, provider PASS or broader90% acceptance is claimed.

After this evidence-only push, fresh exact-head hosted checks, complete expected topology, zero active review threads and the seven-minute review floor remain mandatory before guarded squash and actual compare. Source2336 remainsOPEN.
