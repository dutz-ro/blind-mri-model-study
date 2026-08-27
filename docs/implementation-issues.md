# Implementation Issues And Resolutions

This log records material issues discovered while implementing the DICOM-aware MedGemma arm. It
contains no patient identity, DICOM pixels, credentials, or clinical conclusions.

## Data and DICOM

- The source DVD contained non-DICOM viewer files. Inventory had to ignore them without treating
  file extensions as authoritative.
- One DICOM object was non-imaging and had no pixel data. Contact-sheet and export code initially
  attempted to decode it; both now skip non-image instances and record the reason.
- The anonymized and original Series Instance UIDs differ by design. Series selection initially
  tried to join exports and inventory by UID; it was corrected to use the controlled relative-path
  mapping and later the anonymous series catalog.
- Orthanc WADO-RS returns DICOM instances as `multipart/related`. Requests for bare
  `application/dicom` returned HTTP 400. The gateway and runtime now request and parse multipart
  DICOM explicitly.
- QIDO response order is not slice order. Medical planning now orders locally from
  `ImageOrientationPatient` and `ImagePositionPatient`, with an explicit warning-only fallback to
  `InstanceNumber` and anonymous SOP Instance UID.
- Uniformly sampling up to 32 images left most of the 430-section 3D series unanalyzed. It was
  replaced with deterministic balanced contiguous windows and complete-coverage checks.
- The verified Orthanc manifest originally stored only the aggregate SOP Instance UID hash, not the
  exact UID list. That was insufficient to prove that every planned window UID existed in Orthanc.
  The local Orthanc verification manifest was regenerated to retain all 619 anonymous SOP Instance
  UIDs while preserving the existing DICOM objects.
- The selected meniscal series contain 617 image instances, while Orthanc contains 619 objects in
  total. The planner therefore has to select exactly the five approved roles and explicitly exclude
  the remaining series rather than infer that every Orthanc object belongs in the medical run.
- Geometry can be partially absent. An early fallback implementation mixed geometry-sorted slices
  with InstanceNumber-sorted slices in the same series, despite labeling the whole order as a
  fallback. The corrected behavior uses geometry for the entire series only when geometry is
  complete; otherwise it orders the entire series by InstanceNumber and anonymous SOP Instance UID
  and emits an explicit warning.
- SOP Instance UID uniqueness originally was checked only inside each series. A duplicate UID across
  two selected series could have overwritten the global UID-to-evidence lookup. Global duplicate
  detection was added before evidence IDs are assigned.

## Window planning

- A naive sequence of full 32-image windows can leave a very small final window or create more
  overlap than necessary. The final algorithm computes the minimum number of windows and balances
  total memberships across them, producing deterministic lengths that differ by at most one.
- Window count cannot be fixed to 26. It must result from the actual series inventory and the
  constraints `maximum_window_size=32` and `minimum_overlap=8`. Tests verify the expected current
  distribution of 18, 2, 2, 3, and 1 windows without placing 26 in production code.
- Coverage fields stored in a JSON plan are assertions, not proof. The first implementation trusted
  `coverage_percent` and `plan_sha256` from the file. Integrity checks now recompute the plan digest,
  unique UID coverage, window count, maximum size, duplicated memberships, series endpoints, and
  adjacent overlap before any inference or resume operation.
- Plan coverage cannot be validated only as a global UID union. Per-role checks were added to prove
  that each series starts at index zero, ends at its final geometric index, uses a consistent source
  count, and has valid overlap metadata at every boundary.
- The planner and Scaleway plan initially had a circular dependency: the MedGemma plan depended on
  the provision plan for its runtime digest, while the provision plan needed the MedGemma plan
  digest. Planning was reordered so the frozen runtime digest creates the MedGemma run plan first;
  the Scaleway plan is then generated and bound to that run-plan and evidence-map digest.
- Timestamps make otherwise equivalent plan files differ. Determinism is enforced for window
  membership, ordering, IDs, and input hashes; the signed plan digest intentionally also binds the
  creation timestamp, so regeneration requires a new approval rather than being treated as the same
  approved artifact.

## Evidence map

- The old runtime restarted IDs at `image-000001` for every pass. That made overlap and multiplanar
  provenance ambiguous. Evidence IDs are now assigned once across the complete MedGemma arm in a
  fixed role order and never restart per window or series.
- Reusing an evidence ID in overlapping windows is required, but the evidence map also has to list
  every window membership. The map is finalized only after all windows are created.
- Loading a map by filename was not sufficient. A modified map could change geometry, role, series,
  or window membership while keeping familiar evidence IDs. The runner now recomputes the map digest
  and cross-checks every `(evidence_id, SOPInstanceUID, series, role, window)` relationship against
  the immutable run plan.
- The evidence-map entry count must equal the number of source slices. This is checked separately
  from UID uniqueness so a missing unused entry cannot silently pass.

## Privacy and storage

- Metadata de-identification cannot prove absence of burned-in identity. The workflow stopped for
  human review of 31 contact sheets before any medical-image egress.
- Orthanc and OHIF initially had port-binding conflicts with an existing Docker environment. They
  were moved to dedicated loopback ports and a private Docker network.
- OHIF needs browser-reachable DICOMweb URLs and Orthanc authentication. A loopback Nginx gateway
  now proxies OHIF and DICOMweb while injecting Orthanc credentials server-side.
- All Orthanc and OHIF ports were verified inaccessible through the server's public IPv4.

## Routera and generalist models

- The SSH server's LiteLLM upstream networking was broken. The successful path used an encrypted
  SSH tunnel to the working local LiteLLM instance.
- Alias names alone did not prove the upstream route. Before egress, model metadata was checked to
  require `api.routera.one` for both GPT and Grok.
- The original LiteLLM timeout of 60 seconds was too short for the expanded schema. It was raised to
  300 seconds.
- Some model responses used unsupported enum values. Structural correction attempts are preserved
  separately, limited to two, and instructed not to alter the interpretation merely to validate.
- Interrupted supplement runs produced partial directories. Resume support was added so validated
  batches are skipped and partial artifacts are rejected.

## Google and provider migration

- Google Cloud had billing but zero GPU quota; quota approval was not timely. No DICOM was uploaded
  to Google.
- The pending Google plan was migrated explicitly to Scaleway. The state stayed at
  `CLOUD_PLAN_READY`, legacy plans remained intact, and no approval was carried between providers.
- Scaleway CLI response envelopes differed between resource types. Provisioning initially failed
  after creating only an SSH key and security group. Resource creation became idempotent and
  response-envelope aware; all partial resources were deleted and verified absent.
- Every runtime change invalidated the existing approval. Several exploratory containers were
  therefore destroyed before medical inference. The final implementation now requires one frozen
  release candidate before plan generation.
- A stale provision plan can remain on disk after runtime development. File presence is not treated
  as authorization: provisioning recomputes the plan digest and verifies its bindings to the current
  run plan and evidence map before checking the explicit approval artifact.
- Provider inventory had to be checked independently of local state files. The final audit confirmed
  zero active servers, security groups, volumes, public IPs, and temporary MRIStudy SSH keys in the
  target zone before release-candidate work continued.

## GPU runtime

- Hugging Face model metadata was readable anonymously, while gated weight download required an
  authenticated account. The pinned snapshot is now downloaded on the trusted SSH server and sent
  to the GPU without exposing a Hugging Face token there.
- The initial runtime was only a bootstrap service, not inference-capable. Later revisions added
  actual Transformers inference, DICOM retrieval, codecs, schema validation, and provenance.
- A newer PyTorch package selected a Triton native kernel that attempted runtime C compilation; the
  image lacked a compiler. The runtime environment now disables that optional native JIT path.
- Synthetic model output frequently used fenced JSON. Parsing now accepts a complete JSON code
  fence without relaxing the schema.
- The first model-backed probe of the approved window runtime returned the valid fenced JSON after a
  standalone `info` preamble. The parser accepted fences only when they began at the first character,
  so the probe failed closed before any medical DICOM access. The temporary GPU was destroyed and
  absence verified; the parser now accepts that observed non-clinical preamble and has a regression
  test for the exact output shape.
- The next approved runtime passed the one-image probe but the eight-image probe emitted a complete
  prose report followed by the requested valid fenced JSON. The prefix-specific parser still failed
  closed. Again, no medical DICOM was accessed, all GPU resources were destroyed, and the approval
  was invalidated. The parser now extracts the final fenced JSON block regardless of preceding prose,
  with regression coverage for the observed response form.
- A subsequent retry showed that the eight-image response is not deterministic: it sometimes emits
  only prose and omits JSON entirely, despite successfully processing the image batch. This made JSON
  generation an unreliable capacity-test criterion. Synthetic capacity probes now require an exact,
  count-specific non-medical marker from the model, while the canonical empty v2 result is validated
  independently in the same request and the exact evidence ID sequence is retained in the probe
  artifact. Medical inference still requires full schema and evidence validation.
- The first medical preflight exposed incorrect WADO media negotiation and was stopped before any
  result lock. Resources were destroyed and verified absent before replanning.
- A release candidate had technically valid bootstrap probes but not full medical inference. The
  final acceptance sequence now requires model-backed probes at 1/8/16/32 images and 32 synthetic
  DICOM objects before medical access.
- The first medical endpoint allowed the caller to request a series and let the runtime choose
  samples. This could not enforce an approved evidence list. The broad endpoint and its uniform
  sampling code were removed; `/analyze-window` accepts only exact SOP Instance UID and evidence ID
  arrays from the mounted plan.
- Exact request equality was initially enforced by source-code assertions only. The validation was
  extracted into an importable runtime contract and tested directly for empty, oversized,
  duplicated, reordered, and unapproved lists.
- Verifying the request against a mounted JSON file is insufficient if that file can change. The
  runtime now recomputes the mounted plan digest and compares it with an independently supplied
  approved digest before accepting a window.
- A CLI-only probe gate could be bypassed by calling `/analyze-window` directly. The runtime now
  requires a mounted probe-authorization artifact bound to the run plan, provision plan, runtime
  digest, model revision, and snapshot manifest before it performs even the medical QIDO lookup.
- Probe responses originally proved only that some model returned the expected structure. They did
  not prove the approved model, snapshot, runtime, or plan. Probe validation now checks all returned
  identity and digest fields before medical authorization.
- The 32-DICOM probe initially received weaker checks than the RGB probes. It now must report exactly
  32 inputs, the expected global evidence IDs, the contiguous synthetic-DICOM source type, valid v2
  schema, no OOM, and the approved runtime/model provenance.
- Hub fallback was still possible when `MEDGEMMA_MODEL_PATH` was absent. The runtime now refuses to
  start inference without the local snapshot and runs with Hugging Face and Transformers offline.
- The snapshot manifest digest alone does not prove its listed files remain intact. Planning and
  provisioning now execute `sha256sum -c` across the full snapshot manifest.
- Two copies of the clinical schema diverged: the runtime copy accepted any non-empty evidence ID,
  while the packaged schema required `image-NNNNNN`. The container now copies the canonical schema
  directly from the Python package, eliminating the duplicate maintained copy.
- Generation parameters and correction policy were initially caller-controlled. `max_new_tokens`
  and the maximum of two structural corrections are now stored per approved window and checked as
  part of plan integrity.
- Accepting correction text from the caller allowed unapproved prompt material. Structural
  corrections were moved inside the runtime: the server retains the previous output, appends only a
  fixed correction instruction, and returns every attempt for immutable storage.
- Runtime responses originally were trusted for their clinical JSON while manifests copied expected
  model metadata from the plan. The runner now rejects a response whose model, revision, window,
  ordered UIDs, evidence IDs, prompt hash, runtime digest, run-plan digest, provision-plan digest, or
  snapshot manifest differs from the approved values.
- Docker build context changed when the runtime began consuming the canonical schema and shared
  contract module. The Dockerfile now expects the repository root as build context and copies all
  runtime inputs explicitly.
- The first frozen-image health test failed because Uvicorn listened on `127.0.0.1` inside the
  container. Docker's host-loopback port publication could not reach that container-local loopback
  socket. The service now listens on the container's private `0.0.0.0:8000`, while the host still
  publishes it only on `127.0.0.1` and exposes no public application port.

## Result artifacts and resume

- The first medical window reached the model but generated thousands of repeated normal statements,
  exhausted the token budget, and never emitted JSON. Both structural corrections repeated the same
  behavior. Integration stopped, no result was locked, and the GPU was destroyed. The MedGemma-only
  prompt is now compact, explicitly permits an empty findings array instead of enumerating normal
  slices, and places the full JSON output contract after the image sequence so the final instruction
  is adjacent to generation. GPT/Grok prompts and locked outputs remain unchanged.
- The compact prompt allowed four windows to validate, but window 5 repeatedly omitted two required
  `confidence` fields and used `segment="entire"`, which is not a v2 enum. The raw findings were
  otherwise parseable and stable across corrections. A narrow deterministic normalizer now maps
  `entire` to `multiple_segments` and supplies the conservative missing value `confidence="low"`.
  It deep-copies the model result, records every path and operation, and does not alter clinical
  claims, evidence references, tear status, or measurements.
- With the normalizer enabled, nine windows validated before window 10 returned a malformed nested
  value: `vascular_zone_estimate` was a string rather than an object. The semantic evidence validator
  called `.get()` on it and raised `AttributeError`, preventing the normal structural-correction loop.
  The validator now treats malformed nested types as validation errors, never exceptions, so the two
  permitted model corrections can run and all raw attempts remain available for audit.
- After that validator fix, window 10 consistently returned a shorthand object: combined segment
  `posterior_horn_body` plus scalar values where v2 requires nested objects. Two corrections repeated
  the same stable claims. The audited normalizer now maps the combined segment to
  `multiple_segments` and expands only those shorthand fields into conservative objects. Explicit
  claims such as `displaced` and `degenerative_appearing` are preserved in their corresponding v2
  fields; every unspecified subfield becomes `indeterminate`, `null`, or `low` rather than being
  invented.
- That expansion allowed thirteen windows to validate. Window 14 then returned the correct empty
  findings array but an empty `treatment_boundary` object after both corrections. Because the three
  boundary fields are immutable contract constants rather than clinical interpretations, the
  normalizer now restores only those missing values as `false` and records each insertion. It never
  changes a present `true` value into `false`; an explicit boundary violation still fails.
- Fourteen windows then validated before window 15 entered a deterministic phrase-repetition loop,
  consumed the full generation budget, and never emitted JSON in any of its three attempts. The
  approved per-window generation contract now pins `do_sample=false`, a conservative
  `repetition_penalty=1.05`, and `no_repeat_ngram_size=8`. These parameters prevent exact phrase
  loops without sampling, changing the image list, or introducing a provider-dependent fallback.
- The first window under that policy produced valid structured output, but the runtime response did
  not echo the approved generation object. The runner correctly rejected it as a provenance mismatch
  before writing a valid-window manifest. The runtime now returns the exact generation policy used,
  allowing the runner to prove that repetition controls match the approved plan.
- Although the generation policy was correctly echoed, window 15 still repeated prose in every
  correction because the runtime resent the full 32-image medical input for structural repair. The
  correction path now becomes text-only after the first attempt: it receives the preserved prior
  response, validation errors, and schema, and is explicitly forbidden to inspect or reinterpret
  images. This reduces visual-token pressure while keeping the original medical attempt and all
  correction artifacts immutable.
- Repetition controls alone did not guarantee JSON: the model could still choose long non-JSON prose
  before any structural correction was possible. Medical generation is now constrained token by
  token with the canonical `knee-meniscus-finding-v2` JSON Schema. The approved window records
  `json_schema_enforced=true`, and runtime request/response provenance binds that policy. Post-output
  schema, evidence, treatment-boundary, and normalization checks remain mandatory defense in depth.
- Inspection of the exact pinned Transformers implementation found why the first constrained-decoder
  probe was ineffective: `ImageTextToTextPipeline._sanitize_parameters()` forwards generation
  controls only when nested under `generate_kwargs`. Direct `prefix_allowed_tokens_fn`, repetition,
  and sampling keywords were routed to preprocessing instead of `model.generate`. Every runtime
  inference path now uses nested generation kwargs, and a regression test exercises the pinned
  pipeline sanitizer to prove the callback reaches generation.
- Before another GPU plan was created, the exact pinned model snapshot and container were exercised
  on CPU with one synthetic image and the corrected nested generation contract. The model output
  began with `{`, contained one complete candidate, and passed the canonical v2 schema and semantic
  validator. This closes the constrained-decoder integration issue without another exploratory GPU
  cycle.
- The first GPU constrained-JSON probe then returned one `exact_valid` v2 object, but used a different
  valid `global_limitations` list than the synthetic fixture. The gate compared the entire object and
  failed despite proving the decoder contract. The corrected criterion requires the fixed schema and
  study purpose, an empty findings array, an array-valued limitations field, and all three mandatory
  false boundaries; it does not require model-generated free text to equal fixture wording.
- The CLI and runner both attempted to create a new run directory. The first real windowed medical
  invocation therefore stopped with `FileExistsError` before the first runtime request. Directory
  ownership now belongs only to the runner, and a regression test verifies creation of a fresh run.
- A raw file without its structured result or manifest is an unsafe partial state. Resume rejects
  any partial triplet rather than inferring that a window completed.
- The accepted correction response can differ from the initial raw response. Early resume validation
  checked only the initial raw file. Window manifests now identify the accepted attempt explicitly,
  hash it, and preserve every attempt as a separate artifact.
- A manifest hash is useless if it is never recomputed. Resume now verifies the manifest digest,
  raw and structured hashes, accepted-attempt hash, all listed attempt files, and the equality of the
  accepted structured result with the stored structured JSON.
- Extra unlisted files can indicate an interrupted write or manual modification. Resume compares the
  actual per-window artifact set with the manifest and fails closed on missing or unexpected files.
- A resume run must use exactly the same complete plan object, not merely the same string stored in a
  `plan_sha256` field. Both the current and stored plans are independently revalidated before any
  completed window is skipped.
- A runtime identity mismatch originally produced a misleading schema failure. It is now a hard
  provenance failure and the window is marked invalid without integration.
- The next approved runtime passed every synthetic gate and returned an `exact_valid` first medical
  window, but `/analyze-window` omitted the approved decoder contract from its response. The runner
  correctly rejected the otherwise valid result as a provenance mismatch, no result lock was
  created, and the GPU was stopped. The runtime now echoes the LM Format Enforcer identity, version,
  policy digest, and configuration in medical responses, with regression coverage requiring the
  field on health, constrained-probe, and medical response paths.

## Clinical contract

- Conventional MRI does not directly measure meniscal perfusion. Vascular zones are recorded only
  as anatomical estimates and must set `directly_observed_vascularity=false`.
- Imaging may provide factors relevant to specialist review but does not determine repairability or
  recommend treatment. These boundaries are constants in the schema and are revalidated after every
  window and integration stage.
- Window overlaps can repeat the same evidence. Global evidence IDs and evidence-linked components
  prevent overlap from being counted as independent corroboration.
- Contradictions between neighboring windows and different planes are retained as separate claim
  variants instead of being resolved by majority vote.
- Duplicate evidence cannot be rejected globally across a response because one slice may legitimately
  support two distinct findings. Validation now rejects duplicates within one finding and rejects
  using the same evidence as both supporting and counter-evidence for that finding.
- Schema validation alone does not enforce evidence semantics. The validator now requires
  `traceable_evidence` entries to use role `supporting` and `counter_evidence` entries to use role
  `counter`.
- Deduplicating solely by anatomy would incorrectly merge separate lesions. Series integration only
  links findings from different windows when they have the same anatomical key and share an exact
  global evidence ID.
- Shared evidence can be supporting in one window and counter-evidence in another. Claim grouping now
  includes the evidence-role signature so this disagreement remains as separate variants.
- Repetition in overlapping windows is not independent support. The initial multiplanar classifier
  could promote a finding merely because the same slice appeared twice. Classification no longer
  uses repeated overlap membership as corroboration.
- Group-level multiplanar provenance was not sufficient when a group contained several contradictory
  output variants. The final matrix now maps each output finding to its exact source finding IDs,
  source roles, evidence IDs, and claim key.
- Integration output can be structurally valid while containing an unknown evidence ID. Both series
  and multiplanar outputs are revalidated against the global evidence map and the v2 clinical
  boundary constants before being written.

## Testing and tooling

- Test fixtures initially reused generated Series and SOP UIDs across independently created
  synthetic series, triggering legitimate global-uniqueness checks. Fixtures now generate stable,
  role-specific UIDs.
- Importing helpers from `tests` failed because the directory was not a package. A test package marker
  was added so shared valid-schema fixtures can be reused safely.
- Several old tests asserted the removed `/analyze` endpoint and per-request evidence numbering. They
  were replaced with explicit `/analyze-window`, canonical-schema, exact-UID, and no-sampling
  contracts.
- Static endpoint tests alone were too weak. Direct contract tests now exercise oversized windows,
  duplicates, reordered or invented UIDs, evidence-role errors, treatment-boundary violations,
  digest recomputation, stale maps, correction preservation, and resume integrity.
- A code review found that stored coverage, probe, model, and provenance claims were being trusted in
  several places. Those findings led to independent digest recomputation and cross-binding checks at
  planning, provisioning, probing, execution, resume, and integration.

## Current release boundary

- Local and server test suites pass, Ruff passes, and `compileall` passes.
- The Orthanc manifest now records the exact anonymous SOP Instance UID inventory.
- GPT/Grok result locks were hashed before and after synchronization and remained unchanged.
- No Scaleway resource or new medical egress was created during this implementation.
- The final container build, frozen runtime digest, 617-slice medical run plan, evidence-map digest,
  and replacement Scaleway plan still have to be generated as one release-candidate sequence.
- Work must stop at `CLOUD_PLAN_READY` after that plan is generated. GPU provisioning and model-backed
  1/8/16/32 probes require explicit approval of the new provision-plan digest.

## Replay hardening reset

- Iterative GPU patching was stopped after repeated output-shape failures. A local replay corpus was
  built from 70 historical raw-response artifacts and 82 attempts without displaying their contents
  or reading DICOM. The first hardened pass classified 48 attempts as exactly valid, 25 as valid
  after audited normalization, and 9 as unrecoverable prose or truncated generations with no JSON
  candidate. No ambiguous pair of distinct valid JSON results was found.
- Output extraction now scans complete JSON objects with string and escape awareness, rejects
  duplicate keys and non-finite numbers, validates every candidate independently, and fails closed
  if multiple distinct valid semantic results are present.
- The normalization policy is versioned and verified through a structural before/after diff. Every
  changed path must have exactly one allowlisted action, evidence and measurement paths are protected,
  and normalization must be idempotent.
- Replay reports retain their creation timestamp, but the replay digest excludes that timestamp and
  binds only policies, classifications, source hashes, and decisions. Re-running an unchanged corpus
  therefore produces the same approval-bound digest instead of invalidating a plan because of time.
- The runtime and local replay now share the same pure output-contract module. Resume replays the
  accepted raw response through this module instead of trusting the stored structured result.
- Medical generation uses LM Format Enforcer with a schema-derived grammar, pinned implementation
  version, strict field order, and an explicit array cap of 128. A decoder-specific schema relaxes
  only boolean `const` handling required by the library; canonical Draft 2020-12 validation remains
  authoritative after generation.
- The first model-backed constrained-JSON probe proved that merely passing
  `prefix_allowed_tokens_fn` through the multimodal pipeline did not guarantee a valid result: the
  callback produced one JSON candidate, but canonical validation still rejected it. The dedicated
  probe now records only response hashes and lengths, requires the first non-whitespace character to
  be `{`, and fails before medical access if the callback permits a non-JSON prefix or produces an
  invalid candidate. The GPU used for this failed decoder probe is treated as invalid and cleaned up
  before any new release candidate.
- Deterministic fuzz tests exercise prose wrappers, fences, truncation, braces and escapes in strings,
  duplicate keys, non-finite numbers, multiple candidates, and ambiguity. The hardened local suite
  currently contains 123 passing tests.

## Feedback loop and verification integrity

- The runtime contract tests asserted on the text of `app.py`, for example
  `assert source.count('"decoder": _decoder_metadata') == 3`. Counting occurrences of a source
  string proves that a literal was typed, not that the response satisfies the runner. Every
  pre-lock provenance failure was therefore only observable on a paid GPU, after the model had
  already produced a medical result. `tests/test_runtime_response_contract.py` now derives the
  required key set from `execute_window` itself and compares it against the keys the runtime
  returns, so an omitted provenance field fails locally in milliseconds. The test was validated by
  mutation: deleting the decoder field from the medical response makes it fail.
- The occurrence-count assertion was also actively harmful, because adding a legitimate fourth
  decoder response path broke an otherwise correct change. Structural checks replaced it.
- The synthetic capacity probes at 1, 8, 16, and 32 images and the 32-DICOM probe did not report
  decoder provenance, and `run_required_probes` did not require it. Only the constrained JSON probe
  checked the field. Because these probes are what authorize medical access, medical generation
  could have run under a decoder that no gate had proven. Both the runtime payload and probe
  validation now carry and require decoder identity.
- Provider absence verification listed only servers and security groups. A GPU instance root volume
  is an SBS volume that survives server deletion, and a reserved public IP continues to bill, so a
  cleanup could report `verified_absent` while billable resources remained. Absence is now proven
  for volumes across both the instance and Block APIs, for public IPs, and for temporary SSH keys.
- The destroy plan recorded `volume_ids: ["0"]`. That is the instance volume map slot key, not a
  volume identifier, so it could never prove absence. Real SBS identifiers are now resolved from the
  provider by server reference before deletion, without modifying the approved plan.
- The repository had no version control, so eleven consecutive pre-lock failures could not be
  bisected or compared. Git was initialized. `.gitignore` did not exclude the source DICOM media,
  because DICOM instances are stored without a file extension; directory-level exclusions were added
  before the first commit and the staged-file policy check was run.
