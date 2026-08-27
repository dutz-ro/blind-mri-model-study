# Routera and LiteLLM

Configure the proxy and exact aliases discovered on the server:

```bash
export LITELLM_BASE_URL=https://proxy.example/v1-root-without-trailing-v1
read -rs LITELLM_API_KEY
export LITELLM_API_KEY
export MRI_STUDY_GPT_MODEL=<verified-alias>
export MRI_STUDY_GROK_MODEL=<verified-alias>
```

Discover without medical inputs:

```bash
./bin/mristudy models probe --study <id>
```

Executing probes may incur cost:

```bash
./bin/mristudy models probe --execute --model <alias>
```

Create and inspect the exact egress plan:

```bash
./bin/mristudy models select --study <id>
./bin/mristudy models plan --study <id> --batch-size 12
./bin/mristudy approve provider-egress --study <id>
./bin/mristudy models run --study <id>
```

The knee profile selects sagittal, coronal, and axial DP-FS MPR series plus sagittal T1 for the
generalist arms. The dense source 3D series is excluded from generalist egress and reserved for
the DICOM-aware volumetric arm.
