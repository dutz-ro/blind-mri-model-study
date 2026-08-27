# Installation

## Native

```bash
git clone <private-repository>
cd blind-mri-model-study
./install.sh --mode=native
./bin/mristudy doctor
```

The installer creates `.venv`, installs pinned dependency ranges, creates the protected data root,
installs the Google/Vertex adapter, and runs environment checks. It does not install system
packages with `sudo`.

## Docker

```bash
./install.sh --mode=docker
docker compose run --rm mristudy doctor
```

Copy `.env.example` to `.env` only on the server. Never commit `.env`.
