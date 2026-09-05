---
description: Bootstrap the dev environment
---

# Bootstrap Dev Environment

Walk the user through first-time setup of this project's dev environment (or recovery after a wipe of local state — containers, databases, generated secrets). Detect the project's actual tooling rather than assuming a specific stack; do not invent steps the project doesn't have.

## Step 1 — Discover the project's setup story

Before doing anything, look for existing setup instructions and infer the stack:

- A `README.md` / `CONTRIBUTING.md` "Getting Started" or "Setup" section
- `docker-compose.yml` / `compose.yaml` (containerised services)
- `.env.example`, `.env.template`, or a `Setup/`-style directory of config templates
- Package manifests (`package.json`, `*.csproj`, `pyproject.toml`, etc.) for install/run scripts
- A `Makefile`, `justfile`, or similar task runner

If the project has clear existing setup docs, follow those rather than improvising. If nothing exists, ask the user how they currently set this project up locally before proceeding.

## Step 2 — Ensure local config exists

Check for the project's env/config file(s) (e.g. `.env`, `appsettings.Development.json`). If missing but a template exists (`.env.example`, `*.template`), copy it:

```bash
cp <template> <target>
```

Leave any secret placeholders blank for now unless the user already has real values to provide.

## Step 3 — Detect a two-phase credential bootstrap, if applicable

Some projects can't start fully until a secret/API key/credential is minted by one of their own services (e.g. a UI-generated API key, a first-run admin token). If the project's setup docs describe this kind of chicken-and-egg step:

1. Start only the services that don't need the credential yet.
2. Wait for the service that will produce the credential to become ready (poll its logs/health endpoint with a bounded timeout — don't sleep forever).
3. Tell the user exactly what to click/run to obtain the credential, and where to paste it back.
4. Write the credential into the local config with a targeted edit (don't invent or guess a value).
5. Start the remaining services.

If the project has no such step, skip straight to bringing everything up.

## Step 4 — Bring everything up

Run whatever the project's own tooling uses to start the stack (`docker compose up -d`, `npm run dev`, a task-runner target, etc.), following what Step 1 discovered.

## Step 5 — Verify

Confirm the services/processes are actually up and healthy (`docker compose ps`, a health-check endpoint, a smoke-tested URL — whatever fits the stack). Report back to the user:

- What's running and healthy.
- Anything unhealthy or restarting, with the relevant log tail.
- How to confirm the app is actually usable (e.g. a URL to open, a request to try).

## Recovery note

If the user says they wiped local state (containers, volumes, a database), treat any previously-minted credentials as gone too — clear them from the local config and restart from Step 3.

## What not to do

- Don't start services that depend on a not-yet-minted credential before that credential exists — this just burns restart attempts.
- Don't invent a credential/secret value. If the project mints it via a UI or first-run flow, it must come from there.
- Don't run a destructive reset (wiping volumes/databases) as a "recovery step" without checking with the user first.
