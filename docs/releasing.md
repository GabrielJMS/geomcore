# Releasing

Releases are fully automated: pushing a tag `vX.Y.Z` triggers
`.github/workflows/release.yml`, which runs the test gate, builds wheels for
Linux (x86_64, aarch64), macOS, and Windows plus an sdist, publishes to PyPI
and crates.io via trusted publishing (no long-lived secrets), and creates a
GitHub release with the artifacts attached.

## One-time setup

### PyPI (can be done before the first release)

1. Create an account at https://pypi.org (2FA is mandatory).
2. Add a **pending trusted publisher** at
   https://pypi.org/manage/account/publishing/ with exactly:
   - PyPI project name: `geomcore`
   - Owner: `GabrielJMS`
   - Repository: `geomcore`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   The first CI publish then creates the project and claims the name — no
   token needed, ever.

### crates.io

1. Log in at https://crates.io with GitHub and verify your email address
   (Account Settings → Profile).
2. Trusted publishing can only be configured for a crate that already exists,
   so the **first publish is manual**: create an API token (Account Settings →
   API Tokens, scope `publish-new`), then locally:

   ```sh
   cargo login   # paste the token
   cargo publish -p geomcore
   ```

3. After the first publish, open the crate's Settings → Trusted Publishing and
   add:
   - Repository: `GabrielJMS/geomcore`
   - Workflow filename: `release.yml`
   - Environment: `crates-io`

   Then revoke the API token; subsequent releases publish token-free from CI.
   (The workflow skips the crates.io publish when the tagged version is
   already there, so doing the manual publish before pushing the first tag is
   fine.)

## Cutting a release

1. Bump `version` under `[workspace.package]` in the root `Cargo.toml` — the
   single source of truth. Both crates inherit it (`version.workspace = true`),
   and the Python package version follows the crate version automatically via
   maturin. The tag check refuses a release if they ever disagree.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new
   `[X.Y.Z] - YYYY-MM-DD` heading and update the compare links at the bottom.
3. PR, review, merge to `main`.
4. Tag and push:

   ```sh
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. Watch the Release workflow. It will refuse to publish if the tag doesn't
   match the crate versions, or if tests fail.

First release only: remove the "Not yet published" lines from `README.md`
beforehand.
