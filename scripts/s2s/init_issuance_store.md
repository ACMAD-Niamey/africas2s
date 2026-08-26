# Bootstrap the `issuance-store` orphan branch (one-time)

The S2S testbed workflow persists per-issuance NetCDF outputs to a long-lived
orphan branch named `issuance-store`. "Orphan" means the branch has no
shared history with `main` — it's a separate root commit — so it can be
force-pushed, squashed, and pruned independently of the codebase.

Run these commands once from a clean checkout of the africas2s repo:

```bash
cd /path/to/africas2s

# 1. Create the orphan branch (no parent commit).
git checkout --orphan issuance-store

# 2. Remove every file from the index — the orphan starts as an empty
#    tree, but the working directory still has main's files staged.
git rm -rf .

# 3. Create a minimal README explaining what this branch is for.
cat > README.md <<'EOF'
# S2S testbed issuance store

This orphan branch holds the raw per-issuance NetCDF outputs of the
africas2s S2S testbed. It is written to by `.github/workflows/s2s_testbed.yml`
and consumed by the same workflow's `verify` job.

**Do not** merge this branch into `main`. It is intentionally
history-isolated so it can be pruned and force-pushed without affecting
the codebase.

See `docs/superpowers/specs/2026-05-19-s2s-downscaling-testbed-design.md`
on `main` for the full design and the on-disk layout under `issuances/`.
EOF

# 4. Commit and push.
git add README.md
git commit -m "Initialize issuance-store orphan branch"
git push -u origin issuance-store

# 5. Return to your working branch.
git checkout main
```

After this, the workflow can `actions/checkout` the branch into a sibling
directory and append/commit/push without ever touching `main`.

To prune the branch later (drop NetCDFs older than 18 months — see the
spec's "Risks: Storage growth" section), repeat the bootstrap pattern:
checkout the branch, `git rm` old directories, commit, force-push.
