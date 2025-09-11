# What is Rebasing (in plain terms)?

**Rebase** moves your commits to a new base commit. Instead of merging two branches (which creates a merge commit), rebase **replays** your branch’s commits *on top of* another branch, producing a **linear history**.

**Merge (no history rewrite):**

```
A---B---C  (main)
         \ 
          D---E (feature)          --> merge --> A---B---C---M
                                  /                         \
                                D---E                     (merge commit M)
```

**Rebase (history rewrite on feature):**

```
A---B---C  (main)
         \
          D---E (feature)       --> rebase --> A---B---C---D'---E'
                               (D,E replayed on top of C)
```

---

# Step-by-Step: Rebase a Feature Branch onto `main`

## 0) Setup a sample repo

```bash
mkdir git-rebase-demo && cd git-rebase-demo
git init
echo "hello" > app.txt
git add app.txt
git commit -m "A: initial commit"
```

Make two more commits on `main`:

```bash
echo "line 2" >> app.txt
git commit -am "B: add line 2"

echo "line 3" >> app.txt
git commit -am "C: add line 3"
```

Check log:

```bash
git log --oneline
# Example output (your SHAs will differ)
# 7f1e2c3 C: add line 3
# 0a9b8d4 B: add line 2
# 1a2b3c4 A: initial commit
```

## 1) Create a feature branch and make commits

```bash
git checkout -b feature/add-title
echo "title=MyApp" > config.ini
git add config.ini
git commit -m "D: add config.ini with title"

echo "env=prod" >> config.ini
git commit -am "E: set env to prod"
```

At this point, history diverged:

```
main:    A --- B --- C
                     \
feature:              D --- E
```

## 2) Meanwhile, `main` moves forward

Switch back to main and add a commit:

```bash
git switch main   # or: git checkout main
echo "line 4" >> app.txt
git commit -am "F: add line 4"
```

`main` is ahead by `F`.

## 3) Rebase feature onto the latest `main`

Now bring your feature on top of `main`:

```bash
git switch feature/add-title
git rebase main
```

**What happens:**

* Git finds the common ancestor of `main` and `feature`.
* It replays `D` then `E` *on top of* `main`’s tip (`F`).
* You’ll end up with new commits `D'` and `E'` (new SHAs).

Inspect:

```bash
git log --oneline --graph --decorate
# Example:
# * abc1234 E': set env to prod
# * def5678 D': add config.ini with title
# * 55aa66f F: add line 4   (main)
# * 7f1e2c3 C: add line 3
# * 0a9b8d4 B: add line 2
# * 1a2b3c4 A: initial commit
```

## 4) Push after a rebase (force-with-lease!)

Because rebase rewrote `feature` history, you must **force push** the branch:

```bash
git push --force-with-lease origin feature/add-title
```

> Use `--force-with-lease` (safer) rather than `--force`. It refuses to overwrite others’ work if the remote moved unexpectedly.

---

# Handling Conflicts During Rebase

Suppose both `main` and `feature` touched the same lines in `config.ini`. During `git rebase main`, Git will pause and show conflicts.

### Typical flow

```bash
git rebase main
# ... stops with conflict
git status
# fix conflicted files in your editor
git add path/to/conflicted-file
git rebase --continue
# repeat if more conflicts
```

If things get messy:

```bash
git rebase --abort    # go back to pre-rebase state
```

**Tip:** Turn on reuse recorded resolutions to speed repeated conflict resolution:

```bash
git config --global rerere.enabled true
```

---

# Interactive Rebase (Clean Up Your Commits)

Before opening a pull request, teams often clean up noisy “WIP” commits by **squashing** or **rewording**.

Start an interactive rebase for the last N commits (here N=3):

```bash
git rebase -i HEAD~3
```

In your editor you’ll see something like:

```
pick def5678 D: add config.ini with title
pick abc1234 E: set env to prod
pick 55aa66f F: add line 4
```

Change actions:

* `pick` — keep as is
* `reword` — keep commit, edit message
* `squash` — combine with previous commit, edit combined message
* `fixup` — combine and discard this commit’s message
* `edit` — stop mid-rebase to amend the commit’s content
* `drop` — remove commit

Example (squash E into D):

```
pick def5678 D: add config.ini with title
squash abc1234 E: set env to prod
pick 55aa66f F: add line 4
```

Save/close; you’ll be prompted to write a new combined message for D+E. Finish the rebase; your history is now concise.

---

# Pull with Rebase (Keep Your Local History Linear)

Instead of `git pull` (fetch + merge), many teams prefer:

```bash
git pull --rebase
```

This fetches new commits and **replays your local commits on top** of them, avoiding merge bubbles.

Make it the default:

```bash
git config --global pull.rebase true
# optionally also:
git config --global rebase.autoStash true  # stashes/un-stashes uncommitted changes around a rebase
```

---

# Advanced: `rebase --onto` (Move a Subset of Commits)

Use `--onto` to move a range of commits elsewhere—handy for splitting a branch or changing base precisely.

**Scenario:**
You have `A - B - C - D - E` on `feature`, and you want to move commits `D..E` to sit on top of `main` (ignoring `C`).

```bash
# Syntax: git rebase --onto <newbase> <upstream> <branch>
git rebase --onto main C feature
```

This replays commits after `C` (i.e., `D` and `E`) onto `main`.

---

# Rebase vs Merge: Quick Comparison

| Aspect               | Merge                                | Rebase                                              |
| -------------------- | ------------------------------------ | --------------------------------------------------- |
| History shape        | Non-linear (merge commits)           | Linear (no merge commits)                           |
| Rewrites history     | No                                   | Yes (new SHAs)                                      |
| Collaboration safety | Safe for shared branches             | Risky if already pushed (requires force-with-lease) |
| Traceability         | Preserves true chronology            | Presents a clean, linear story                      |
| Conflict surface     | Fewer, larger conflicts (merge time) | More, smaller conflicts (during replay)             |
| Typical usage        | Integrate long-lived branches        | Clean up local/feature branches before sharing      |

---

# Advantages of Rebasing

* **Cleaner, linear history** (easier to read and bisect).
* **Easier review**: PRs show a straight series of meaningful commits.
* **Re-validate work**: replaying commits against the latest base catches issues early.

# Disadvantages of Rebasing

* **Rewrites history**: dangerous on public/shared branches.
* **Requires force-push**: potential to clobber teammates’ updates if misused.
* **Lose merge context**: you can’t see when/how lines converged via merge commits.
* **Learning curve**: conflicts may appear multiple times during replay.

---

# When (and Where) to Rebase

### Good candidates (do rebase)

* **Local feature/topic branches** before opening a PR.
* **Your own fork/branch** when syncing with upstream to avoid merge noise.
* **Short-lived branches** you alone control.

### Avoid rebasing (don’t rebase)

* **Protected branches**: `main`, `develop`, `release` that others pull from.
* **Shared/public branches already pushed** (unless you coordinate and everyone is okay with force pushes).
* **Release/hotfix histories** where merge commits document decisions.

---

# Common Workflows

## A) Keep your feature branch current

```bash
git switch feature/my-change
git fetch origin
git rebase origin/main
# resolve conflicts as needed
git push --force-with-lease
```

## B) Clean up commits before PR

```bash
git rebase -i HEAD~5
# squash/reword/fixup
git push --force-with-lease
```

## C) Always pull with rebase

```bash
git config --global pull.rebase true
git pull     # now fetch + rebase
```

---

# Recovery Safety Net

Messed up a rebase? **Reflog** to the rescue:

```bash
git reflog
# find the previous HEAD state (e.g., HEAD@{3})
git reset --hard HEAD@{3}
```

> Reflog is local history of where HEAD and branch tips moved. It’s your undo button.

---

## TL;DR Cheat Sheet

```bash
# Rebase feature onto main
git switch feature
git fetch origin
git rebase origin/main

# Continue/Abort during conflicts
git add <files>
git rebase --continue
git rebase --abort

# Interactive cleanup (last 4 commits)
git rebase -i HEAD~4

# Safer push after rebase
git push --force-with-lease
```

---