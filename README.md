# Advance Git Commands and GitHub Features

Git is more than `add`, `commit`, and `push`. Once you start collaborating on real projects, you’ll often need to rewrite history, clean up commits, rebase branches, or even recover “lost” work. This guide explores advanced Git features with explanations and space for practical demonstrations.

## 1. Commit History Rewriting in Git

### 1.1 Amending Git Commits
Amending in Git is like fixing or updating your most recent commit without creating a brand-new one. Instead of stacking another commit on top, you “rewrite” the last one.

```bash
git log --oneline
0832375 (HEAD -> main, origin/main) HTML File
f86213d first commit


git commit -am "main:modified the title - blog.html"
[main ab40261] main:modified the title - blog.html
 2 files changed, 9 insertions(+), 1 deletion(-)


git log --oneline                                   
ab40261 (HEAD -> main) main:modified the title - blog.html
0832375 (origin/main) HTML File
f86213d first commit
```

- Lets's suppose you have a file named 404.html and you want to commit the changes made inside this file as well in the previous commit.
 
```bash
# Stage the changes
git add 404.html

git status
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
  (use "git push" to publish your local commits)

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   404.html

```
In other words, now we want to rewrite the previous commit SHA "ab40261" and 
amend it to reflect the changes made in both blog.html and 404.html file.

```bash
# Commit the changes with amend and no edit flag
git commit --amend --no-edit

[main 4bf9f69] main:modified the title - blog.html
 Date: Thu Sep 11 20:15:36 2025 +0200
 3 files changed, 60 insertions(+), 1 deletion(-)
 create mode 100644 404.html

# The commit has been made with new SHA and the old commit shaw is deleted
git log --oneline           
4bf9f69 (HEAD -> main) main:modified the title - blog.html
0832375 (origin/main) HTML File
f86213d first commit

```
- "--no-edit" flag let's you make the amendment to your commit without changing its commit message.
- Amending rewrites history. If you already pushed that commit to a shared branch, amending it means the hash changes, which can cause confusion for others. Safe rule of thumb:
    - Amend freely on local commits you haven’t shared.
    - Avoid amending commits that are already pushed to remote unless you know how to handle a force-push.

#### Tomake changes to the message of the commit:

```bash
git add config.yaml
git commit --amend

[main 65cf379] main:modified the title - blog.html and added 404.html; added config.yaml
 Date: Thu Sep 11 20:15:36 2025 +0200
 4 files changed, 80 insertions(+), 1 deletion(-)
 create mode 100644 404.html
 create mode 100644 config.yaml

git log --oneline  
65cf379 (HEAD -> main) main:modified the title - blog.html and added 404.html; added config.yaml
0832375 (origin/main) HTML File
f86213d first commit

```
If you only want to fix the **commit message** and not change the files, Git gives you a flag for that:

```bash
git commit --amend -m "New, corrected commit message"
```

That command replaces the old message with the new one while keeping the commit’s content exactly the same.

Alternatively, if you want to open your editor to edit the message interactively:

```bash
git commit --amend
```

Just don’t stage any new files before running it—otherwise Git will also include those changes in the amended commit.

## 2. Git Rebase
**Rebase** moves your commits to a new base commit. Instead of merging two branches (which creates a merge commit), rebase **replays** your branch’s commits *on top of* another branch, producing a **linear history**.

```bash
# Create a new branch named feature-branch
git checkout -b feature-branch 

# sample command for rebase
git rebase <BASE>
# base can be: an ID, a branch name, a tag, or a relative referance to HEAD
git rebase main
```

```bash
git log --oneline --decorate --graph --all
* 89293e4 (HEAD -> main, origin/main) Rebasing
* 86bcbc6 Updated Readme with future section
* 65cf379 main:modified the title - blog.html and added 404.html; added config.yaml
* 0832375 HTML File
* f86213d first commit

```

#### Practical Example (with Conflict Simulation)
In this walkthrough, we will simulate a real scenario: creating a feature branch, making changes, modifying `main`, and then rebasing with conflicts.

---

#### Step 1: Create a New Feature Branch

```bash
git checkout -b feature-branch
```

**Output:**

```
Switched to a new branch 'feature-branch'
```

At this point, both `main` and `feature-branch` point to the same commit.

---

#### Step 2: Add Commits to Feature Branch

```bash
git commit -am "feature-branch: modified title - 404.html"
[feature-branch 82ff182] feature-branch: modified title - 404.html
 2 files changed, 21 insertions(+), 1 deletion(-)

git commit -am "feature-branch: modified Error Message - 404.html"
[feature-branch e7f8a2e] feature-branch: modified Error Message - 404.html
 2 files changed, 8 insertions(+), 2 deletions(-)
```

Now, `feature-branch` has two commits that are not present in `main`.

---

#### Step 3: Add Commits to Main

Switch back to `main` and simulate independent changes:

```bash
git checkout main
git commit -am "main: modified title again- 404.html"
[main 84900fe] main: modified title again- 404.html
 1 file changed, 2 insertions(+), 2 deletions(-)

git commit -am "main: modified Error Message again- 404.html"
[main 9cbea38] main: modified Error Message again- 404.html
 1 file changed, 2 insertions(+), 2 deletions(-)
```

**Visual Log (`git log --oneline --decorate --graph --all`):**

```
* 9cbea38 (HEAD -> main) main: modified Error Message again- 404.html
* 84900fe main: modified title again- 404.html
| * e7f8a2e (feature-branch) feature-branch: modified Error Message - 404.html
| * 82ff182 feature-branch: modified title - 404.html
|/  
* 89293e4 (origin/main) Rebasing
* 86bcbc6 Updated Readme with future section
```

Now both branches have diverged.

---

#### Step 4: Start Rebasing

```bash
git checkout feature-branch
Switched to branch 'feature-branch'

git rebase main
```

**Output:**

```
Auto-merging 404.html
CONFLICT (content): Merge conflict in 404.html
error: could not apply 82ff182... feature-branch: modified title - 404.html
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add/rm <conflicted_files>", then run "git rebase --continue".
```

Git stopped because of a conflict in `404.html`.

---

#### Step 5: Resolve the First Conflict

```bash
git mergetool
```

**Output:**

```
Merging:
404.html

Normal merge conflict for '404.html':
  {local}: modified file
  {remote}: modified file
```

Open the file, resolve the conflict, then stage the changes:

```bash
git add 404.html
git rebase --continue
```

**Output:**

```
[detached HEAD 352a722] feature-branch: Rebasing the feature branch - 404.html
 2 files changed, 21 insertions(+), 1 deletion(-)
Auto-merging 404.html
CONFLICT (content): Merge conflict in 404.html
error: could not apply e7f8a2e... feature-branch: modified Error Message - 404.html
```

Another conflict occurred in the next commit.

---

#### Step 6: Resolve the Second Conflict

Fix the file again, then run:

```bash
git add 404.html
git rebase --continue
```

**Output:**

```
[detached HEAD 24d3a81] feature-branch: rebase in progress - 404.html
 2 files changed, 8 insertions(+), 2 deletions(-)
Successfully rebased and updated refs/heads/feature-branch.
```

All conflicts are resolved, and rebasing is complete.

---

#### Step 7: Verify Commit History

```bash
git log --oneline --decorate --graph --all
```

**Output:**

```
* fca2f48 (HEAD -> feature-branch) feature-branch: modified README.md File - README.md
* 24d3a81 feature-branch: rebase in progress - 404.html
* 352a722 feature-branch: Rebasing the feature branch - 404.html
* 9cbea38 (main) main: modified Error Message again- 404.html
* 84900fe main: modified title again- 404.html
* 89293e4 (origin/main) Rebasing
```

Notice how `feature-branch` commits are now **on top of `main`**, forming a linear history.

---

#### Step 8: Merge Feature Branch into Main

Since `feature-branch` is rebased, merging results in a **fast-forward**:

```bash
git checkout main
git merge feature-branch
```

**Output:**

```
Updating 9cbea38..fca2f48
Fast-forward
 404.html  |  4 ++--
 README.md | 38 ++++++++++++++++++++++++++++++++++++++
```

`main` now contains all rebased commits.

---

#### Step 9: Cleanup

```bash
git branch -d feature-branch
Deleted branch feature-branch (was fca2f48).
```

---

#### Handy Rebase Commands

* **Abort rebase and go back**:

  ```bash
  git rebase --abort
  ```
* **Skip a conflicting commit**:

  ```bash
  git rebase --skip
  ```
* **Continue after fixing conflicts**:

  ```bash
  git rebase --continue
  ```

---

#### Key Takeaways

* Rebasing helps maintain a **clean, linear history**.
* Conflicts must be **resolved commit by commit** during rebase.
* Rebasing is typically done on **feature branches** before merging into `main` to ensure history clarity.

---

This simulation demonstrated a real-world rebasing scenario with conflicts, including how to resolve them and preserve a clean history.

---

## 3. Git Fetch

When working with Git in collaborative projects, it’s important to keep your local repository **aware of changes** that have happened on the remote without immediately applying them. That’s where **`git fetch`** comes in.

The command:

```bash
git fetch origin
```

* **Downloads** objects and refs from the remote repository (`origin`).
* **Updates your remote-tracking branches** (e.g., `origin/main`) to match the remote.
* **Does not modify your local branches** (e.g., `main`).

This makes `git fetch` a safe operation—you can inspect remote changes before deciding whether to merge or rebase them into your local branch.

---

## Step 1: Checking Branches

List all local and remote branches:

```bash
git branch -a
```

Output:

```
* main
  remotes/origin/main
```

* `main` → local branch
* `remotes/origin/main` → remote-tracking branch (your local copy of the remote branch)

Check only local branches:

```bash
git branch -v
* main 453b2c9 Rebasing Done with Updated Readme
```

Check only remote branches:

```bash
git branch -r
  origin/main
```

Check remote-tracking with last commit:

```bash
git branch -rv
  origin/main 453b2c9 Rebasing Done with Updated Readme
```

See everything:

```bash
git branch -av
* main                453b2c9 Rebasing Done with Updated Readme
  remotes/origin/main 453b2c9 Rebasing Done with Updated Readme
```

---

### Explanation

* **`main`**: local branch, currently checked out (`*`).
* **`remotes/origin/main`**: remote-tracking branch, representing the `main` branch on `origin`.

Both point to the same commit `453b2c9`, meaning your local branch is **in sync** with remote.

---

## Step 2: Detecting Divergence

Suppose new commits were added on the remote repository. Before fetching:

```bash
git log --oneline main
453b2c9 Rebasing Done with Updated Readme
... (older commits)
```

```bash
git log --oneline origin/main
453b2c9 Rebasing Done with Updated Readme
... (older commits)
```

Both look the same.

Now, after someone pushes **two new commits** (`Included Hyperparameter tuning`, `Created svm_classification.py`) to remote, your local copy is outdated.

Run:

```bash
git fetch origin main
```

Output:

```
From github.com:SurajBhar/advance_git
 * branch            main       -> FETCH_HEAD
   453b2c9..e5c4de0  main       -> origin/main
```

Remote-tracking branch `origin/main` is now updated, but **local `main` is untouched**.

---

## Step 3: Inspect Remote Changes

Check remote branch logs:

```bash
git log --oneline origin/main
e5c4de0 (origin/main) Included Hyperparameter tuning
86f6a84 Created svm_classification.py
453b2c9 Rebasing Done with Updated Readme
... (older commits)
```

Compare with local:

```bash
git status
On branch main
Your branch is behind 'origin/main' by 2 commits, and can be fast-forwarded.
```

At this point:

* `origin/main` knows about the **2 new commits**.
* `main` is **2 commits behind**.

---

## Step 4: Merging Remote Changes

To bring your local branch up to date:

```bash
git merge origin/main
```

Output:

```
Updating 453b2c9..e5c4de0
Fast-forward
 svm_classification.py | 125 +++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 125 insertions(+)
 create mode 100644 svm_classification.py
```

Now:

```bash
git log --oneline --decorate --graph --all
* e5c4de0 (HEAD -> main, origin/main) Included Hyperparameter tuning
* 86f6a84 Created svm_classification.py
* 453b2c9 Rebasing Done with Updated Readme
...
```

Your local `main` is synced with `origin/main`.

---

## Advantages of `git fetch`

- **Safe** – does not change your working branch.
- **Informative** – lets you see remote changes before merging.
- **Collaboration-friendly** – helps you stay aware of your teammates’ work.
- **Keeps remote-tracking branches up-to-date**.

---

## Disadvantages / Gotchas

- `git fetch` alone does not update your local branch—you still need `merge` or `rebase`.
- Can cause confusion for beginners because `origin/main` is updated but `main` is not.
- Requires extra step compared to `git pull` (which does fetch + merge).

---

## Tips for Users

- **Fetch daily**: Run `git fetch` before starting your workday.
- **Fetch before pushing**: Ensures you don’t accidentally push outdated work.
- **Inspect before merging**: Use `git log origin/main` to review remote commits.
- **Combine with prune**: `git fetch --prune` cleans up deleted remote branches.

---

### Key Takeaway

* `git fetch` updates your **view of the remote repository**.
* Your local branch remains unchanged until you explicitly merge/rebase.
* Think of it as “checking for mail” without opening the letters.

---

## 4. Git Pull and Pull with Rebase
When collaborating with others, keeping your local repository in sync with the remote repository is crucial. After all, you’re not the only one pushing changes!

* **`git fetch`** brings your remote-tracking branches (like `origin/main`) up to date but does not touch your local working branch.
* **`git pull`** goes further: it fetches changes **and merges them into your local branch**.

In other words:

```text
git pull = git fetch + git merge
```

---

## Step 1: Basic Git Pull

Let’s say you are on `main` and new commits have appeared on the remote repository. Running:

```bash
git pull origin main
```

* Fetches the new commits from `origin/main`.
* Merges them into your local `main`.
* If there are no conflicts, you get a **fast-forward merge**.
* If there are conflicts, Git will stop and ask you to resolve them.

- **Advantage**: quick and simple, your local branch matches remote.
- **Disadvantage**: history may become cluttered with merge commits, especially when you pull frequently.

---

## Step 2: Why Pull with Rebase?

Sometimes merging is not the cleanest solution. If you are working on **local changes** but also want to integrate the latest updates from remote:

* A normal merge will produce a merge commit and mix histories.
* A rebase will **reapply your local commits on top of the remote commits**, producing a cleaner, linear history.

The command:

```bash
git pull --rebase origin main
```

works like this:

1. `git fetch` → gets latest changes from `origin/main`.
2. `git rebase` → temporarily “removes” your local commits, applies the new commits from `origin/main`, then replays your local commits on top.

---

## Step 3: Example Walkthrough

Before pulling, the commit history looks like this:

```bash
git log --oneline --decorate --graph --all
* e5c4de0 (HEAD -> main, origin/main) Included Hyperparameter tuning
* 86f6a84 Created svm_classification.py
* 453b2c9 Rebasing Done with Updated Readme
* fca2f48 feature-branch:modified README.md File - README.md
* 24d3a81 feature-branch: rebase in progress - 404.html
* 352a722 feature-branch: Rebasing the feature branch - 404.html
* 9cbea38 main: modified Error Message again- 404.html
* 84900fe main: modified title again- 404.html
...
```

Two new commits (`Update 2 config.yaml`, `Updated 1 config.yaml`) exist on the remote but are not yet on local.

Now run:

```bash
git pull --rebase origin main
```

Output:

```
From github.com:SurajBhar/advance_git
 * branch            main       -> FETCH_HEAD
   e5c4de0..0535e79  main       -> origin/main
Successfully rebased and updated refs/heads/main.
```

---

## Step 4: After Rebase

Check the new commit history:

```bash
git log --oneline --decorate --graph --all
* ebf7c5c (HEAD -> main) main: modified readme - README.md
* 0535e79 (origin/main) Update 2 config.yaml
* d9b0764 Updated 1 config.yaml
* e5c4de0 Included Hyperparameter tuning
* 86f6a84 Created svm_classification.py
* 453b2c9 Rebasing Done with Updated Readme
...
```

Notice the difference:

* The remote commits (`d9b0764`, `0535e79`) were applied first.
* Your local commit (`ebf7c5c main: modified readme`) was **rebased on top**.
* The history is linear, without extra merge commits.

---

## Advantages of `git pull --rebase`

- Produces a **clean, linear history**.
- Easier to follow commit logs (`git log` looks tidy).
- Avoids unnecessary merge commits when working on small feature changes.

---

## Disadvantages / Cautions

- Can be risky on **shared branches** — rebasing rewrites history.
- Requires more caution when conflicts occur; you must resolve them commit by commit.
- For long-running feature branches, sometimes merge is the safer and more transparent option.

---

## Tips for Using Pull with Rebase

- Use `git pull --rebase` for personal feature branches or small sets of local commits.
- Avoid rebasing on shared/team branches unless everyone agrees to it.
- Set rebase as default for pulls:

```bash
git config --global pull.rebase true
```

- If conflicts occur, use:

```bash
git rebase --continue
git rebase --abort
```

---

### Key Takeaway

* **`git pull`** = fetch + merge → quick but may clutter history.
* **`git pull --rebase`** = fetch + rebase → cleaner history, local commits reapplied on top of remote.
* Choose **merge** for long-lived, collaborative branches and **rebase** for short-lived, personal branches where you want a tidy history.

---

## 5. Git Reference Logs (Reflog)
Git’s **reference logs (reflog)** act as a **safety net** for your repository. While `git log` shows the commits that are part of your branch’s history, `git reflog` records **all changes to the tips of branches and other references**—even if those commits don’t appear in the visible history anymore.

Think of the reflog as a **personal diary of your Git actions**, maintained locally. Every time you:

* Checkout a branch
* Commit changes
* Reset or rebase
* Merge or pull

…the `HEAD` reference is updated, and the reflog records it.

---

## Step 1: Viewing the Reflog

Run:

```bash
git reflog
```

Example output:

```
ebf7c5c (HEAD -> main, origin/main) HEAD@{0}: pull --rebase origin main (finish): returning to refs/heads/main
ebf7c5c (HEAD -> main, origin/main) HEAD@{1}: pull --rebase origin main (pick): main: modified readme - README.md
0535e79 HEAD@{2}: pull --rebase origin main (start): checkout 0535e79...
e7d0f19 HEAD@{3}: commit: main: modified readme - README.md
e5c4de0 HEAD@{4}: merge origin/main: Fast-forward
453b2c9 HEAD@{5}: commit: Rebasing Done with Updated Readme
...
```

Here’s what’s happening:

* `HEAD@{0}` – The most recent action. In this case, completing a pull with rebase.
* `HEAD@{3}` – A commit made earlier (`main: modified readme`).
* `HEAD@{5}` – A commit you thought might be “lost” if you reset later.

Unlike `git log`, these entries **remain visible even after resets or rebases**.

---

## Step 2: Why Reflog Matters

Let’s say you reset a branch or made changes you regret.

* `git log` may no longer show those commits (they are detached from the branch).
* But reflog keeps them recorded, allowing you to **recover lost commits**.

That’s why reflog is Git’s **time machine**.

---

## Step 3: Inspecting Historical States

You can inspect any past state of the repository with:

```bash
git show HEAD@{n}
```

Example:

```bash
git show HEAD@{29}
```

Output shows the very **first commit**:

```diff
commit f86213de...
Author: Suraj Bhardwaj
Date:   Thu Sep 11 17:45:43 2025 +0200

    first commit
```

---

You can also query reflog by **time references**:

```bash
git show main@{1.hour.ago}
git show main@{yesterday}
git show main@{1.week.ago}
```

Example:

```bash
git show main@{1.hour.ago}
```

Output:

```
commit e5c4de0...
Author: Suraj Bhardwaj
Date:   Fri Sep 12 10:29:35 2025 +0200

    Included Hyperparameter tuning
```

---

## Step 4: Using Reflog to Recover from Mistakes

### Scenario: Hard Reset

Suppose you ran:

```bash
git reset --hard bee45fc
```

This moved your branch pointer back to an older commit. At first glance, it looks like you lost your newer commits. But reflog recorded them:

```bash
git reflog
bee45fc (HEAD -> main) HEAD@{0}: reset: moving to bee45fc
ebf7c5c (origin/main) HEAD@{1}: reset: moving to ebf7c5c
bee45fc (HEAD -> main) HEAD@{2}: commit: main: deleted server info - config.yaml
...
```

You can still recover the commits via their reflog entries.

---

### Scenario: Restoring Lost Commits

You decide those commits weren’t mistakes after all. To bring them back:

```bash
# Checkout an old reflog entry in detached HEAD state
git checkout HEAD@{2}

# Create a new branch to save those commits
git checkout -b restore-branch

# Merge back into main
git checkout main
git merge restore-branch

# Delete the temporary branch
git branch -d restore-branch
```

Now your “lost” commits are restored into your main history.

---

## Step 5: Where Reflog Data Lives

* Reflog entries are stored inside `.git/logs/`
* They are **local only** – other collaborators cannot see your reflog.
* Git garbage collection (`git gc`) may eventually clean up very old reflog entries.

---

## Step 6: Cheat Sheet

Here’s a quick summary of useful reflog commands:

```bash
# Show reflog for HEAD (all actions)
git reflog

# Show reflog for a specific branch
git reflog show main

# Show commit at a specific reflog entry
git show HEAD@{5}

# Checkout an old state
git checkout HEAD@{10}

# Recover work by branching from a reflog entry
git checkout -b restore-branch HEAD@{8}

# Show history by time references
git show main@{1.hour.ago}
git show main@{yesterday}

# Diff between old state and current
git diff @{1.hour.ago}

# Show reflog in log format
git log -g
```

---

## Key Takeaways

* **Reflog is your safety net**: it tracks all branch tip movements.
* You can **restore commits** even after resets, rebases, and amends.
* It is **local-only** – don’t expect teammates to see your reflog.
* Always check reflog when you think “I lost my commits!”

---