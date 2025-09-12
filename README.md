# Advance Git Commands and GitHub Features

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

## 4. Git Pull with Rebase

## 5. Git Reference Logs