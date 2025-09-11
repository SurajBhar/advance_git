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

# Create a new branch named feature-branch
git checkout -b feature-branch 

git log --oneline --decorate --graph --all
* 89293e4 (HEAD -> feature-branch, origin/main, main) Rebasing
* 86bcbc6 Updated Readme with future section
* 65cf379 main:modified the title - blog.html and added 404.html; added config.yaml
* 0832375 HTML File
* f86213d first commit

```

### 2.1 Rebase implementation

### 2.2 Resolving Git Rebase Conflict

### 2.3 Git Fetch

### 2.4 Git Pull with Rebase

### 2.5 Git Reference Logs