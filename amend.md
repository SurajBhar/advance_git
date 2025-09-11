Amending in Git is like fixing or updating your **most recent commit** without creating a brand-new one. Instead of stacking another commit on top, you “rewrite” the last one.

Here’s how it works:

* If you forgot to add a file, or need to tweak the commit message, you can run:

  ```bash
  git commit --amend
  ```
* By default, this opens your editor with the previous commit message. You can edit it or leave it as is.
* If you just staged new changes, `--amend` will include them into the same last commit.

### Example

1. Make changes, stage them:

   ```bash
   git add file.txt
   git commit -m "Initial commit"
   ```
2. Realize you forgot `config.yml`. Stage it:

   ```bash
   git add config.yml
   ```
3. Amend the last commit:

   ```bash
   git commit --amend
   ```

   → Now both `file.txt` and `config.yml` are inside one commit.

### A caution

Amending **rewrites history**. If you already pushed that commit to a shared branch, amending it means the hash changes, which can cause confusion for others. Safe rule of thumb:

* Amend freely on local commits you haven’t shared.
* Avoid amending commits that are already pushed to remote unless you know how to handle a force-push.

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


