## Git Reviews – The Human Layer of Pull Requests

A Pull Request isn’t just about merging code—it’s about ensuring **quality, security, readability, and maintainability**. This is where **code reviews** come into play.

Code review is the process of having other developers **read, critique, and approve changes** before they are merged. It serves as both a **technical safeguard** and a **collaborative learning tool**.

---

### Why Git Reviews Matter

1. **Bug Detection Early**
   Reviews help catch logical, syntactical, or architectural flaws before they hit production.

2. **Consistency & Standards**
   Enforces coding conventions, architecture guidelines, and project rules.

3. **Knowledge Sharing**
   Junior developers learn best practices from senior engineers. Teams stay aligned on how features are built.

4. **Accountability & Transparency**
   Every line of code has a reviewer, ensuring no changes slip in unnoticed.

5. **Security**
   Sensitive logic (e.g., authentication, payments) gets extra scrutiny to avoid vulnerabilities.

---

### GitHub Reviews

* GitHub supports **review types**:

  * *Approve* → Reviewer accepts the changes.
  * *Comment* → Reviewer leaves feedback but doesn’t block merging.
  * *Request Changes* → PR cannot be merged until issues are resolved.

* Reviews are tied to the PR workflow. You can comment **inline on specific lines of code**, reply in threads, and even suggest changes directly.

* Teams can enforce **branch protection rules**:

  * Require at least X approvals before merging.
  * Require status checks (CI/CD tests) to pass before merging.
  * Prevent force pushes or direct commits to `main`.

Example: A PR to add login functionality might require **2 approvals** and all unit tests passing before it can be merged.

---

### Bitbucket Reviews

* Bitbucket PRs integrate tightly with **Jira**. Reviews are linked to issues automatically.
* You can assign **mandatory reviewers** (e.g., team leads, QA engineers).
* Approvals work similarly to GitHub: reviewers can approve, decline, or leave comments.
* Bitbucket supports **merge checks**:

  * A minimum number of approvals required.
  * All tasks in the PR must be resolved.
  * Builds must pass before merging.

Example: A Jira issue `ECOM-142` linked to a PR cannot be merged until at least one **QA reviewer** approves and Bitbucket Pipelines succeed.

---

### GitLab Reviews (for completeness)

* GitLab calls them **Merge Requests (MRs)** instead of PRs.
* Code review happens inside the MR view.
* GitLab allows **suggested changes** that authors can apply with a single click.
* Teams often combine GitLab MRs with **merge request approvals** (like requiring 1 backend and 1 frontend engineer’s approval).

---

### Real-World PR + Review Workflow

Let’s expand the earlier **E-commerce project** example with reviews included:

1. **Developer Creates PR**
   Alice pushes `feature/ECOM-142-discount-coupons` to Bitbucket and opens a PR.

2. **Automated Checks**
   Jenkins runs unit tests, code coverage, and lint checks.

3. **Reviewer Assignments**

   * Bob (team lead) → reviews architecture and logic.
   * Carol (QA) → reviews test coverage and edge cases.

4. **Review Cycle**

   * Bob requests changes to error handling.
   * Alice pushes a new commit fixing the logic.
   * Carol adds a comment about test cases → Alice adds them.
   * Both approve once satisfied.

5. **Merge Conditions**

   * 2 approvals 
   * All pipelines passing 
   * Jira issue `ECOM-142` automatically transitions to **Done**.

6. **Post-Merge**

   * The `develop` branch updates.
   * A staging deployment runs.
   * Release notes are auto-generated from merged PRs.

---

### Common Mistakes in Git Reviews

1. **Rubber-Stamp Approvals** – Approving without actually reviewing.
2. **Nitpicking Style Only** – Ignoring architecture and logic while focusing on trivial formatting.
3. **Unclear Review Comments** – Writing vague comments like *“Fix this”* without explanation.
4. **Review Bottlenecks** – Only one senior reviewer available, slowing merges.
5. **Skipping Reviews in Emergencies** – Merging hotfixes without at least minimal review.

---

### Best Practices in Git Reviews

1. **Automate What You Can**
   Use linters, formatters, and CI pipelines to catch simple issues. Focus human reviews on **logic, architecture, and maintainability**.

2. **Use Checklists**
   Teams often adopt review checklists:

   * Is the code readable and maintainable?
   * Are there enough tests?
   * Does it follow coding standards?
   * Any security implications?

3. **Be Respectful & Constructive**
   Reviews should improve the code, not criticize the developer. Use clear, actionable feedback.

4. **Rotate Reviewers**
   Avoid bottlenecks by spreading review responsibilities. It also improves team knowledge sharing.

5. **Define Clear Rules**
   Agree on how many approvals are required, who reviews what type of code, and how to handle urgent fixes.

---

## Extended PR + Review Workflow Cheat Sheet

| Stage | Action                | Tool/Command                                           | Notes                                 |
| ----- | --------------------- | ------------------------------------------------------ | ------------------------------------- |
| 1     | Create feature branch | `git checkout -b feature/ECOM-142`                     | Branch linked to Jira issue           |
| 2     | Commit changes        | `git commit -m "feat(ECOM-142): add discount coupons"` | Follow commit convention              |
| 3     | Push branch           | `git push origin feature/ECOM-142`                     | Push to remote repo                   |
| 4     | Open PR               | GitHub/Bitbucket UI → “Create Pull Request”            | Fill description, reviewers, issue ID |
| 5     | Automated checks      | CI/CD pipelines run                                    | Must pass before review               |
| 6     | Assign reviewers      | UI or auto-rules                                       | e.g., 1 lead + 1 QA reviewer          |
| 7     | Review feedback       | Inline comments, requests                              | Push new commits to update            |
| 8     | Approval              | “Approve” in UI                                        | Minimum approvals required            |
| 9     | Merge PR              | “Squash & Merge” or “Rebase & Merge”                   | Follow team policy                    |
| 10    | Jira sync             | PR merge updates issue status                          | Issue moves to “Done”                 |

---