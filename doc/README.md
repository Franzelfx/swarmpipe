# Documentation

Principle for this stage: **less is more.** Documentation that runs ahead of
the code is stale before it is read. What is written here must be true; what is
no longer true gets deleted, not "updated later".

## Where things live

| Place | Content |
|---|---|
| `README.md` | Problem, approach, status, how this differs from existing projects, installation and usage as soon as there is something to use, licence. The only place for prose about the project. |
| `README.de.md` | German translation of the README, kept in sync. |
| `CONTRIBUTING.md` | Conventions, rules, pitfalls — everything someone must know before their first change. |
| Docstrings | Purpose and *why* of every public symbol. The API reference is the code. |
| Tests | The behaviour. A test is the binding specification, not a document. |
| Issues | Open questions, ideas, anything beyond the current milestone. |
| `doc/` | This file. Later at most `decisions.md`, see below. |

Milestones, schedule and funding rationale live in the project sketch, not in
the repository.

## What belongs in `doc/`

Only what cannot be read off the code and is still binding for everyone:

- **`decisions.md`** — decisions already made whose consequences reach beyond a
  single file (licence, dependency, an interface that may no longer change).
  Three sentences per entry: what was decided, why, what was rejected. Decided
  decisions only; open questions are issues.

One document per topic, at most one page, with the date of the last review in
the first line.

## What does not belong in `doc/`

- Designs for code that does not exist yet: design documents, requirements
  lists, porting guides. The design becomes docstrings and tests when the code
  is written.
- Roadmaps beyond the current milestone. Those are issues and milestones on the
  repository host.
- Reading lists. A citation belongs where it is used: a docstring or a README
  table.
- Second copies of README content.
- Guides and tutorials before the API is stable.
- Application texts and funding narratives.
- Generated content: API reference, changelog before the first release.

## Rules

1. If a document contradicts the code, the code wins. The document is corrected
   or deleted in the same change.
2. Documentation is written in the same change as the thing it describes —
   never before, never "added later".
3. Language: English everywhere — documents, code, docstrings, commit messages.
   `README.de.md` is the one deliberate translation.
4. When in doubt, do not document. An issue costs less than a stale document.
5. Before every release and every funding round, walk through `doc/`: whatever
   is no longer true goes.
