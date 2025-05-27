# Learning-AI-DS
Daily hands on learning for AI/ML/DS, covering libraries, algorithms, and CI/CD practices.

This is a daily learning journey towards AI/DS for everyday.
Each folder represents a new day exploring libraries, concepts, or Techniques, with clear environment setup and notebooks.

## Structure
- `notebooks/`- Daily practice  folders(Day wise topics)
- `requirements.txt` – Global tools needed across days
- `.github/workflows/`- CI pipelines to automate checks

## CI/CD 
I am  using GitHub Actions to:
- Install daily dependencies
- Lint code with `black` and `ruff`
- Test notebooks using `pytest` and `nbval`

> Every folder is a self-contained mini-project with its own dependencies and README.

---
