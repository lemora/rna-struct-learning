RNA Secondary Structure Learning from Chemical Reactivity Data
---

This project implements JAX-based RNA secondary structure prediction and parameter
learning for a differentiable, ensemble-based McCaskill-Nussinov model.
It learns base-pair energies directly from chemical reactivity data using gradient-based optimisation.

Requirements
---

- python >= 3.10
- the `pixi` package manager and workflow tool:
  - Show all runnable tasks: `pixi task list`
  - Run prediction: `pixi run predict --seq GGGAUUACCCC -p params/NussinovParamsDefault.v1`

`pixi` Tasks
---

All functionality can be accessed through the predefined tasks:

| Task           | Command                                                          |
|----------------|------------------------------------------------------------------|
| **predict**    | Predict RNA structure                       |
| **train**      | Train the model                               |
| **evaluate**   | Evaluate predictions, logs, loss landscape |
| **visualize**  | Visualise structure from `.dbn`           |
| **test**       | Run tests                                    |

---


