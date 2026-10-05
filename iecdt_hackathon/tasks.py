from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Task:
    """One probe target.

    Attributes:
        kind: "regression" or "classification".
        variable: name in the label NetCDF.
        n_classes: number of classes, classification only.
        drop_values: label values meaning "undefined for this tile"; rows
            carrying them are excluded from both fitting and scoring. NaN is
            always dropped and does not need listing here.
    """

    kind: str
    variable: str
    n_classes: Optional[int] = None
    drop_values: tuple = field(default_factory=tuple)

    @property
    def description(self):
        return self.variable.replace("_", " ").title()

    @property
    def chance_level(self):
        """Balanced accuracy of a uniform random guesser."""
        return 1.0 / self.n_classes if self.n_classes else None


def _regression(name):
    return Task("regression", name)


def _classification(name, n_classes, drop_values=(-1,)):
    return Task("classification", name, n_classes=n_classes, drop_values=drop_values)


#: The four tasks with published labels. `evaluate.py` iterates this, so a
#: participant's local score covers these and nothing else; the organisers'
#: scorer passes it the full ten explicitly.
TASKS = {
    "task_4": _regression("task_4"),
    "task_5": _regression("task_5"),
    "task_6": _classification("task_6", 10),
    "task_7": _regression("task_7"),
}

REGRESSION_TASKS = [k for k, t in TASKS.items() if t.kind == "regression"]
CLASSIFICATION_TASKS = [k for k, t in TASKS.items() if t.kind == "classification"]
