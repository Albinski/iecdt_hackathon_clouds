import numpy as np

_UNSET = object()


def rank_within_task(scores):
    """Rank submissions on one task, best first.

    Args:
        scores: {submission: value}, higher is better. A submission whose value
            is None or non-finite is not ranked on this task at all -- it does
            not occupy a rank that would push the others down.

    Returns:
        {submission: rank}, 1-based, standard competition ranking.
    """
    ranked = {
        name: float(value)
        for name, value in scores.items()
        if value is not None and np.isfinite(value)
    }
    ranks, position = {}, 0
    previous = None
    for i, (name, value) in enumerate(
        sorted(ranked.items(), key=lambda kv: (-kv[1], kv[0])), start=1
    ):
        if previous is None or value != previous:
            position = i  # a tie keeps the earlier position
        ranks[name] = position
        previous = value
    return ranks


def scored_tasks(per_task_values, submission):
    """The tasks this submission carries a usable value for, sorted."""
    return sorted(
        task
        for task, values in per_task_values.items()
        if values.get(submission) is not None and np.isfinite(values[submission])
    )


def mean_score(per_task_values, submission, tasks=None):
    """A submission's global score: the mean of its per-task metrics.

    The mean is over the tasks this submission was actually scored on. A task
    is normally skipped for the whole field or for none of it -- every
    submission covers the same tiles and is scored against the same labels -- so
    all the means are over the same tasks and are directly comparable. When that
    is not so, `global_ranking` reports the count alongside the mean and
    `unequal_task_counts` flags it, because means over different task sets are
    not strictly comparable.

    Returns None when nothing was scored, which `global_ranking` places last
    rather than treating as a zero.
    """
    if tasks is None:
        tasks = scored_tasks(per_task_values, submission)
    values = [float(per_task_values[task][submission]) for task in tasks]
    return float(np.mean(values)) if values else None


def unequal_task_counts(order):
    """True if the ranked submissions' means are over different task counts.

    The condition this exists to catch is a submission whose probe collapsed on
    a task the rest survived: its mean would then skip that task rather than be
    dragged down by it. Callers surface this rather than silently comparing.
    """
    counts = {row["n_tasks"] for row in order if row["mean_score"] is not None}
    return len(counts) > 1


def global_ranking(per_task_values, submissions):
    """Order submissions by their mean score, best first.

    Args:
        per_task_values: {task: {submission: value}}, higher is better.
        submissions: every submission to place.

    Returns:
        A list of dicts in ranked order, each with `submission`, `rank`
        (global, 1-based, ties shared), `mean_score` (None if nothing was
        scored) and `n_tasks` (how many tasks that mean is over).
    """
    submissions = list(submissions)
    tasks = {name: scored_tasks(per_task_values, name) for name in submissions}
    means = {
        name: mean_score(per_task_values, name, tasks[name]) for name in submissions
    }

    # Descending by mean; anything unscorable sorts last. The name breaks
    # residual ties so the output order is deterministic, though the shared rank
    # still records that they tied.
    order = sorted(
        submissions, key=lambda name: (means[name] is None, -(means[name] or 0.0), name)
    )
    # `None` is a legitimate mean here (nothing scored), so the "no previous
    # row yet" sentinel has to be something a mean can never be.
    out, position, previous = [], 0, _UNSET
    for i, name in enumerate(order, start=1):
        if previous is _UNSET or means[name] != previous:
            position = i
        out.append(
            {
                "submission": name,
                "rank": position,
                "mean_score": means[name],
                "n_tasks": len(tasks[name]),
            }
        )
        previous = means[name]
    return out


def rank_everything(per_task_values, submissions=None):
    """One call from raw per-task metrics to the global order.

    Args:
        per_task_values: {task: {submission: value}}, higher is better.
        submissions: optional explicit roster, so a submission that every task
            skipped still appears (ranked last).

    Returns:
        (ranks_by_task, global_order) -- the per-task ranks for the detail
        tables, and the list `global_ranking` returns.
    """
    if submissions is None:
        submissions = sorted(
            {name for values in per_task_values.values() for name in values}
        )
    ranks_by_task = {
        task: rank_within_task(values) for task, values in per_task_values.items()
    }
    return ranks_by_task, global_ranking(per_task_values, submissions)
