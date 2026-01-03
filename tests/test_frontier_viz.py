import pandas as pd

from pa_core.viz.frontier import make


def test_frontier_viz_marks_constraints():
    df = pd.DataFrame(
        {
            "frontier_risk": [0.01, 0.02, 0.03],
            "frontier_return": [0.05, 0.06, 0.04],
            "frontier_cvar": [0.02, 0.03, 0.08],
            "constraints_satisfied": [True, False, True],
            "is_frontier": [True, False, True],
        }
    )

    fig = make(
        df,
        max_te=0.02,
        max_cvar=0.05,
        max_breach=0.1,
        max_shortfall=0.1,
    )

    trace_names = {trace.name for trace in fig.data}
    assert {"Feasible", "Infeasible", "Frontier"} <= trace_names
    assert fig.layout.shapes is not None
    assert len(fig.layout.shapes) >= 1
