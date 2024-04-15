import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich import inspect

df = pd.DataFrame(
    {
        "Training Loss": list(range(100)),
        "Validation Loss": [x / 2 for x in range(100)],
        "Accuracy": list(range(100)),
        "Epoch": range(100),
    }
)

inspect(df)

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(x=df["Epoch"], y=df["Training Loss"], name="Training Loss"), row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df["Epoch"], y=df["Validation Loss"], name="Validation Loss"),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(x=df["Epoch"], y=df["Accuracy"], name="Accuracy"), row=1, col=2
)

fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
fig.show()
