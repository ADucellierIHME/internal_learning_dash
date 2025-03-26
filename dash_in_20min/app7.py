# Import packages
from dash import Dash, html, dcc, callback, Output, Input
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Define the dataset
N = 100
x = np.linspace(0, 1, N)
y1 = np.random.randn(N) + 5
y2 = np.random.randn(N)
y3 = np.random.randn(N) - 5
df = pd.DataFrame(data={'x': x, 'y1': y1, 'y2': y2, 'y3': y3})

# Initialize the app
app = Dash()

# App layout
app.layout = [
    html.Div(children='Dashboard with layers'),
    html.Hr(),
    dcc.Checklist(options=['y1', 'y2', 'y3'], value=['y1'], id='columns'),
    dcc.Graph(figure={}, id='plot')
]

# Plot the chosen columns
@callback(
    Output(component_id='plot', component_property='figure'),
    Input(component_id='columns', component_property='value')
)
def make_figure(col_chosen):
    fig = go.Figure()
    for col in col_chosen:
        fig.add_trace(go.Scatter(x=df.x, y=df[col], mode='lines', name=col, showlegend=True))
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

