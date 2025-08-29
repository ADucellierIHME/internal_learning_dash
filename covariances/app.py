import altair as alt
import dash
import dash_bootstrap_components as dbc
import dash_vega_components as dvc
import pandas as pd
import numpy as np

from dash import Input, Output, dcc, html

pd.options.mode.chained_assignment = None

# Data set
df = pd.read_csv('observations_25.csv')
df.sort_values(by=['county', 'race', 'cause', 'samples'], inplace=True)
df_mean = df.groupby(['cause', 'race', 'county']).agg({'value': 'mean'}).reset_index()
df_mean = df_mean.sort_values(by=['county', 'race', 'cause']).reset_index(drop=True)
data = np.reshape(df['value'].to_numpy(), (100, 72), order='F')
m = df_mean['value'].to_numpy()
S = np.matmul(np.transpose(data - m), data - m) / 100

# Start the app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = dbc.Container([
    html.P('Choose the dimension of the random vector, the number of samples and the number of bootstrap simulations.'),
    dbc.Card([
        html.Div([
            dbc.Label('Dimension'),
            dcc.Dropdown(
                id='p',
                options=[3, 9, 15, 45, 72],
                value=3,
            ),
        ]),        
        html.Div([
            dbc.Label('Samples'),
            dcc.Dropdown(
                id='n',
                options=[100, 250, 500, 1000],
                value=100,
            ),
        ]),
        html.Div([
            dbc.Label('Bootstrap'),
            dcc.Dropdown(
                id='B',
                options=[200, 500],
                value=200,
            ),
        ]),
    ], body=True)
])

app.layout = dbc.Container([
    html.H1('Sample covariance matrix'),
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(dvc.Vega(
            id='chart1',
            opt={'renderer': 'svg', 'actions': False},
            spec={}
        ), md=4),
        dbc.Col(dvc.Vega(
            id='chart2',
            opt={'renderer': 'svg', 'actions': False},
            spec={}
        ), md=4),
    ]),
    ],
    fluid=True,
)

@app.callback(
    [
        Output('chart1', 'spec'),
        Output('chart2', 'spec')
    ],
    [
        Input('p', 'value'),
        Input('n', 'value'),
        Input('B', 'value')
    ],
)
def make_graph(p, n, B):

    rng = np.random.default_rng(0)

    if p == 3:
        indices = [0, 24, 48]
    elif p == 9:
        indices=[1, 2, 3, 25, 26, 27, 49, 50, 51]
    elif p == 15:
        indices=[4, 8, 12, 16, 20, 28, 32, 36, 40, 44, 52, 56, 60, 64, 68]
    elif p == 45:
        indices = [5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, \
           29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, \
           53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71]
    else:
        indices = np.arange(0, p).tolist()
    S_sub = S[np.ix_(indices, indices)]
    m_sub = m[np.ix_(indices)]

    S_bar = np.zeros((p, p, B))    
    for b in range(0, B):
        X = rng.multivariate_normal(m_sub, S_sub, n)
        m_bar = np.mean(X, axis=0)
        S_bar[:, :, b] = np.matmul(np.transpose(X - m_bar), X - m_bar) / n

    S_true = S_sub[np.triu_indices(p, k=1)]
    S_mean = np.mean(S_bar, axis=2)[np.triu_indices(p, k=1)]
    S_lower = np.quantile(S_bar, 0.025, axis=2)[np.triu_indices(p, k=1)]
    S_upper = np.quantile(S_bar, 0.975, axis=2)[np.triu_indices(p, k=1)]

    num_coeff = len(S_true)
    num_uncertain = len(np.where((S_lower < 0.0) & (S_upper > 0))[0])
    perc_uncertain = 100 * num_uncertain / num_coeff

    sign_S_sub = np.repeat(np.sign(S_sub)[:, :, np.newaxis], B, axis=2)
    sign_S_bar = (sign_S_sub * S_bar > 0)
    sign_correct = (sign_S_bar.sum(axis=2) / B)[np.triu_indices(p, k=1)]

    # First plot
    df1 = pd.DataFrame(data={ \
        'true': S_true, \
        'mean': S_mean, \
        'lower': S_lower, \
        'upper': S_upper}).sort_values(by=['true']).reset_index(drop=True)
    df1['num'] = np.arange(0, num_coeff)

    true = alt.Chart(df1).mark_line(color='black', strokeDash=[2,2]).encode(
        x=alt.X('num:O', axis=alt.Axis(title='', labels=False, ticks=False)),
        y=alt.Y('mean:Q', axis=alt.Axis(title=''))
    )
    mean = alt.Chart(df1).mark_line(color='black').encode(
        x=alt.X('num:O', axis=alt.Axis(title='Index', labels=False, ticks=False)),
        y=alt.Y('mean:Q', axis=alt.Axis(title='Covariance'))
    )
    upper = alt.Chart(df1).mark_line(color='lightgrey').encode(
        x=alt.X('num:O', axis=alt.Axis(title='', labels=False, ticks=False)),
        y=alt.Y('upper:Q', axis=alt.Axis(title=''))
    )
    lower = alt.Chart(df1).mark_line(color='lightgrey').encode(
        x=alt.X('num:O', axis=alt.Axis(title='', labels=False, ticks=False)),
        y=alt.Y('lower:Q', axis=alt.Axis(title=''))
    )
    chart1 = alt.layer(
        true,
        upper,
        lower,
        mean
    ).properties(
        height=300,
        width=300,
        title=alt.Title('Off-diagonal coefficients',
               subtitle='{:.2f} % of the confidence intervals contain 0'.format(perc_uncertain))
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_title(
        fontSize=20,
        anchor='middle'
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16
    )

    # Second plot
    df2 = pd.DataFrame(data={ \
        'true': S_true, \
        'perc_correct': sign_correct}).sort_values(by=['true']).reset_index(drop=True)
    df2['num'] = np.arange(0, num_coeff)

    chart2 = alt.Chart(df2).mark_bar().encode(
        x=alt.X('num:O', axis=alt.Axis(title='', labels=False, ticks=False)),
        y=alt.Y('perc_correct:Q', axis=alt.Axis(title='Percentage'))
    ).properties(
        height=300,
        width=300,
        title=alt.Title('Percentage of coefficients with correct sign')
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_title(
        fontSize=16,
        anchor='middle'
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16
    )
    
    return (chart1.to_dict(format='vega'), chart2.to_dict(format='vega'))

if __name__ == "__main__":
    app.run(debug=True)

