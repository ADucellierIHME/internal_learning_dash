import altair as alt
import dash
import dash_bootstrap_components as dbc
import dash_vega_components as dvc
import pandas as pd
import numpy as np

from dash import Input, Output, dcc, html

from create_data import create_income, create_race, create_ell
from create_plots import plot_gap, plot_success

pd.options.mode.chained_assignment = None

# Data set
df = pd.read_csv("dataset.csv")

# List of schools
schools = df.SchoolName.unique().tolist()
schools.remove("District Total")
schools.remove("State Total")

# Start the app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = dbc.Container([
    html.P("The graphs show the percentage of students who met Level 3 or Level 4 on their assessment and the gap in percentage points between groups of students. All grade levels are considered. The test is the SBAC test. Best schools include all schools where at least 70% of students met Level 3 or Level 4 on their assessment. Only elementary schools are considered."),
    dbc.Card([
        html.Div([
            dbc.Label("Comparison between student groups"),
            dcc.Dropdown(
                    id="category",
                    options=[
                        {"label": "Low-income and non low-income", "value": "Income"},
                        {"label": "Hispanic and white", "value": "Race"},
                        {"label": "ELL and non ELL", "value": "ELL Status"}
                    ],
                    value="Income",
                ),
            ]),
        html.Div([
                dbc.Label("Test subject"),
                dcc.Dropdown(
                    id="subject",
                    options=[
                        {"label": "Mathematics", "value": "Math"},
                        {"label": "English Language Arts", "value": "ELA"}
                    ],
                    value="Math",
                ),
            ]),
        html.Div([
                dbc.Label("Choice of graph"),
                dcc.Dropdown(
                    id="type_chart",
                    options=[
                        {"label": "Gap between student groups", "value": "Gap"},
                        {"label": "Gap in schools with best test performances", "value": "Gap_best"},
                        {"label": "Schools where students perform best", "value": "Best"}
                    ],
                    value="Gap",
                ),
            ]),
        html.Div([
                dbc.Label("Highlighted school"),
                dcc.Dropdown(
                    id="highlighted",
                    options=schools,
                    value="McDonald",
                ),
            ])
        ], body=True)
])

app.layout = dbc.Container(
    [
        html.H1("How do students in the DLI program perform compared to Seattle schools?"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dvc.Vega(
                    id="chart",
                    opt={"renderer": "svg", "actions": False},
                    spec={}
                ), md=8),
            ],
            align="center",
        ),
        html.Hr(),
        dbc.Row([
            html.A("Source: Washington Office of Superintendent of Public Instruction - " + \
                       "Report Card Assessment Data 2023-24 School Year", \
                   href="https://data.wa.gov/education/Report-Card-Assessment-Data-2023-24-School-Year/x73g-mrqp/about_data", \
                   target="_blank"),
        ]),
    ],
    fluid=True,
)

@app.callback(
    Output("chart", "spec"),
    [
        Input("category", "value"),
        Input("subject", "value"),
        Input("type_chart", "value"),
        Input("highlighted", "value")
    ],
)
def make_graph(category, subject, type_chart, highlighted):
    if category == "Income":
        df_loc = create_income(df, subject)
    elif category == "Race":
        df_loc = create_race(df, subject)
    elif category == "ELL Status":
        df_loc = create_ell(df, subject)

    df_schools = df_loc.loc[~df_loc.SchoolName.isin(["State Total", "District Total"])]
    df_context = df_loc.loc[df_loc.SchoolName.isin(["State Total", "District Total"])]

    df_schools["our"] = 0
    df_schools["our"].loc[df_schools.SchoolName==highlighted] = 1

    if type_chart == "Gap":

        if category == "Income":
            title = "Gap between low-income and non low-income students in " + subject
        elif category == "Race":
            title = "Gap between Hispanic and white students in " + subject
        elif category == "ELL Status":
            title = "Gap between ELL and non ELL students in " + subject
        chart = plot_gap(df_schools, df_context, title)

    elif type_chart == "Gap_best":

        df_best = df_schools.loc[df_schools['All Students'] > 70.0]
        if category == "Income":
            title = "Gap between low-income and non low-income students in " + subject + " (best schools)"
        elif category == "Race":
            title = "Gap between Hispanic and white students in " + subject + " (best schools)"
        elif category == 'ELL Status':
            title = "Gap between ELL and non-ELL students in " + subject + " (best schools)"
        chart = plot_gap(df_best, df_context, title)

    elif type_chart == "Best":

        if category == "Income":
            title = "Performance of low-income students in " + subject
            column = "Low-Income:Q"
        elif category == "Race":
            title = "Performance of Hispanic students in " + subject
            column = "Hispanic/ Latino of any race(s):Q"
        elif category == "ELL Status":
            title = "Performance of ELL students in " + subject
            column = "English Language Learners:Q"
        chart = plot_success(df_schools, df_context, title, column)

    return chart.to_dict()

if __name__ == "__main__":
    app.run(debug=True)

