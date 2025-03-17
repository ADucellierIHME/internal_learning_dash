import altair as alt
import dash
import dash_bootstrap_components as dbc
import dash_vega_components as dvc
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output

from plotting_functions import plot_initial_raked_1, plot_initial_raked_2, \
    plot_effect_of_1_initial_on_all_raked, plot_effect_of_all_initials_on_1_raked, \
    plot_comparison_mean, plot_comparison_variance

from raking_functions import rake_mean, rake_draws

# Start Dash application
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Raking problems', children=[
            dcc.Dropdown(
                id='raking_problem',
                options={'1D': '1-dimensional raking',
                         '2D': '2-dimensional raking',
                         '3D': '3-dimensional raking',
                         'USHD': "USHD team's raking problem"},
                value='2D'
            ),
            html.Img(id='image_raking', height=500)
        ]),
        dcc.Tab(label='Raking results', children=[
            html.I('Enter the number of possible values for X1'),
            dcc.Slider(id='value_I',
                value=3,
                min=2,
                max=10,
                step=1
            ),
            html.I('Enter the number of possible values for X2'),
            dcc.Slider(id='value_J',
                value=5,
                min=2,
                max=10,
                step=1
            ),
            html.H1('Initial observations and corresponding raked values'),
            dvc.Vega(
                id='raked_values_1',
                opt={'renderer': 'svg', 'actions': False},
                spec={},
            ),
            dvc.Vega(
                id='raked_values_2',
                opt={'renderer': 'svg', 'actions': False},
                spec={},
            ),
        ]),
        dcc.Tab(label='Raking uncertainty', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col(
                        dbc.Row([
                            html.H2('Effect of a single observation on all the raked values'),
                            dcc.Slider(id='effect_of_obs_X1',
                                value=1,
                                min=1,
                                step=1
                            ),
                            dcc.Slider(id='effect_of_obs_X2',
                                value=1,
                                min=1,
                                step=1
                            ),
                            dvc.Vega(
                                id='effect_of_1_initial',
                                opt={'renderer': 'svg', 'actions': False},
                                spec={},
                            )],
                            justify='center',
                        )
                    ),
                    dbc.Col(
                        dbc.Row([
                            html.H2('Effect of all observations on a single raked value'),
                            dcc.Slider(id='effect_on_raked_X1',
                                value=1,
                                min=1,
                                step=1
                            ),
                            dcc.Slider(id='effect_on_raked_X2',
                                value=1,
                                min=1,
                                step=1
                            ),
                            dvc.Vega(
                                id='effect_on_1_raked',
                                opt={'renderer': 'svg', 'actions': False},
                                spec={},
                            )],
                            justify='center',
                        )
                    )],
                    justify='center',
                )
            ])
        ]),
        dcc.Tab(label='Comparison with Monte Carlo', children=[
            dbc.Container([
                dbc.Row([
                    html.I('Change the number of samples for the Monte Carlo simulation and compare the results with the delta method.'),
                    dcc.Input(id='num_samples',
                        type='number', 
                        min=10,
                        max=100000,
                        value=1000
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Row([
                            html.H2('Comparison between the means'),
                            dvc.Vega(
                                id='comparison_mean',
                                opt={'renderer': 'svg', 'actions': False},
                                spec={},
                            )],
                            justify='center'
                        )
                    ),
                    dbc.Col(
                        dbc.Row([
                            html.H2('Comparison between the variances'),
                            dvc.Vega(
                                id='comparison_variance',
                                opt={'renderer': 'svg', 'actions': False},
                                spec={},
                            )],
                            justify='center'
                        )
                    )],
                    justify='center',
                )
            ])
       ])
    ])
])

@app.callback(
    Output('image_raking', 'src'),
    Input('raking_problem', 'value'))
def image_raking_path(filename):
    image_path = app.get_asset_url('raking_' + filename + '.png')
    return image_path

@app.callback(
    Output(component_id='raked_values_1', component_property='spec'),
    Output(component_id='raked_values_2', component_property='spec'),
    Input('value_I', 'value'),
    Input('value_J', 'value'))
def plot_raked_uncertainties(I, J):
    (X1, X2, df_raked, df_x, df_y, covariance_mean) = rake_mean(I, J)

    initial = pd.DataFrame({'X1': X1, \
                            'X2': X2, \
                            'variance': np.arange(0.01, 0.01 * ( I * J + 1), 0.01)})

    variance = pd.DataFrame({'X1': X1, \
                             'X2': X2, \
                             'variance': np.diag(covariance_mean)})

    df_obs = df_raked.drop(columns=['raked_values']).rename(columns={'observations': 'Value'})
    df_obs['Type'] = 'Initial'
    df_obs['width'] = 1
    df_obs = df_obs.merge(initial, how='inner', on=['X1', 'X2'])

    df_plot = df_raked.drop(columns=['observations']).rename(columns={'raked_values': 'Value'})
    df_plot['Type'] = 'Raked'
    df_plot['width'] = 2
    df_plot = df_plot.merge(variance, how='inner', on=['X1', 'X2'])

    df_plot = pd.concat([df_obs, df_plot])

    df_plot['Upper'] = df_plot['Value'] + np.sqrt(df_plot['variance'])
    df_plot['Lower'] = df_plot['Value'] - np.sqrt(df_plot['variance'])

    chart1 = plot_initial_raked_1(df_plot)
    chart2 = plot_initial_raked_2(df_plot)
    return (chart1.to_dict(), chart2.to_dict())

@app.callback(
    Output(component_id='effect_of_obs_X1', component_property='max'),
    Input('value_I', 'value'))    
def update_slider_X1(I):
    return I

@app.callback(
    Output(component_id='effect_of_obs_X2', component_property='max'),
    Input('value_J', 'value'))    
def update_slider_X2(J):
    return J 

@app.callback(
    Output(component_id='effect_of_1_initial', component_property='spec'),
    Input('value_I', 'value'),
    Input('value_J', 'value'),
    Input('effect_of_obs_X1', 'value'),
    Input('effect_of_obs_X2', 'value'))
def effect_of_1_initial_on_all_raked(I, J, index1, index2):
    (X1, X2, df_raked, df_x, df_y, covariance_mean) = rake_mean(I, J)
    df_x_loc = df_x.loc[(df_x.X1==index1)&(df_x.X2==index2)]
    max_scale = max(abs(df_x_loc['grad_x'].min()), abs(df_x_loc['grad_x'].max()))
    chart = plot_effect_of_1_initial_on_all_raked(df_x_loc, max_scale, index1, index2)
    return chart.to_dict()

@app.callback(
    Output(component_id='effect_on_raked_X1', component_property='max'),
    Input('value_I', 'value'))    
def update_slider_X1(I):
    return I

@app.callback(
    Output(component_id='effect_on_raked_X2', component_property='max'),
    Input('value_J', 'value'))    
def update_slider_X2(J):
    return J 

@app.callback(
    Output(component_id='effect_on_1_raked', component_property='spec'),
    Input('value_I', 'value'),
    Input('value_J', 'value'),
    Input('effect_on_raked_X1', 'value'),
    Input('effect_on_raked_X2', 'value'))
def effect_of_all_initials_on_1_raked(I, J, index1, index2):
    (X1, X2, df_raked, df_x, df_y, covariance_mean) = rake_mean(I, J)
    df_x_loc = df_x.loc[(df_x.raked_1==index1)&(df_x.raked_2==index2)]
    max_scale = max(abs(df_x_loc['grad_x'].min()), abs(df_x_loc['grad_x'].max()))
    chart = plot_effect_of_all_initials_on_1_raked(df_x_loc, max_scale, index1, index2)
    return chart.to_dict()

@app.callback(
    Output(component_id='comparison_mean', component_property='spec'),
    Output(component_id='comparison_variance', component_property='spec'),
    Input('value_I', 'value'),
    Input('value_J', 'value'),
    Input('num_samples', 'value'))
def compare_mean_variance(I, J, num_samples):
    (X1, X2, df_raked, mean_draws, covariance_mean, covariance_draws) = rake_draws(I, J, num_samples)

    delta_method = pd.DataFrame({'X1': X1, \
                             'X2': X2, \
                             'mean': df_raked['raked_values']})
    all_draws = pd.DataFrame({'X1': X1, \
                          'X2': X2, \
                          'all_draws': mean_draws})
    df_both = delta_method.merge(all_draws, how='inner', on=['X1', 'X2'])
    chart_mean = plot_comparison_mean(df_both)

    delta_method = pd.DataFrame({'X1': X1, \
                             'X2': X2, \
                             'delta_method': np.diag(covariance_mean)})
    all_draws = pd.DataFrame({'X1': X1, \
                          'X2': X2, \
                          'all_draws': np.diag(covariance_draws)})
    df_both = delta_method.merge(all_draws, how='inner', on=['X1', 'X2'])
    chart_variance = plot_comparison_variance(df_both)

    return (chart_mean.to_dict(), chart_variance.to_dict())
    
if __name__ == '__main__':
    app.run(debug=True)

