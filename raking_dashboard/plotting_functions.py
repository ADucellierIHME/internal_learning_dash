import altair as alt

def plot_initial_raked_1(df):
    bar = alt.Chart(df).mark_errorbar(clip=True, opacity=0.5).encode(
        alt.X('Upper:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='Raked value')),
        alt.X2('Lower:Q'),
        alt.Y('X2:N', axis=alt.Axis(title='X2')),
        color=alt.Color('Type:N', legend=None),
        strokeWidth=alt.StrokeWidth('width:Q', legend=None)
    )
    point = alt.Chart(df).mark_point(
        filled=True
    ).encode(
        alt.X('Value:Q'),
        alt.Y('X2:N'),
        color=alt.Color('Type:N'),
        shape=alt.Shape('Type:N')
    )
    chart = alt.layer(point, bar).resolve_scale(
        shape='independent',
        color='independent'
    ).properties(
        width=200,
        height=100
    ).facet(
        column=alt.Column('X1:N', header=alt.Header(title='X1', titleFontSize=24, labelFontSize=24)),
    ).configure_axis(
        labelFontSize=24,
        titleFontSize=24
    ).configure_legend(
        labelFontSize=24,
        titleFontSize=24
    )
    return chart

def plot_initial_raked_2(df):
    bar = alt.Chart(df).mark_errorbar(clip=True, opacity=0.5).encode(
        alt.X('Upper:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='Raked value')),
        alt.X2('Lower:Q'),
        alt.Y('X1:N', axis=alt.Axis(title='X1')),
        color=alt.Color('Type:N', legend=None),
        strokeWidth=alt.StrokeWidth('width:Q', legend=None)
    )
    point = alt.Chart(df).mark_point(
        filled=True
    ).encode(
        alt.X('Value:Q'),
        alt.Y('X1:N'),
        color=alt.Color('Type:N'),
        shape=alt.Shape('Type:N')
    )
    chart = alt.layer(point, bar).resolve_scale(
        shape='independent',
        color='independent'
    ).properties(
        width=200,
        height=100
    ).facet(
        column=alt.Column('X2:N', header=alt.Header(title='X2', titleFontSize=24, labelFontSize=24)),
    ).configure_axis(
        labelFontSize=24,
        titleFontSize=24
    ).configure_legend(
        labelFontSize=24,
        titleFontSize=24
    )
    return chart

def plot_effect_of_1_initial_on_all_raked(df, max_scale, index1, index2):
    base = alt.Chart(df).encode(
        x=alt.X('raked_1:N', axis=alt.Axis(title='X1')),
        y=alt.Y('raked_2:N', axis=alt.Axis(title='X2')),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color('grad_x:Q',
        scale=alt.Scale(scheme='redblue', domain=[-max_scale, max_scale]),
        legend=alt.Legend(title=['Effect of', 'one obs.']))
    )
    text = base.mark_text(baseline='middle', fontSize=20).encode(
        alt.Text('grad_x:Q', format='.2f')
    )
    chart = alt.layer(heatmap, text
    ).properties(
        title='X1 = ' + str(int(index1)) + ' - X2 = ' + str(int(index2)),
        width=240,
        height=360
    ).configure_title(
        fontSize=20
    ).configure_axis(
        labelFontSize=20,
        titleFontSize=20
    ).configure_legend(
        labelFontSize=20,
        titleFontSize=20
    )
    return chart

def plot_effect_of_all_initials_on_1_raked(df, max_scale, index1, index2):
    base = alt.Chart(df).encode(
        x=alt.X('X1:N', axis=alt.Axis(title='X1')),
        y=alt.Y('X2:N', axis=alt.Axis(title='X2')),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color('grad_x:Q',
        scale=alt.Scale(scheme='redblue', domain=[-max_scale, max_scale]),
        legend=alt.Legend(title=['Effect of', 'all obs.']))
    )
    text = base.mark_text(baseline='middle', fontSize=20).encode(
        alt.Text('grad_x:Q', format='.2f')
    )
    chart = alt.layer(heatmap, text
    ).properties(
        title='X1 = ' + str(int(index1)) + ' - X2 = ' + str(int(index2)),
        width=240,
        height=360
    ).configure_title(
        fontSize=20
    ).configure_axis(
        labelFontSize=20,
        titleFontSize=20
    ).configure_legend(
        labelFontSize=20,
        titleFontSize=20
    )
    return chart

def plot_comparison_mean(df):
    min_x = min(df['mean'].min(), df['all_draws'].min())
    max_x = max(df['mean'].max(), df['all_draws'].max())
    chart = alt.Chart(df).mark_point(size=60).encode(
        x=alt.X('all_draws:Q', scale=alt.Scale(domain=[min_x, max_x], zero=False), axis=alt.Axis(title='Using all draws')),
        y=alt.Y('mean:Q', scale=alt.Scale(domain=[min_x, max_x], zero=False), axis=alt.Axis(title='Using delta method')),
        color=alt.Color('X1:N', legend=alt.Legend(title='X1')),
        shape=alt.Shape('X2:N', legend=alt.Legend(title='X2'))
    ).configure_axis(
        labelFontSize=24,
        titleFontSize=24
    )
    return chart

def plot_comparison_variance(df):
    min_x = min(df['delta_method'].min(), df['all_draws'].min())
    max_x = max(df['delta_method'].max(), df['all_draws'].max())
    chart = alt.Chart(df).mark_point(size=60).encode(
        x=alt.X('all_draws:Q', scale=alt.Scale(domain=[min_x, max_x], zero=False), axis=alt.Axis(title='Using all draws')),
        y=alt.Y('delta_method:Q', scale=alt.Scale(domain=[min_x, max_x], zero=False), axis=alt.Axis(title='Using delta method')),
        color=alt.Color('X1:N', legend=alt.Legend(title='X1')),
        shape=alt.Shape('X2:N', legend=alt.Legend(title='X2'))
    ).configure_axis(
        labelFontSize=24,
        titleFontSize=24
    )
    return chart

