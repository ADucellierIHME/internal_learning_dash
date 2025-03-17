import altair as alt

def plot_gap(df_schools, df_context, title):
    schools = alt.Chart(df_schools).mark_bar().encode(
        x=alt.X('SchoolName:N',
                axis=alt.Axis(title='School'),
                sort='y'
        ),
        y=alt.Y('gap:Q',
                axis=alt.Axis(title='Gap (%)')
        ),
        color=alt.Color('our', legend=None)
    )
    context = alt.Chart(df_context).mark_rule().encode(
        y=alt.Y('gap:Q'),
        color=alt.Color('SchoolName:N', \
            legend=alt.Legend(title='', orient='top-left'))
    )
    chart = alt.layer(
        schools,
        context
    ).properties(
        title=title,
        width='container'
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16
    ) 
    return chart

def plot_success(df_schools, df_context, title, column):
    schools = alt.Chart(df_schools).mark_bar().encode(
        x=alt.X('SchoolName:N',
                axis=alt.Axis(title='School'),
                sort='y'
        ),
        y=alt.Y(column,
                axis=alt.Axis(title='Percentage meeting Levels 3 or 4 ')
        ),
        color=alt.Color('our', legend=None)
    )
    context = alt.Chart(df_context).mark_rule().encode(
        y=alt.Y(column),
        color=alt.Color('SchoolName:N', \
            legend=alt.Legend(title='', orient='top-left'))
    )
    chart = alt.layer(
        schools,
        context
    ).properties(
        title=title,
        width='container'
    ).configure_axis(
        labelFontSize=16,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    ).configure_legend(
        labelFontSize=16,
        titleFontSize=16
    ) 
    return chart
