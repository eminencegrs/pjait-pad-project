import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input

data_frame = pd.read_csv('winequality.csv')
data_frame.columns = [col.capitalize() if col != 'pH' else col for col in data_frame.columns]
data_frame.rename(columns={data_frame.columns[0]: ''}, inplace=True)

df_for_dropdown = data_frame[data_frame.columns[1:]]

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Wine Quality Data"),
    html.Div(
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in data_frame.columns],
            data=data_frame.to_dict('records'),
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0},
            style_cell_conditional=[{'if': {'column_id': c}, 'textAlign': 'left'} for c in data_frame.columns],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_table={'maxHeight': '500px', 'overflowY': 'scroll', 'overflowX': 'scroll'}
        )
    ),
    html.Div([
        html.H1('Data Analysis'),
        dcc.Dropdown(
            id='model-choice',
            options=[
                {'label': 'Regression', 'value': 'regression'},
                {'label': 'Classification', 'value': 'classification'},
            ],
            value='regression',
        ),
        dcc.Dropdown(
            id='variable-choice',
            options=[{'label': i, 'value': i} for i in df_for_dropdown.columns if i not in ['pH', 'Target']],
            value=df_for_dropdown.columns[0],
        ),
        dcc.Dropdown(
            id='chart-type',
            options=[
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Pie', 'value': 'pie'},
            ],
            value='pie',
            disabled=True
        ),
        dcc.Graph(id='graph'),
    ]),
])

@app.callback(
    Output('chart-type', 'disabled'),
    Input('model-choice', 'value')
)
def update_chart_dropdown_disabled(selected_model):
    if selected_model == 'regression':
        return True
    else:
        return False

@app.callback(
    Output('graph', 'figure'),
    Input('model-choice', 'value'),
    Input('variable-choice', 'value'),
    Input('chart-type', 'value')
)
def update_figure(selected_model, selected_variable, chart_type):
    if selected_model == 'regression':
        figure = px.scatter(data_frame, x=selected_variable, y='pH', trendline="ols")
    else:
        if chart_type == 'histogram':
            figure = px.histogram(
                data_frame,
                x=selected_variable,
                color='Target',
                color_discrete_sequence=['purple', 'orange'])
        else:
            figure = px.pie(
                data_frame,
                names='Target',
                values=selected_variable,
                title='Distribution of ' + selected_variable,
                color_discrete_sequence=['purple', 'orange'])
    
    figure.update_layout(transition_duration=500)
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
