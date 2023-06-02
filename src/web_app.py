import sys
import os
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'dataset'))

dataset_path = os.path.join(parent_dir, 'dataset', 'questions_small.csv')

data_frame = pd.read_csv(dataset_path)
columns = [
    'question_id',
    'title',
    'language',
    'score',
    'is_answered',
    'accepted_answer_id',
    'view_count',
    'answer_count',
    'creation_date',
    'last_edit_date',
    'last_activity_date',
    'closed_date',
    'closed_reason',
    'owner.account_id',
    'owner.reputation',
    'tags'
]

data_frame = data_frame[columns]

data_frame['accepted_answer_id'] = data_frame['accepted_answer_id'].fillna('N/A')
data_frame['last_edit_date'] = data_frame['last_edit_date'].fillna('N/A')
data_frame['closed_date'] = data_frame['closed_date'].fillna('N/A')
data_frame['closed_reason'] = data_frame['closed_reason'].fillna('N/A')

data_frame.rename(
    columns = {
        'question_id': 'ID',
        'title': 'Title',
        'language': 'Language',
        'score': 'Score',
        'is_answered': 'Is Answered?',
        'accepted_answer_id': 'Answer ID',
        'view_count': 'View Count',
        'answer_count': 'Answer Count',
        'creation_date': 'Creation Date',
        'last_edit_date': 'Edit Date',
        'last_activity_date': 'Last Activity Date',
        'closed_date': 'Closed Date',
        'closed_reason': 'Closed Reason',
        'owner.account_id': 'Owner ID',
        'owner.reputation': 'Owner Reputation',
        'tags': 'Tags'
    },
    inplace = True)

Q1 = data_frame['Score'].quantile(0.25)
Q3 = data_frame['Score'].quantile(0.75)
IQR = Q3 - Q1
filter = (data_frame['Score'] >= Q1 - 1.5 * IQR) & (data_frame['Score'] <= Q3 + 1.5 *IQR)
df_clean = data_frame.loc[filter]  


df_for_dropdown = data_frame[data_frame.columns[1:]]
df_for_dropdown = df_for_dropdown.select_dtypes(include=['number'])

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("StackOverflow Data: Questions"),
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
        html.H1('StackOverflow Data: Analysis'),
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
            options=[{'label': i, 'value': i} for i in df_for_dropdown.columns],
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
        figure = px.scatter(data_frame, x=selected_variable, y='Score', trendline="ols")
    else:
        if chart_type == 'histogram':
            figure = px.histogram(
                data_frame,
                x=selected_variable,
                color='Is Answered?',
                color_discrete_sequence=['purple', 'orange'])
        else:
            figure = px.pie(
                data_frame,
                names='Is Answered?',
                values=selected_variable,
                title='Distribution of ' + selected_variable,
                color_discrete_sequence=['purple', 'orange'])
    
    figure.update_layout(transition_duration=500)
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
