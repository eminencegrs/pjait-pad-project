import dash
import dash_table

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from datetime import datetime

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from column_renamer import ColumnRenamer
from column_filter import ColumnFilter
from data_cleaner import DataCleaner
from data_enhancer import DataEnhancer
from data_reader import DataReader
from predictor import QuestionsNumberPredictor

# Read data.
data_reader = DataReader('questions.csv')
data_frame = data_reader.read_data()

# Choose the required columns only.
column_filter = ColumnFilter()
data_frame = column_filter.filter_data(data_frame)

# Clear data.
data_cleaner = DataCleaner()
data_frame = data_cleaner.clean_data(data_frame)

# Rename columns to make them readable.
renamer = ColumnRenamer()
data_frame = renamer.rename_columns(data_frame)

# Extend the data frame with additional columns.
enhancer = DataEnhancer()
data_frame = enhancer.enhance(data_frame)

# The variable dropdown.
variable_options = ['View Count', 'Answer Count', 'Owner Reputation', 'Tags Count']

# The variables for 3D graph's dropdown.
variable_options_for_3d_graph = ['Owner Reputation', 'Score', 'Tags Count', 'View Count']

# The language dropdown.
languages = data_frame['Language'].unique()
language_options = [{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in languages]

# Configure the Dash application.
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Stack Overflow Data Analysis"),
    html.Div([
        html.H3("Questions Dataset"),
        html.Hr(),
        dash_table.DataTable(
            id = 'table',
            columns = [{"name": i, "id": i} for i in data_frame.columns],
            data = data_frame.to_dict('records'),
            style_data = {'whiteSpace': 'normal', 'height': 'auto'},
            style_cell = {'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0},
            style_cell_conditional = [{'if': {'column_id': c}, 'textAlign': 'left'} for c in data_frame.columns],
            style_header = {'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_table = {'maxHeight': '500px', 'overflowY': 'scroll', 'overflowX': 'scroll'}
        ),
        html.Div([
            html.H3('Top Languages', style={ 'justify-content': 'center' }),
            html.Hr(),
            html.Div([
                html.Label('Order by:', style={'margin-right': '10px', 'text-align': 'center'}),
                dcc.Dropdown(
                    id = 'sorting-order-options',
                    options = [
                        {'label': 'Language Name', 'value': 'laguange_name'},
                        {'label': 'Total Questions', 'value': 'total_count'},
                        {'label': 'Answered Questions', 'value': 'answered_count'},
                    ],
                    value = 'laguange_name',
                    clearable=False,
                    style={ 'width': '250px', 'justify-content': 'center' }
                ),
            ], style={ 'display': 'flex', 'justify-content': 'center', 'gap': '10px' }),
            dcc.Graph(id='top-languages-bar')
        ])
    ]),
    html.Div([
        html.H3('Questions Over Time'),
        html.Hr(),
        html.Div([
            dcc.Dropdown(
                id='questions-over-time-chart-type',
                options=[
                    {'label': 'Scatter', 'value': 'scatter'},
                    {'label': 'Bar', 'value': 'bar'}
                ],
                value='scatter',
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Dropdown(
                id='data-granularity',
                options=[
                    {'label': 'Hour', 'value': 'hour'},
                    {'label': 'Day', 'value': 'day'},
                    {'label': 'Week', 'value': 'week'},
                    {'label': 'Month', 'value': 'month'}
                ],
                value='month',
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'display': 'flex', 'justify-content': 'center', 'gap': '10px'}),
        dcc.Graph(id='questions-over-time-graph')
    ]),
    html.Div([
        html.H3('Questions by Language Over Time'),
        html.Hr(),
        html.Div([
            dcc.Dropdown(
                id='questions-by-language-over-time-chart-type',
                options=[
                    {'label': 'Scatter', 'value': 'scatter'},
                    {'label': 'Bar', 'value': 'bar'}
                ],
                value='scatter',
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Dropdown(
                id='questions-by-language-over-time-data-granularity',
                options=[
                    {'label': 'Hour', 'value': 'hour'},
                    {'label': 'Day', 'value': 'day'},
                    {'label': 'Week', 'value': 'week'},
                    {'label': 'Month', 'value': 'month'}
                ],
                value='month',
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'display': 'flex', 'justify-content': 'center', 'gap': '10px'}),
        dcc.Graph(id='questions-by-language-over-time-graph')
    ]),
    html.Div([
        html.H3('Correlation Heatmap'),
        html.Hr(),
        dcc.Graph(id='heatmap'),
        dcc.Interval(
            id='interval-component',
            interval = 60 * 1000,
            n_intervals = 0
        )
    ]),
    html.Div([
        html.H3('Questions in 3D'),
        html.Hr(),
        html.Div([
            dcc.Dropdown(
                id='3d-graph-y-axis-dropdown',
                options = [{'label': i, 'value': i} for i in variable_options_for_3d_graph],
                value = variable_options_for_3d_graph[0],
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Dropdown(
                id='3d-graph-z-axis-dropdown',
                options = [{'label': i, 'value': i} for i in variable_options_for_3d_graph],
                value = variable_options_for_3d_graph[1],
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Dropdown(
                id='3d-graph-data-granularity',
                options=[
                    {'label': 'Hour', 'value': 'hour'},
                    {'label': 'Day', 'value': 'day'},
                    {'label': 'Week', 'value': 'week'},
                    {'label': 'Month', 'value': 'month'}
                ],
                value='month',
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'display': 'flex', 'justify-content': 'center', 'gap': '10px'}),
        dcc.Graph(id='3d-graph'),
    ]),
    html.Div([
        html.H3('Data Analysis: Distribution'),
        html.Hr(),
        dcc.Dropdown(
            id = 'data-distribution-model-choice',
            options = [
                {'label': 'Regression', 'value': 'regression'},
                {'label': 'Classification', 'value': 'classification'},
            ],
            value = 'regression',
        ),
        dcc.Dropdown(
            id = 'data-distribution-variable-choice',
            options = [{'label': i, 'value': i} for i in variable_options],
            value = variable_options[0],
        ),
        dcc.Dropdown(
            id = 'data-distribution-chart-type',
            options = [
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Pie', 'value': 'pie'},
            ],
            value = 'pie',
            disabled = True
        ),
        dcc.Dropdown(id='data-distribution-language-choice', options=language_options, value='All'),
        dcc.Graph(id = 'data-distribution-graph'),
    ]),
    html.Div([
        html.H3('Data Analysis: Classification'),
        html.Hr(),
        dcc.Dropdown(
            id='regression-model-choice',
            options=[
                {'label': 'Logistic Regression (liblinear)', 'value': 'lr_liblinear'},
                {'label': 'Logistic Regression (newton-cg)', 'value': 'lr_newton'},
                {'label': 'Logistic Regression (lbfgs)', 'value': 'lr_lbfgs'},
                {'label': 'Logistic Regression (sag)', 'value': 'lr_sag'},
                {'label': 'Logistic Regression (saga)', 'value': 'lr_saga'},
                {'label': 'Decision Tree', 'value': 'decision_tree'},
                {'label': 'Random Forest', 'value': 'random_forest'},
                {'label': 'Support Vector Machine', 'value': 'svm'},
            ],
            value='lr_liblinear',
        ),
        html.Div('Test Size', style={'margin':'10px 0'}),
        dcc.Slider(
            id='test-size-slider',
            min=10, max=90, step=5, value=20,
            marks={i: f'{i}%' for i in range(10, 100, 10)},
        ),
        html.Div([
            html.H3('Classification Report', id='classification-report-label'),
            dash_table.DataTable(id='classification-report', data=[])
        ]),
        html.Div([
            html.H3('Prediction Results'),
            html.Hr(),
            html.Div([
                dcc.Dropdown(
                    id='prediction-results-language-choice', 
                    options=language_options, 
                    value='All',
                    style={'width': '150px'}
                )
            ], style={'display': 'flex', 'justify-content': 'center', 'gap': '10px' }),
            dcc.Graph(id='prediction-results-graph'),
        ]),
    ])
])

### Top Languages (bar)
@app.callback(
    Output('top-languages-bar', 'figure'),
    Input('sorting-order-options', 'value')
)
def update_top_languages_bar(sorting_order):
    languages_df = pd.DataFrame(data_frame)
    languages_df['Is Answered?'] = languages_df['Is Answered?'].astype(int)
    total_count = languages_df.groupby('Language').size()
    answered_count = languages_df.groupby('Language')['Is Answered?'].sum()
    open_count = total_count - answered_count
    
    plot_df = pd.DataFrame({'Answered Count': answered_count, 'Total Count': total_count, 'Open Count': open_count}).reset_index()

    if sorting_order == 'language_name':
        plot_df.sort_values('Language', inplace=True)
    elif sorting_order == 'total_count':
        plot_df.sort_values('Total Count', inplace=True)
    elif sorting_order == 'answered_count':
        plot_df.sort_values('Answered Count', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=plot_df['Language'], y=plot_df['Total Count'], name='All', marker_color='#466ba3'))
    fig.add_trace(go.Bar(x=plot_df['Language'], y=plot_df['Answered Count'], name='Answered Questions', marker_color='#328570'))
    fig.add_trace(go.Bar(x=plot_df['Language'], y=plot_df['Open Count'], name='Open Questions', marker_color='#cc4b27'))
    fig.update_layout(xaxis_title='Language', yaxis_title='Count')
    
    return fig



### Questions over time.
@app.callback(
    Output('questions-over-time-graph', 'figure'),
    [Input('questions-over-time-chart-type', 'value'),
     Input('data-granularity', 'value')]
)
def update_questions_over_time_graph(chart_type, data_granularity):
    timeline_df = pd.DataFrame(data_frame)
    timeline_df['Creation Date'] = pd.to_datetime(timeline_df['Creation Date'])

    if data_granularity == 'hour':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('H')
    elif data_granularity == 'day':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('D')
    elif data_granularity == 'week':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('W')
    elif data_granularity == 'month':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('M')

    answered_count = timeline_df.groupby('time_period')['Is Answered?'].sum()
    total_count = timeline_df.groupby('time_period').size()
    plot_df = pd.DataFrame({'Answered Count': answered_count, 'Total Count': total_count}).reset_index()
    plot_df['time_period'] = plot_df['time_period'].astype(str)

    fig = go.Figure()

    if chart_type == 'scatter':
        fig.add_trace(go.Scatter(x=plot_df['time_period'], y=plot_df['Answered Count'], mode='lines', name='Answered Count'))
        fig.add_trace(go.Scatter(x=plot_df['time_period'], y=plot_df['Total Count'], mode='lines', name='Total Count'))
    else:
        fig.add_trace(go.Bar(x=plot_df['time_period'], y=plot_df['Answered Count'], name='Answered Count'))
        fig.add_trace(go.Bar(x=plot_df['time_period'], y=plot_df['Total Count'], name='Total Count'))

    if data_granularity == 'hour':
        xaxis_title = 'Hour'
    elif data_granularity == 'day':
        xaxis_title = 'Day'
    elif data_granularity == 'week':
        xaxis_title = 'Week'
    elif data_granularity == 'month':
        xaxis_title = 'Month'

    fig.update_layout(xaxis_title=xaxis_title, yaxis_title='Count')

    return fig



### Questions by language over time.
@app.callback(
    Output('questions-by-language-over-time-graph', 'figure'),
    [Input('questions-by-language-over-time-chart-type', 'value'),
     Input('questions-by-language-over-time-data-granularity', 'value')]
)
def update_questions_by_language_over_time_graph(chart_type, data_granularity):
    timeline_df = pd.DataFrame(data_frame)
    timeline_df['Creation Date'] = pd.to_datetime(timeline_df['Creation Date'])

    if data_granularity == 'hour':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('H')
    elif data_granularity == 'day':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('D')
    elif data_granularity == 'week':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('W')
    elif data_granularity == 'month':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('M')

    figure = go.Figure()

    for language in timeline_df['Language'].unique():
        filtered_df = timeline_df[timeline_df['Language'] == language]
        total_count = filtered_df.groupby('time_period').size()
        plot_df = pd.DataFrame({'Count': total_count}).reset_index()
        plot_df['time_period'] = plot_df['time_period'].astype(str)
        if chart_type == 'scatter':
            figure.add_trace(go.Scatter(x=plot_df['time_period'], y=plot_df['Count'], mode='lines', name=language))
        else:
            figure.add_trace(go.Bar(x=plot_df['time_period'], y=plot_df['Count'], name=language))

    if data_granularity == 'hour':
        xaxis_title = 'Hour'
    elif data_granularity == 'day':
        xaxis_title = 'Day'
    elif data_granularity == 'week':
        xaxis_title = 'Week'
    elif data_granularity == 'month':
        xaxis_title = 'Month'

    figure.update_layout(xaxis_title=xaxis_title, yaxis_title='Count')

    return figure



### Correlation heatmap.
@app.callback(
    Output('heatmap', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_heatmap(fake_param):
    heatmap_df = pd.DataFrame(data_frame)
    heatmap_df['Is Answered?'] = heatmap_df['Is Answered?'].astype(int)
    chosen_columns = data_frame[['Score', 'Is Answered?', 'View Count', 'Answer Count', 'Owner Reputation', 'Tags Count']]
    correlation_matrix = chosen_columns.corr()

    figure = px.imshow(correlation_matrix)
    figure.update_layout(
        annotations=[
            dict(
                x=i,
                y=j,
                text=f'{correlation_matrix.iloc[i, j]:.2f}',
                showarrow=False,
                font=dict(color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
            )
            for i in range(correlation_matrix.shape[0])
            for j in range(correlation_matrix.shape[1])
        ]
    )

    return figure



### 3D graph.
@app.callback(
    Output('3d-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('3d-graph-data-granularity', 'value'),
     Input('3d-graph-y-axis-dropdown', 'value'),
     Input('3d-graph-z-axis-dropdown', 'value')]
)
def update_3d_graph(fake_param, data_granularity, y_axis, z_axis):
    timeline_df = pd.DataFrame(data_frame)
    timeline_df['Creation Date'] = pd.to_datetime(timeline_df['Creation Date'])
    
    if data_granularity == 'hour':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('H')
    elif data_granularity == 'day':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('D')
    elif data_granularity == 'week':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('W')
    elif data_granularity == 'month':
        timeline_df['time_period'] = timeline_df['Creation Date'].dt.to_period('M')

    timeline_df['time_period'] = timeline_df['time_period'].astype(str)

    y_values = None
    if y_axis == 'Owner Reputation':
        y_values = timeline_df['Owner Reputation']
    elif y_axis == 'Score':
        y_values = timeline_df['Score']
    elif y_axis == 'Tags Count':
        y_values = timeline_df['Tags Count']
    elif y_axis == 'View Count':
        y_values = timeline_df['View Count']

    z_values = None
    if z_axis == 'Owner Reputation':
        z_values = timeline_df['Owner Reputation']
    elif z_axis == 'Score':
        z_values = timeline_df['Score']
    elif z_axis == 'Tags Count':
        z_values = timeline_df['Tags Count']
    elif z_axis == 'View Count':
        z_values = timeline_df['View Count']

    plot_df = pd.DataFrame({
        'Time Period': timeline_df['time_period'],
        'Y Axis': y_values,
        'Z Axis': z_values
    })

    trace = go.Scatter3d(
        x=plot_df['Time Period'],
        y=plot_df['Y Axis'],
        z=plot_df['Z Axis'],
        mode='markers',
        marker=dict(
            size=5,
            #color='blue',
            color=plot_df['Y Axis'],
            colorscale='Viridis',
        )
    )

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='Time Period',
            yaxis_title=y_axis,
            zaxis_title=z_axis
        )
    )

    figure = go.Figure(data=[trace], layout=layout)

    return figure



### Data Analysis: Distribution.
@app.callback(
    Output('data-distribution-chart-type', 'disabled'),
    Input('data-distribution-model-choice', 'value')
)
def update_chart_dropdown_disabled(selected_model):
    if selected_model == 'regression':
        return True
    else:
        return False

@app.callback(
    Output('data-distribution-graph', 'figure'),
    [Input('data-distribution-model-choice', 'value'),
     Input('data-distribution-variable-choice', 'value'),
     Input('data-distribution-chart-type', 'value'),
     Input('data-distribution-language-choice', 'value')]
)
def update_figure(selected_model, selected_variable, chart_type, selected_language):
    if selected_language == 'All':
        filtered_df = data_frame
    else:
        filtered_df = data_frame[data_frame['Language'] == selected_language]
    
    # This trendline, specified as "ols" (Ordinary Least Squares),
    # is a line of best fit that minimizes the sum of the squared residuals in the dataset,
    # effectively trying to capture the trend in the data.
    # This can help in identifying the general relationship between the two variables.
    
    if selected_model == 'regression':
        figure = px.scatter(filtered_df, x=selected_variable, y='Score', trendline="ols")
    else:
        if chart_type == 'histogram':
            figure = px.histogram(
                filtered_df,
                x = selected_variable,
                color = 'Is Answered?',
                color_discrete_sequence = ['#2785cc', '#cc4b27'])
        else:
            figure = px.pie(
                filtered_df,
                names = 'Is Answered?',
                values = selected_variable,
                title = 'Distribution of ' + selected_variable,
                color_discrete_sequence = ['#2785cc', '#cc4b27'])
    
    figure.update_layout(transition_duration=500)
    return figure



### Data Analysis: Classification.
@app.callback(
    Output('classification-report-label', 'children'),
    Output('classification-report', 'data'),
    Input('regression-model-choice', 'value'),
    Input('test-size-slider', 'value'),
    State('regression-model-choice', 'options')
)
def update_classification_report(model_choice, test_size, model_options):
    selected_label = next((option['label'] for option in model_options if option['value'] == model_choice), '')
    classification_report_label = 'Classification Report [' + selected_label + ']'
    
    chosen_size = test_size / 100
    prediction_data_frame = pd.DataFrame(data_frame)
    prediction_data_frame['Is Answered?'] = prediction_data_frame['Is Answered?'].astype(int)
    feature_columns = ['Score', 'View Count', 'Answer Count', 'Owner Reputation', 'Tags Count']
    X = prediction_data_frame[feature_columns]
    y = data_frame['Is Answered?'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=chosen_size, random_state=27)

    if model_choice == 'lr_liblinear':
        classifier = LogisticRegression(solver='liblinear', random_state=27)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    elif model_choice == 'lr_newton':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        classifier = LogisticRegression(solver='newton-cg', random_state=27, max_iter=10000)
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
    elif model_choice == 'lr_lbfgs':
        classifier = LogisticRegression(solver='lbfgs', random_state=27)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    elif model_choice == 'lr_sag':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        classifier = LogisticRegression(solver='sag', random_state=27, max_iter=10000, C=10)
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
    elif model_choice == 'lr_saga':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        classifier = LogisticRegression(solver='saga', random_state=27, max_iter=10000, C=10)
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
    elif model_choice == 'decision_tree':
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    elif model_choice == 'random_forest':
        classifier = RandomForestClassifier(random_state=27)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    elif model_choice == 'svm':
        classifier = svm.SVC(random_state=27)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose().reset_index()
    df_report = df_report.round(2)
    classification_report_data = df_report.to_dict('records')

    return classification_report_label, classification_report_data



### Prediction: The Number of Questions in Future.
@app.callback(
    Output('prediction-results-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('prediction-results-language-choice', 'value')]
)
def update_prediction_results(n, chosen_language):
    df = QuestionsNumberPredictor.fetch_prediction_results(data_frame)

    if chosen_language != 'All':
        df = df[df['Language'] == chosen_language]
    
    grouped = df.groupby('Language')

    fig = go.Figure()

    for name, group in grouped:
        fig.add_trace(go.Scatter(x=group['date'], y=group['Actual'], mode='lines', name=f'{name} - Actual'))
        fig.add_trace(go.Scatter(x=group['date'], y=group['Predicted'], mode='lines', name=f'{name} - Predicted'))

    fig.update_layout(title='Prediction results over time', xaxis_title='Date', yaxis_title='Count')

    return fig



### Run web app.
if __name__ == '__main__':
    app.run_server(debug=True)
