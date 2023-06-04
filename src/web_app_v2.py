import sys
import os
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.graph_objs import Scatter3d, Layout, Figure
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Set working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'dataset'))

# Read data from the file.
dataset_path = os.path.join(parent_dir, 'dataset', 'questions.csv')
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

# Clearing data. 
data_frame = data_frame[columns]
data_frame['accepted_answer_id'] = data_frame['accepted_answer_id'].fillna('N/A')
data_frame['last_edit_date'] = data_frame['last_edit_date'].fillna('N/A')
data_frame['closed_date'] = data_frame['closed_date'].fillna('N/A')
data_frame['closed_reason'] = data_frame['closed_reason'].fillna('N/A')

# Renaming columns to make them readable.
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

# The variable dropdown.
df_for_dropdown = data_frame[['View Count', 'Answer Count', 'Owner Reputation']]

# The language dropdown.
language_values = data_frame['Language'].unique()
language_options = [{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in language_values]

# Configure the Dash application.
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Stack Overflow Data Analysis"),
    html.Div([
        html.H2("Questions Dataset"),
        dash_table.DataTable(
            id = 'table',
            columns = [{"name": i, "id": i} for i in data_frame.columns],
            data = data_frame.to_dict('records'),
            style_data = {'whiteSpace': 'normal', 'height': 'auto'},
            style_cell = {'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0},
            style_cell_conditional = [{'if': {'column_id': c}, 'textAlign': 'left'} for c in data_frame.columns],
            style_header = {'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_table = {'maxHeight': '500px', 'overflowY': 'scroll', 'overflowX': 'scroll'}
        )
    ]),
    html.Div([
        html.H2('Questions Over Time (v1)'),
        dcc.Graph(id='questions-year-graph')
    ]),
    html.Div([
        html.H2('Heatmap'),
        dcc.Graph(id='heatmap-graph')
    ]),
    html.Div([
        html.H2('Questions Over Time (v2)'),
        dcc.Graph(id='timeline-graph'),
    ]),
    html.Div([
        html.H2('Questions by Language Over Time'),
        dcc.Graph(id='language-graph'),
    ]),
    html.Div([
        html.H2('3D Analysis Over Time'),
        dcc.Graph(id='3d-graph'),
    ]),
    html.Div([
        html.H2('Data Analysis'),
        dcc.Dropdown(
            id = 'model-choice',
            options = [
                {'label': 'Regression', 'value': 'regression'},
                {'label': 'Classification', 'value': 'classification'},
            ],
            value = 'regression',
        ),
        dcc.Dropdown(
            id = 'variable-choice',
            options = [{'label': i, 'value': i} for i in df_for_dropdown.columns],
            value = df_for_dropdown.columns[0],
        ),
        dcc.Dropdown(
            id = 'chart-type',
            options = [
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Pie', 'value': 'pie'},
            ],
            value = 'pie',
            disabled = True
        ),
        dcc.Dropdown(id='language-choice', options=language_options, value='All'),
        dcc.Graph(id = 'graph'),
    ]),
    html.Div([
        html.H2('Data Analysis'),
        dcc.Dropdown(
            id='model-choice',
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
        ])
    ])
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
    [Input('model-choice', 'value'),
     Input('variable-choice', 'value'),
     Input('chart-type', 'value'),
     Input('language-choice', 'value')]
)
def update_figure(selected_model, selected_variable, chart_type, selected_language):
    if selected_language == 'All':
        filtered_df = data_frame
    else:
        filtered_df = data_frame[data_frame['Language'] == selected_language]
    
    if selected_model == 'regression':
        figure = px.scatter(filtered_df, x=selected_variable, y='Score', trendline="ols")
    else:
        if chart_type == 'histogram':
            figure = px.histogram(
                filtered_df,
                x = selected_variable,
                color = 'Is Answered?',
                color_discrete_sequence = ['purple', 'orange'])
        else:
            figure = px.pie(
                filtered_df,
                names = 'Is Answered?',
                values = selected_variable,
                title = 'Distribution of ' + selected_variable,
                color_discrete_sequence = ['purple', 'orange'])
    
    figure.update_layout(transition_duration=500)
    return figure

@app.callback(
    Output('timeline-graph', 'figure'),
    [Input('model-choice', 'value')]
)
def update_timeline_graph(selected_model):
    timeline_data_frame = pd.DataFrame(data_frame)
    timeline_data_frame['Creation Date'] = pd.to_datetime(timeline_data_frame['Creation Date'])
    timeline_data_frame['day'] = timeline_data_frame['Creation Date'].dt.to_period('D')
    timeline_data_frame['Is Answered?'] = timeline_data_frame['Is Answered?'].astype(int)
    question_counts = timeline_data_frame.groupby('day')['Is Answered?'].sum()
    question_total = timeline_data_frame.groupby('day').size()
    plot_df = pd.DataFrame({'Answered Count': question_counts, 'Total Count': question_total}).reset_index()
    plot_df['day'] = plot_df['day'].dt.strftime('%Y-%m-%d')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['day'], y=plot_df['Answered Count'], mode='lines', name='Answered Count'))
    fig.add_trace(go.Scatter(x=plot_df['day'], y=plot_df['Total Count'], mode='lines', name='Total Count'))
    fig.update_layout(title='Count of Questions Over Time', xaxis_title='Day', yaxis_title='Count')
    return fig

@app.callback(
    Output('language-graph', 'figure'),
    [Input('model-choice', 'value')]
)
def update_language_graph(selected_model):
    timeline_data_frame = pd.DataFrame(data_frame)
    timeline_data_frame['Creation Date'] = pd.to_datetime(timeline_data_frame['Creation Date'])
    timeline_data_frame['day'] = timeline_data_frame['Creation Date'].dt.to_period('D')
    timeline_data_frame['Is Answered?'] = timeline_data_frame['Is Answered?'].astype(int)
    fig = go.Figure()

    for language in timeline_data_frame['Language'].unique():
        filtered_df = timeline_data_frame[timeline_data_frame['Language'] == language]
        question_counts = filtered_df.groupby('day').size()
        plot_df = pd.DataFrame({'Count': question_counts}).reset_index()
        plot_df['day'] = plot_df['day'].dt.strftime('%Y-%m-%d')
        fig.add_trace(go.Scatter(x=plot_df['day'], y=plot_df['Count'], mode='lines', name=language))

    fig.update_layout(title='Count of Questions by Language Over Time', xaxis_title='Day', yaxis_title='Count')

    return fig

@app.callback(
    Output('3d-graph', 'figure'),
    [Input('model-choice', 'value')]
)
def update_multivariable_graph(selected_model):
    timeline_data_frame = pd.DataFrame(data_frame)
    timeline_data_frame['Creation Date'] = pd.to_datetime(timeline_data_frame['Creation Date'])
    timeline_data_frame['day'] = timeline_data_frame['Creation Date'].dt.to_period('D')
    timeline_data_frame['Is Answered?'] = timeline_data_frame['Is Answered?'].astype(int)
    answered_counts = timeline_data_frame.groupby('day')['Is Answered?'].sum()
    total_counts = timeline_data_frame.groupby('day').size()
    view_counts = timeline_data_frame.groupby('day')['View Count'].sum()
    
    plot_df = pd.DataFrame({'Answered Count': answered_counts, 'Total Count': total_counts, 'View Count': view_counts}).reset_index()
    plot_df['day'] = plot_df['day'].dt.strftime('%Y-%m-%d')

    trace = Scatter3d(
        x=plot_df['day'],
        y=plot_df['Answered Count'],
        z=plot_df['View Count'],
        mode='markers',
        marker=dict(
            size=5,
            color=plot_df['View Count'], 
            colorscale='Viridis',
        )
    )

    layout = Layout(
        margin=dict(l = 0, r = 0, b = 0, t = 0),
        scene=dict(
            xaxis_title='Day',
            yaxis_title='Answered Questions',
            zaxis_title='View Count',
        )
    )

    fig = Figure(data=[trace], layout=layout)

    return fig


### Questions a year.
@app.callback(
    Output('questions-year-graph', 'figure'),
    [Input('model-choice', 'value')]
)
def update_questions_year_plot(n):
    timeline_data_frame = pd.DataFrame(data_frame)
    timeline_data_frame['Creation Date'] = pd.to_datetime(timeline_data_frame['Creation Date'])
    timeline_data_frame['month'] = timeline_data_frame['Creation Date'].dt.to_period('M')
    timeline_data_frame['Is Answered?'] = timeline_data_frame['Is Answered?'].astype(int)
    question_counts = timeline_data_frame.groupby('month')['Is Answered?'].sum()
    question_total = timeline_data_frame.groupby('month').size()
    plot_df = pd.DataFrame({'Answered Count': question_counts, 'Total Count': question_total}).reset_index()
    plot_df['month'] = plot_df['month'].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=plot_df['month'], y=plot_df['Answered Count'], name='Answered Count'))
    fig.add_trace(go.Bar(x=plot_df['month'], y=plot_df['Total Count'], name='Total Count'))
    fig.update_layout(title='Count of Questions Over Time', xaxis_title='month', yaxis_title='Count')
    return fig


### Correlation heatmap
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('model-choice', 'value')]
)
def update_heatmap(n):
    heatmap_data_frame = pd.DataFrame(data_frame)
    heatmap_data_frame['Is Answered?'] = heatmap_data_frame['Is Answered?'].astype(int)
    numerical_data = data_frame[['Score', 'Is Answered?', 'View Count', 'Answer Count', 'Owner Reputation']]
    correlation_matrix = numerical_data.corr()
    fig = px.imshow(correlation_matrix)
    fig.update_layout(title_text='Correlation Heatmap')
    return fig



@app.callback(
    Output('classification-report-label', 'children'),
    Output('classification-report', 'data'),
    Input('model-choice', 'value'),
    Input('test-size-slider', 'value'),
    State('model-choice', 'options')
)
def update_classification_report(model_choice, test_size, model_options):
    selected_label = next((option['label'] for option in model_options if option['value'] == model_choice), '')
    classification_report_label = 'Classification Report [' + selected_label + ']'
    
    chosen_size = test_size / 100
    prediction_data_frame = pd.DataFrame(data_frame)
    prediction_data_frame['Is Answered?'] = prediction_data_frame['Is Answered?'].astype(int)
    feature_columns = ['Score', 'View Count', 'Answer Count']
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
    elif model_choice == 'decision_tree':
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    # Add other model conditions as needed

    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose().reset_index()
    df_report = df_report.round(2)
    classification_report_data = df_report.to_dict('records')
    
    return classification_report_label, classification_report_data



# @app.callback(
#     Output('classification-report-label', 'children'),
#     Input('model-choice', 'value'),
#     State('model-choice', 'options')
# )
# def update_classification_report_label(model_value, model_options):
#     selected_label = next((option['label'] for option in model_options if option['value'] == model_value), '')
#     return 'Classification Report ' + '[' + selected_label + ']'




if __name__ == '__main__':
    app.run_server(debug=True)


### Heatmap:
# The heatmap in this context is used to display the correlation matrix of the numerical data from the DataFrame.
# A correlation matrix is a table showing the correlation coefficients between many variables.
# Each cell in the table shows the correlation between two variables.

# In this case, the heatmap shows the correlation between the following numerical variables:
# 'Score', 'View Count', 'Answer Count', and 'Owner Reputation'. 
# Correlation is a statistical measure that expresses the extent to which two variables are linearly related
# (meaning they change together at a constant rate).
# It's a common tool for understanding the relationship between multiple variables and features in your dataset.

# The correlation coefficient ranges from -1 to 1:
# ==> A correlation of -1 indicates a perfect negative correlation, meaning that as one variable goes up, the other goes down.
# ==> A correlation of +1 indicates a perfect positive correlation, meaning that as one variable goes up, the other goes up.
# ==> A correlation of 0 indicates that there is no linear relationship between the variables.

# In the heatmap, the closer the color of the cell is to 1 (or to -1), 
# the stronger the positive (or negative) correlation between the two variables.
# The closer the color of the cell is to 0, the weaker the correlation.
# Usually, the colors will be represented in a gradient form, so you can visualize the strength of the correlations. 



### Predictions:
# There is being used logistic regression, which is a good choice for binary classification problems.
# There is also used 'liblinear' solver which is appropriate for small datasets and binary classification.

# Our model has an overall accuracy of 80%, which means 
# it correctly predicts whether a question is answered or not 80% of the time. 

# Precision, recall, and F1-score are all reasonable.
# In particular, the model has high precision for class 0 (questions that are not answered),
# meaning when it predicts a question won't be answered, it's right 93% of the time.
# On the other hand, it has high recall for class 1 (questions that are answered),
# meaning it correctly identifies 92% of answered questions.

# The only potential issue here is the difference between class 0 and class 1 results.
# The model performs significantly better for class 0 in terms of precision, and for class 1 in terms of recall.
# This could be due to an imbalance in the data, or it might just reflect the inherent difficulty of the prediction task.

# Considering all the above, we could improve the model using the following options:
# ==> Feature engineering:
#     Create new features that might be relevant for the task. This might involve domain knowledge about the data.
# ==> Model selection:
#     Try out different types of models and see which performs best.
#     Decision trees, random forest, support vector machines, or neural networks could be options.
# ==> Hyperparameter tuning:
#     Experiment with different settings of the model.
#     In the case of logistic regression, we could adjust the regularization strength ('C' parameter) or try a different solver.
# ==> Handling class imbalance:
#     If the classes are imbalanced, we could use techniques such as oversampling the minority class,
#     undersampling the majority class, or using a more advanced method such as SMOTE.
