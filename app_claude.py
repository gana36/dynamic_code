import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
from flask import Flask, request, jsonify
import requests
import json
import numpy as np
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Initialize Dash app and Flask server
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load initial data and set global variables
df = pd.read_csv("global_terror.csv")
token = 'pk.eyJ1IjoiZGVldmVzaGl6bSIsImEiOiJja2VrenI3amEwZXJtMnNwd242YW42ajJpIn0.FqsxGZu5Q5vUAjNRIq6IvA'

# Convert numeric columns to Python native types
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].astype(float)

@server.route('/update-data', methods=['POST'])
def update_data():
    """Endpoint to handle incoming HTTP requests and filter data."""
    try:
        data = request.json

        # Extract filter criteria from JSON data
        region = data.get("region", [])
        country = data.get("country", [])
        attack_type = data.get("attack_type", [])
        year_start = data.get("year_start", df['iyear'].min())
        year_end = data.get("year_end", df['iyear'].max())

        # Apply filters to the DataFrame
        filtered_df = df[df['iyear'].between(year_start, year_end)]
        if region:
            filtered_df = filtered_df[filtered_df['region_txt'].isin(region)]
        if country:
            filtered_df = filtered_df[filtered_df['country_txt'].isin(country)]
        if attack_type:
            filtered_df = filtered_df[filtered_df['attacktype1_txt'].isin(attack_type)]

        # Convert to dictionary and serialize with custom encoder
        response_data = filtered_df.to_dict(orient='records')
        return json.dumps(response_data, cls=NumpyEncoder)

    except Exception as e:
        print(f"Error in update_data: {e}")
        return jsonify([])

def create_app_ui():
    """Define the layout of the Dash app."""
    return html.Div([
        html.H1("Terrorism Analysis and Insights", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        # Filters Section
        html.Div([
            html.Div([
                html.Label("Select Region(s)"),
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[{'label': i, 'value': i} for i in sorted(df['region_txt'].unique())],
                    multi=True,
                    placeholder="Select Regions"
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Select Country(s)"),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[],
                    multi=True,
                    placeholder="Select Countries"
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Select Attack Type(s)"),
                dcc.Dropdown(
                    id='attack-type-dropdown',
                    options=[{'label': i, 'value': i} for i in sorted(df['attacktype1_txt'].unique())],
                    multi=True,
                    placeholder="Select Attack Types"
                )
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Select Year Range"),
            dcc.RangeSlider(
                id='year-slider',
                min=df['iyear'].min(),
                max=df['iyear'].max(),
                value=[df['iyear'].min(), df['iyear'].max()],
                marks={str(year): str(year) for year in range(int(df['iyear'].min()), int(df['iyear'].max())+1, 5)},
                step=1
            )
        ], style={'marginBottom': '30px'}),
        
        # Store components for data
        dcc.Store(id='filtered-data-store'),
        dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
        
        # Loading spinner and graph container
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id='graph-container')
        ),
        
        # Statistics Section
        html.Div(id='statistics-container', style={'marginTop': '30px'})
    ])

@app.callback(
    Output('country-dropdown', 'options'),
    Input('region-dropdown', 'value')
)
def update_country_dropdown(selected_regions):
    if not selected_regions:
        return [{'label': i, 'value': i} for i in sorted(df['country_txt'].unique())]
    filtered_countries = df[df['region_txt'].isin(selected_regions)]['country_txt'].unique()
    return [{'label': i, 'value': i} for i in sorted(filtered_countries)]

@app.callback(
    Output('filtered-data-store', 'data'),
    [Input('interval-component', 'n_intervals'),
     Input('region-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('attack-type-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def fetch_filtered_data(n_intervals, regions, countries, attack_types, years):
    """Fetch filtered data based on user selections."""
    try:
        response = requests.post('http://127.0.0.1:8080/update-data', json={
            "region": regions if regions else [],
            "country": countries if countries else [],
            "attack_type": attack_types if attack_types else [],
            "year_start": years[0] if years else df['iyear'].min(),
            "year_end": years[1] if years else df['iyear'].max()
        })
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return []

@app.callback(
    [Output('graph-container', 'children'),
     Output('statistics-container', 'children')],
    Input('filtered-data-store', 'data')
)
def update_visualizations(data):
    """Update the map and statistics based on the filtered data."""
    if not data:
        return html.Div("No data available for the selected filters."), None

    # Convert the JSON data into a DataFrame
    new_df = pd.DataFrame(data)

    # Create the map figure
    fig = px.scatter_mapbox(
        new_df,
        lat='latitude',
        lon='longitude',
        hover_data=['region_txt', 'country_txt', 'provstate', 'city', 
                   'attacktype1_txt', 'nkill', 'iyear'],
        zoom=1,
        color='attacktype1_txt',
        height=650,
        title='Global Terrorism Incidents'
    )

    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_accesstoken=token,
        autosize=True,
        margin=dict(l=0, r=0, b=25, t=40),
        template='plotly_dark'
    )

    # Calculate statistics
    total_incidents = len(new_df)
    total_casualties = new_df['nkill'].sum()
    most_affected_country = new_df['country_txt'].value_counts().index[0]
    most_common_attack = new_df['attacktype1_txt'].value_counts().index[0]

    statistics = html.Div([
        html.H3("Key Statistics", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H4("Total Incidents"),
                html.P(f"{total_incidents:,}")
            ], className='stat-box'),
            html.Div([
                html.H4("Total Casualties"),
                html.P(f"{total_casualties:,.0f}")
            ], className='stat-box'),
            html.Div([
                html.H4("Most Affected Country"),
                html.P(most_affected_country)
            ], className='stat-box'),
            html.Div([
                html.H4("Most Common Attack Type"),
                html.P(most_common_attack)
            ], className='stat-box')
        ], style={'display': 'flex', 'justifyContent': 'space-around'})
    ])

    return dcc.Graph(figure=fig), statistics

# Set the layout of the app
app.layout = create_app_ui()

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8080)