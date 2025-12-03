import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dash
from dash import html, dcc, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
import os

# ------------------------
# Carregando dados
# ------------------------
DATA_PATH = "data/clientes.csv"
df = pd.read_csv(DATA_PATH)

le = LabelEncoder()
df['mensagem_encoded'] = le.fit_transform(df['mensagem'])

X = df[['mensagem_encoded']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
df['prob_churn'] = model.predict_proba(X[['mensagem_encoded']])[:,1]

CRITICO = 0.7
df['alerta'] = df['prob_churn'] >= CRITICO

# ------------------------
# App Dash
# ------------------------
app = dash.Dash(__name__)
app.title = "Churn Dashboard Profissional"

# ------------------------
# Estilo neutro e corporativo
# ------------------------
BODY_STYLE = {
    'fontFamily':'Arial, sans-serif', 
    'backgroundColor':'#e0e0e0',  # fundo cinza
    'color':'#333333',
    'padding':'20px'
}

CARD_STYLE = {
    'display':'inline-block',
    'padding':'20px',
    'margin':'10px',
    'borderRadius':'10px',
    'width':'23%',
    'backgroundColor':'#ffffff',
    'boxShadow':'0 2px 8px rgba(0,0,0,0.15)',
    'textAlign':'center',
    'transition':'all 0.5s ease'
}

BUTTON_STYLE = {
    'backgroundColor':'#007bff',
    'border':'none',
    'padding':'10px 20px',
    'color':'#ffffff',
    'fontWeight':'bold',
    'cursor':'pointer',
    'borderRadius':'5px',
    'transition':'all 0.3s ease'
}

H1_STYLE = {
    'textAlign':'center',
    'color':'#007bff',
    'fontSize':'36px',
    'marginBottom':'30px'
}

# ------------------------
# Layout
# ------------------------
app.layout = html.Div(style=BODY_STYLE, children=[
    html.H1("Churn Dashboard Profissional", style=H1_STYLE),

    html.Div([
        html.Div(id='kpi_total', style=CARD_STYLE),
        html.Div(id='kpi_churn', style=CARD_STYLE),
        html.Div(id='kpi_media', style=CARD_STYLE),
        html.Div(id='kpi_critico', style=CARD_STYLE)
    ], style={'textAlign':'center'}),

    html.Hr(),

    html.Div([
        html.Label("Filtrar por probabilidade mínima de churn:", style={'fontSize':'16px','marginRight':'20px'}),
        dcc.Slider(
            id='prob_slider',
            min=0, max=1, step=0.01, value=0,
            marks={0:'0',0.25:'0.25',0.5:'0.5',0.75:'0.75',1:'1'}
        )
    ], style={'padding':'20px'}),

    dcc.Graph(id='grafico_barras'),

    html.Div([
        html.H3("Tabela de Clientes"),
        dcc.Graph(id='tabela_clientes')
    ]),

    html.Hr(),

    # ------------------------
    # Nova Seção Teste e Exportação
    # ------------------------
    html.Div([
        html.Div([
            html.H3("Teste de Churn em Tempo Real", style={'marginBottom':'10px'}),
            dcc.Input(
                id='mensagem_input', 
                type='text', 
                placeholder="Digite mensagem do cliente", 
                style={'width':'100%', 'padding':'10px', 'borderRadius':'5px', 'border':'1px solid #ccc', 'marginBottom':'10px'}
            ),
            html.Button("Prever Churn", id='btn_prever', n_clicks=0, style={**BUTTON_STYLE, 'width':'100%'}),
            html.Div(id='resultado_churn', style={'marginTop':'10px','fontSize':'18px','fontWeight':'bold','textAlign':'center'})
        ], style={
            'backgroundColor':'#f0f0f0',  # card cinza claro
            'padding':'20px',
            'borderRadius':'10px',
            'boxShadow':'0 2px 8px rgba(0,0,0,0.1)',
            'maxWidth':'500px',
            'margin':'0 auto'
        }),

        html.Div([
            html.Button("Exportar CSV Filtrado", id="btn_csv", style={**BUTTON_STYLE, 'width':'100%'}),
            dcc.Download(id="download_csv")
        ], style={
            'marginTop':'20px',
            'maxWidth':'500px',
            'margin':'20px auto 0 auto',
            'textAlign':'center'
        })
    ], style={'marginTop':'30px'}),

    dcc.Interval(id='intervalo_kpi', interval=2000, n_intervals=0)
])

# ------------------------
# Funções utilitárias
# ------------------------
def contador_animado(titulo, valor):
    valor_str = "{:,}".format(int(valor))
    return html.Div([
        html.Div(titulo, style={'fontSize':'14px','marginBottom':'5px'}),
        html.Div(valor_str, style={'fontSize':'26px','fontWeight':'bold','fontFamily':'Courier New, monospace'})
    ])

# ------------------------
# Callbacks
# ------------------------
@app.callback(
    Output('kpi_total','children'),
    Output('kpi_churn','children'),
    Output('kpi_media','children'),
    Output('kpi_critico','children'),
    Input('intervalo_kpi','n_intervals')
)
def atualizar_kpis(n):
    total = len(df)
    churn_total = df['churn'].sum()
    media_churn = df['prob_churn'].mean()
    critico = df['alerta'].sum()
    return (
        html.Div(contador_animado("Total de Clientes", total), style=CARD_STYLE),
        html.Div(contador_animado("Clientes com Churn", churn_total), style=CARD_STYLE),
        html.Div(contador_animado("Probabilidade Média (%)", media_churn*100), style=CARD_STYLE),
        html.Div(contador_animado("Clientes Críticos", critico), style=CARD_STYLE)
    )

@app.callback(
    Output('grafico_barras', 'figure'),
    Input('prob_slider', 'value')
)
def atualizar_grafico(prob_min):
    df_filtrado = df[df['prob_churn'] >= prob_min]
    fig = px.bar(df_filtrado, x='cliente_id', y='prob_churn', color='alerta',
                 color_discrete_map={True:'#dc3545', False:'#28a745'},
                 labels={'cliente_id':'Cliente', 'prob_churn':'Probabilidade Churn'})
    fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#333333',
                      title={'text':'Probabilidade de Churn por Cliente','font_size':20,'x':0.5},
                      xaxis_tickangle=-45, transition={'duration':500})
    fig.update_traces(marker_line_color='black', marker_line_width=1, opacity=0.9)
    return fig

@app.callback(
    Output('tabela_clientes', 'figure'),
    Input('prob_slider', 'value')
)
def atualizar_tabela(prob_min):
    df_filtrado = df[df['prob_churn'] >= prob_min]
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_filtrado.columns),
                    fill_color='#007bff',
                    font=dict(color='white', size=12)),
        cells=dict(values=[df_filtrado[col] for col in df_filtrado.columns],
                   fill_color='#f8f9fa',
                   font=dict(color='black', size=11)))
    ])
    fig.update_layout(title={'text':'Clientes Filtrados','font_size':18,'x':0.5})
    return fig

@app.callback(
    Output("download_csv", "data"),
    Input("btn_csv", "n_clicks"),
    Input('prob_slider', 'value'),
    prevent_initial_call=True
)
def exportar_csv(n_clicks, prob_min):
    df_filtrado = df[df['prob_churn'] >= prob_min]
    return dcc.send_data_frame(df_filtrado.to_csv, "clientes_filtrados.csv")

@app.callback(
    Output('resultado_churn', 'children'),
    Input('btn_prever', 'n_clicks'),
    State('mensagem_input', 'value')
)
def prever_churn(n_clicks, mensagem):
    if n_clicks > 0 and mensagem:
        try:
            msg_enc = le.transform([mensagem])[0]
        except:
            msg_enc = 0
        prob = model.predict_proba([[msg_enc]])[0][1]
        if prob >= CRITICO:
            return f"⚠️ Alta probabilidade de churn: {prob:.2f}"
        else:
            return f"Baixa probabilidade de churn: {prob:.2f} ✅"
    return ""

# ------------------------
# Servidor
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
