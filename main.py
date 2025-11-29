import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Ler CSV apenas uma vez
df = pd.read_csv("data/clientes.csv")

# Transformar texto em números
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['mensagem'])
y = df['churn']

# Treinar modelo uma única vez
model = RandomForestClassifier()
model.fit(X, y)

# Prever probabilidade de churn
df['probabilidade_churn'] = model.predict_proba(X)[:,1]

# Criar app Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sistema de Churn - IA"),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Atualiza a cada 5 segundos
        n_intervals=0
    ),
    dcc.Graph(id='graph-churn')
])

@app.callback(
    Output('graph-churn', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    # Ler CSV novamente para novos clientes
    df_new = pd.read_csv("data/clientes.csv")

    # Manter as probabilidades já calculadas para clientes existentes
    df_new = df_new.merge(df[['cliente_id', 'probabilidade_churn']], 
                          on='cliente_id', how='left')
    
    # Calcular probabilidade apenas para novos clientes
    mask = df_new['probabilidade_churn'].isna()
    if mask.any():
        X_new = vectorizer.transform(df_new.loc[mask, 'mensagem'])
        df_new.loc[mask, 'probabilidade_churn'] = model.predict_proba(X_new)[:,1]

    # Criar gráfico
    fig = px.bar(df_new, x='cliente_id', y='probabilidade_churn',
                 hover_data=['mensagem', 'churn'],
                 labels={'probabilidade_churn':'Probabilidade de Churn'},
                 title='Clientes em Risco')
    return fig

if __name__ == '__main__':
    app.run(debug=True)