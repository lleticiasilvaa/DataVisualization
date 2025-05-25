import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def tokens_to_price(input_tokens, output_tokens, price_input, price_output):
    return input_tokens / 1_000_000 * price_input + output_tokens / 1_000_000 * price_output


# Main page content
st.title("📊 Análise de Custo por Requisição")
st.sidebar.markdown("# Parametros")

st.sidebar.header("🖥️ GPU - Configuração")
instance_name = st.sidebar.text_input("Nome da Instânica", value='Oracle-NVIDIA A10')
gpu_hour_price = st.sidebar.number_input("Preço por Hora da GPU (R$)", value=11.10, min_value=0.0)
gpu_memory_total = st.sidebar.number_input("Memória Total da GPU (GB)", value=24.0, min_value=0.1)

st.sidebar.header("⚙️ API")
st.sidebar.subheader("Preço por Milhão de Tokens")

model_name = st.sidebar.text_input("Nome do Modelo", value='gpt-4o-mini')
price_input = st.sidebar.number_input("Tokens de Entrada (R$)", value=0.60, min_value=0.0)
price_output = st.sidebar.number_input("Tokens de Saída (R$)", value=2.50, min_value=0.0)

st.sidebar.header("🧠 Estratégia")
model_memory = st.sidebar.number_input("Memória Ocupada pelo Modelo (GB)", value=12.0, min_value=0.1)
request_time = st.sidebar.number_input("Tempo por Requisição (seg)", value=1.0, min_value=0.1)
input_tokens = st.sidebar.number_input("Tokens de Entrada por Requisição", value=800)
output_tokens = st.sidebar.number_input("Tokens de Saída por Requisição", value=1200)



if model_memory > gpu_memory_total:
    st.error("Erro: o modelo ocupa mais memória do que a disponível na GPU.")
else:
    concurrent_models = int(gpu_memory_total // model_memory)
    max_reqs_per_model = int(3600 // request_time)
    max_reqs_total = concurrent_models * max_reqs_per_model

    st.markdown(f"""
    - **Modelos simultâneos na GPU:** {concurrent_models}  
    - **Máx. requisições por modelo/hora:** {max_reqs_per_model}  
    - **Máx. requisições totais por hora:** {max_reqs_total}
    """)

    x = list(range(1, max_reqs_total + 1))
    y_gpu = [gpu_hour_price / r for r in x]
    y_api = [tokens_to_price(input_tokens, output_tokens, price_input, price_output)] * len(x)

    df_plot = pd.DataFrame({
        "Requisições por hora": x + x,
        "Custo (R$/req)": y_gpu + y_api,
        "Origem": [instance_name] * len(x) + [model_name] * len(x)
    })

    fig = px.line(df_plot, x="Requisições por hora", y="Custo (R$/req)", color="Origem",
                  log_x=True, log_y=True,
                  title="Custo por Requisição: GPU remota vs API")

    diff = np.array(y_gpu) - np.array(y_api)
    cross_indices = np.where(np.diff(np.sign(diff)))[0]

    if len(cross_indices) > 0:
        idx = cross_indices[0]
        x_cross = x[idx]
        y_cross = y_gpu[idx]
        fig.add_scatter(x=[x_cross], y=[y_cross], mode='markers+text',
                        marker=dict(color='black', size=10),
                        text=[f"{x_cross} req/h"], textposition="top center",
                        name="Ponto de cruzamento")

    st.plotly_chart(fig, use_container_width=True)