import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

def tokens_to_price(input_tokens, output_tokens, price_input, price_output):
    return input_tokens / 1_000_000 * price_input + output_tokens / 1_000_000 * price_output

st.set_page_config(page_title='👩‍💻💸', layout='wide') #💰🤑
st.title("Compensa pagar por uma GPU remota?")

st.sidebar.title('Parametros')

# Sidebar: GPU Info
st.sidebar.header("🖥️ GPU - Configuração")
instance_name = st.sidebar.text_input("Nome da Instânica", value='Oracle-NVIDIA A10')
gpu_hour_price = st.sidebar.number_input("Preço por Hora da GPU (R$)", value=11.10, min_value=0.0)
gpu_memory_total = st.sidebar.number_input("Memória Total da GPU (GB)", value=24.0, min_value=0.1)

# Sidebar: API Info
st.sidebar.header("⚙️ API")
st.sidebar.subheader("Preço por Milhão de Tokens")

model_name = st.sidebar.text_input("Nome do Modelo", value='gpt-4o-mini')
price_input = st.sidebar.number_input("Tokens de Entrada (R$)", value=0.60, min_value=0.0)
price_output = st.sidebar.number_input("Tokens de Saída (R$)", value=2.50, min_value=0.0)

# Sidebar: Estratégia
st.sidebar.header("🧠 Estratégia")
model_memory = st.sidebar.number_input("Memória Ocupada pelo Modelo (GB)", value=12.0, min_value=0.1)
request_time = st.sidebar.number_input("Tempo por Requisição (seg)", value=1.0, min_value=0.1)
input_tokens = st.sidebar.number_input("Tokens de Entrada por Requisição", value=800)
output_tokens = st.sidebar.number_input("Tokens de Saída por Requisição", value=1200)


# Corpo principal: cálculo e gráfico
st.subheader("📊 Análise de Custo por Requisição")

# Cálculo de requisições possíveis por hora na GPU
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

    # Ponto de cruzamento
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



                                   
# st.title("Custo por Requisição: GPU remota vs API")

# st.sidebar.title("")
# st.sidebar.caption("GPU Remota")

# def tokens_to_price(input_tokens, output_tokens, price_input, price_output):
#     return input_tokens / 1_000_000 * price_input + output_tokens / 1_000_000 * price_output

# st.subheader("Parâmetros da API GPT")
# price_input = st.number_input("Preço por milhão de tokens de entrada (R$)", value=0.60, min_value=0.0)
# price_output = st.number_input("Preço por milhão de tokens de saída (R$)", value=2.50, min_value=0.0)

# st.subheader("Parâmetros da GPU Local")
# gpu_hour_price = st.number_input("Preço por hora da GPU (R$)", value=11.10, min_value=0.0)
# gpu_memory_total = st.number_input("Memória total da GPU (GB)", value=24.0, min_value=0.1)
# model_memory = st.number_input("Memória ocupada pelo modelo (GB)", value=12.0, min_value=0.1)
# request_time = st.number_input("Tempo de uma requisição (em segundos)", value=1.0, min_value=0.1)

# st.subheader("Parâmetros do Modelo")
# input_tokens = st.number_input("Tokens de entrada por requisição", value=800)
# output_tokens = st.number_input("Tokens de saída por requisição", value=1200)

# # Cálculo de requisições possíveis por hora na GPU
# if model_memory > gpu_memory_total:
#     st.error("Erro: o modelo ocupa mais memória do que a disponível na GPU.")
# else:
#     concurrent_models = int(gpu_memory_total // model_memory)
#     max_reqs_per_model = int(3600 // request_time)
#     max_reqs_total = concurrent_models * max_reqs_per_model

#     st.markdown(f"""
#     **Modelos simultâneos possíveis na GPU:** {concurrent_models}  
#     **Máx. requisições por modelo por hora:** {max_reqs_per_model}  
#     **Máx. requisições por hora (GPU):** {max_reqs_total}
#     """)

#     # Gerar dados
#     x = list(range(1, max_reqs_total + 1))
#     y_gpu = [gpu_hour_price / r for r in x]
#     y_api = [tokens_to_price(input_tokens, output_tokens, price_input, price_output)] * len(x)

#     df_plot = pd.DataFrame({
#         "Requisições por hora": x + x,
#         "Custo (R$/h)": y_gpu + y_api,
#         "Origem": ["GPU Local"] * len(x) + ["API GPT"] * len(x)
#     })

#     fig = px.line(df_plot, x="Requisições por hora", y="Custo (R$/h)", color="Origem",
#                   log_x=True, log_y=True,
#                   title="Custo por Hora: GPU Local vs API GPT")

#     # Ponto de cruzamento (se houver)
#     diff = np.array(y_gpu) - np.array(y_api)
#     cross_indices = np.where(np.diff(np.sign(diff)))[0]

#     if len(cross_indices) > 0:
#         idx = cross_indices[0]
#         x_cross = x[idx]
#         y_cross = y_gpu[idx]
#         fig.add_scatter(x=[x_cross], y=[y_cross], mode='markers+text',
#                         marker=dict(color='black', size=10),
#                         text=[f"{x_cross} req/h"], textposition="top center",
#                         name="Ponto de cruzamento")

#     st.plotly_chart(fig, use_container_width=True)


# df = pd.read_csv("custos.csv")

# def tokens_to_price(input, output, price_input, price_output):
#     return input / 1_000_000 * price_input + output / 1_000_000 * price_output

# def plot_cost_comparison_plotly(df):
#     tabs = []

#     grupos = df.groupby(['parametros', 'estrategia', 'esquema'], sort=False)
    
#     for i, ((modelo, estrategia, esquema), grupo) in enumerate(grupos):
#         row = grupo.iloc[0]
#         input_tokens = row['Input']
#         output_tokens = row['Output']
#         max_reqs = int(row['Req/Hora max_modelo'])

#         x = list(range(1, max_reqs + 1))
#         y_oracle = [11.10 / i for i in x]
#         y_gpt = [tokens_to_price(input_tokens, output_tokens, 3.54, 8.5) for _ in x]

#         df_plot = pd.DataFrame({
#             "requisições": x + x,
#             "custo": y_oracle + y_gpt,
#             "origem": ["GPU local"] * max_reqs + ["API GPT"] * max_reqs
#         })

#         fig = px.line(
#             df_plot, x="requisições", y="custo", color="origem",
#             log_x=True, log_y=True,
#             labels={"requisições": "Requisições por hora", "custo": "Custo (R$)"},
#             title=f"{modelo} - {estrategia} ({esquema})"
#         )

#         diff = np.array(y_oracle) - np.array(y_gpt)
#         cross_indices = np.where(np.diff(np.sign(diff)))[0]

#         if len(cross_indices) > 0:
#             idx = cross_indices[0]
#             x_cross = x[idx]
#             y_cross = y_oracle[idx]
#             fig.add_trace(go.Scatter(
#                 x=[x_cross], y=[y_cross],
#                 mode="markers+text",
#                 marker=dict(color="black", size=10),
#                 text=[f"{x_cross} req/h"],
#                 textposition="top center",
#                 name="Ponto de cruzamento"
#             ))
#         else:
#             x_red = row['Req/Hora max_modelo']
#             y_red = 11.10 / x_red
#             fig.add_trace(go.Scatter(
#                 x=[x_red], y=[y_red],
#                 mode="markers+text",
#                 marker=dict(color="red", size=10),
#                 text=[f"{int(x_red)} req/h"],
#                 textposition="top center",
#                 name="Limite máximo"
#             ))

#         tabs.append((f"{modelo} - {estrategia} ({esquema})", fig))

#     return tabs

# Streamlit App


# Supondo que o df já esteja carregado
# df = pd.read_csv("...")

# tabs = plot_cost_comparison_plotly(df)

# tab_names = [label for label, _ in tabs]
# tab_objs = st.tabs(tab_names)

# for tab, (_, fig) in zip(tab_objs, tabs):
#     with tab:
#         st.plotly_chart(fig, use_container_width=True)
