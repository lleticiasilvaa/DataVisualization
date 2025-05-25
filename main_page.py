import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

if "gpus" not in st.session_state:
    st.session_state.gpus = [
        {
            "name": 'Oracle-NVIDIA-A10',
            "memory": 24,
            "hour_price": 11.10
        }
    ]

if "apis" not in st.session_state:
    st.session_state.apis = [
        {
            "name": 'gpt-4o-mini',
            "input_tokens": 0.15,
            "output_tokens": 0.60,
            "ex_all_spider-dev": 74.3,
            "ex_all_cnpj": 83.0
        },
        {
            "name": 'gpt-4o',
            "input_tokens": 2.5,
            "output_tokens": 10.00,
            "ex_all_spider-dev": 76.6,
            "ex_all_cnpj": 80.5
        },
        {
            "name": 'gpt-4.1-nano',
            "input_tokens": 0.10,
            "output_tokens": 0.40,
            "ex_all_spider-dev": 76.4,
            "ex_all_cnpj": 75.0
        },
        {
            "name": 'gpt-4.1-mini',
            "input_tokens": 0.40,
            "output_tokens": 1.60,
            "ex_all_spider-dev": 77.3,
            "ex_all_cnpj": 82.0
        },
        {
            "name": 'gpt-4.1',
            "input_tokens": 2.00,
            "output_tokens": 8.00,
            "ex_all_spider-dev": 76.4,
            "ex_all_cnpj": 81.5
        },
                
    ]

if "estrategias" not in st.session_state:
    st.session_state.estrategias = []

    # ------ ler estratégias do csv ---------------- #
    df = pd.read_csv('/home/user/Documentos/UFV/Mestrado/VisualizacaoDados/DataVisualization/estrategias.csv')

    # Verifica se as colunas obrigatórias estão presentes
    colunas_esperadas = {"name", "memory", "input_tokens", "output_tokens", "time","ex_all_spider-dev","ex_all_cnpj" }

    novas_estrategias = df.to_dict(orient="records")
    st.session_state.estrategias.extend(novas_estrategias)

def tokens_to_price(input_tokens, output_tokens, price_input, price_output):
    return input_tokens / 1_000_000 * price_input + output_tokens / 1_000_000 * price_output

# # Main page content
st.title("📊 Análise de Custo por Requisição")

st.sidebar.markdown("# Parâmetros")

# === GPU ===
st.sidebar.header("🖥️ GPU")
gpu_opcoes = [gpu["name"] for gpu in st.session_state.gpus]
gpu_nome = st.sidebar.selectbox("Escolha a GPU", gpu_opcoes)

gpu = next((g for g in st.session_state.gpus if g["name"] == gpu_nome), None)

# === Estratégia ===
st.sidebar.header("🧠 Estratégia")
estrategia_opcoes = [e["name"] for e in st.session_state.estrategias]
estrategia_nome = st.sidebar.selectbox("Escolha a Estratégia", estrategia_opcoes)

estrategia = next((e for e in st.session_state.estrategias if e["name"] == estrategia_nome), None)

# === APIs ===
st.sidebar.header("⚙️ APIs")
api_nomes = [api["name"] for api in st.session_state.apis]
apis_escolhidas = st.sidebar.multiselect("Escolha as APIs", api_nomes)



# st.sidebar.markdown("# Parametros")

# st.sidebar.header("🖥️ GPU - Configuração")
# instance_name = st.sidebar.text_input("Nome da Instânica", value='Oracle-NVIDIA A10')
# gpu_hour_price = st.sidebar.number_input("Preço por Hora da GPU (R$)", value=11.10, min_value=0.0)
# gpu_memory_total = st.sidebar.number_input("Memória Total da GPU (GB)", value=24.0, min_value=0.1)

# st.sidebar.header("⚙️ API")
# st.sidebar.subheader("Preço por Milhão de Tokens")

# model_name = st.sidebar.text_input("Nome do Modelo", value='gpt-4o-mini')
# price_input = st.sidebar.number_input("Tokens de Entrada (R$)", value=0.60, min_value=0.0)
# price_output = st.sidebar.number_input("Tokens de Saída (R$)", value=2.50, min_value=0.0)

# st.sidebar.header("🧠 Estratégia")
# model_memory = st.sidebar.number_input("Memória Ocupada pelo Modelo (GB)", value=12.0, min_value=0.1)
# request_time = st.sidebar.number_input("Tempo por Requisição (seg)", value=1.0, min_value=0.1)
# input_tokens = st.sidebar.number_input("Tokens de Entrada por Requisição", value=800)
# output_tokens = st.sidebar.number_input("Tokens de Saída por Requisição", value=1200)



# if model_memory > gpu_memory_total:
#     st.error("Erro: o modelo ocupa mais memória do que a disponível na GPU.")
# else:
#     concurrent_models = int(gpu_memory_total // model_memory)
#     max_reqs_per_model = int(3600 // request_time)
#     max_reqs_total = concurrent_models * max_reqs_per_model

#     st.markdown(f"""
#     - **Modelos simultâneos na GPU:** {concurrent_models}  
#     - **Máx. requisições por modelo/hora:** {max_reqs_per_model}  
#     - **Máx. requisições totais por hora:** {max_reqs_total}
#     """)

#     x = list(range(1, max_reqs_total + 1))
#     y_gpu = [gpu_hour_price / r for r in x]
#     y_api = [tokens_to_price(input_tokens, output_tokens, price_input, price_output)] * len(x)

#     df_plot = pd.DataFrame({
#         "Requisições por hora": x + x,
#         "Custo (R$/req)": y_gpu + y_api,
#         "Origem": [instance_name] * len(x) + [model_name] * len(x)
#     })

#     fig = px.line(df_plot, x="Requisições por hora", y="Custo (R$/req)", color="Origem",
#                   log_x=True, log_y=True,
#                   title="Custo por Requisição: GPU remota vs API")

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

if estrategia and gpu and apis_escolhidas:
    if estrategia["memory"] > gpu["memory"]:
        st.error("Erro: o modelo ocupa mais memória do que a disponível na GPU.")
    else:
        concurrent_models = int(gpu["memory"] // estrategia["memory"])
        max_reqs_per_model = int(3600 // estrategia["time"])
        max_reqs_total = concurrent_models * max_reqs_per_model

        st.markdown(f"""
        - **Modelos simultâneos na GPU:** {concurrent_models}  
        - **Máx. requisições por modelo/hora:** {max_reqs_per_model}  
        - **Máx. requisições totais por hora:** {max_reqs_total}
        """)

        x = list(range(1, max_reqs_total + 1))
        y_gpu = [gpu["hour_price"] / r for r in x]

        df_plot = pd.DataFrame({
            "Requisições por hora": x,
            "Custo (R$/req)": y_gpu,
            "Origem": [gpu["name"]] * len(x)
        })

        for nome_api in apis_escolhidas:
            api = next(a for a in st.session_state.apis if a["name"] == nome_api)
            custo_api = tokens_to_price(estrategia["input_tokens"],
                                        estrategia["output_tokens"],
                                        api["input_tokens"],
                                        api["output_tokens"])
            df_api = pd.DataFrame({
                "Requisições por hora": x,
                "Custo (R$/req)": [custo_api] * len(x),
                "Origem": [api["name"]] * len(x)
            })
            df_plot = pd.concat([df_plot, df_api], ignore_index=True)

        fig = px.line(df_plot, x="Requisições por hora", y="Custo (R$/req)", color="Origem",
                      log_x=True, log_y=True,
                      title="Custo por Requisição: GPU remota vs APIs")

        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Selecione uma GPU, uma estratégia e ao menos uma API.")
