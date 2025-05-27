import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# # Inicializa a flag do popup se ainda não existir
# if "show_popup" not in st.session_state:
#     st.session_state.show_popup = False

# # Botão para abrir o gráfico
# if st.button("Comparar ex_all em popup"):
#     st.session_state.show_popup = True

# # Exemplo de dados (substitua pelos reais do seu contexto)
# dados_barras = [
#     {"Origem": "GPU", "Conjunto": "Spider", "ex_all": 0.81},
#     {"Origem": "GPU", "Conjunto": "CNPJ", "ex_all": 0.74},
#     {"Origem": "OpenAI", "Conjunto": "Spider", "ex_all": 0.85},
#     {"Origem": "OpenAI", "Conjunto": "CNPJ", "ex_all": 0.79},
# ]

# df_barras = pd.DataFrame(dados_barras)

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

dolar = 5.9
def tokens_to_price(input_tokens, output_tokens, price_input, price_output):
    return input_tokens / 1_000_000 * price_input * dolar + output_tokens / 1_000_000 * price_output * dolar

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

        x = list(range(1, max_reqs_total + 1))
        y_gpu = [gpu["hour_price"] / r for r in x]

        df_plot = pd.DataFrame({
            "Requisições por hora": x,
            "Custo (R$/req)": y_gpu,
            "Origem": [gpu["name"]] * len(x)
        })
        
        fig = None  # Vamos declarar o fig fora para usar depois
        cross_points = []

        for nome_api in apis_escolhidas:
            api = next(a for a in st.session_state.apis if a["name"] == nome_api)
            custo_api = tokens_to_price(estrategia["input_tokens"],
                                        estrategia["output_tokens"],
                                        api["input_tokens"],
                                        api["output_tokens"])
            
            y_api = [custo_api] * len(x)
            df_api = pd.DataFrame({
                "Requisições por hora": x,
                "Custo (R$/req)": y_api,
                "Origem": [api["name"]] * len(x)
            })
            
            df_plot = pd.concat([df_plot, df_api], ignore_index=True)
    
            # Calcula ponto de cruzamento entre GPU e esta API
            diff = np.array(y_gpu) - np.array(y_api)
            cross_indices = np.where(np.diff(np.sign(diff)))[0]

            if len(cross_indices) > 0:
                idx = cross_indices[0]
                x_cross = x[idx]
                y_cross = y_gpu[idx]
                cross_points.append((x_cross, y_cross, api["name"]))

        fig = px.line(df_plot, x="Requisições por hora", y="Custo (R$/req)", color="Origem",
                      log_x=True, log_y=True,
                      title="Custo por Requisição: GPU remota vs APIs")
        
        for i, (x_cross, y_cross, nome_api) in enumerate(cross_points):
            fig.add_scatter(
                x=[x_cross], y=[y_cross],
                mode='markers+text', #'markers',
                marker=dict(color='green', size=10), #symbol='x'
                #hovertext=[f"{x_cross} req/h — {nome_api}"],
                text=[f"{x_cross} req/h"],
                textposition="top right",
                name="Ponto de cruzamento" if i == 0 else None,
                showlegend=(i == 0)
            )

        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown(f"""
        - **Modelos simultâneos na GPU:** {concurrent_models}  
        - **Máx. requisições por modelo/hora:** {max_reqs_per_model}  
        - **Máx. requisições totais por hora:** {max_reqs_total}
        """)
    
        st.divider()
        
        col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1.5, 1.5, 1.5])
        col1.markdown("<p style='text-align: left'><b>-</b></p>", unsafe_allow_html=True)
        col2.markdown("<p style='text-align: center'><b>Memory</b></p>", unsafe_allow_html=True)
        col3.markdown("<p style='text-align: center'><b>Preço por Hora</b></p>", unsafe_allow_html=True)
        col1.markdown(f"{gpu['name']}", unsafe_allow_html=True)
        col2.markdown(f"<p style='text-align: center'> {gpu['memory']} GB</p>", unsafe_allow_html=True)
        col3.markdown(f"<p style='text-align: center'> R$ {gpu['hour_price']}</p>", unsafe_allow_html=True)
        
        st.divider()

        col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1.5, 1.5, 1.5])
        col1.markdown("<p style='text-align: left'><b>-</b></p>", unsafe_allow_html=True)
        col2.markdown("<p style='text-align: center'><b>Model Memory</b></p>", unsafe_allow_html=True)
        col3.markdown("<p style='text-align: center'><b>Request Time</b></p>", unsafe_allow_html=True)
        col4.markdown("<p style='text-align: center'><b>Input Tokens</b></p>", unsafe_allow_html=True)
        col5.markdown("<p style='text-align: center'><b>Output Tokens</b></p>", unsafe_allow_html=True)
        col1.markdown(f"{estrategia['name']}", unsafe_allow_html=True)
        col2.markdown(f"<p style='text-align: center'>{estrategia['memory']} GB</p>", unsafe_allow_html=True)
        col3.markdown(f"<p style='text-align: center'>{estrategia['time']} s</p>", unsafe_allow_html=True)
        col4.markdown(f"<p style='text-align: center'>{estrategia['input_tokens']}</p>", unsafe_allow_html=True)
        col5.markdown(f"<p style='text-align: center'>{estrategia['output_tokens']}</p>", unsafe_allow_html=True)
        
        st.divider()
        col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1.5, 1.5, 1.5])
        col1.markdown("<p style='text-align: left'><b>-</b></p>", unsafe_allow_html=True)
        col4.markdown("<p style='text-align: center'><b>Input Tokens</b></p>", unsafe_allow_html=True)
        col5.markdown("<p style='text-align: center'><b>Output Tokens</b></p>", unsafe_allow_html=True)

        
        for nome_api in apis_escolhidas:
            api = next(a for a in st.session_state.apis if a["name"] == nome_api)
            col1.markdown(f"{api['name']}", unsafe_allow_html=True)
            col4.markdown(f"<p style='text-align: center'> $ {api['input_tokens']} = R$ {api['input_tokens']*dolar:.2f}</p>", unsafe_allow_html=True)
            col5.markdown(f"<p style='text-align: center'> $ {api['output_tokens']} = R$ {api['output_tokens']*dolar:.2f}</p>", unsafe_allow_html=True)
        
        col5.markdown("<p style='text-align: right'>Preço/1M de tokens</p>", unsafe_allow_html=True)
        
        # linhas = st.number_input("Número de linhas do grid de gráficos", min_value=1, value=4)
        # colunas = st.number_input("Número de colunas do grid de gráficos", min_value=1, value=6)
        
        # # Estratégias disponíveis
        # estrategias = st.session_state.estrategias  # Lista de dicionários

        # # Subplots com tamanho ajustável
        # fig = make_subplots(rows=linhas, cols=colunas,
        #                     subplot_titles=[e["name"] for e in estrategias],
        #                     shared_xaxes=True, shared_yaxes=True)

        # total = len(estrategias)
        # for idx, estrategia in enumerate(estrategias):
        #     row = idx // colunas + 1
        #     col = idx % colunas + 1

        #     if estrategia["memory"] > gpu["memory"]:
        #         st.warning(f"A estratégia '{estrategia['name']}' excede a memória da GPU.")
        #         continue

        #     concurrent_models = int(gpu["memory"] // estrategia["memory"])
        #     max_reqs_per_model = int(3600 // estrategia["time"])
        #     max_reqs_total = concurrent_models * max_reqs_per_model

        #     x = list(range(1, max_reqs_total + 1))
        #     y_gpu = [gpu["hour_price"] / r for r in x]

        #     fig.add_trace(
        #         go.Scatter(x=x, y=y_gpu, mode='lines', name=gpu["name"], legendgroup=gpu["name"],
        #                 line=dict(color='blue')),
        #         row=row, col=col
        #     )

        #     for i, nome_api in enumerate(apis_escolhidas):
        #         api = next(a for a in st.session_state.apis if a["name"] == nome_api)
        #         custo_api = tokens_to_price(estrategia["input_tokens"],
        #                                     estrategia["output_tokens"],
        #                                     api["input_tokens"],
        #                                     api["output_tokens"])
        #         y_api = [custo_api] * len(x)

        #         fig.add_trace(
        #             go.Scatter(x=x, y=y_api, mode='lines', name=api["name"], legendgroup=api["name"],
        #                     line=dict(dash='dot')),
        #             row=row, col=col
        #         )

        #         # Cruzamento
        #         diff = np.array(y_gpu) - np.array(y_api)
        #         cross_idx = np.where(np.diff(np.sign(diff)))[0]
        #         if len(cross_idx) > 0:
        #             idx_cross = cross_idx[0]
        #             fig.add_trace(
        #                 go.Scatter(
        #                     x=[x[idx_cross]], y=[y_gpu[idx_cross]],
        #                     mode='markers+text',
        #                     marker=dict(color='green', size=10),
        #                     text=[f"{x[idx_cross]} req/h"],
        #                     textposition="top right",
        #                     name="Cruzamento" if i == 0 else None,
        #                     showlegend=(row == 1 and col == 1 and i == 0)
        #                 ),
        #                 row=row, col=col
        #             )

        # # Layout global
        # fig.update_layout(
        #     height=400 * linhas,
        #     width=500 * colunas,
        #     title_text="Custo por Requisição: GPU vs APIs por Estratégia",
        #     showlegend=False
        # )

        # # Eixos log em todos os subgráficos
        # fig.update_xaxes(type="log", title_text="Requisições por hora")
        # fig.update_yaxes(type="log", title_text="Custo (R$/req)")

        # # Exibir na página
        # st.plotly_chart(fig, use_container_width=True)


        
else:
    st.warning("Selecione uma GPU, uma estratégia e ao menos uma API.")


# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # Dados de exemplo (substitua pelos reais)
# dados_barras = [
#     {"Origem": "GPU", "Conjunto": "Spider", "ex_all": 0.81},
#     {"Origem": "OpenAI", "Conjunto": "Spider", "ex_all": 0.85},
#     {"Origem": "GPU", "Conjunto": "CNPJ", "ex_all": 0.74},
#     {"Origem": "OpenAI", "Conjunto": "CNPJ", "ex_all": 0.79},
# ]

# df_barras = pd.DataFrame(dados_barras)

# # Cria colunas lado a lado
# col1, col2 = st.columns(2)

# # Gráfico do Spider
# with col1:
#     df_spider = df_barras[df_barras["Conjunto"] == "Spider"]
#     fig_spider = px.bar(
#         df_spider,
#         x="Origem",
#         y="ex_all",
#         title="ex_all no Spider",
#         color="Origem",
#         text="ex_all"
#     )
#     fig_spider.update_layout(showlegend=False)
#     st.plotly_chart(fig_spider, use_container_width=True)

# # Gráfico do CNPJ
# with col2:
#     df_cnpj = df_barras[df_barras["Conjunto"] == "CNPJ"]
#     fig_cnpj = px.bar(
#         df_cnpj,
#         x="Origem",
#         y="ex_all",
#         title="ex_all no CNPJ",
#         color="Origem",
#         text="ex_all"
#     )
#     fig_cnpj.update_layout(showlegend=False)
#     st.plotly_chart(fig_cnpj, use_container_width=True)
