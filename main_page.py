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


if estrategia and gpu and apis_escolhidas:
    
    if estrategia["memory"] > gpu["memory"]:
        st.error("Erro: o modelo ocupa mais memória do que a disponível na GPU.")
    else:
    
        col1, col2 = st.columns(2)
        
        with col1:

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
                    showlegend=False #(i == 0)
                )
            
            #fig.update_layout(showlegend=False) 
            fig.update_layout(
                legend=dict(
                    orientation="h",           # horizontal
                    yanchor="bottom",          # âncora inferior
                    y=-0.5,                    # um pouco abaixo do gráfico
                    xanchor="center",          # centralizado horizontalmente
                    x=0.5                      # centro da largura
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        
            st.markdown(f"""
            - **Modelos simultâneos na GPU:** {concurrent_models}  
            - **Máx. requisições por modelo/hora:** {max_reqs_per_model}  
            - **Máx. requisições totais por hora:** {max_reqs_total}
            """)
            
        with col2:
            # Exemplo de dados (substitua pelos reais do seu contexto)
            dados_barras = [
                {"Origem": estrategia["name"], "Conjunto": "Spider-Dev", "ex_all": estrategia["ex_all_spider-dev"]*100},
                {"Origem": estrategia["name"], "Conjunto": "CNPJ", "ex_all": estrategia["ex_all_cnpj"]*100}
            ]
            
            for nome_api in apis_escolhidas:
                api = next(a for a in st.session_state.apis if a["name"] == nome_api)
                dados_barras.append({"Origem": api["name"], "Conjunto": "Spider-Dev", "ex_all": api["ex_all_spider-dev"]*100})
                dados_barras.append({"Origem": api["name"], "Conjunto": "CNPJ", "ex_all": api["ex_all_cnpj"]*100})

            df_barras = pd.DataFrame(dados_barras)

            fig_barras = px.bar(
                df_barras,
                x="Conjunto",
                y="ex_all",
                color="Origem",
                barmode="group",
                title="Comparação de Desempenho",
            )
            # fig_barras.update_layout(showlegend=False)  # Oculta a legend
            
            fig_barras.update_layout(
                legend=dict(
                    orientation="h",           # horizontal
                    yanchor="bottom",          # âncora inferior
                    y=-0.5,                    # um pouco abaixo do gráfico
                    xanchor="center",          # centralizado horizontalmente
                    x=0.5                      # centro da largura
                )
            )
            
            st.plotly_chart(fig_barras, use_container_width=True)            
        
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
