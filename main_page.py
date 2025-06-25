import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots


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
    df = pd.read_csv('estrategias.csv')

    # Verifica se as colunas obrigatórias estão presentes
    colunas_esperadas = {"name", "memory", "input_tokens", "output_tokens", "time","ex_all_spider-dev","ex_all_cnpj" }

    novas_estrategias = df.to_dict(orient="records")
    st.session_state.estrategias.extend(novas_estrategias)

dolar = 5.9
def tokens_to_price(input_tokens, output_tokens, price_input, price_output):
    return input_tokens / 1_000_000 * price_input * dolar + output_tokens / 1_000_000 * price_output * dolar

# # Main page content
st.title("📊 Cost Analysis by Request")

st.sidebar.markdown("# Parameters")

# === GPU ===
st.sidebar.header("🖥️ GPU")
gpu_opcoes = [gpu["name"] for gpu in st.session_state.gpus]
gpu_nome = st.sidebar.selectbox("Choose the GPU", gpu_opcoes)

gpu = next((g for g in st.session_state.gpus if g["name"] == gpu_nome), None)

# === Estratégia ===
st.sidebar.header("🧠 Strategy")
estrategia_opcoes = [e["name"] for e in st.session_state.estrategias]
estrategia_nome = st.sidebar.selectbox("Choose the Strategy", estrategia_opcoes)

estrategia = next((e for e in st.session_state.estrategias if e["name"] == estrategia_nome), None)

# === APIs ===
st.sidebar.header("⚙️ APIs")
api_nomes = [api["name"] for api in st.session_state.apis]
apis_escolhidas = st.sidebar.multiselect("Choose APIs", api_nomes)


if estrategia and gpu and apis_escolhidas:
    
    if estrategia["memory"] > gpu["memory"]:
        st.error("Error: Model occupies more memory than available on GPU.")
    else:
    
        col1, col2 = st.columns(2)
        
        with col1:

            concurrent_models = int(gpu["memory"] // estrategia["memory"])
            max_reqs_per_model = int(3600 // estrategia["time"])
            max_reqs_total = concurrent_models * max_reqs_per_model

            x = list(range(1, max_reqs_total + 1))
            y_gpu = [gpu["hour_price"] / r for r in x]

            df_plot = pd.DataFrame({
                "Requests per hour": x,
                "Cost (R$/req)": y_gpu,
                "Source": [gpu["name"]] * len(x)
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
                    "Requests per hour": x,
                    "Cost (R$/req)": y_api,
                    "Source": [api["name"]] * len(x)
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

            fig = px.line(df_plot, x="Requests per hour", y="Cost (R$/req)", color="Source",
                        log_x=True, log_y=True,
                        title="Cost per Request: Cloud GPU vs APIs")
            
            for i, (x_cross, y_cross, nome_api) in enumerate(cross_points):
                fig.add_scatter(
                    x=[x_cross], y=[y_cross],
                    mode='markers+text', #'markers',
                    marker=dict(color='green', size=10), #symbol='x'
                    #hovertext=[f"{x_cross} req/h — {nome_api}"],
                    text=[f"{x_cross} req/h"],
                    textposition="top right",
                    name="break-even point" if i == 0 else None,
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
            - **Concurrent models on GPU:** {concurrent_models}
            - **Max requests per model/hour:** {max_reqs_per_model}
            - **Max total requests per hour:** {max_reqs_total}
            """)
            
        with col2:
            # Exemplo de dados (substitua pelos reais do seu contexto)
            dados_barras = [
                {"Source": estrategia["name"], "Set": "Spider-Dev", "ex_all": estrategia["ex_all_spider-dev"]},
                {"Source": estrategia["name"], "Set": "CNPJ", "ex_all": estrategia["ex_all_cnpj"]}
            ]
            
            for nome_api in apis_escolhidas:
                api = next(a for a in st.session_state.apis if a["name"] == nome_api)
                dados_barras.append({"Source": api["name"], "Set": "Spider-Dev", "ex_all": api["ex_all_spider-dev"]})
                dados_barras.append({"Source": api["name"], "Set": "CNPJ", "ex_all": api["ex_all_cnpj"]})

            df_barras = pd.DataFrame(dados_barras)

            fig_barras = px.bar(
                df_barras,
                x="Set",
                y="ex_all",
                color="Source",
                barmode="group",
                title="Performance Comparison",
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
        col3.markdown("<p style='text-align: center'><b>Hour Price</b></p>", unsafe_allow_html=True)
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
        
        col5.markdown("<p style='text-align: right'>Price/1M tokens</p>", unsafe_allow_html=True)
    
    
    with st.expander("📚 Compare Multiple Strategies", expanded=True):
        select = False
        
        num_colunas = st.number_input("Number of columns", min_value=1, max_value=6, value=6, step=1)
        num_linhas = 1

        total_celulas = num_colunas * num_linhas

        estrategias_selecionadas = st.multiselect(
            "Select strategies",
            options=estrategia_opcoes,
            key="estrategias_disponiveis"
        )
        
        if len(estrategias_selecionadas) > total_celulas:
            st.warning(f"You have selected more strategies ({len(estrategias_selecionadas)}) than available columns ({total_celulas}).")
        
        if len(estrategias_selecionadas) == num_colunas*num_linhas:
            select = True
        else:
            st.warning(f"There is a lack of strategies!")
            select = False

        posicoes_grid = [[None for _ in range(num_colunas)] for _ in range(num_linhas)]
        
        count = 0
        for idx in range(min(len(estrategias_selecionadas), total_celulas)):
            linha = idx // num_colunas
            coluna = idx % num_colunas
            posicoes_grid[linha][coluna] = estrategias_selecionadas[count]
            count += 1

        if gpu and apis_escolhidas and select:
            estrategias_dict = {e["name"]: e for e in st.session_state.estrategias}
            apis_dict = {a["name"]: a for a in st.session_state.apis}
            x = list(range(1, max_reqs_total + 1))

            celulas_grid = []

            # Etapa 1: montar o grid de containers
            for i in range(num_linhas):
                with st.container():
                    colunas = st.columns(num_colunas)
                    celulas_grid.append(colunas)

            # Etapa 2: preencher os gráficos no grid
            for i in range(num_linhas):
                for j in range(num_colunas):
                    with celulas_grid[i][j]:
                        try:
                            nome_estrategia = posicoes_grid[i][j]
                            if not nome_estrategia:
                                st.info("Sem estratégia.")
                                continue

                            estrategia = estrategias_dict.get(nome_estrategia)
                            if not estrategia:
                                st.warning("Estratégia não encontrada.")
                                continue

                            if estrategia["memory"] > gpu["memory"]:
                                st.error("Memória excedida.")
                                continue

                            # Gerar os dados do gráfico
                            dfs = []
                            cross_points = []

                            y_gpu = [gpu["hour_price"] / r for r in x]
                            dfs.append(pd.DataFrame({
                                "Requests per hour": x,
                                "Cost (R$/req)": y_gpu,
                                "Source": [f"{estrategia['name']} (GPU)"] * len(x)
                            }))

                            for nome_api in apis_escolhidas:
                                api = apis_dict[nome_api]
                                custo_api = tokens_to_price(
                                    estrategia["input_tokens"],
                                    estrategia["output_tokens"],
                                    api["input_tokens"],
                                    api["output_tokens"]
                                )
                                y_api = [custo_api] * len(x)

                                dfs.append(pd.DataFrame({
                                    "Requests per hour": x,
                                    "Cost (R$/req)": y_api,
                                    "Source": [f"{estrategia['name']} vs {api['name']}"] * len(x)
                                }))

                                diff = np.array(y_gpu) - np.array(y_api)
                                cross_indices = np.where(np.diff(np.sign(diff)))[0]
                                if len(cross_indices) > 0:
                                    idx = cross_indices[0]
                                    cross_points.append((x[idx], y_gpu[idx]))

                            df_plot = pd.concat(dfs, ignore_index=True)
                            
                            fig = px.line(
                                df_plot,
                                x="Requests per hour",
                                y="Cost (R$/req)",
                                color="Source",
                                log_x=True,
                                log_y=True,
                                title=f"{estrategia['name'].replace('Qwen2.5-','')}"
                            )

                            for x_cross, y_cross in cross_points:
                                fig.add_scatter(
                                    x=[x_cross], y=[y_cross],
                                    mode='markers+text',
                                    marker=dict(color='green', size=10),
                                    text=[f"{x_cross} req/h"],
                                    textposition="top right",
                                    showlegend=False
                                )

                            fig.update_yaxes(showticklabels=False, title_text=None)

                            fig.update_layout(
                                showlegend=False,
                                margin=dict(l=20, r=20, t=40, b=20),
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Erro ({i},{j}): {e}")

        
            dados_barras = []
            for i in range(num_linhas):
                for j in range(num_colunas):
                    nome_estrategia = posicoes_grid[i][j]
                    if not nome_estrategia:
                        continue
                    
                    estrategia = estrategias_dict.get(nome_estrategia)
                    if not estrategia:
                        continue
                    
                    dados_barras.append({
                        "Source": estrategia["name"],
                        "Set": "Spider-Dev",
                        "ex_all": estrategia.get("ex_all_spider-dev", None)
                    })
                    dados_barras.append({
                        "Source": estrategia["name"],
                        "Set": "CNPJ",
                        "ex_all": estrategia.get("ex_all_cnpj", None)
                    })

            # Para as APIs selecionadas
            for nome_api in apis_escolhidas:
                api = next(a for a in st.session_state.apis if a["name"] == nome_api)
                if not api:
                    continue
                dados_barras.append({
                    "Source": api["name"],
                    "Set": "Spider-Dev",
                    "ex_all": api.get("ex_all_spider-dev", None)
                })
                dados_barras.append({
                    "Source": api["name"],
                    "Set": "CNPJ",
                    "ex_all": api.get("ex_all_cnpj", None)
                })

            df_barras = pd.DataFrame(dados_barras)

            fig_barras = px.bar(
                df_barras,
                x="Set",
                y="ex_all",
                color="Source",
                barmode="group",
                title="Performance Comparison"
            )

            fig_barras.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.5,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig_barras, use_container_width=True)
               
else:
    st.warning("Select a GPU, a strategy, and at least one API.")


