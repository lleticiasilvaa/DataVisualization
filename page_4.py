import streamlit as st
import pandas as pd

st.markdown("# 🧠 Strategy")

with st.expander(f"{len(st.session_state.estrategias)} Registered Strategies", expanded=True):
    if len(st.session_state.estrategias) > 0:

        col1, col2, col3, col4, col5, col6 = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1.5])
        col1.markdown("", unsafe_allow_html=True)
        col2.markdown("<p style='text-align: center'><b>Model Memory</b></p>", unsafe_allow_html=True)
        col3.markdown("<p style='text-align: center'><b>Request Time</b></p>", unsafe_allow_html=True)
        col4.markdown("<p style='text-align: center'><b>Input Tokens</b></p>", unsafe_allow_html=True)
        col5.markdown("<p style='text-align: center'><b>Output Tokens</b></p>", unsafe_allow_html=True)
        col6.markdown("<p style='text-align: right'><b>Delete</b></p>", unsafe_allow_html=True)

        # Linhas de dados
        for i, est in enumerate(st.session_state.estrategias):
            col1, col2, col3, col4, col5, col6 = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1.5])
            col1.markdown(f"{est['name']}", unsafe_allow_html=True)
            col2.markdown(f"<p style='text-align: center'>{est['memory']} GB</p>", unsafe_allow_html=True)
            col3.markdown(f"<p style='text-align: center'>{est['time']} s</p>", unsafe_allow_html=True)
            col4.markdown(f"<p style='text-align: center'>{est['input_tokens']}</p>", unsafe_allow_html=True)
            col5.markdown(f"<p style='text-align: center'>{est['output_tokens']}</p>", unsafe_allow_html=True)
            with col6:
                _, right_col = st.columns([1, 1])
                with right_col:
                    if st.button("❌", key=f"delete_{i}"):
                        st.session_state.estrategias.pop(i)
                        st.rerun()
    else:
        st.info("No Strategy registered yet.")

st.divider()

st.markdown("### Register Strategy")

with st.form("form_cadastro"):
    nome = st.text_input("Name")
    model_memory = st.text_input("Model Memory (GB)")
    request_time = st.text_input("Request Time (s)")
    input_tokens = st.number_input("Input Tokens", min_value=0)
    output_tokens = st.number_input("Output Tokens", min_value=0)
    ex_all_spider_dev = st.number_input("EX_all Spider-Dev", min_value=0.0, format="%.4f")
    ex_all_cnpj = st.number_input("EX_all CNPJ", min_value=0.0, format="%.4f")
    cadastrar = st.form_submit_button("Register")

    if cadastrar and nome:
        st.session_state.estrategias.append({
            "name": nome,
            "memory": model_memory,
            "time": request_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ex_all_spider-dev":ex_all_spider_dev,
            "ex_all_cnpj": ex_all_cnpj
        })
        st.success(f"Strategy '{nome}' successfully registered!")
        st.rerun()

st.divider()
st.markdown("##### Upload Strategies with .CSV File")

# Upload de CSV
arquivo_csv = st.file_uploader("Select .csv file", type="csv")

if arquivo_csv:
    st.error(f"Feature still under development!")
    # try:
    #     df = pd.read_csv(arquivo_csv)

    #     # Verifica se as colunas obrigatórias estão presentes
    #     colunas_esperadas = {"nome", "model_memory", "request_time", "input_tokens", "output_tokens"}
    #     if not colunas_esperadas.issubset(df.columns):
    #         st.error(f"O CSV deve conter as colunas: {', '.join(colunas_esperadas)}")
    #     else:
    #         # Adiciona estratégias do CSV ao estado da sessão
    #         novas_estrategias = df.to_dict(orient="records")
    #         st.session_state.estrategias.extend(novas_estrategias)
    #         st.success(f"{len(novas_estrategias)} estratégias importadas com sucesso!")

    # except Exception as e:
    #     st.error(f"Erro ao ler o CSV: {e}")

