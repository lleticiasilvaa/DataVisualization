import streamlit as st

st.markdown("# ⚙️ API")



# st.markdown("### Estratégias Cadastradas")
with st.expander(f"{len(st.session_state.apis)} APIs Cadastradas | Price per 1M tokens", expanded=True):
    if len(st.session_state.apis) > 0:

        col1, col2, col3, col4 = st.columns([4, 2.5, 2.5, 1.5])
        col1.markdown("", unsafe_allow_html=True)
        col2.markdown("<p style='text-align: center'><b>Input Tokens</b></p>", unsafe_allow_html=True)
        col3.markdown("<p style='text-align: center'><b>Output Tokens</b></p>", unsafe_allow_html=True)
        col4.markdown("<p style='text-align: right'><b>Excluir</b></p>", unsafe_allow_html=True)

        # Linhas de dados
        for i, est in enumerate(st.session_state.apis):
            col1, col2, col3, col4 = st.columns([4, 2.5, 2.5, 1.5])
            col1.markdown(f"{est['name']}", unsafe_allow_html=True)
            col2.markdown(f"<p style='text-align: center'> $ {est['input_tokens']}</p>", unsafe_allow_html=True)
            col3.markdown(f"<p style='text-align: center'> $ {est['output_tokens']}</p>", unsafe_allow_html=True)
            with col4:
                _, right_col = st.columns([1, 1])
                with right_col:
                    if st.button("❌", key=f"delete_{i}"):
                        st.session_state.apis.pop(i)
                        st.rerun()
    else:
        st.info("Nenhuma API cadastrada ainda.")

st.divider()

st.markdown("### Cadastrar API")

# Formulário para cadastro
with st.form("form_cadastro"):
    nome = st.text_input("Nome do Modelo")
    input_tokens = st.number_input("Input Tokens Price ($)", min_value=0.0, format="%.4f")
    output_tokens = st.number_input("Output Tokens Price ($)", min_value=0.0, format="%.4f")
    ex_all_spider_dev = st.number_input("EX_all Spider-Dev", min_value=0.0, format="%.4f")
    ex_all_cnpj = st.number_input("EX_all CNPJ", min_value=0.0, format="%.4f")
    cadastrar = st.form_submit_button("Cadastrar")

    if cadastrar and nome:
        st.session_state.apis.append({
            "name": nome,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ex_all_spider-dev":ex_all_spider_dev,
            "ex_all_cnpj": ex_all_cnpj
        })
        st.success(f"API '{nome}' cadastrada com sucesso!")
        st.rerun()
