import streamlit as st

st.markdown("# 🖥️ GPU - Configuração")

# st.markdown("### Estratégias Cadastradas")
with st.expander(f"{len(st.session_state.gpus)} GPUs Cadastradas | Price per 1M tokens", expanded=True):
    if len(st.session_state.gpus) > 0:

        col1, col2, col3, col4 = st.columns([4, 2.5, 2.5, 1.5])
        col1.markdown("", unsafe_allow_html=True)
        col2.markdown("<p style='text-align: center'><b>Memory</b></p>", unsafe_allow_html=True)
        col3.markdown("<p style='text-align: center'><b>Preço por Hora</b></p>", unsafe_allow_html=True)
        col4.markdown("<p style='text-align: right'><b>Excluir</b></p>", unsafe_allow_html=True)

        # Linhas de dados
        for i, est in enumerate(st.session_state.gpus):
            col1, col2, col3, col4 = st.columns([4, 2.5, 2.5, 1.5])
            col1.markdown(f"{est['name']}", unsafe_allow_html=True)
            col2.markdown(f"<p style='text-align: center'> $ {est['memory']}</p>", unsafe_allow_html=True)
            col3.markdown(f"<p style='text-align: center'> $ {est['hour_price']}</p>", unsafe_allow_html=True)
            with col4:
                _, right_col = st.columns([1, 1])
                with right_col:
                    if st.button("❌", key=f"delete_{i}"):
                        st.session_state.gpus.pop(i)
                        st.rerun()
    else:
        st.info("Nenhuma GPU cadastrada ainda.")

st.divider()

st.markdown("### Cadastrar API")

# Formulário para cadastro
with st.form("form_cadastro"):
    nome = st.text_input("Nome da Instância")
    memory = st.number_input("Memória (GB)", min_value=0.0, format="%.4f")
    hour_price = st.number_input("Preço por Hora (R$)", min_value=0.0, format="%.4f")
    cadastrar = st.form_submit_button("Cadastrar")

    if cadastrar and nome:
        st.session_state.gpus.append({
            "name": nome,
            "memory": memory,
            "hour_price": hour_price,
        })
        st.success(f"GPU '{nome}' cadastrada com sucesso!")
        st.rerun()