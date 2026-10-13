import uuid

import requests
import streamlit as st

# Configuración de la página de Streamlit
st.set_page_config(page_title="Agente IA - Local", page_icon="🤖", layout="centered")

st.title("💬 Chat con mi Agente IA (LangGraph)")
st.markdown("Conectado a tu servidor local en `http://localhost:8123`")

# URL base de tu API de LangGraph
API_URL = "http://localhost:8123"

# Inicializar un thread_id único para la sesión actual del navegador si no existe
if "thread_id" not in st.session_state:
    try:
        # Creamos un hilo automáticamente al iniciar la app
        response = requests.post(f"{API_URL}/threads", json={})
        if response.status_code == 200:
            st.session_state.thread_id = response.json().get("thread_id")
        else:
            st.session_state.thread_id = str(uuid.uuid4())  # Fallback si falla
    except Exception:
        st.session_state.thread_id = str(uuid.uuid4())

# Inicializar el historial de mensajes en la sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar los mensajes anteriores del chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obtener la entrada del usuario mediante el widget de chat inferior
if prompt := st.chat_input("Escribe tu mensaje aquí..."):
    # Añadir mensaje del usuario al historial visual
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preparar la llamada a la API de LangGraph (/runs/wait) con tu assistant_id correcto
    assistant_id = "pygenai"
    payload = {"assistant_id": assistant_id, "input": {"messages": [{"role": "human", "content": prompt}]}}

    with st.chat_message("assistant"), st.spinner("Pensando..."):
        try:
            url = f"{API_URL}/threads/{st.session_state.thread_id}/runs/wait"
            response = requests.post(url, json=payload, timeout=60)

            # Si el servidor se reinició y el hilo ya no existe (Error 404)
            if response.status_code == 404:
                new_thread_res = requests.post(f"{API_URL}/threads", json={})
                if new_thread_res.status_code == 200:
                    st.session_state.thread_id = new_thread_res.json().get("thread_id")
                    # Reintentamos la petición con el nuevo thread_id recién creado
                    url = f"{API_URL}/threads/{st.session_state.thread_id}/runs/wait"
                    response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                # Extraer los mensajes devueltos por el grafo
                messages_history = data.get("messages", [])

                # Buscamos el último mensaje generado por el asistente (AI)
                ai_response = "No se recibió respuesta del agente."
                for msg in reversed(messages_history):
                    if isinstance(msg, dict) and (msg.get("type") == "ai" or msg.get("role") == "assistant"):
                        ai_response = msg.get("content", "")
                        break
                    elif hasattr(msg, "type") and msg.type == "ai":
                        ai_response = msg.content
                        break
                    elif isinstance(msg, dict) and "content" in msg and msg.get("role") != "human":
                        ai_response = msg.get("content")
                        if ai_response != prompt:
                            break

                if ai_response == "No se recibió respuesta del agente." and len(messages_history) > 0:
                    last_msg = messages_history[-1]
                    if isinstance(last_msg, dict):
                        ai_response = last_msg.get("content", str(last_msg))
                    else:
                        ai_response = getattr(last_msg, "content", str(last_msg))

                st.markdown(ai_response)
                # Guardar respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                error_msg = f"Error en la API ({response.status_code}): {response.text}"
                st.error(error_msg)
        except Exception as e:
            st.error(f"No se pudo conectar con el servidor en {API_URL}. Detalles: {e}")
