import streamlit as st
from openai import OpenAI


def init_client():
    return OpenAI()


def get_default_system_prompt():
    return (
        "You are a helpful assistant and expert Python programming tutor. "
        "Answer questions clearly and concisely. Provide runnable Python code examples when helpful, "
        "explain reasoning, and mention version-specific features if relevant (e.g., Python 3.10 match/case). "
        "Prefer standard library solutions unless a third-party library is essential. "
        "When showing code, use complete examples and highlight potential pitfalls. "
        "If the user's question is ambiguous, ask a brief clarifying question."
    )


def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": get_default_system_prompt()}
        ]
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2


def render_sidebar():
    with st.sidebar:
        st.header("Python Q&A Assistant")
        model = st.selectbox(
            "Model",
            options=["gpt-3.5-turbo", "gpt-4"],
            index=0 if st.session_state.get("model") == "gpt-3.5-turbo" else 1,
            help="Use gpt-3.5-turbo for speed/cost, gpt-4 for higher quality."
        )
        st.session_state.model = model

        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("temperature", 0.2),
            step=0.05,
            help="Lower = more deterministic; higher = more creative."
        )
        st.session_state.temperature = temperature

        if st.button("New conversation"):
            system_msg = {"role": "system", "content": get_default_system_prompt()}
            st.session_state.messages = [system_msg]
            st.experimental_rerun()

        st.markdown("---")
        st.caption(
            "Set your OpenAI API key as the environment variable OPENAI_API_KEY "
            "before running this app."
        )
        st.caption("Tip: Ask me about Python basics, data structures, OOP, async, typing, pandas, NumPy, testing, packaging, and more.")


def build_messages():
    # Return a copy to avoid accidental mutation
    return list(st.session_state.messages)


def ask_openai(client: OpenAI, messages, model: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,  # "gpt-3.5-turbo" or "gpt-4"
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def render_chat_history():
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])


def main():
    st.set_page_config(page_title="Python Programming Chatbot", page_icon="🐍")
    st.title("🐍 Python Programming Chatbot")

    ensure_session_state()
    render_sidebar()

    # Chat history
    render_chat_history()

    # Chat input
    user_input = st.chat_input("Ask a Python question (e.g., 'How do I use list comprehensions?')")
    if user_input:
        # Append user message
        user_msg = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call OpenAI
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = init_client()
                    messages = build_messages()
                    reply = ask_openai(
                        client=client,
                        messages=messages,
                        model=st.session_state.model,
                        temperature=st.session_state.temperature,
                    )
                except Exception as e:
                    reply = (
                        "Sorry, I couldn't process your request. "
                        f"Details: {e}"
                    )
                st.markdown(reply)
        # Append assistant reply
        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()