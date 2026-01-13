import hmac
import streamlit as st

def require_password():
    """
    Simple shared-password gate.
    Password is stored in Streamlit secrets: st.secrets["APP_PASSWORD"]
    """
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if st.session_state.authed:
        return True

    st.title("This app is password-protected")
    pw = st.text_input("Password", type="password")

    # Don't leak timing info; use hmac.compare_digest
    if pw:
        expected = st.secrets.get("APP_PASSWORD", "")
        if expected and hmac.compare_digest(pw, expected):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password.")

    st.stop()
