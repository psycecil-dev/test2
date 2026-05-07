"""Supabase 이메일/비밀번호 회원가입 (확인 메일). 메인 앱: ../multi-users-ref.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

_app_dir = Path(__file__).resolve().parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from supabase_auth_shared import init_supabase, streamlit_secrets_into_environ, supabase_client_key

st.set_page_config(page_title="회원가입 · PDF 멀티유저 RAG", page_icon="✉️", layout="wide")

streamlit_secrets_into_environ()
_u, _k = supabase_client_key()
_supabase = init_supabase(_u, _k)

st.markdown(
    """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important; color: #ffd700 !important; }
.stButton > button {
  background-color: #ff69b4 !important; color: white !important; border: none !important;
  border-radius: 5px !important; padding: 0.5rem 1rem !important; font-weight: bold !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("# ✉️ 회원가입")
st.caption("이메일로 확인 메일을 보냅니다. **Authentication → Providers → Email → Confirm email** 이 켜져 있어야 합니다.")

if _supabase is None:
    st.error("Supabase에 연결할 수 없습니다. `SUPABASE_URL` 과 `SUPABASE_ANON_KEY`(또는 서비스 롤)를 Secrets에 설정하세요.")
    st.stop()

with st.expander("확인 메일이 오지 않을 때", expanded=False):
    st.markdown(
        """
- 스팸함을 확인하세요.
- Supabase **Authentication → URL Configuration** 에서 **Site URL** 과 **Redirect URLs** 가 올바른지 확인하세요.
- 확인 링크는 보통 회원가입 시 지정한 `email_redirect_to` 또는 프로젝트 기본 Site URL 로 갑니다.
- Streamlit Cloud 배포 주소가 Redirect URLs 에 등록되어 있어야 할 수 있습니다.
"""
    )


def _auth_error_text(exc: BaseException) -> str:
    parts: list[str] = []
    s = str(exc).strip()
    if s:
        parts.append(s)
    blob = " ".join(parts).lower()
    for attr in ("message", "msg", "error_description"):
        v = getattr(exc, attr, None)
        if v is None:
            continue
        vs = str(v).strip()
        if vs and vs.lower() not in blob:
            parts.append(vs)
            blob = " ".join(parts).lower()
    return " ".join(parts) if parts else repr(exc)


with st.form("signup_form"):
    email = st.text_input("이메일", placeholder="you@example.com", key="su_email")
    pw1 = st.text_input("비밀번호", type="password", key="su_pw1")
    pw2 = st.text_input("비밀번호 확인", type="password", key="su_pw2")
    submitted = st.form_submit_button("가입 요청 (확인 메일 발송)", use_container_width=True)

if submitted:
    em = email.strip().lower()
    if not em or not pw1:
        st.warning("이메일과 비밀번호를 입력하세요.")
    elif pw1 != pw2:
        st.warning("비밀번호가 일치하지 않습니다.")
    elif len(pw1) < 6:
        st.warning("비밀번호는 Supabase 설정에 따라 보통 6자 이상이 필요합니다.")
    else:
        streamlit_secrets_into_environ()
        redir = (os.getenv("SUPABASE_EMAIL_REDIRECT_URL") or "").strip()
        opts: dict = {}
        if redir and str(redir).strip():
            opts["email_redirect_to"] = str(redir).strip()
        try:
            res = _supabase.auth.sign_up({"email": em, "password": pw1, "options": opts or {}})
            if res.user:
                st.success(
                    "가입 요청이 접수되었습니다. 받은편지함(및 스팸함)에서 **확인 링크**를 눌러 인증을 완료한 뒤, "
                    "메인 앱에서 같은 이메일·비밀번호로 로그인하세요."
                )
                if res.session:
                    st.info("이 프로젝트 설정상 확인 없이 세션이 바로 발급되었습니다. 메인 화면으로 이동해 사용할 수 있습니다.")
            else:
                st.warning("응답에 사용자 정보가 없습니다. 대시보드 Authentication → Users 를 확인하세요.")
        except Exception as e:
            st.error(f"가입 실패: {_auth_error_text(e)}")

st.markdown("---")
st.markdown("**이미 계정이 있으신가요?**")
if hasattr(st, "page_link"):
    st.page_link("multi-users-ref.py", label="메인 앱(로그인)으로 돌아가기", icon="🏠")
else:
    st.caption("상단 사이드바에서 `multi-users-ref` 페이지를 선택하세요.")
