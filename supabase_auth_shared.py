"""
Supabase 초기화·Secrets 동기화 (multi-users-ref.py · pages/회원가입.py 공통)
- Streamlit Cloud: Settings → Secrets 에 SUPABASE_* 키를 넣으면 os.environ 으로 복사
- 클라이언트는 SUPABASE_ANON_KEY 우선 (Auth·RLS 호환). 없으면 SUPABASE_SERVICE_ROLE_KEY
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import streamlit as st
from supabase import Client, create_client


def streamlit_secrets_into_environ() -> None:
    """st.secrets 값을 코드 수정 없이 os.getenv 로 읽을 수 있게 동기화합니다."""
    keys = (
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_EMAIL_REDIRECT_URL",
    )
    try:
        sec = getattr(st, "secrets", None)
        if sec is None:
            return
        for k in keys:
            if os.getenv(k):
                continue
            try:
                if k in sec:
                    os.environ[k] = str(sec[k]).strip()
            except Exception:
                continue
    except Exception:
        return


def supabase_client_key() -> Tuple[Optional[str], Optional[str]]:
    url = (os.getenv("SUPABASE_URL") or "").strip() or None
    key = (
        (os.getenv("SUPABASE_ANON_KEY") or "").strip()
        or (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        or None
    )
    return url, key


def init_supabase(url: Optional[str], key: Optional[str]) -> Optional[Client]:
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None
