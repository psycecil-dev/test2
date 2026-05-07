"""
PDF 기반 멀티유저 RAG 챗봇
- Supabase Auth(이메일/비밀번호), 세션/메시지/벡터(pgvector), OpenAI 임베딩
- LLM API 키는 사이드바 상단 입력 → os.environ 반영 (멀티유저·Streamlit Cloud)
- Supabase 설정은 Streamlit Secrets(SUPABASE_URL / SUPABASE_ANON_KEY 등)에서 직접 읽음
- 세션 UI·버튼: multi-session-ref.py 와 동일(세션저장/로드/삭제/화면초기화/제목보정/vectordb)
- 회원가입: `pages/회원가입.py` 전용 페이지 (공통 로직: `supabase_auth_shared.py`)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field, PrivateAttr
from supabase import Client, create_client

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None  # type: ignore
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from supabase_auth_shared import init_supabase

# --- 로깅 ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"multi_users_rag_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(name).setLevel(logging.WARNING)

MODEL_GPT = "gpt-5.5"
MODEL_CLAUDE = "claude-opus-4-7"
MODEL_GEMINI = "gemini-3-pro-preview"
ALL_CHAT_MODELS = [MODEL_GPT, MODEL_CLAUDE, MODEL_GEMINI]


def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\n\s*-{3,}\s*\n", "\n\n", text)
    text = re.sub(r"\n\s*={3,}\s*\n", "\n\n", text)
    text = re.sub(r"\n\s*_{3,}\s*\n", "\n\n", text)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*={3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*_{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned


def ensure_api_keys(openai_key: str, anthropic_key: str, gemini_key: str) -> None:
    st.session_state.openai_api_key = (openai_key or "").strip()
    st.session_state.anthropic_api_key = (anthropic_key or "").strip()
    st.session_state.gemini_api_key = (gemini_key or "").strip()


def _api_key(name: str) -> str:
    if name == "OPENAI_API_KEY":
        return str(st.session_state.get("openai_api_key") or "").strip()
    if name == "ANTHROPIC_API_KEY":
        return str(st.session_state.get("anthropic_api_key") or "").strip()
    if name == "GOOGLE_API_KEY":
        return str(st.session_state.get("gemini_api_key") or "").strip()
    return ""


def sync_supabase_session_from_state() -> None:
    if not supabase:
        return
    at = st.session_state.get("sb_access_token")
    rt = st.session_state.get("sb_refresh_token")
    if at and rt:
        try:
            supabase.auth.set_session(at, rt)
        except Exception:
            logger.exception("set_session 실패")


def get_supabase_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "has_url": bool(_sb_url),
        "has_key": bool(_sb_key),
        "connected": supabase is not None,
        "query_ok": False,
        "error": None,
    }
    if supabase:
        try:
            supabase.table("sessions").select("id").limit(1).execute()
            status["query_ok"] = True
        except Exception as e:
            status["error"] = str(e)
    return status


def _auth_error_text(exc: BaseException) -> str:
    """GoTrue 예외에 message 등이 따로 있을 때까지 합쳐서 표시."""
    parts: List[str] = []
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


def send_password_reset_email(email: str) -> None:
    """비밀번호 재설정 메일(SUPABASE_EMAIL_REDIRECT_URL + Auth Redirect URLs 필요)."""
    if not supabase:
        st.error("Supabase가 연결되지 않았습니다.")
        return
    em = email.strip().lower()
    if not em:
        st.warning("이메일을 입력하세요.")
        return
    redir = ""
    try:
        redir = str(st.secrets.get("SUPABASE_EMAIL_REDIRECT_URL", "")).strip()
    except Exception:
        redir = ""
    if not redir:
        st.warning(
            "`.env` 또는 Streamlit Secrets에 **SUPABASE_EMAIL_REDIRECT_URL** 을 넣고, "
            "Supabase → Authentication → URL Configuration → **Redirect URLs** 에도 같은 URL을 등록하세요."
        )
        return
    try:
        supabase.auth.reset_password_for_email(em, {"redirect_to": redir})
        st.success("재설정 메일을 보냈습니다. 받은편지함·스팸함을 확인하세요.")
    except Exception as e:
        st.error(f"전송 실패: {_auth_error_text(e)}")


def _auth_failure_hint(message: str) -> str:
    m = (message or "").lower()
    if "invalid login credentials" in m or "invalid_credentials" in m:
        return (
            "**이 메시지는 여러 경우에 똑같이 나옵니다**(Supabase가 구분을 숨기는 경우가 많음).\n\n"
            "**1) 이메일 미인증 (Confirm email 켜짐)**\n"
            "- 사용자는 이미 있지만 **확인 메일을 아직 안 누른 상태**면 로그인이 막히고, "
            "여기서도 `Invalid login credentials` 만 보일 수 있습니다.\n"
            "- Dashboard → **Authentication → Users** → 해당 이메일 행을 열어 "
            "**Email confirmed / Confirmed at** 이 비어 있는지 확인하세요.\n"
            "- 비어 있으면: 받은편지함·스팸함의 **확인 링크** 클릭, 또는 사용자 메뉴에서 "
            "**Confirm user** / **Send magic link** 등(대시보드 버전에 따라 이름 상이)으로 처리합니다.\n\n"
            "**2) 비밀번호가 틀림**\n"
            "- 앱·대시보드에서 가입할 때 쓴 비밀번호와 다르면 **같은** 오류가 납니다.\n"
            "- 아래 **비밀번호 재설정**으로 새 비밀번호를 잡으세요.\n\n"
            "**3) 초대만 되고 앱에서 비밀번호 가입을 안 한 경우**\n"
            "- 대시보드 초대는 실패했어도(이미 등록됨) **예전에 만든 계정**이 남아 있을 수 있습니다.\n"
            "- Users에서 확인 후 **재설정 메일** 또는 **회원가입 페이지**에서 같은 이메일로 비번을 다시 정합니다.\n\n"
            "**4) 그 외**\n"
            "- **다른 Supabase 프로젝트**의 URL/키를 앱에 넣은 경우\n"
            "- 사이드바 **Supabase 상태** → **연결 캐시 새로고침** 후 재시도\n"
            "- 개발 중이면 **Confirm email** 을 잠시 끄고(새 가입부터 적용) 다시 가입해 테스트할 수 있습니다."
        )
    if "email not confirmed" in m:
        return (
            "**이메일 미인증:** 메일함의 확인 링크를 누르거나, 개발 중에는 "
            "Authentication → Providers → Email → **Confirm email** 을 끄세요."
        )
    return ""


def sign_in(email: str, password: str) -> bool:
    if not supabase:
        st.error("Supabase가 연결되지 않았습니다.")
        return False
    em = email.strip().lower()
    try:
        res = supabase.auth.sign_in_with_password({"email": em, "password": password})
        if res and res.session:
            st.session_state.user_email = em
            st.session_state.user_id = res.user.id
            st.session_state.sb_access_token = res.session.access_token
            st.session_state.sb_refresh_token = res.session.refresh_token
            return True
        st.error("로그인에 실패했습니다.")
        return False
    except Exception as e:
        raw = _auth_error_text(e)
        st.error(f"로그인 오류: {raw}")
        hint = _auth_failure_hint(raw)
        if hint:
            st.markdown(hint)
        return False


def sign_out() -> None:
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.session_state.sb_access_token = None
    st.session_state.sb_refresh_token = None
    st.session_state.sessions_bootstrapped = False


def current_user_id() -> Optional[str]:
    uid = st.session_state.get("user_id")
    return str(uid) if uid else None


def get_chat_llm(model_name: str, *, streaming: bool, temperature: float = 1.0) -> Any:
    if model_name == MODEL_GPT:
        api_key = _api_key("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API 키가 필요합니다. 사이드바에서 입력하세요.")
        return ChatOpenAI(model=MODEL_GPT, temperature=temperature, openai_api_key=api_key, streaming=streaming)
    if model_name == MODEL_CLAUDE:
        if ChatAnthropic is None:
            raise RuntimeError("langchain_anthropic 패키지가 필요합니다.")
        ak = _api_key("ANTHROPIC_API_KEY")
        if not ak:
            raise RuntimeError("Anthropic API 키가 필요합니다. 사이드바에서 입력하세요.")
        return ChatAnthropic(model=MODEL_CLAUDE, temperature=temperature, anthropic_api_key=ak, streaming=streaming)
    if model_name == MODEL_GEMINI:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain_google_genai 패키지가 필요합니다.")
        gk = _api_key("GOOGLE_API_KEY")
        if not gk:
            raise RuntimeError("Google(Gemini) API 키가 필요합니다. 사이드바에서 입력하세요.")
        return ChatGoogleGenerativeAI(
            model=MODEL_GEMINI, temperature=temperature, google_api_key=gk, streaming=streaming
        )
    return get_chat_llm(MODEL_GPT, streaming=streaming, temperature=temperature)


class SessionRetriever(BaseRetriever):
    k: int = Field(default=10)
    _supabase: Client = PrivateAttr()
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _session_id: Optional[str] = PrivateAttr()
    _filter_user_id: Optional[str] = PrivateAttr()

    def __init__(
        self,
        supabase_client: Client,
        embeddings: OpenAIEmbeddings,
        session_id: Optional[str],
        filter_user_id: Optional[str],
        k: int = 10,
    ):
        super().__init__(k=k)
        self._supabase = supabase_client
        self._embeddings = embeddings
        self._session_id = session_id
        self._filter_user_id = filter_user_id

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        try:
            qe = self._embeddings.embed_query(query)
            result = self._supabase.rpc(
                "match_documents",
                {
                    "query_embedding": qe,
                    "match_threshold": 0.28,
                    "match_count": self.k * 4,
                    "filter_user_id": self._filter_user_id,
                },
            ).execute()
            docs: List[Document] = []
            if not result.data:
                return docs
            for item in result.data:
                meta = item.get("metadata") or {}
                if not isinstance(meta, dict):
                    meta = {}
                sid = meta.get("session_id")
                if self._session_id is None or sid == self._session_id:
                    docs.append(Document(page_content=item.get("content", ""), metadata=meta))
                if len(docs) >= self.k:
                    break
            return docs
        except Exception as e:
            logger.exception("Retriever 오류: %s", e)
            st.error(f"Retriever 오류: {e}")
            return []


def get_sessions() -> List[Dict]:
    uid = current_user_id()
    if not supabase or not uid:
        return []
    try:
        r = (
            supabase.table("sessions")
            .select("id, title, created_at, updated_at, session_id")
            .eq("user_id", uid)
            .order("updated_at", desc=True)
            .limit(200)
            .execute()
        )
        return r.data or []
    except Exception as e:
        st.error(f"세션 목록 조회 실패: {e}")
        return []


def create_session_row() -> Optional[str]:
    uid = current_user_id()
    if not supabase or not uid:
        return None
    sid = str(uuid.uuid4())
    try:
        supabase.table("sessions").insert({"id": sid, "session_id": sid, "title": "New Chat", "user_id": uid}).execute()
        return sid
    except Exception as e:
        st.error(f"세션 생성 실패: {e}")
        return None


def _first_user_assistant_pair(history: List[Dict]) -> tuple[str, str]:
    u, a = "", ""
    for m in history:
        if m.get("role") == "user" and not u:
            u = str(m.get("content") or "")
    for m in history:
        if m.get("role") == "assistant" and u:
            a = str(m.get("content") or "")
            break
    return u, a


def _title_from_history(history: List[Dict], default_title: str = "New Chat") -> str:
    u, a = _first_user_assistant_pair(history)
    if not u:
        return default_title
    if a:
        return generate_session_title(u, a)
    return (u[:19] + "…") if len(u) > 20 else u


def generate_session_title(user_q: str, ai_a: str) -> str:
    ak = _api_key("OPENAI_API_KEY")
    if not ak or not user_q:
        return (user_q[:19] + "…") if len(user_q) > 20 else (user_q or "New Chat")
    try:
        llm = ChatOpenAI(model=MODEL_GPT, temperature=0.5, openai_api_key=ak)
        prompt = f"""다음 질문과 답변을 한 줄로 요약해 세션 제목을 만드세요.

질문: {user_q[:400]}
답변: {ai_a[:500]}

규칙: 한글, 20자 이내, 따옴표 없이 제목만 출력."""
        title = (llm.invoke(prompt).content or "").strip().strip('"').strip("'")
        return (title[:19] + "…") if len(title) > 20 else (title or "New Chat")
    except Exception:
        return (user_q[:19] + "…") if len(user_q) > 20 else user_q


def _normalize_embedding(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def _doc_user_id_for_row() -> str:
    uid = current_user_id()
    return uid if uid else "anon"


def _copy_documents_to_session(old_sid: str, new_sid: str) -> int:
    if not supabase:
        return 0
    copied = 0
    page_size = 200
    offset = 0
    uid = _doc_user_id_for_row()
    while True:
        q = (
            supabase.table("documents")
            .select("content, metadata, embedding, user_id")
            .contains("metadata", {"session_id": old_sid})
            .range(offset, offset + page_size - 1)
        )
        r = q.execute()
        rows = r.data or []
        if not rows:
            break
        batch: List[Dict] = []
        for row in rows:
            meta = dict(row.get("metadata") or {})
            meta["session_id"] = new_sid
            emb = _normalize_embedding(row.get("embedding"))
            if emb is None:
                continue
            batch.append(
                {
                    "content": row.get("content") or "",
                    "metadata": meta,
                    "embedding": emb,
                    "user_id": uid,
                }
            )
        if batch:
            try:
                supabase.table("documents").insert(batch).execute()
                copied += len(batch)
            except Exception as e:
                st.warning(f"문서 복제 일부 실패: {e}")
        if len(rows) < page_size:
            break
        offset += page_size
    return copied


def _dedupe_sessions_by_title(user_id: str, title: str, keep_session_id: str) -> int:
    """같은 사용자+같은 제목 세션 중 keep_session_id 만 남기고 정리."""
    if not supabase or not user_id or not title or not keep_session_id:
        return 0
    deleted = 0
    try:
        q = (
            supabase.table("sessions")
            .select("id")
            .eq("user_id", user_id)
            .eq("title", title)
            .neq("id", keep_session_id)
            .limit(200)
        )
        rows = q.execute().data or []
        for row in rows:
            sid = str(row.get("id") or "")
            if not sid:
                continue
            try:
                supabase.table("messages").delete().eq("session_id", sid).execute()
            except Exception:
                pass
            try:
                supabase.table("documents").delete().contains("metadata", {"session_id": sid}).execute()
            except Exception:
                pass
            try:
                supabase.table("sessions").delete().eq("id", sid).eq("user_id", user_id).execute()
                deleted += 1
            except Exception:
                pass
    except Exception as e:
        logger.warning("세션 중복 정리 경고: %s", e)
    return deleted


def snapshot_append_new_session() -> bool:
    if not supabase:
        st.error("Supabase가 연결되지 않았습니다.")
        return False
    if not current_user_id():
        st.error("로그인이 필요합니다.")
        return False
    old_sid = st.session_state.current_session_id
    if not st.session_state.chat_history:
        st.warning("저장할 대화가 없습니다.")
        return False
    new_sid = str(uuid.uuid4())
    title = _title_from_history(st.session_state.chat_history)
    uid = current_user_id()
    try:
        supabase.table("sessions").insert({"id": new_sid, "session_id": new_sid, "title": title, "user_id": uid}).execute()
    except Exception as e:
        st.error(f"새 세션 행 생성 실패: {e}")
        return False
    try:
        for msg in st.session_state.chat_history:
            role = msg.get("role")
            if role == "assistant":
                role = "ai"
            if role not in ("user", "ai"):
                continue
            content = sanitize_text(str(msg.get("content") or ""))
            if not content.strip():
                continue
            supabase.table("messages").insert({"session_id": new_sid, "role": role, "content": content}).execute()
    except Exception as e:
        st.error(f"메시지 복제 실패: {e}")
        return False
    n = _copy_documents_to_session(old_sid, new_sid)
    d = _dedupe_sessions_by_title(uid, title, new_sid)
    if d > 0:
        st.success(f"새 세션으로 저장했습니다. (벡터 청크 {n}건 복제, 중복 세션 {d}건 정리)")
    else:
        st.success(f"새 세션으로 저장했습니다. (벡터 청크 {n}건 복제)")
    return True


def persist_working_session(session_id: str) -> bool:
    if not supabase or not session_id or not current_user_id():
        return False
    uid = current_user_id()
    title = _title_from_history(st.session_state.chat_history, default_title="")
    try:
        ex = supabase.table("sessions").select("id, title").eq("id", session_id).eq("user_id", uid).execute()
        if not title:
            if ex.data:
                title = ex.data[0].get("title") or "New Chat"
            else:
                title = "New Chat"
        if ex.data:
            supabase.table("sessions").update({"title": title}).eq("id", session_id).eq("user_id", uid).execute()
        else:
            supabase.table("sessions").insert({"id": session_id, "session_id": session_id, "title": title, "user_id": uid}).execute()
    except Exception as e:
        logger.warning("세션 upsert 경고: %s", e)
    try:
        supabase.table("messages").delete().eq("session_id", session_id).execute()
        for msg in st.session_state.chat_history:
            role = msg.get("role")
            if role == "assistant":
                role = "ai"
            if role not in ("user", "ai"):
                continue
            content = sanitize_text(str(msg.get("content") or ""))
            if not content.strip():
                continue
            supabase.table("messages").insert({"session_id": session_id, "role": role, "content": content}).execute()
    except Exception as e:
        st.error(f"메시지 동기화 실패: {e}")
        return False
    if title:
        _dedupe_sessions_by_title(uid, title, session_id)
    return True


def backfill_new_chat_titles() -> tuple[int, int]:
    if not supabase or not current_user_id():
        return 0, 0
    uid = current_user_id()
    updated = 0
    skipped = 0
    try:
        sessions_res = supabase.table("sessions").select("id, title").eq("user_id", uid).execute()
        sessions_rows = sessions_res.data or []
        for s in sessions_rows:
            sid = s.get("id")
            title = (s.get("title") or "").strip()
            if not sid:
                skipped += 1
                continue
            if title and title.lower() not in {"new chat", "new chat..."}:
                skipped += 1
                continue
            msgs_res = (
                supabase.table("messages")
                .select("role, content, created_at")
                .eq("session_id", sid)
                .execute()
            )
            msgs = msgs_res.data or []
            if not msgs:
                skipped += 1
                continue
            msgs.sort(key=lambda x: x.get("created_at") or "")
            history = []
            for m in msgs:
                role = m.get("role")
                if role == "ai":
                    role = "assistant"
                history.append({"role": role, "content": m.get("content") or ""})
            new_title = _title_from_history(history, default_title="").strip()
            if not new_title or new_title.lower() in {"new chat", "new chat..."}:
                skipped += 1
                continue
            supabase.table("sessions").update({"title": new_title}).eq("id", sid).eq("user_id", uid).execute()
            updated += 1
    except Exception as e:
        st.error(f"제목 일괄 보정 실패: {e}")
    return updated, skipped


def load_session(session_id: str) -> bool:
    if not supabase or not current_user_id():
        return False
    uid = current_user_id()
    try:
        own = supabase.table("sessions").select("id").eq("id", session_id).eq("user_id", uid).limit(1).execute()
        if not own.data:
            st.error("세션을 찾을 수 없거나 권한이 없습니다.")
            return False
        r = supabase.table("messages").select("id, role, content, created_at").eq("session_id", session_id).execute()
        rows = r.data or []
        rows.sort(key=lambda x: x.get("created_at") or "")
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        for msg in rows:
            role = msg.get("role") or ""
            content = msg.get("content") or ""
            if not content:
                continue
            disp = "assistant" if role == "ai" else role
            st.session_state.chat_history.append({"role": disp, "content": content})
            if role == "user":
                st.session_state.conversation_memory.append(f"사용자: {content}")
            elif role == "ai":
                st.session_state.conversation_memory.append(f"AI: {content}")
        ak = _api_key("OPENAI_API_KEY")
        if ak:
            emb = OpenAIEmbeddings(openai_api_key=ak)
            st.session_state.retriever = SessionRetriever(supabase, emb, session_id, uid, k=10)
        else:
            st.session_state.retriever = None
        srcs: set[str] = set()
        try:
            off = 0
            while True:
                dr = (
                    supabase.table("documents")
                    .select("metadata")
                    .contains("metadata", {"session_id": session_id})
                    .range(off, off + 499)
                    .execute()
                )
                part = dr.data or []
                for row in part:
                    m = row.get("metadata") or {}
                    s = m.get("source")
                    if s:
                        srcs.add(str(s))
                if len(part) < 500:
                    break
                off += 500
        except Exception:
            pass
        st.session_state.processed_files = sorted(srcs)
        return True
    except Exception as e:
        st.error(f"세션 로드 실패: {e}")
        return False


def delete_session(session_id: str) -> bool:
    if not supabase or not current_user_id():
        return False
    uid = current_user_id()
    try:
        own = supabase.table("sessions").select("id").eq("id", session_id).eq("user_id", uid).limit(1).execute()
        if not own.data:
            st.error("삭제 권한이 없습니다.")
            return False
        supabase.table("messages").delete().eq("session_id", session_id).execute()
        off = 0
        while True:
            r = (
                supabase.table("documents")
                .select("id")
                .contains("metadata", {"session_id": session_id})
                .range(off, off + 499)
                .execute()
            )
            rows = r.data or []
            for doc in rows:
                try:
                    supabase.table("documents").delete().eq("id", doc["id"]).execute()
                except Exception:
                    pass
            if len(rows) < 500:
                break
            off += 500
    except Exception:
        try:
            all_docs = supabase.table("documents").select("id, metadata").execute()
            for doc in all_docs.data or []:
                meta = doc.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("session_id") == session_id:
                    supabase.table("documents").delete().eq("id", doc["id"]).execute()
        except Exception as e:
            st.warning(f"문서 삭제 중 경고: {e}")
    try:
        supabase.table("sessions").delete().eq("id", session_id).eq("user_id", uid).execute()
        return True
    except Exception as e:
        st.error(f"세션 삭제 실패: {e}")
        return False


def save_documents_to_supabase(chunks: List[Any], embeddings: OpenAIEmbeddings, session_id: str) -> bool:
    if not supabase or not chunks or not current_user_id():
        return False
    uid = _doc_user_id_for_row()
    batch_size = 40
    saved_any = False
    for i in range(0, len(chunks), batch_size):
        part = chunks[i : i + batch_size]
        texts: List[str] = []
        metas: List[Dict] = []
        for ch in part:
            txt = sanitize_text(ch.page_content or "")
            if not txt.strip():
                continue
            meta = dict(ch.metadata or {})
            for k, v in list(meta.items()):
                if isinstance(v, str):
                    meta[k] = sanitize_text(v)
            meta["session_id"] = session_id
            texts.append(txt)
            metas.append(meta)
        if not texts:
            continue
        embs = embeddings.embed_documents(texts)
        rows = [{"content": t, "metadata": m, "embedding": e, "user_id": uid} for t, m, e in zip(texts, metas, embs)]
        try:
            supabase.table("documents").insert(rows).execute()
            saved_any = True
        except Exception as e:
            st.warning(f"문서 저장 실패: {e}")
    return saved_any


def file_already_embedded(session_id: str, source_name: str) -> bool:
    if not supabase:
        return False
    try:
        r = (
            supabase.table("documents")
            .select("id")
            .contains("metadata", {"session_id": session_id, "source": source_name})
            .limit(1)
            .execute()
        )
        return bool(r.data)
    except Exception:
        return False


def generate_followup_questions(user_q: str, answer: str, context_text: str, model_name: str) -> List[str]:
    try:
        llm = get_chat_llm(model_name, streaming=False, temperature=1.0)
        prompt = f"""질문과 답변·문맥을 보고 후속 질문 3개만 줄바꿈으로 출력하세요.

질문: {user_q}
답변: {answer[:2000]}
문맥: {context_text[:1200]}
형식: 질문만 3줄, 번호 없이."""
        txt = (llm.invoke(prompt).content or "").strip()
        qs = [q.strip() for q in txt.splitlines() if q.strip()]
        out: List[str] = []
        for line in qs:
            line = re.sub(r"^\d+[\).\s]+", "", line).strip()
            if line.startswith(("-", "•")):
                line = line[1:].strip()
            if len(line) > 5:
                out.append(line)
        return out[:3]
    except Exception as e:
        logger.warning("후속 질문 생성 실패: %s", e)
        return []


# --- Streamlit (set_page_config 를 최우선) ---
st.set_page_config(
    page_title="PDF 기반 멀티유저 RAG 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
_sb_url = ""
_sb_anon_key = ""
_sb_service_key = ""
try:
    _sb_url = str(st.secrets.get("SUPABASE_URL", "")).strip()
    _sb_anon_key = str(st.secrets.get("SUPABASE_ANON_KEY", "")).strip()
    _sb_service_key = str(st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
except Exception:
    pass
if not _sb_url:
    _sb_url = str(st.session_state.get("supabase_url_input") or "").strip()
if not _sb_anon_key and not _sb_service_key:
    _sb_anon_key = str(st.session_state.get("supabase_key_input") or "").strip()
_sb_key = _sb_anon_key or _sb_service_key
supabase = init_supabase(_sb_url, _sb_key)

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_GPT
if "sessions_bootstrapped" not in st.session_state:
    st.session_state.sessions_bootstrapped = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "sb_access_token" not in st.session_state:
    st.session_state.sb_access_token = None
if "sb_refresh_token" not in st.session_state:
    st.session_state.sb_refresh_token = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "supabase_url_input" not in st.session_state:
    st.session_state.supabase_url_input = ""
if "supabase_key_input" not in st.session_state:
    st.session_state.supabase_key_input = ""

sync_supabase_session_from_state()

st.markdown(
    """
<style>
h1 { font-size: 0.82rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 0.72rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 0.68rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
html, body, [data-testid="stAppViewContainer"] { font-size: 10px !important; }
[data-testid="stSidebar"] { font-size: 9px !important; min-width: 220px !important; }
[data-testid="stSidebar"] { position: sticky !important; top: 0 !important; height: 100vh !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarNav"] [data-testid="stSidebarNavCollapseButton"] { display: none !important; }
[data-testid="stSidebarUserContent"] { overflow-y: auto !important; max-height: 100vh !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
  font-size: 9px !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
  font-size: 9px !important;
}
.stChatMessage { font-size: 0.66rem !important; line-height: 1.15 !important; }
.stChatMessage p { font-size: 0.66rem !important; line-height: 1.15 !important; margin: 0.15rem 0 !important; }
.stButton > button {
  background-color: #ff69b4 !important; color: white !important; border: none !important;
  border-radius: 4px !important; padding: 0.12rem 0.3rem !important; font-weight: bold !important;
  font-size: 0.52rem !important;
}
.stButton > button:hover { background-color: #ff1493 !important; }
.stSidebar .stButton > button { font-size: 0.5rem !important; padding: 0.1rem 0.25rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align: center; margin-top: -3rem; margin-bottom: 0.5rem;">
  <h1 style="font-size: 0.95rem; font-weight: bold; margin: 0;">
    <span style="color: #1f77b4;">PDF 기반 멀티유저 RAG 챗봇</span>
  </h1>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("로그인 후 모델·API 키를 설정하고 PDF를 처리하세요. 세션은 Supabase에 사용자별로 저장됩니다.")

with st.sidebar:
    st.markdown('<h2 style="color: #16a085;">Supabase 연결</h2>', unsafe_allow_html=True)
    st.caption("Secrets가 없으면 아래에 직접 입력 후 연결하세요.")
    st.text_input("SUPABASE_URL", key="supabase_url_input", placeholder="https://<project-ref>.supabase.co")
    st.text_input("SUPABASE_ANON_KEY", type="password", key="supabase_key_input", placeholder="eyJ...")
    if st.button("Supabase 연결 적용", use_container_width=True):
        st.rerun()

    st.markdown('<h2 style="color: #1f77b4;">API 키</h2>', unsafe_allow_html=True)
    oa = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="inp_openai")
    an = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...", key="inp_anthropic")
    gm = st.text_input("Google (Gemini) API Key", type="password", placeholder="AIza...", key="inp_gemini")
    ensure_api_keys(oa or "", an or "", gm or "")

    st.markdown('<h2 style="color: #9b59b6;">로그인 (Supabase Auth)</h2>', unsafe_allow_html=True)
    st.caption("이 프로젝트(`SUPABASE_URL`)에 **가입된 이메일**만 로그인됩니다. 앱 전용 API 키와 무관합니다.")
    with st.form("sb_login_form", clear_on_submit=False):
        st.text_input("Login ID (이메일)", key="sb_login_id")
        st.text_input("Password", type="password", key="sb_login_pw")
        login_submitted = st.form_submit_button("로그인", use_container_width=True)
    if login_submitted:
        login_id = (st.session_state.get("sb_login_id") or "").strip()
        login_pw = st.session_state.get("sb_login_pw") or ""
        if login_id and login_pw:
            if sign_in(login_id, login_pw):
                st.session_state.sessions_bootstrapped = False
                st.success("로그인되었습니다.")
                st.rerun()
        else:
            st.warning("이메일과 비밀번호를 입력하세요.")

    if st.button("로그아웃", use_container_width=True):
        sign_out()
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.session_state.processed_files = []
        st.session_state.retriever = None
        st.success("로그아웃했습니다.")
        st.rerun()

    with st.expander("비밀번호 재설정", expanded=False):
        st.caption(
            "위 **Login ID (이메일)** 칸에 적은 주소로 재설정 메일을 보냅니다. "
            "`SUPABASE_EMAIL_REDIRECT_URL` 과 Supabase **Redirect URLs** 설정이 필요합니다."
        )
        if st.button("재설정 메일 보내기", use_container_width=True, key="btn_pw_reset"):
            send_password_reset_email((st.session_state.get("sb_login_id") or "").strip())

    st.markdown("**계정이 없으신가요?**")
    st.caption(
        "가입은 전용 페이지에서 진행합니다. 확인 메일이 안 오면 대개 **Confirm email** 이 꺼져 있거나 "
        "스팸함·Redirect URL 설정 문제입니다. 가입 페이지의 **「확인 메일이 오지 않을 때」** 를 보세요."
    )
    if hasattr(st, "page_link"):
        st.page_link("pages/회원가입.py", label="회원가입 페이지로 이동", icon="✉️")
    elif hasattr(st, "switch_page"):
        if st.button("회원가입 페이지로 이동", use_container_width=True):
            st.switch_page("pages/회원가입.py")
    else:
        st.caption("Streamlit 1.30 이상에서 `pages/회원가입.py` 멀티페이지가 표시됩니다.")

    if st.session_state.user_email:
        st.info(f"로그인: {st.session_state.user_email}")
    else:
        st.warning("로그인 후 세션·PDF 저장이 가능합니다.")

    st.markdown('<h2 style="color: #1f77b4;">LLM 선택</h2>', unsafe_allow_html=True)
    want_model = (
        st.session_state.selected_model
        if st.session_state.selected_model in ALL_CHAT_MODELS
        else ALL_CHAT_MODELS[0]
    )
    if "model_sb" not in st.session_state or st.session_state.model_sb not in ALL_CHAT_MODELS:
        st.session_state.model_sb = want_model
    st.session_state.selected_model = st.selectbox("모델", ALL_CHAT_MODELS, key="model_sb")

    with st.expander("Supabase 상태", expanded=False):
        stt = get_supabase_status()
        st.write("URL:", "OK" if stt["has_url"] else "없음")
        st.write("키:", "OK" if stt["has_key"] else "없음")
        st.write("쿼리:", "OK" if stt["query_ok"] else "실패")
        if stt.get("error"):
            st.caption(stt["error"])
        if _sb_url:
            try:
                ref = _sb_url.split("//")[1].split(".")[0] if "//" in _sb_url else _sb_url[:40]
                st.caption(f"프로젝트 ref(앞부분): `{ref}`")
            except Exception:
                pass
        if _sb_anon_key:
            st.caption("Auth용 키: **anon** (권장)")
        elif _sb_service_key:
            st.caption("Auth용 키: **service_role** (가능하나 공개 앱에서는 비권장)")
        elif st.session_state.get("supabase_key_input"):
            st.caption("Auth용 키: **sidebar 입력값 사용 중**")
        if st.button("연결 캐시 새로고침", help=".env 또는 Secrets 변경 후 클라이언트를 다시 만듭니다."):
            st.cache_resource.clear()
            st.rerun()

    st.markdown('<h2 style="color: #ffd700;">세션 관리</h2>', unsafe_allow_html=True)
    if not supabase:
        st.warning("Supabase가 연결되지 않았습니다. Secrets의 SUPABASE_URL / SUPABASE_ANON_KEY 를 확인하세요.")
    elif not current_user_id():
        st.info("세션 기능을 쓰려면 먼저 로그인하세요.")
    else:
        sessions = get_sessions()
        labels = ["(새 작업 — 목록에서 고르거나 유지)"]
        label_to_id: Dict[str, str] = {}
        for s in sessions:
            title = s.get("title") or "New Chat"
            sid = s.get("id")
            ts = s.get("updated_at") or s.get("created_at") or ""
            lab = f"{title}"
            if ts:
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    kst = timezone(timedelta(hours=9))
                    lab += f" · {dt.astimezone(kst).strftime('%m/%d %H:%M')}"
                except Exception:
                    pass
            if lab in label_to_id:
                lab = f"{lab} [{str(sid)[:8]}]"
            labels.append(lab)
            label_to_id[lab] = str(sid)

        cur_sid = st.session_state.current_session_id
        default_i = 0
        for i, s in enumerate(sessions):
            if s.get("id") == cur_sid:
                default_i = i + 1
                break
        want_label = labels[min(default_i, len(labels) - 1)]
        if st.session_state.get("reset_to_new_after_clear"):
            st.session_state.sess_sel = labels[0]
            st.session_state.reset_to_new_after_clear = False
        if "sess_sel" not in st.session_state or st.session_state.sess_sel not in labels:
            st.session_state.sess_sel = want_label
        choice = st.selectbox("세션 선택", labels, key="sess_sel")
        chosen_id = label_to_id.get(choice)

        if chosen_id and chosen_id != cur_sid:
            if persist_working_session(cur_sid):
                pass
            if load_session(chosen_id):
                st.session_state.current_session_id = chosen_id
                st.success("세션을 불러왔습니다.")
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("세션저장", use_container_width=True):
                if snapshot_append_new_session():
                    st.rerun()
        with c2:
            if st.button("세션로드", use_container_width=True):
                if not chosen_id:
                    st.warning("목록에서 세션을 먼저 선택하세요.")
                elif chosen_id == cur_sid:
                    st.info("이미 이 세션입니다.")
                else:
                    if persist_working_session(cur_sid) and load_session(chosen_id):
                        st.session_state.current_session_id = chosen_id
                        st.success("세션을 로드했습니다.")
                        st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            if st.button("세션삭제", use_container_width=True):
                if not chosen_id:
                    st.warning("삭제할 세션을 선택하세요.")
                else:
                    if delete_session(chosen_id):
                        st.success("삭제했습니다.")
                        if chosen_id == st.session_state.current_session_id:
                            nid = create_session_row() or str(uuid.uuid4())
                            st.session_state.current_session_id = nid
                            st.session_state.chat_history = []
                            st.session_state.conversation_memory = []
                            st.session_state.processed_files = []
                            st.session_state.retriever = None
                        st.rerun()
        with c4:
            if st.button("화면초기화", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_memory = []
                st.session_state.processed_files = []
                st.session_state.retriever = None
                nid = create_session_row() or str(uuid.uuid4())
                st.session_state.current_session_id = nid
                st.session_state.reset_to_new_after_clear = True
                st.success("화면을 초기화했습니다.")
                st.rerun()

        if st.button("제목 보정(New Chat)", use_container_width=True):
            with st.spinner("세션 제목 보정 중..."):
                u, s = backfill_new_chat_titles()
            st.success(f"완료: {u}개 보정, {s}개 건너뜀")
            st.rerun()

        if st.button("vectordb", use_container_width=True):
            names = set()
            sid = st.session_state.current_session_id
            uid = current_user_id()
            if supabase and uid:
                try:
                    off = 0
                    while True:
                        r = (
                            supabase.table("documents")
                            .select("metadata, user_id")
                            .contains("metadata", {"session_id": sid})
                            .range(off, off + 499)
                            .execute()
                        )
                        chunk = r.data or []
                        for d in chunk:
                            if str(d.get("user_id") or "") != str(uid):
                                continue
                            m = d.get("metadata") or {}
                            src = m.get("source")
                            if src:
                                names.add(str(src))
                        if len(chunk) < 500:
                            break
                        off += 500
                except Exception as e:
                    st.error(str(e))
            for f in st.session_state.processed_files:
                names.add(str(f))
            if names:
                st.info("벡터 DB 파일명:\n" + "\n".join(sorted(names)))
            else:
                st.warning("현재 세션에 저장된 파일명이 없습니다.")

    st.markdown("---")
    st.markdown('<h2 style="color: #ff69b4;">PDF 업로드</h2>', unsafe_allow_html=True)
    uploads = st.file_uploader("PDF", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("파일 처리하기"):
        if not current_user_id():
            st.error("PDF 저장을 위해 로그인하세요.")
        else:
            with st.spinner("PDF 처리 중…"):
                tmp = tempfile.TemporaryDirectory()
                all_docs = []
                new_names: List[str] = []
                for uf in uploads:
                    if uf.name in st.session_state.processed_files:
                        continue
                    if file_already_embedded(st.session_state.current_session_id, uf.name):
                        st.info(f"{uf.name} 은(는) 이미 임베딩되어 건너뜁니다.")
                        if uf.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uf.name)
                        continue
                    path = os.path.join(tmp.name, uf.name)
                    with open(path, "wb") as f:
                        f.write(uf.getbuffer())
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = uf.name
                    all_docs.extend(docs)
                    new_names.append(uf.name)
                if not all_docs and not new_names:
                    st.success("새로 처리할 파일이 없습니다.")
                elif all_docs:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    chunks = splitter.split_documents(all_docs)
                    ak = _api_key("OPENAI_API_KEY")
                    if not ak:
                        st.error("OpenAI API 키가 필요합니다(임베딩). 사이드바에서 입력하세요.")
                    else:
                        emb = OpenAIEmbeddings(openai_api_key=ak)
                        uid = current_user_id()
                        ok = False
                        if supabase and uid:
                            ok = save_documents_to_supabase(chunks, emb, st.session_state.current_session_id)
                            if ok:
                                st.session_state.retriever = SessionRetriever(
                                    supabase, emb, st.session_state.current_session_id, uid, k=10
                                )
                        if not ok:
                            vs = FAISS.from_documents(chunks, emb)
                            st.session_state.vectorstore = vs
                            st.session_state.retriever = vs.as_retriever(search_kwargs={"k": 10})
                            st.caption("Supabase 저장 실패·미연결 — 로컬 FAISS 로 검색합니다.")
                        for n in new_names:
                            if n not in st.session_state.processed_files:
                                st.session_state.processed_files.append(n)
                        if supabase and uid:
                            persist_working_session(st.session_state.current_session_id)
                        st.success("파일 처리 완료")
                if new_names and supabase and st.session_state.get("retriever") and current_user_id():
                    try:
                        emb2 = OpenAIEmbeddings(openai_api_key=_api_key("OPENAI_API_KEY"))
                        st.session_state.retriever = SessionRetriever(
                            supabase, emb2, st.session_state.current_session_id, current_user_id(), k=10
                        )
                    except Exception:
                        pass

    if st.session_state.processed_files:
        st.markdown("**처리된 파일**")
        for fn in st.session_state.processed_files:
            st.write("- ", fn)

# 로그인 후 최초: 최근 세션 복원 또는 현재 UUID로 sessions 행 생성
if supabase and current_user_id() and not st.session_state.sessions_bootstrapped:
    st.session_state.sessions_bootstrapped = True
    sessions = get_sessions()
    if sessions:
        latest = sessions[0]["id"]
        load_session(latest)
        st.session_state.current_session_id = latest
    else:
        cur = st.session_state.current_session_id
        uid = current_user_id()
        try:
            supabase.table("sessions").insert({"id": cur, "session_id": cur, "title": "New Chat", "user_id": uid}).execute()
        except Exception:
            nid = create_session_row()
            if nid:
                st.session_state.current_session_id = nid

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(remove_separators(str(msg.get("content", ""))), unsafe_allow_html=False)
        else:
            st.write(msg.get("content", ""))

if prompt := st.chat_input("질문을 입력하세요"):
    if not current_user_id():
        st.warning("로그인 후 질문할 수 있습니다.")
        st.stop()
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.write("먼저 PDF를 업로드하고 파일 처리하기를 눌러주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 PDF를 업로드하고 파일 처리하기를 눌러주세요."})
    else:
        try:
            retrieved = st.session_state.retriever.invoke(prompt)
            if not retrieved:
                ans = f"'{prompt}' 와 관련된 문서를 찾지 못했습니다."
                with st.chat_message("assistant"):
                    st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
            else:
                top = retrieved[:3]
                ctx = ""
                tot = 0
                for i, d in enumerate(top):
                    piece = f"[문서 {i+1}]\n{d.page_content}\n\n"
                    if tot + len(piece) > 8000:
                        break
                    ctx += piece
                    tot += len(piece)
                mem = ""
                if st.session_state.conversation_memory:
                    mem = "\n=== 이전 대화 ===\n" + "\n".join(st.session_state.conversation_memory[-50:]) + "\n"
                sys_prompt = f"""질문: {prompt}

관련 문서:
{ctx}{mem}

문서와 맥락을 바탕으로 한국어 존댓말로 답하세요. 헤딩(# ## ###)으로 구조화하세요.
문서 번호·출처 표기·구분선(---)·취소선은 쓰지 마세요."""

                llm = get_chat_llm(st.session_state.selected_model, streaming=True, temperature=1.0)
                full = ""
                with st.chat_message("assistant"):
                    ph = st.empty()
                    for ch in llm.stream(sys_prompt):
                        piece = ch.content if hasattr(ch, "content") else str(ch)
                        if piece:
                            full += piece
                            ph.markdown(remove_separators(full) + "▌")
                    fu = generate_followup_questions(prompt, full, ctx, st.session_state.selected_model)
                    if fu:
                        full += "\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n"
                        for i, q in enumerate(fu, 1):
                            full += f"{i}. {q}\n\n"
                    ph.markdown(remove_separators(full))
                full = remove_separators(full)
                st.session_state.chat_history.append({"role": "assistant", "content": full})
                st.session_state.conversation_memory.extend([f"사용자: {prompt}", f"AI: {full}"])
                st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                if supabase and current_user_id():
                    persist_working_session(st.session_state.current_session_id)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(str(e))
            st.session_state.chat_history.append({"role": "assistant", "content": f"오류: {e}"})
