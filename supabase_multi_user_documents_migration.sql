-- multi-session-ref.sql 기반 테이블(sessions / messages / documents)에 멀티유저·RLS 적용
-- PDF 기반 멀티유저 RAG(multi-users-ref.py)는 documents.user_id 에 auth.uid 문자열을 넣습니다.
-- SQL Editor 에서 순서대로 실행하세요. (기존 DB에 사용자 없이 anon 정책만 있었다면 이 스크립트로 교체)

-- --- 1. sessions 에 소유자 컬럼 ---
alter table public.sessions add column if not exists user_id uuid references auth.users (id) on delete cascade;

-- 기존 데이터가 있다면 마이그레이션 필요(개발용 초기 데이터는 삭제 후 재생성 가능)
create index if not exists idx_sessions_user_id on public.sessions (user_id);

-- --- 2. RLS 재정의: 본인 세션만 ---
alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.documents enable row level security;

drop policy if exists "sessions_select" on public.sessions;
drop policy if exists "sessions_insert" on public.sessions;
drop policy if exists "sessions_update" on public.sessions;
drop policy if exists "sessions_delete" on public.sessions;

create policy "sessions_select_own" on public.sessions
  for select to authenticated using (auth.uid() = user_id);

create policy "sessions_insert_own" on public.sessions
  for insert to authenticated with check (auth.uid() = user_id);

create policy "sessions_update_own" on public.sessions
  for update to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "sessions_delete_own" on public.sessions
  for delete to authenticated using (auth.uid() = user_id);

-- --- 3. messages: 소유 세션에만 연결 ---
drop policy if exists "messages_select" on public.messages;
drop policy if exists "messages_insert" on public.messages;
drop policy if exists "messages_update" on public.messages;
drop policy if exists "messages_delete" on public.messages;

create policy "messages_select_own" on public.messages
  for select to authenticated
  using (exists (select 1 from public.sessions s where s.id = messages.session_id and s.user_id = auth.uid()));

create policy "messages_insert_own" on public.messages
  for insert to authenticated
  with check (exists (select 1 from public.sessions s where s.id = messages.session_id and s.user_id = auth.uid()));

create policy "messages_update_own" on public.messages
  for update to authenticated
  using (exists (select 1 from public.sessions s where s.id = messages.session_id and s.user_id = auth.uid()))
  with check (exists (select 1 from public.sessions s where s.id = messages.session_id and s.user_id = auth.uid()));

create policy "messages_delete_own" on public.messages
  for delete to authenticated
  using (exists (select 1 from public.sessions s where s.id = messages.session_id and s.user_id = auth.uid()));

-- --- 4. documents: user_id 텍스트 = auth.uid() ---
drop policy if exists "documents_select" on public.documents;
drop policy if exists "documents_insert" on public.documents;
drop policy if exists "documents_update" on public.documents;
drop policy if exists "documents_delete" on public.documents;

grant usage on schema public to authenticated;
grant execute on function public.match_documents(vector, double precision, integer, text) to authenticated;

create policy "documents_select_own" on public.documents
  for select to authenticated using (user_id = auth.uid()::text);

create policy "documents_insert_own" on public.documents
  for insert to authenticated with check (user_id = auth.uid()::text);

create policy "documents_update_own" on public.documents
  for update to authenticated using (user_id = auth.uid()::text) with check (user_id = auth.uid()::text);

create policy "documents_delete_own" on public.documents
  for delete to authenticated using (user_id = auth.uid()::text);
