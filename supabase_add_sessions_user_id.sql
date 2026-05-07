-- 에러 해결: column sessions.user_id does not exist
-- Supabase Dashboard → SQL Editor 에서 실행하세요.

-- 로그인한 사용자(auth.users)별로 세션을 구분하기 위한 필수 컬럼
ALTER TABLE public.sessions
ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES auth.users (id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions (user_id);

-- 참고: 이전 멀티세션(anon) 테스트로 만든 행은 user_id 가 NULL 일 수 있습니다.
-- 멀티유저 앱에서는 조회 시 user_id 로 필터하므로, 필요하면 초기 데이터를 비우세요:
-- TRUNCATE public.messages CASCADE; TRUNCATE public.sessions CASCADE;
-- (문서 임베딩까지 지우려면 documents 도 별도 정리)
