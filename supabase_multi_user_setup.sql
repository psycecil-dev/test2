-- Supabase 멀티유저 벡터 데이터베이스 설정 SQL
-- Supabase SQL Editor에서 실행하세요

-- 1. pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. embeddings 테이블 생성 (사용자별 분리)
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,  -- 사용자 ID 추가
    session_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI embeddings는 1536 차원
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, session_id, file_name, chunk_index)
);

-- 3. 인덱스 생성 (벡터 검색 성능 향상)
CREATE INDEX IF NOT EXISTS embeddings_user_id_idx ON embeddings(user_id);
CREATE INDEX IF NOT EXISTS embeddings_session_id_idx ON embeddings(session_id);
CREATE INDEX IF NOT EXISTS embeddings_file_name_idx ON embeddings(file_name);
CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops);

-- 4. 벡터 검색 함수 생성 (RPC) - 사용자별 필터링
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10,
    user_id_filter UUID DEFAULT NULL,
    session_id_filter text DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    user_id uuid,
    session_id text,
    file_name text,
    chunk_index integer,
    chunk_text text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.user_id,
        e.session_id,
        e.file_name,
        e.chunk_index,
        e.chunk_text,
        e.metadata,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM embeddings e
    WHERE 
        (user_id_filter IS NULL OR e.user_id = user_id_filter)
        AND (session_id_filter IS NULL OR e.session_id = session_id_filter)
        AND (1 - (e.embedding <=> query_embedding)) >= match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 5. sessions 테이블 생성 (사용자별 세션 관리)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,  -- 사용자 ID 추가
    session_id TEXT NOT NULL,
    title TEXT,
    chat_history TEXT,  -- JSON 문자열
    conversation_memory TEXT,  -- JSON 문자열
    processed_files TEXT,  -- JSON 문자열
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, session_id)
);

-- 6. sessions 테이블 인덱스 생성
CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_session_id_idx ON sessions(session_id);
CREATE INDEX IF NOT EXISTS sessions_updated_at_idx ON sessions(updated_at DESC);

-- 7. 세션 삭제 시 관련 임베딩도 삭제하는 트리거 함수
CREATE OR REPLACE FUNCTION delete_session_embeddings()
RETURNS TRIGGER AS $$
BEGIN
    DELETE FROM embeddings 
    WHERE user_id = OLD.user_id AND session_id = OLD.session_id;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- 8. 트리거 생성
DROP TRIGGER IF EXISTS delete_embeddings_on_session_delete ON sessions;
CREATE TRIGGER delete_embeddings_on_session_delete
AFTER DELETE ON sessions
FOR EACH ROW
EXECUTE FUNCTION delete_session_embeddings();

-- 9. RLS (Row Level Security) 정책 설정 (선택사항)
-- 사용자별 데이터 분리를 위해 RLS 활성화 권장
ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

-- 기존 정책이 있으면 삭제
DROP POLICY IF EXISTS "Users can view their own embeddings" ON embeddings;
DROP POLICY IF EXISTS "Users can insert their own embeddings" ON embeddings;
DROP POLICY IF EXISTS "Users can delete their own embeddings" ON embeddings;
DROP POLICY IF EXISTS "Users can view their own sessions" ON sessions;
DROP POLICY IF EXISTS "Users can insert their own sessions" ON sessions;
DROP POLICY IF EXISTS "Users can update their own sessions" ON sessions;
DROP POLICY IF EXISTS "Users can delete their own sessions" ON sessions;

-- 사용자가 자신의 데이터만 볼 수 있도록 정책 생성
CREATE POLICY "Users can view their own embeddings"
ON embeddings FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own embeddings"
ON embeddings FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own embeddings"
ON embeddings FOR DELETE
USING (auth.uid() = user_id);

CREATE POLICY "Users can view their own sessions"
ON sessions FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own sessions"
ON sessions FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own sessions"
ON sessions FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own sessions"
ON sessions FOR DELETE
USING (auth.uid() = user_id);

-- 10. updated_at 자동 업데이트 트리거
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 기존 트리거가 있으면 삭제
DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;

-- 트리거 생성
CREATE TRIGGER update_sessions_updated_at
BEFORE UPDATE ON sessions
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

