CREATE TABLE public.users (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_users_email ON users (email);

CREATE VIEW active_users AS
  SELECT id, email FROM users WHERE created_at > now() - INTERVAL '30 days';

CREATE FUNCTION normalize_email(address TEXT) RETURNS TEXT AS $$
  SELECT lower(trim(address));
$$ LANGUAGE SQL;

CREATE TRIGGER users_audit AFTER INSERT ON users
  FOR EACH ROW EXECUTE FUNCTION audit_users();

CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy');
