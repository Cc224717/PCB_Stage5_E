-- PCB 缺陷检测系统 - MySQL 库表
-- Database: pcb_defect

CREATE TABLE IF NOT EXISTS users (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    username        VARCHAR(64) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    salt            VARCHAR(64) NOT NULL,
    display_name    VARCHAR(128) NULL,
    created_at      DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    last_login_at   DATETIME(3) NULL,
    is_active       TINYINT(1) NOT NULL DEFAULT 1,
    INDEX idx_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sessions (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id      VARCHAR(64) NOT NULL UNIQUE,
    streamlit_session_id VARCHAR(128) NULL,
    operator        VARCHAR(128) NOT NULL DEFAULT '未登录',
    purpose         VARCHAR(255) NULL,
    opened_at       DATETIME(3) NOT NULL,
    closed_at       DATETIME(3) NULL,
    close_reason    VARCHAR(64) NULL,
    click_count     INT NOT NULL DEFAULT 0,
    file_count      INT NOT NULL DEFAULT 0,
    status          VARCHAR(32) NOT NULL DEFAULT 'open',
    created_at      DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    INDEX idx_sessions_operator (operator),
    INDEX idx_sessions_opened (opened_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS audit_events (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id      VARCHAR(64) NOT NULL,
    operator        VARCHAR(128) NOT NULL DEFAULT '未登录',
    event_type      VARCHAR(64) NOT NULL,
    message         TEXT NULL,
    action          VARCHAR(128) NULL,
    level           VARCHAR(16) NOT NULL DEFAULT 'INFO',
    payload_json    JSON NULL,
    created_at      DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_events_session (session_id),
    INDEX idx_events_type (event_type),
    INDEX idx_events_created (created_at),
    CONSTRAINT fk_events_session
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS detect_jobs (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id      VARCHAR(64) NOT NULL,
    operator        VARCHAR(128) NOT NULL,
    mode            VARCHAR(32) NOT NULL,
    file_name       VARCHAR(255) NOT NULL,
    file_path       TEXT NULL,
    source_type     VARCHAR(32) NULL,
    model_name      VARCHAR(128) NULL,
    conf            DOUBLE NULL,
    iou             DOUBLE NULL,
    imgsz           INT NULL,
    status          VARCHAR(32) NOT NULL,
    detection_count INT NOT NULL DEFAULT 0,
    class_counter_json JSON NULL,
    predict_call_ms DOUBLE NULL,
    total_ms        DOUBLE NULL,
    annotated_image TEXT NULL,
    result_json     TEXT NULL,
    output_dir      TEXT NULL,
    error_message   TEXT NULL,
    started_at      DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    finished_at     DATETIME(3) NULL,
    INDEX idx_jobs_session (session_id),
    INDEX idx_jobs_file (file_name),
    INDEX idx_jobs_status (status),
    CONSTRAINT fk_jobs_session
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS detections (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    job_id          BIGINT NOT NULL,
    session_id      VARCHAR(64) NOT NULL,
    file_name       VARCHAR(255) NOT NULL,
    class_id        INT NULL,
    class_name      VARCHAR(64) NOT NULL,
    confidence      DOUBLE NOT NULL,
    x1              DOUBLE NOT NULL,
    y1              DOUBLE NOT NULL,
    x2              DOUBLE NOT NULL,
    y2              DOUBLE NOT NULL,
    created_at      DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    INDEX idx_det_job (job_id),
    INDEX idx_det_class (class_name),
    CONSTRAINT fk_det_job
        FOREIGN KEY (job_id) REFERENCES detect_jobs(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
