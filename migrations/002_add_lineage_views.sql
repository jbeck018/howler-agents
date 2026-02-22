-- Lineage statistics view
CREATE OR REPLACE VIEW agent_lineage_stats AS
SELECT
    a.run_id,
    a.id AS agent_id,
    a.generation,
    a.parent_id,
    a.performance_score,
    a.novelty_score,
    a.combined_score,
    COUNT(DISTINCT c.id) AS child_count,
    COALESCE(AVG(c.combined_score), 0) AS avg_child_score,
    COUNT(DISTINCT t.id) AS trace_count,
    COUNT(DISTINCT p.id) AS patch_count
FROM agents a
LEFT JOIN agents c ON c.parent_id = a.id
LEFT JOIN evolutionary_traces t ON t.agent_id = a.id
LEFT JOIN framework_patches p ON p.agent_id = a.id
GROUP BY a.run_id, a.id, a.generation, a.parent_id,
         a.performance_score, a.novelty_score, a.combined_score;

-- Run progress view
CREATE OR REPLACE VIEW run_progress AS
SELECT
    r.id AS run_id,
    r.status,
    r.current_generation,
    r.total_generations,
    r.best_score,
    COUNT(DISTINCT a.id) AS total_agents,
    COUNT(DISTINCT g.id) AS total_groups,
    COUNT(DISTINCT t.id) AS total_traces,
    COALESCE(AVG(a.combined_score), 0) AS avg_fitness,
    COALESCE(MAX(a.combined_score), 0) AS max_fitness,
    COALESCE(STDDEV(a.combined_score), 0) AS fitness_stddev,
    r.created_at,
    r.updated_at
FROM evolution_runs r
LEFT JOIN agents a ON a.run_id = r.id
LEFT JOIN agent_groups g ON g.run_id = r.id
LEFT JOIN evolutionary_traces t ON t.run_id = r.id
GROUP BY r.id, r.status, r.current_generation, r.total_generations,
         r.best_score, r.created_at, r.updated_at;
