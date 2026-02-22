"""Run management endpoint tests."""

import uuid


def test_create_run(client):
    response = client.post("/api/v1/runs", json={
        "config": {
            "population_size": 10,
            "group_size": 3,
            "num_iterations": 5,
        }
    })
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"
    assert data["total_generations"] == 5
    # Verify ID is a valid UUID
    uuid.UUID(data["id"])


def test_get_run(client):
    create_resp = client.post("/api/v1/runs", json={"config": {"num_iterations": 3}})
    run_id = create_resp.json()["id"]

    resp = client.get(f"/api/v1/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run_id


def test_get_run_not_found(client):
    fake_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/runs/{fake_id}")
    assert resp.status_code == 404


def test_list_runs(client):
    client.post("/api/v1/runs", json={"config": {}})
    resp = client.get("/api/v1/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert "total" in data


def test_list_runs_pagination(client):
    # Create multiple runs
    for _ in range(3):
        client.post("/api/v1/runs", json={"config": {}})

    resp = client.get("/api/v1/runs?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) <= 2


def test_step_evolution(client):
    create_resp = client.post("/api/v1/runs", json={"config": {"num_iterations": 2}})
    run_id = create_resp.json()["id"]

    resp = client.post(f"/api/v1/runs/{run_id}/step")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["current_generation"] == 1


def test_step_evolution_completes(client):
    create_resp = client.post("/api/v1/runs", json={"config": {"num_iterations": 1}})
    run_id = create_resp.json()["id"]

    resp = client.post(f"/api/v1/runs/{run_id}/step")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_step_completed_run_fails(client):
    create_resp = client.post("/api/v1/runs", json={"config": {"num_iterations": 1}})
    run_id = create_resp.json()["id"]

    client.post(f"/api/v1/runs/{run_id}/step")  # completes
    resp = client.post(f"/api/v1/runs/{run_id}/step")  # should fail
    assert resp.status_code == 400
