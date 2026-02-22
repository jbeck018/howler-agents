#!/usr/bin/env bash
# Scale howler-agents and orochi-cloud deployments up/down.
# Usage:
#   ./scripts/scale.sh up     # Scale everything to 1 replica
#   ./scripts/scale.sh down   # Scale everything to 0 (DB + Redis stay up)
#   ./scripts/scale.sh status # Show current replica counts
set -euo pipefail

case "${1:-status}" in
  up)
    echo "Scaling howler-agents UP..."
    kubectl scale deployment howler-agents-service --replicas=1 -n howler-agents
    kubectl scale deployment howler-agents-ui --replicas=1 -n howler-agents
    kubectl scale deployment howler-agents-docs --replicas=1 -n howler-agents
    echo ""
    echo "Scaling orochi-cloud UP..."
    kubectl scale deployment control-plane --replicas=1 -n orochi-cloud
    kubectl scale deployment dashboard --replicas=1 -n orochi-cloud
    kubectl scale deployment pgdog --replicas=1 -n orochi-cloud
    kubectl scale deployment provisioner --replicas=1 -n orochi-cloud
    echo ""
    echo "Done. All deployments scaled to 1."
    ;;
  down)
    echo "Scaling howler-agents DOWN..."
    kubectl scale deployment howler-agents-service --replicas=0 -n howler-agents
    kubectl scale deployment howler-agents-ui --replicas=0 -n howler-agents
    kubectl scale deployment howler-agents-docs --replicas=0 -n howler-agents
    echo ""
    echo "Scaling orochi-cloud DOWN..."
    kubectl scale deployment control-plane --replicas=0 -n orochi-cloud
    kubectl scale deployment dashboard --replicas=0 -n orochi-cloud
    kubectl scale deployment pgdog --replicas=0 -n orochi-cloud
    kubectl scale deployment provisioner --replicas=0 -n orochi-cloud
    echo ""
    echo "Done. App deployments at 0. DB/Redis/Ingress still running."
    ;;
  status)
    echo "=== howler-agents ==="
    kubectl get deploy -n howler-agents -o custom-columns='NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas'
    echo ""
    echo "=== orochi-cloud ==="
    kubectl get deploy -n orochi-cloud -o custom-columns='NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas'
    echo ""
    echo "=== Infrastructure ==="
    kubectl get pods -n howler-agents -l 'app.kubernetes.io/component=database' --no-headers 2>/dev/null || true
    kubectl get pods -n howler-agents -l 'cnpg.io/cluster=howler-pg' --no-headers 2>/dev/null || true
    kubectl get pods -n howler-agents -l 'app.kubernetes.io/name=redis' --no-headers 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {up|down|status}"
    exit 1
    ;;
esac
