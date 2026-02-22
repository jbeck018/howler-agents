#!/usr/bin/env bash
# =============================================================================
# doks-setup.sh — Initial cluster bootstrap for Howler Agents on DOKS
#
# Run once against a freshly provisioned DigitalOcean Kubernetes cluster.
# Prerequisites:
#   - kubectl configured with the DOKS cluster context
#   - helm 3.x installed
#   - doctl authenticated (optional, for context management)
#
# Usage:
#   chmod +x scripts/doks-setup.sh
#   ./scripts/doks-setup.sh
# =============================================================================
set -euo pipefail

# ---- Configuration -----------------------------------------------------------
NAMESPACE="howler-agents"
CLUSTER_CONTEXT="${KUBE_CONTEXT:-}"   # set KUBE_CONTEXT env var or leave blank
HELM_TIMEOUT="10m"

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---- Preflight checks --------------------------------------------------------
command -v kubectl >/dev/null 2>&1 || error "kubectl not found"
command -v helm    >/dev/null 2>&1 || error "helm not found"

if [[ -n "${CLUSTER_CONTEXT}" ]]; then
  info "Switching kubectl context to: ${CLUSTER_CONTEXT}"
  kubectl config use-context "${CLUSTER_CONTEXT}"
fi

info "Target cluster: $(kubectl config current-context)"
kubectl cluster-info --request-timeout=10s || error "Cannot reach cluster"

# ---- Namespace ---------------------------------------------------------------
info "Creating namespace: ${NAMESPACE}"
kubectl apply -f deploy/k8s/base/namespace.yaml

# ---- Secrets (interactive prompt — do NOT script passwords into this file) ---
info "Creating application secrets..."
warn "You will be prompted for secret values. These are NOT stored in this script."

read -rsp "Enter DATABASE_PASSWORD: " DB_PASS;      echo
read -rsp "Enter JWT_SECRET (min 32 chars): " JWT;   echo
read -rsp "Enter REDIS_PASSWORD: " REDIS_PASS;       echo
read -rsp "Enter DO Spaces ACCESS_KEY_ID: " S3_KEY;  echo
read -rsp "Enter DO Spaces SECRET_ACCESS_KEY: " S3_SECRET; echo

kubectl create secret generic howler-agents-secrets \
  --namespace "${NAMESPACE}" \
  --from-literal=DATABASE_PASSWORD="${DB_PASS}" \
  --from-literal=JWT_SECRET="${JWT}" \
  --from-literal=REDIS_PASSWORD="${REDIS_PASS}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic do-spaces-creds \
  --namespace "${NAMESPACE}" \
  --from-literal=ACCESS_KEY_ID="${S3_KEY}" \
  --from-literal=SECRET_ACCESS_KEY="${S3_SECRET}" \
  --dry-run=client -o yaml | kubectl apply -f -

# CloudNativePG expects a secret for the application DB owner in the format:
#   username / password
kubectl create secret generic howler-pg-app \
  --namespace "${NAMESPACE}" \
  --from-literal=username=howler \
  --from-literal=password="${DB_PASS}" \
  --dry-run=client -o yaml | kubectl apply -f -

unset DB_PASS JWT REDIS_PASS S3_KEY S3_SECRET

# ---- 1. CloudNativePG Operator -----------------------------------------------
info "Installing CloudNativePG operator..."
CNPG_VERSION="1.23.1"
kubectl apply --server-side \
  -f "https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-${CNPG_VERSION}/releases/cnpg-${CNPG_VERSION}.yaml"

info "Waiting for CloudNativePG webhook to become ready..."
kubectl rollout status deployment/cnpg-controller-manager \
  -n cnpg-system --timeout=180s

# ---- 2. cert-manager ---------------------------------------------------------
info "Installing cert-manager..."
CERT_MANAGER_VERSION="v1.15.0"
helm repo add jetstack https://charts.jetstack.io --force-update
helm repo update jetstack

helm upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version "${CERT_MANAGER_VERSION}" \
  --set installCRDs=true \
  --wait \
  --timeout "${HELM_TIMEOUT}"

info "Waiting for cert-manager webhook..."
kubectl rollout status deployment/cert-manager-webhook \
  -n cert-manager --timeout=120s

# ---- 3. NGINX Ingress Controller ---------------------------------------------
info "Installing NGINX Ingress Controller..."
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx --force-update
helm repo update ingress-nginx

helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/do-loadbalancer-enable-proxy-protocol"="true" \
  --set controller.config.use-proxy-protocol="true" \
  --wait \
  --timeout "${HELM_TIMEOUT}"

info "Waiting for LoadBalancer IP assignment (may take ~2 minutes on DOKS)..."
kubectl wait svc/ingress-nginx-controller \
  -n ingress-nginx \
  --for=jsonpath='{.status.loadBalancer.ingress[0].ip}' \
  --timeout=180s

LB_IP=$(kubectl get svc ingress-nginx-controller \
  -n ingress-nginx \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
info "LoadBalancer IP: ${LB_IP}"
warn "Point api.howler.dev and app.howler.dev (and staging variants) to: ${LB_IP}"

# ---- 4. Bitnami Redis --------------------------------------------------------
info "Installing Redis via Bitnami Helm chart..."
helm repo add bitnami https://charts.bitnami.com/bitnami --force-update
helm repo update bitnami

# Read the Redis password that was already stored in the secret above
REDIS_PASS=$(kubectl get secret howler-agents-secrets \
  -n "${NAMESPACE}" \
  -o jsonpath='{.data.REDIS_PASSWORD}' | base64 -d)

helm upgrade --install howler-redis bitnami/redis \
  --namespace "${NAMESPACE}" \
  --values deploy/k8s/redis/values.yaml \
  --set auth.existingSecret="" \
  --set auth.password="${REDIS_PASS}" \
  --wait \
  --timeout "${HELM_TIMEOUT}"

unset REDIS_PASS

# ---- 5. ClusterIssuer (Let's Encrypt) ----------------------------------------
info "Applying ClusterIssuer manifests..."
kubectl apply -f deploy/k8s/ingress/cluster-issuer.yaml

# ---- 6. PostgreSQL Cluster ---------------------------------------------------
info "Deploying CloudNativePG Cluster..."
kubectl apply -f deploy/k8s/postgres/cluster.yaml
kubectl apply -f deploy/k8s/postgres/scheduled-backup.yaml

info "Waiting for PostgreSQL primary to become ready (may take ~3 minutes)..."
kubectl wait cluster/howler-pg \
  -n "${NAMESPACE}" \
  --for=condition=Ready \
  --timeout=300s

# ---- 7. Application base manifests -------------------------------------------
info "Applying base Kubernetes manifests..."
kubectl apply -k deploy/k8s/base

# ---- 8. Database migrations --------------------------------------------------
info "Running database migrations..."
# Grab connection string from the CloudNativePG RW service
PG_HOST="howler-pg-rw.${NAMESPACE}.svc.cluster.local"
DB_PASS=$(kubectl get secret howler-pg-app \
  -n "${NAMESPACE}" \
  -o jsonpath='{.data.password}' | base64 -d)

# Launch a one-off migration Job using the service image
kubectl run howler-migrate \
  --namespace "${NAMESPACE}" \
  --image="ghcr.io/howler-agents/service:latest" \
  --restart=Never \
  --rm \
  --attach \
  --env="DATABASE_URL=postgresql+asyncpg://howler:${DB_PASS}@${PG_HOST}:5432/howler_agents" \
  -- python -m alembic upgrade head

unset DB_PASS

# ---- Done --------------------------------------------------------------------
info "Cluster setup complete."
info ""
info "Next steps:"
info "  1. Update your DNS: api.howler.dev -> ${LB_IP}"
info "                      app.howler.dev -> ${LB_IP}"
info "  2. For staging:     api.staging.howler.dev -> ${LB_IP}"
info "                      app.staging.howler.dev -> ${LB_IP}"
info "  3. Apply the desired overlay:"
info "     kubectl apply -k deploy/k8s/overlays/production"
info "     kubectl apply -k deploy/k8s/overlays/staging"
info "  4. Monitor cert provisioning:"
info "     kubectl get certificate -n ${NAMESPACE} -w"
