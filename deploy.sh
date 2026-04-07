#!/usr/bin/env bash
# deploy.sh — Deploy WADA Compliance Agent to Google Cloud Run
# Usage: ./deploy.sh [--project PROJECT_ID] [--region REGION]
set -euo pipefail

# ── Config (override via args or env) ────────────────────────────────────────
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
SERVICE_NAME="wada-compliance-agent"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
TAG="${TAG:-latest}"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --region)  REGION="$2";     shift 2 ;;
    --tag)     TAG="$2";        shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

IMAGE="${IMAGE_NAME}:${TAG}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  WADA Compliance Agent — Cloud Run Deployment"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Image   : ${IMAGE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Authenticate & set project ────────────────────────────────────────────
echo ""
echo "▶ Step 1 — Set GCP project"
gcloud config set project "${PROJECT_ID}"

# ── 2. Enable required APIs ───────────────────────────────────────────────────
echo ""
echo "▶ Step 2 — Enable required APIs"
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  containerregistry.googleapis.com \
  aiplatform.googleapis.com \
  --project "${PROJECT_ID}"

# ── 3. Store GEMINI_API_KEY in Secret Manager ─────────────────────────────────
echo ""
echo "▶ Step 3 — Configure GEMINI_API_KEY secret"

if ! gcloud secrets describe gemini-api-key --project "${PROJECT_ID}" &>/dev/null; then
  echo "Creating secret 'gemini-api-key'..."
  if [ -z "${GEMINI_API_KEY:-}" ]; then
    read -rsp "Enter your Gemini API key: " GEMINI_API_KEY
    echo
  fi
  echo -n "${GEMINI_API_KEY}" | gcloud secrets create gemini-api-key \
    --data-file=- \
    --project "${PROJECT_ID}"
  echo "Secret created."
else
  echo "Secret 'gemini-api-key' already exists. Skipping."
fi

# ── 4. Build Docker image via Cloud Build ────────────────────────────────────
echo ""
echo "▶ Step 4 — Build Docker image with Cloud Build"
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --timeout=15m \
  .

echo "Image built: ${IMAGE}"

# ── 5. Deploy to Cloud Run ────────────────────────────────────────────────────
echo ""
echo "▶ Step 5 — Deploy to Cloud Run"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --port 8080 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300s \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_REGION=${REGION},LOG_LEVEL=INFO,WADA_CACHE_TTL_HOURS=6"

# ── 6. Get service URL ────────────────────────────────────────────────────────
echo ""
echo "▶ Step 6 — Fetching service URL"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Deployment complete!"
echo ""
echo "  Service URL : ${SERVICE_URL}"
echo "  Health      : ${SERVICE_URL}/health"
echo "  API Docs    : ${SERVICE_URL}/docs"
echo ""
echo "  Test the deployment:"
echo "  curl -X POST ${SERVICE_URL}/analyze-supplement \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"input_type\": \"text\", \"data\": \"caffeine 200mg, vitamin c 500mg\"}'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
