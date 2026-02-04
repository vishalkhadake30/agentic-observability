#!/bin/bash

# Quick Start Script for Agentic Observability Platform
# This script starts the full infrastructure stack

set -e

echo "Starting Agentic Observability Platform..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Start Docker services
echo "Starting Docker services (TimescaleDB, Redis, Qdrant, Jaeger, Prometheus, Grafana)..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check service health
echo "Checking service health..."
docker-compose ps

echo ""
echo "Services are ready:"
echo "   - Grafana:    http://localhost:3000 (admin/admin)"
echo "   - Prometheus: http://localhost:9090"
echo "   - Jaeger:     http://localhost:16686"
echo "   - TimescaleDB: localhost:5432"
echo "   - Redis:       localhost:6379"
echo "   - Qdrant:      localhost:6333"
echo ""

# Activate virtual environment and start API
echo "Starting FastAPI server..."
source .venv/bin/activate
python -m uvicorn agentic_observability.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 5

echo ""
echo "Platform is ready!"
echo ""
echo "API Documentation:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc:      http://localhost:8000/redoc"
echo ""
echo "Health Check:"
curl -s http://localhost:8000/api/v1/health | python -m json.tool
echo ""
echo ""
echo "View Grafana Dashboard:"
echo "   1. Open http://localhost:3000"
echo "   2. Login with admin/admin"
echo "   3. Navigate to: Dashboards → Agentic Observability Platform"
echo ""
echo "Test Pipeline:"
echo "   curl -X POST http://localhost:8000/api/v1/pipeline/process \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{"
echo "       \"service_name\": \"test-service\","
echo "       \"metric_name\": \"response_time\","
echo "       \"metrics\": [95, 96, 97, 98, 150, 200, 250, 300]"
echo "     }'"
echo ""
echo "To stop: Press Ctrl+C, then run: make docker-down"
echo ""

# Wait for API process
wait $API_PID
