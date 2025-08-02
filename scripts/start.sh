#!/bin/bash

# VigilantBytes AI - Startup Script
# This script sets up and starts the VigilantBytes AI fraud detection system

set -e

echo "🚀 Starting VigilantBytes AI Fraud Detection System"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_color $RED "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_color $RED "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    print_color $BLUE "🔍 Checking if $service is running on port $port..."
    
    if curl -s -f "http://localhost:$port" > /dev/null 2>&1; then
        print_color $GREEN "✅ $service is running"
        return 0
    else
        print_color $YELLOW "⏳ $service is not ready yet"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    print_color $BLUE "⏳ Waiting for $service to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if check_service "$service" "$port"; then
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 2
    done
    
    print_color $RED "❌ $service failed to start within expected time"
    return 1
}

# Parse command line arguments
PROFILE=""
GENERATE_DATA=false
TRAIN_MODELS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-monitoring)
            PROFILE="--profile monitoring"
            shift
            ;;
        --generate-data)
            GENERATE_DATA=true
            shift
            ;;
        --train-models)
            TRAIN_MODELS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-monitoring    Start with monitoring stack (Prometheus & Grafana)"
            echo "  --generate-data      Generate synthetic training data"
            echo "  --train-models       Train ML models after startup"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
print_color $BLUE "📁 Creating necessary directories..."
mkdir -p data/{models,raw,processed} backend/logs

# Start services
print_color $BLUE "🐳 Starting Docker services..."
docker-compose up -d $PROFILE

# Wait for services to be ready
print_color $BLUE "⏳ Waiting for services to start..."

# Wait for backend API
if wait_for_service "Backend API" "8000"; then
    print_color $GREEN "✅ Backend API is ready"
else
    print_color $RED "❌ Backend API failed to start"
    docker-compose logs backend
    exit 1
fi

# Wait for frontend
if wait_for_service "Frontend" "3000"; then
    print_color $GREEN "✅ Frontend is ready"
else
    print_color $RED "❌ Frontend failed to start"
    docker-compose logs frontend
    exit 1
fi

# Generate synthetic data if requested
if [ "$GENERATE_DATA" = true ]; then
    print_color $BLUE "🎲 Generating synthetic training data..."
    docker-compose exec backend python scripts/generate_training_data.py
    print_color $GREEN "✅ Synthetic data generated"
fi

# Train models if requested
if [ "$TRAIN_MODELS" = true ]; then
    print_color $BLUE "🤖 Training ML models..."
    curl -X POST "http://localhost:8000/api/models/retrain"
    print_color $GREEN "✅ Model training initiated"
fi

# Show service status
print_color $GREEN "🎉 VigilantBytes AI is running!"
echo ""
print_color $BLUE "📊 Service URLs:"
echo "  • Main Dashboard:     http://localhost:3000"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • Health Check:      http://localhost:8000/health"

if echo "$PROFILE" | grep -q "monitoring"; then
    echo "  • Prometheus:        http://localhost:9090"
    echo "  • Grafana:          http://localhost:3001 (admin/admin)"
fi

echo ""
print_color $BLUE "🔧 Management Commands:"
echo "  • View logs:         docker-compose logs -f [service]"
echo "  • Stop services:     docker-compose down"
echo "  • Restart service:   docker-compose restart [service]"
echo "  • Generate data:     $0 --generate-data"
echo "  • Train models:      $0 --train-models"

echo ""
print_color $YELLOW "📋 Next Steps:"
echo "  1. Open http://localhost:3000 to access the dashboard"
echo "  2. Generate training data if this is your first run"
echo "  3. Test the fraud detection by submitting transactions"
echo "  4. Monitor real-time alerts and analytics"

print_color $GREEN "✨ Happy fraud detecting! ✨"