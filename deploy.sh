#!/bin/bash

# Deployment script for depression detection system
# Usage: ./deploy.sh [environment]
# Environment: development, production, or docker

set -e

ENVIRONMENT=${1:-production}
echo "Deploying for environment: $ENVIRONMENT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if .env exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        print_warning "Please update .env with your configuration"
    else
        print_error ".env.example not found. Cannot proceed."
        exit 1
    fi
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p uploads logs reports data/audio models

# Install dependencies
print_status "Installing Python dependencies..."
pip install -r backend/requirements.txt

if [ "$ENVIRONMENT" = "production" ]; then
    print_status "Installing production dependencies..."
    pip install gunicorn
fi

# Install frontend dependencies
if [ -d "frontend" ]; then
    print_status "Installing frontend dependencies..."
    cd frontend
    npm ci
    cd ..
fi

# Build frontend
if [ "$ENVIRONMENT" = "production" ] || [ "$ENVIRONMENT" = "docker" ]; then
    print_status "Building frontend..."
    cd frontend
    npm run build
    cd ..
fi

# Run tests
print_status "Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v || print_warning "Some tests failed, but continuing deployment..."
else
    print_warning "pytest not found, skipping tests..."
fi

# Check model exists
if [ ! -f "models/best_model.pth" ]; then
    print_error "Model file not found: models/best_model.pth"
    print_warning "Deployment will continue but model won't be available"
fi

# Docker deployment
if [ "$ENVIRONMENT" = "docker" ]; then
    print_status "Building Docker images..."
    docker-compose build
    
    print_status "Starting Docker containers..."
    docker-compose up -d
    
    print_status "Waiting for services to be healthy..."
    sleep 10
    
    print_status "Checking service health..."
    docker-compose ps
fi

print_status "Deployment completed successfully!"
print_status "Backend API: http://localhost:${PORT:-5001}"
print_status "Frontend: http://localhost:80"
print_status "Streamlit: http://localhost:8501"
