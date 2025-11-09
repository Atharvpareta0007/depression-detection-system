# Deployment Structure

This document describes the deployment-ready file structure.

## Directory Structure

```
major2.0_clean/
├── backend/                 # Flask API backend
│   ├── app.py              # Main Flask application
│   ├── wsgi.py             # WSGI entry point for production
│   ├── requirements.txt    # Backend dependencies
│   └── uploads/            # Upload directory
│
├── frontend/               # React frontend
│   ├── src/               # Source code
│   ├── dist/              # Built files (generated)
│   ├── package.json       # Node dependencies
│   └── vite.config.js     # Vite configuration
│
├── src/                   # Core Python source
│   ├── model.py           # Model architecture
│   ├── inference.py       # Inference system
│   ├── preprocessing.py   # Audio preprocessing
│   ├── explain/           # Explainability modules
│   ├── models/            # Alternative model architectures
│   └── preprocessing/     # Preprocessing utilities
│
├── models/                # Trained models
│   └── best_model.pth     # Best model (75% accuracy)
│
├── tests/                 # Test suite
│   ├── test_*.py          # Test files
│   └── assets/            # Test assets
│
├── scripts/               # Utility scripts
│   ├── evaluate.py       # Evaluation script
│   ├── split_dataset.py  # Data splitting
│   └── smoke_inference.py # Smoke test
│
├── docs/                  # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── evaluation.md
│   └── temporal_models.md
│
├── deployment/            # Deployment guides
│   └── *.md               # Platform-specific guides
│
├── config.py              # Configuration management
├── gunicorn_config.py     # Gunicorn configuration
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile.backend     # Backend Dockerfile
├── Dockerfile.frontend    # Frontend Dockerfile
├── Dockerfile.streamlit   # Streamlit Dockerfile
├── nginx.conf             # Nginx configuration
├── .env.example           # Environment variables example
├── .dockerignore          # Docker ignore file
├── .gitignore             # Git ignore file
├── Makefile               # Make commands
├── deploy.sh              # Deployment script
├── requirements.txt       # Production requirements
├── requirements-dev.txt   # Development requirements
└── README.md              # Main README
```

## Key Files for Deployment

### Configuration Files

- **config.py**: Centralized configuration management
- **.env.example**: Environment variables template
- **gunicorn_config.py**: Production WSGI server config

### Docker Files

- **Dockerfile.backend**: Backend container
- **Dockerfile.frontend**: Frontend container
- **Dockerfile.streamlit**: Streamlit container
- **docker-compose.yml**: Multi-container orchestration
- **.dockerignore**: Files to exclude from Docker builds

### Deployment Scripts

- **deploy.sh**: Automated deployment script
- **Makefile**: Common commands

### Entry Points

- **backend/app.py**: Development Flask app
- **backend/wsgi.py**: Production WSGI entry point

## Environment Variables

See `.env.example` for all available environment variables.

Key variables:
- `FLASK_ENV`: Environment (development/production)
- `PORT`: Backend port
- `MODEL_PATH`: Path to model file
- `SECRET_KEY`: Flask secret key
- `CORS_ORIGINS`: Allowed CORS origins

## Quick Deployment

### Docker (Recommended)

```bash
cp .env.example .env
# Edit .env
docker-compose up -d
```

### Production Server

```bash
cp .env.example .env
# Edit .env
pip install -r requirements.txt
gunicorn -c gunicorn_config.py backend.wsgi:app
```

## Health Checks

All services include health checks:
- Backend: `GET /api/metrics`
- Frontend: `GET /health`
- Streamlit: `GET /_stcore/health`

## Monitoring

- Logs: `logs/app.log`
- Access logs: Configured in gunicorn
- Error logs: Configured in gunicorn

