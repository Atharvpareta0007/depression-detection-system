# Project Structure

This document describes the deployment-ready project structure.

## Directory Organization

```
major2.0_clean/
├── backend/                    # Flask Backend API
│   ├── app.py                # Main Flask application
│   ├── wsgi.py                 # WSGI entry point (production)
│   ├── requirements.txt        # Backend dependencies
│   ├── Dockerfile             # Backend Dockerfile (legacy)
│   └── uploads/               # Upload directory
│       └── .gitkeep
│
├── frontend/                   # React Frontend
│   ├── src/                   # Source code
│   │   ├── components/        # React components
│   │   ├── pages/            # Page components
│   │   ├── config.js         # API configuration
│   │   └── ...
│   ├── dist/                  # Built files (generated)
│   ├── package.json          # Node dependencies
│   ├── vite.config.js         # Vite configuration
│   ├── nginx.conf             # Nginx config for frontend
│   └── vercel.json           # Vercel deployment config
│
├── src/                       # Core Python Source
│   ├── __init__.py
│   ├── model.py              # Main model architecture
│   ├── inference.py          # Inference system
│   ├── preprocessing.py     # Audio preprocessing
│   ├── train.py              # Training script
│   ├── explain/              # Explainability modules
│   │   ├── saliency.py       # Saliency maps
│   │   └── shap_explain.py   # SHAP explainability
│   ├── models/               # Alternative architectures
│   │   ├── cnn_lstm.py       # CNN-LSTM model
│   │   └── transformer_model.py  # Transformer model
│   └── preprocessing/        # Preprocessing utilities
│       └── language_detection.py
│
├── models/                    # Trained Models
│   ├── best_model.pth        # Best model (75% accuracy)
│   ├── cross_validation_results.json
│   ├── training_curves.png
│   └── training_history.json
│
├── tests/                     # Test Suite
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_api.py
│   └── assets/               # Test assets
│       └── test_audio.wav
│
├── scripts/                   # Utility Scripts
│   ├── evaluate.py           # Evaluation script
│   ├── split_dataset.py      # Data splitting
│   ├── smoke_inference.py    # Smoke test
│   ├── prepare_data.py
│   └── verify_model.py
│
├── docs/                      # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── evaluation.md
│   └── temporal_models.md
│
├── deployment/               # Deployment Guides
│   ├── DEPLOY_NOW.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── DOCKER_DEPLOYMENT.md
│   └── RAILWAY_DEPLOY.md
│
├── notebooks/                 # Jupyter Notebooks
│   └── explainability_demo.ipynb
│
├── data/                      # Data Directory
│   ├── audio/                # Audio files
│   │   └── .gitkeep
│   └── training_data.csv
│
├── uploads/                   # Upload Directory
│   └── .gitkeep
│
├── logs/                      # Log Directory
│   └── .gitkeep
│
├── reports/                   # Reports Directory
│   └── .gitkeep
│
├── config.py                 # Configuration Management
├── gunicorn_config.py         # Gunicorn Configuration
├── requirements.txt          # Production Requirements
├── requirements-dev.txt      # Development Requirements
│
├── Dockerfile.backend        # Backend Dockerfile
├── Dockerfile.frontend       # Frontend Dockerfile
├── Dockerfile.streamlit      # Streamlit Dockerfile
├── docker-compose.yml        # Docker Compose Config
├── .dockerignore             # Docker Ignore
│
├── .env.example              # Environment Variables Template
├── .gitignore                # Git Ignore
│
├── deploy.sh                 # Deployment Script
├── Makefile                  # Make Commands
│
├── nginx.conf                # Nginx Configuration
│
├── MODEL_CARD.md             # Model Card
├── README.md                 # Main README
├── README_DEPLOYMENT.md      # Deployment README
├── DEPLOYMENT.md             # Deployment Guide
├── CHANGELOG.md              # Changelog
└── PROJECT_STRUCTURE.md      # This file
```

## Key Files

### Configuration
- **config.py**: Centralized configuration management
- **.env.example**: Environment variables template
- **gunicorn_config.py**: Production WSGI server config

### Docker
- **Dockerfile.backend**: Backend container definition
- **Dockerfile.frontend**: Frontend container definition
- **Dockerfile.streamlit**: Streamlit container definition
- **docker-compose.yml**: Multi-container orchestration
- **.dockerignore**: Files excluded from Docker builds

### Deployment
- **deploy.sh**: Automated deployment script
- **Makefile**: Common deployment commands
- **backend/wsgi.py**: WSGI entry point for production

### Documentation
- **README.md**: Main project documentation
- **DEPLOYMENT.md**: Deployment guide
- **README_DEPLOYMENT.md**: Deployment structure
- **MODEL_CARD.md**: Model information
- **CHANGELOG.md**: Version history

## Deployment Structure

### Development
```
backend/app.py (Flask dev server)
frontend/ (Vite dev server)
streamlit_app.py (Streamlit)
```

### Production
```
backend/wsgi.py (Gunicorn)
frontend/dist/ (Static files via Nginx)
docker-compose.yml (Orchestration)
```

## Environment Variables

All configuration is managed through environment variables. See `.env.example` for details.

## Health Checks

All services include health check endpoints:
- Backend: `/api/metrics`
- Frontend: `/health`
- Streamlit: `/_stcore/health`

