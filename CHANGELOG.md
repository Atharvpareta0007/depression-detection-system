# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Model card (MODEL_CARD.md) with comprehensive model information
- Disclaimer in README and UI pages
- Comprehensive test suite (pytest)
- CI/CD pipeline (GitHub Actions)
- Explainability features (saliency maps, SHAP)
- Alternative model architectures (CNN-LSTM, Transformer)
- Language detection and OOD detection
- Evaluation scripts and metrics
- Deployment-ready structure
- Production configuration management
- Docker Compose setup
- Gunicorn configuration for production
- WSGI entry point
- Environment variable management
- Health checks for all services
- Logging configuration
- Makefile for common tasks
- Deployment script

### Changed
- Reorganized file structure for deployment
- Updated backend to use centralized configuration
- Improved OOD detection (non-blocking)
- Fixed frontend API compatibility
- Enhanced error handling

### Fixed
- OOD detection blocking predictions
- Frontend blank page issue
- API response format compatibility
- Training script scheduler issue
- Preprocessing import conflicts

## [1.0.0] - 2024

### Initial Release
- 75% accuracy depression detection model
- React frontend
- Flask backend API
- Streamlit interface
- MFCC feature extraction
- Audio preprocessing pipeline

