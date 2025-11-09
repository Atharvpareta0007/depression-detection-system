# Deployment Guide

This guide covers deploying the Depression Detection System to various platforms.

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Platform-Specific Guides](#platform-specific-guides)
- [Environment Variables](#environment-variables)
- [Health Checks](#health-checks)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Update .env with your configuration
nano .env

# 3. Deploy
./deploy.sh docker

# Or manually:
docker-compose up -d
```

### Manual Deployment

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt
cd frontend && npm install && npm run build && cd ..

# 2. Set environment variables
export FLASK_ENV=production
export PORT=5001

# 3. Run backend
cd backend
gunicorn -c ../gunicorn_config.py wsgi:app

# 4. Serve frontend (separate terminal)
cd frontend
npx serve -s dist -l 80
```

## Docker Deployment

### Prerequisites

- Docker
- Docker Compose

### Steps

1. **Configure Environment**

```bash
cp .env.example .env
# Edit .env with your settings
```

2. **Build and Start**

```bash
docker-compose up -d
```

3. **Check Status**

```bash
docker-compose ps
docker-compose logs -f
```

4. **Stop Services**

```bash
docker-compose down
```

### Services

- **Backend API**: http://localhost:5001
- **Frontend**: http://localhost:80
- **Streamlit**: http://localhost:8501

## Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -c gunicorn_config.py backend.wsgi:app
```

### Using Systemd (Linux)

Create `/etc/systemd/system/depression-detection.service`:

```ini
[Unit]
Description=Depression Detection API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/major2.0_clean
Environment="PATH=/path/to/venv/bin"
Environment="FLASK_ENV=production"
ExecStart=/path/to/venv/bin/gunicorn -c gunicorn_config.py backend.wsgi:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable depression-detection
sudo systemctl start depression-detection
```

## Platform-Specific Guides

### Railway

See `deployment/RAILWAY_DEPLOY.md`

### Render

See `backend/render.yaml` and `deployment/DEPLOYMENT_GUIDE.md`

### Vercel (Frontend)

See `frontend/vercel.json`

### Netlify

See `frontend/netlify.toml`

## Environment Variables

### Required

- `FLASK_ENV`: Environment (development/production)
- `PORT`: Port for backend API
- `MODEL_PATH`: Path to model file

### Optional

- `SECRET_KEY`: Flask secret key (required in production)
- `CORS_ORIGINS`: Comma-separated list of allowed origins
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `WORKERS`: Number of Gunicorn workers

See `.env.example` for all available options.

## Health Checks

### Backend API

```bash
curl http://localhost:5001/api/metrics
```

### Frontend

```bash
curl http://localhost/
```

### Streamlit

```bash
curl http://localhost:8501/_stcore/health
```

## Troubleshooting

### Model Not Loading

- Check `MODEL_PATH` in `.env`
- Ensure model file exists at specified path
- Check file permissions

### CORS Errors

- Update `CORS_ORIGINS` in `.env`
- Ensure frontend URL is included

### Port Already in Use

- Change `PORT` in `.env`
- Kill process using the port: `lsof -ti:5001 | xargs kill`

### Docker Issues

- Check logs: `docker-compose logs`
- Rebuild: `docker-compose build --no-cache`
- Clean up: `docker-compose down -v`

## Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Set strong `SECRET_KEY`
- [ ] Configure `CORS_ORIGINS` properly
- [ ] Set up proper logging
- [ ] Use Gunicorn or similar WSGI server
- [ ] Set up reverse proxy (nginx)
- [ ] Enable HTTPS
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Set up health checks

## Monitoring

### Logs

- Backend: `logs/app.log`
- Access logs: Configured in `gunicorn_config.py`
- Error logs: Configured in `gunicorn_config.py`

### Metrics

- API metrics: `GET /api/metrics`
- Health check: `GET /api/metrics` (returns 200 if healthy)

## Security

- Never commit `.env` file
- Use strong `SECRET_KEY` in production
- Restrict `CORS_ORIGINS` to known domains
- Use HTTPS in production
- Keep dependencies updated
- Regular security audits

