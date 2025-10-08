# Railway.app Docker Deployment - Quick Start

## Perfect for Docker Deployment (No Local Docker Needed!)

Railway.app automatically builds Docker images in the cloud from your GitHub repo.

---

## Deploy in 10 Minutes

### Step 1: Sign Up for Railway

1. Go to: https://railway.app/
2. Click "Login"
3. Sign in with GitHub
4. Authorize Railway to access your repos

### Step 2: Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose: depression-detection-system
4. Railway will scan your repo

### Step 3: Deploy Backend

1. Railway detects Dockerfile.backend automatically
2. Click on the backend service
3. Go to "Variables" tab
4. Add environment variables:
   - PORT = 5001
   - FLASK_ENV = production
5. Click "Settings" then "Networking"
6. Click "Generate Domain" to get public URL
7. Copy your backend URL

Backend is deploying! (Takes 3-5 minutes)

### Step 4: Deploy Frontend

1. Click "+ New" then "GitHub Repo" (same repo)
2. Railway detects Dockerfile.frontend
3. Go to "Variables" tab
4. Add build argument:
   - VITE_API_URL = Your backend Railway URL from Step 3
5. Go to "Settings" then "Networking"
6. Click "Generate Domain"

Frontend is deploying! (Takes 2-3 minutes)

## You're Live!

Once both services show "Active" status:
- Frontend URL: Your Railway domain
- Backend URL: Your Railway domain
- Full-stack app running in Docker containers!

## Pricing

Free Tier: $5 credit/month (perfect for this project)

## Start Now

Visit: https://railway.app/
