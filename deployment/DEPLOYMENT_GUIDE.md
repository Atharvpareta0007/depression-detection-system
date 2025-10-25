# 🚀 Deployment Guide - Depression Detection React App

## ✅ Production Build Complete

Your React app has been successfully built for production! The optimized files are in the `frontend/dist` folder.

**Build Stats:**
- Bundle Size: 335 KB (gzipped: 109 KB)
- CSS Size: 33.66 KB (gzipped: 6.08 KB)
- Total: ~115 KB gzipped

---

## 📦 Deployment Options

### Option 1: Vercel (Recommended) ⚡

**Why Vercel?**
- Free tier available
- Automatic deployments from Git
- Built-in CDN
- Zero configuration needed

**Steps:**

1. **Install Vercel CLI** (optional)
   ```bash
   npm install -g vercel
   ```

2. **Deploy via CLI**
   ```bash
   cd frontend
   vercel
   ```

3. **Or Deploy via Web:**
   - Go to https://vercel.com
   - Import your GitHub repository
   - Vercel auto-detects Vite
   - Set build command: `npm run build`
   - Set output directory: `dist`
   - Deploy! 🎉

**Environment Variables for Vercel:**
- `VITE_API_URL` = Your backend API URL

---

### Option 2: Netlify 🌐

**Why Netlify?**
- Free tier with 100GB bandwidth
- Continuous deployment
- Great for static sites
- Built-in forms and functions

**Steps:**

1. **Install Netlify CLI** (optional)
   ```bash
   npm install -g netlify-cli
   ```

2. **Deploy via CLI**
   ```bash
   cd frontend
   netlify deploy --prod
   ```

3. **Or Deploy via Web:**
   - Go to https://app.netlify.com
   - Drag & drop the `frontend/dist` folder
   - Or connect your GitHub repo
   - Netlify uses the `netlify.toml` config automatically
   - Deploy! 🎉

**Configuration:** Already set in `netlify.toml`

---

### Option 3: GitHub Pages (Free) 📄

**Perfect for:** Public projects, simple hosting

**Steps:**

1. **Add homepage to package.json:**
   ```json
   {
     "homepage": "https://yourusername.github.io/repo-name"
   }
   ```

2. **Install gh-pages:**
   ```bash
   cd frontend
   npm install --save-dev gh-pages
   ```

3. **Add deploy script to package.json:**
   ```json
   {
     "scripts": {
       "deploy": "vite build && gh-pages -d dist"
     }
   }
   ```

4. **Deploy:**
   ```bash
   npm run deploy
   ```

---

### Option 4: Docker + Any Cloud Provider 🐳

**Perfect for:** Full-stack deployment with backend

**Create Dockerfile in frontend:**
```dockerfile
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Build and run:**
```bash
docker build -t depression-detection-frontend .
docker run -p 80:80 depression-detection-frontend
```

---

## 🔧 Backend Deployment

Your React app needs the backend API running. Deploy it separately:

### Backend Options:

1. **Render (Recommended for Python/Flask)**
   - Free tier available
   - Automatic deployments
   - https://render.com

2. **Railway**
   - Easy Python deployment
   - https://railway.app

3. **Heroku**
   - Classic option
   - Free tier available
   - https://heroku.com

4. **AWS EC2 / Google Cloud / Azure**
   - Full control
   - More complex setup

### Backend Deployment Steps (Render):

1. Create `render.yaml` in backend:
   ```yaml
   services:
     - type: web
       name: depression-api
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: python app.py
       envVars:
         - key: PORT
           value: 5001
   ```

2. Push to GitHub
3. Connect to Render
4. Deploy automatically

---

## 🌍 Update API URL

After deploying your backend, update the frontend API URL:

**In `.env.production`:**
```
VITE_API_URL=https://your-backend-url.onrender.com
```

**Or update in your components:**
```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';
```

**Rebuild frontend:**
```bash
npm run build
```

---

## 🔒 CORS Configuration

Update your backend's CORS settings to allow your deployed frontend:

**In `backend/app.py`:**
```python
from flask_cors import CORS

CORS(app, origins=[
    'http://localhost:5173',
    'https://your-app.vercel.app',
    'https://your-app.netlify.app'
])
```

---

## ✨ Quick Deploy Commands

### For Vercel:
```bash
cd frontend
vercel --prod
```

### For Netlify:
```bash
cd frontend
netlify deploy --prod --dir=dist
```

### For GitHub Pages:
```bash
cd frontend
npm run deploy
```

---

## 📊 Performance Optimizations (Already Applied)

✅ **Code Splitting** - React lazy loading
✅ **Minification** - CSS and JS minified  
✅ **Tree Shaking** - Unused code removed
✅ **Gzip Compression** - 109 KB total size
✅ **Modern Build** - ES2020+ features
✅ **CSS Optimization** - Tailwind CSS purged

---

## 🐛 Troubleshooting

### Issue: API calls failing
**Solution:** Update `VITE_API_URL` environment variable

### Issue: Blank page after deployment
**Solution:** Check browser console, ensure routing is configured

### Issue: 404 on refresh
**Solution:** Configure server-side redirects (already in vercel.json and netlify.toml)

---

## 🎯 Recommended Setup

**For Demo/Testing:**
- Frontend: Vercel (free)
- Backend: Render (free)

**For Production:**
- Frontend: Vercel Pro or Netlify Pro
- Backend: AWS/Google Cloud with load balancer
- Database: Managed PostgreSQL
- CDN: Cloudflare

---

## 📝 Post-Deployment Checklist

- [ ] Frontend deployed and accessible
- [ ] Backend deployed and accessible
- [ ] API URL updated in frontend
- [ ] CORS configured correctly
- [ ] Environment variables set
- [ ] SSL/HTTPS enabled
- [ ] Custom domain configured (optional)
- [ ] Analytics added (optional)
- [ ] Error monitoring setup (optional)

---

## 🚀 Your App is Ready to Deploy!

**Fastest Option:** 
1. Push to GitHub
2. Import to Vercel
3. Done in 2 minutes! ⚡

**Questions?** Check the documentation or reach out for help!
