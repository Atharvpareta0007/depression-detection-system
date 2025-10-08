#!/bin/bash

# 🚀 Quick Deploy Script for Depression Detection App

echo "🚀 Depression Detection App - Quick Deploy"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📦 Building React frontend..."
cd frontend
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""
echo "📂 Production files are in: frontend/dist"
echo ""
echo "🌐 Choose your deployment platform:"
echo ""
echo "1. Vercel (Recommended)"
echo "   Command: cd frontend && vercel"
echo ""
echo "2. Netlify"
echo "   Command: cd frontend && netlify deploy --prod"
echo ""
echo "3. GitHub Pages"
echo "   Command: cd frontend && npm run deploy"
echo ""
echo "4. Manual Deploy"
echo "   Upload the 'frontend/dist' folder to your hosting provider"
echo ""
echo "📖 Full guide: See DEPLOYMENT_GUIDE.md"
echo ""
echo "✨ Your app is ready to deploy!"
