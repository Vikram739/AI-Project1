# Deployment Guide

## 📦 Files Created

1. ✅ `requirements.txt` - Python dependencies
2. ✅ `.gitignore` - Git ignore rules
3. ✅ `README.md` - Complete documentation
4. ✅ `.github/workflows/deploy.yml` - CI/CD pipeline
5. ✅ `.streamlit/config.toml` - Streamlit theme configuration

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

Streamlit apps cannot be deployed on GitHub Pages as they require a Python server. Use Streamlit Cloud instead:

#### Steps:

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Add deployment files and update README"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `Vikram739/AI-Project1`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Share the URL:**
   - You'll get a URL like: `https://vikram739-ai-project1.streamlit.app`
   - Update the README.md with this URL
   - Share with your classmates!

### Option 2: Heroku (Alternative)

If you prefer Heroku:

1. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

2. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

3. Deploy to Heroku

### Option 3: Local Network Sharing

Share with people on the same network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then share the Network URL displayed in the terminal.

## 📝 Git Commands to Push

```bash
# Check status
git status

# Add all new files
git add .

# Commit changes
git commit -m "Add deployment files, requirements, and documentation"

# Push to GitHub
git push origin main
```

## ✅ CI/CD Pipeline

The GitHub Actions workflow will automatically:
- ✅ Test Python syntax on every push
- ✅ Verify all input files exist
- ✅ Install and validate dependencies

## 🌐 Accessing the Deployed App

**Streamlit Cloud URL:** After deployment, you'll get a permanent URL

**For Classmates:**
- Share the Streamlit Cloud URL
- No installation needed - just click and use!
- Works on any device with a browser

## 🔧 Environment Variables (if needed)

If you add any secrets later:
1. Go to Streamlit Cloud app settings
2. Click "Secrets"
3. Add your secrets in TOML format

## 📊 Monitoring

- **Streamlit Cloud Dashboard:** View logs, analytics, and resource usage
- **GitHub Actions:** View build status on the Actions tab

## 🆘 Troubleshooting

**If deployment fails:**
1. Check requirements.txt has correct versions
2. Verify app.py has no syntax errors
3. Check Streamlit Cloud logs for specific errors
4. Ensure Python version compatibility (3.7+)

**Common issues:**
- Missing dependencies → Add to requirements.txt
- Import errors → Check package names
- Port issues → Streamlit Cloud handles this automatically
