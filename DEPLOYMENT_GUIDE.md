# 🚀 STREAMLIT CLOUD DEPLOYMENT GUIDE

---

## 📋 **STEP-BY-STEP DEPLOYMENT**

---

## **PHASE 1: PREPARE FILES ON YOUR COMPUTER**

### **Step 1: Create Deployment Folder**

Create a new folder on your computer:
```
PharmacyAI-Deploy/
```

---

### **Step 2: Copy and Rename Files**

Copy these files to the deployment folder:

| Source File | → | Copy To | Rename As |
|-------------|---|---------|-----------|
| `streamlit_app_FINAL.py` | → | PharmacyAI-Deploy/ | `streamlit_app.py` |
| `master_agent_PRODUCTION.py` | → | PharmacyAI-Deploy/ | `master_agent.py` |
| `DEPLOY_requirements.txt` | → | PharmacyAI-Deploy/ | `requirements.txt` |
| `DEPLOY_gitignore.txt` | → | PharmacyAI-Deploy/ | `.gitignore` |
| `DEPLOY_README.md` | → | PharmacyAI-Deploy/ | `README.md` |

---

### **Step 3: Copy Data Folder**

Create `data/` folder inside PharmacyAI-Deploy/:
```
PharmacyAI-Deploy/
  ├── data/
  │   ├── sales_history.csv
  │   └── current_inventory.csv
```

Copy these files:
- `data/sales_history.csv` → `PharmacyAI-Deploy/data/`
- `data/current_inventory.csv` → `PharmacyAI-Deploy/data/`

---

### **Step 4: Verify Folder Structure**

Your `PharmacyAI-Deploy/` folder should look like this:

```
PharmacyAI-Deploy/
  ├── streamlit_app.py          ✅
  ├── master_agent.py            ✅
  ├── requirements.txt           ✅
  ├── .gitignore                 ✅
  ├── README.md                  ✅
  └── data/
      ├── sales_history.csv      ✅
      └── current_inventory.csv  ✅
```

**Total: 7 files**

---

## **PHASE 2: UPLOAD TO GITHUB**

### **Step 1: Create GitHub Account**

1. Go to https://github.com
2. Sign up (if you don't have account)
3. Verify email

---

### **Step 2: Create New Repository**

1. Click **"+"** (top right) → **"New repository"**
2. Fill in:
   - **Repository name**: `pharmacy-ai` (or any name you want)
   - **Description**: "AI-powered pharmacy inventory management"
   - **Public** or **Private**: Choose Public
   - ✅ Check **"Add a README file"** - NO, uncheck this (we have our own)
   - ❌ Don't add .gitignore or license
3. Click **"Create repository"**

---

### **Step 3: Upload Files to GitHub**

On the new repository page:

1. Click **"uploading an existing file"** link

2. **Drag and drop** or click to upload:
   - `streamlit_app.py`
   - `master_agent.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`

3. Click **"Commit changes"**

4. Now upload data folder:
   - Click **"Add file"** → **"Create new file"**
   - In filename box, type: `data/sales_history.csv`
   - Copy-paste contents of your sales_history.csv
   - Click **"Commit new file"**

5. Repeat for `data/current_inventory.csv`:
   - Click **"Add file"** → **"Upload files"**
   - Navigate into `data/` folder first
   - Upload `current_inventory.csv`
   - Commit

---

### **Step 4: Verify GitHub Upload**

Your GitHub repo should show:
```
pharmacy-ai/
  ├── .gitignore
  ├── README.md
  ├── master_agent.py
  ├── requirements.txt
  ├── streamlit_app.py
  └── data/
      ├── current_inventory.csv
      └── sales_history.csv
```

---

## **PHASE 3: DEPLOY TO STREAMLIT CLOUD**

### **Step 1: Create Streamlit Account**

1. Go to https://streamlit.io/cloud
2. Click **"Sign up"**
3. **Sign up with GitHub** (use same GitHub account)
4. Authorize Streamlit to access your GitHub

---

### **Step 2: Deploy App**

1. Click **"New app"** button

2. Fill in deployment settings:
   - **Repository**: `yourusername/pharmacy-ai`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: `pharmacy-ai` (or custom name)

3. Click **"Deploy!"**

---

### **Step 3: Wait for Deployment**

- Initial deployment takes 2-5 minutes
- You'll see build logs
- Wait for **"Your app is live!"** message

---

### **Step 4: View Your Live App**

Your app will be available at:
```
https://pharmacy-ai.streamlit.app
```
(or your custom URL)

---

## **PHASE 4: VERIFY DEPLOYMENT**

### **Test These Features:**

1. ✅ App loads without errors
2. ✅ Sample questions appear on sidebar
3. ✅ Can ask questions and get responses
4. ✅ Data tables display correctly
5. ✅ Metrics show properly
6. ✅ Download buttons work

---

## **TROUBLESHOOTING**

### **Issue: "Error loading data"**

**Cause**: Data files not found

**Fix**:
1. Check `data/` folder exists in GitHub
2. Verify both CSV files are inside
3. Check file names are exactly:
   - `sales_history.csv`
   - `current_inventory.csv`

---

### **Issue: "Module not found"**

**Cause**: Missing package in requirements.txt

**Fix**:
1. Edit `requirements.txt` in GitHub
2. Add missing package
3. Streamlit will auto-redeploy

---

### **Issue: App won't start**

**Cause**: Error in code

**Fix**:
1. Check deployment logs in Streamlit Cloud
2. Look for error message
3. Fix the error in GitHub
4. Streamlit auto-redeploys on commit

---

## **UPDATING YOUR APP**

### **To update code:**

1. Go to your GitHub repository
2. Click on file you want to edit (e.g., `streamlit_app.py`)
3. Click pencil icon (Edit)
4. Make changes
5. Click **"Commit changes"**
6. Streamlit Cloud automatically redeploys (takes 1-2 minutes)

---

## **BONUS: CUSTOM DOMAIN**

### **To use custom domain:**

1. In Streamlit Cloud, go to app settings
2. Click **"Custom domain"**
3. Follow instructions to add CNAME record
4. Your app will be at `pharmacy.yourdomain.com`

---

## **COST**

### **Streamlit Cloud Pricing:**

- **FREE tier**:
  - 1 private app
  - Unlimited public apps
  - 1GB RAM per app
  - Community support

- **Paid tier**: $20/month
  - 3 private apps
  - More resources
  - Priority support

**For your demo: FREE tier is perfect!**

---

## **SECURITY NOTES**

### **✅ Safe to Upload:**
- Code files (.py)
- Data files (.csv) - if not sensitive
- README, requirements, .gitignore

### **❌ NEVER Upload:**
- `.env` file (contains secrets)
- API keys
- Passwords
- Personal information

**The .gitignore file I provided will protect you!**

---

## **FINAL CHECKLIST**

Before going live:

- [ ] All 7 files uploaded to GitHub
- [ ] Data folder with 2 CSV files present
- [ ] Repository is public
- [ ] Streamlit app deployed
- [ ] App loads without errors
- [ ] Test all sample questions
- [ ] Share link with others

---

## **YOUR APP LINK**

After deployment, your app will be at:

```
https://pharmacy-ai.streamlit.app
```

(or whatever custom name you chose)

**Share this link with anyone to demo your app!**

---

## **NEED HELP?**

If you get stuck:

1. Check Streamlit Cloud logs (shows errors)
2. Verify file names are exact
3. Check GitHub repo structure matches guide
4. Make sure data files are not empty

---

## **CONGRATULATIONS! 🎉**

Your Pharmacy AI app is now live on the internet!

Anyone with the link can use it.
No installation needed.
Works on any device.

---

**Total Time: 15-20 minutes** ⏱️

**Difficulty: Easy** 😊

**Cost: FREE** 💰
