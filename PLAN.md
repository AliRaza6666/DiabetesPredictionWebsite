# Plan: Complete Diabetes Prediction ML System

## TL;DR
Build a beginner-friendly machine learning project using Logistic Regression: train a model locally, create an API using Gradio, deploy to Hugging Face, and build a frontend website that connects to it. Each step is independent and can be implemented separately.

---

## Steps

### **PHASE 1: Project Setup**

**Step 1.1: Create Project Directory Structure** *(no dependencies)*
- Create main folders: `/training`, `/model`, `/huggingface-space`, `/website`, `/data`
- Create empty placeholder files in `/model` and `/data` directories
- **Verification**: `ls -la` shows all 5 directories

**Step 1.2: Create requirements.txt** *(no dependencies)*
- File location: Project root
- Include: scikit-learn, pandas, numpy, gradio, huggingface-hub
- **Verification**: `pip install -r requirements.txt` works without errors

---

### **PHASE 2: Machine Learning Model**

**Step 2.1: Create Training Script** *(depends on: Step 1.1)*
- File: `/training/train_model.py`
- Properties:
  - Load diabetes dataset from sklearn
  - Split into 80% train, 20% test
  - Standardize features (StandardScaler)
  - Train LogisticRegression model
  - Print accuracy, precision, recall, f1-score
  - Save 3 pickle files to `/model/`:
    - `diabetes_model.pkl`
    - `scaler.pkl`
    - `feature_names.pkl`
- Expected output: Model metrics + 3 saved files
- **Verification**: Run script, check model files exist, verify metrics print

**Step 2.2: Test Model Locally** *(depends on: Step 2.1)*
- Write small Python script to load saved model
- Test prediction with sample data
- Verify prediction format (0 or 1)
- **Verification**: Script runs, returns prediction value

---

### **PHASE 3: API Deployment (Gradio)**

**Step 3.1: Create Gradio App** *(depends on: Step 2.1)*
- File: `/huggingface-space/app.py`
- Properties:
  - Load model, scaler, feature names from pickle files
  - Create prediction function that takes 8 health inputs
  - Wrap function with Gradio interface
  - Include sliders for all 8 features (age, bmi, bp, glucose, insulin, skin, pedigree, pregnancies)
  - Display prediction + probability scores
  - Add descriptions and disclaimers
- Expected behavior: Creates web interface on localhost:7860
- **Verification**: `python app.py` starts server, interface loads in browser

**Step 3.2: Test Gradio App Locally** *(depends on: Step 3.1)*
- Run Gradio app
- Enter test values
- Verify predictions appear
- Check confidence scores display
- **Verification**: Predictions work, interface is responsive

**Step 3.3: Prepare for Hugging Face Deployment** *(depends on: Step 3.1, 2.1)*
- Copy: app.py and requirements.txt to huggingface-space folder
- Ensure model files are referenced with correct paths
- **Verification**: Files exist in huggingface-space folder

---

### **PHASE 4: Frontend Website**

**Step 4.1: Create HTML Structure** *(no dependencies)*
- File: `/website/index.html`
- Structure:
  - Header with title and subtitle
  - Form with 8 input fields (age, pregnancies, glucose, bp, skin, insulin, bmi, pedigree)
  - Input validation (min/max values)
  - Submit button
  - Results section (hidden initially)
  - Footer with disclaimer
  - Include links to CSS and JavaScript
- **Verification**: HTML validates, no broken links, form appears in browser

**Step 4.2: Create Responsive Styling** *(depends on: Step 4.1)*
- File: `/website/styles.css`
- Properties:
  - CSS variables at top (colors, spacing, transitions)
  - Mobile-first responsive design
  - Professional gradient background
  - Form styling with focus states
  - Results card styling
  - Loading spinner animation
  - Error message styling
  - Breakpoints: 768px (tablet), 480px (mobile)
- **Verification**: Website looks good on desktop and mobile, buttons are clickable

**Step 4.3: Create Frontend Logic** *(depends on: Step 4.1, 4.2)*
- File: `/website/script.js`
- Properties:
  - Form submission handler
  - Collect 8 input values
  - Send POST request to Hugging Face API (later)
  - Parse response
  - Display results with confidence bars
  - Handle errors with user-friendly messages
  - Include demo/mock mode for testing without API
  - Add reset button functionality
- **Verification**: Form collects data, displays mock predictions correctly

**Step 4.4: Test Website Locally** *(depends on: Step 4.3)*
- Start local server: `python -m http.server 8000`
- Test form submission
- Verify results display
- Test mobile responsiveness
- **Verification**: Website works in browser, form submits without errors

---

### **PHASE 5: Hugging Face Deployment**

**Step 5.1: Create Hugging Face Account** *(no dependencies)*
- Go to https://huggingface.co
- Sign up (free)
- Create personal access token
- **Verification**: Can login, token created

**Step 5.2: Create Hugging Face Space** *(depends on: Step 5.1)*
- Create new Space
- Name: diabetes-prediction (or similar)
- SDK: Docker
- Keep public
- **Verification**: Space created, accessible at huggingface.co/spaces/username/diabetes-prediction

**Step 5.3: Deploy Model to Space** *(depends on: Step 5.2, 2.1)*
- Clone Space repository locally
- Create `/model` directory in Space
- Copy 3 pickle files from training to Space `/model`
- Copy `/huggingface-space/app.py` to Space root
- Copy `requirements.txt` to Space root
- Push via Git
- **Verification**: Files appear in Space, build starts automatically

**Step 5.4: Monitor Deployment** *(depends on: Step 5.3)*
- Check Space build logs
- Wait for "Space is running" status
- Space URL: `https://username-diabetes-prediction.hf.space`
- **Verification**: Space is live, can access it

**Step 5.5: Test API Endpoint** *(depends on: Step 5.4)*
- Use browser or curl to test API
- Verify predictions return correctly
- **Verification**: API responds to requests

---

### **PHASE 6: Integration**

**Step 6.1: Connect Website to API** *(depends on: Step 5.4, 4.3)*
- Edit `/website/script.js`
- Update line with API URL: `const HUGGINGFACE_API_URL = "https://username-diabetes-prediction.hf.space"`
- Remove mock/demo mode
- **Verification**: Variable updated with correct URL

**Step 6.2: Test End-to-End** *(depends on: Step 6.1)*
- Start website locally
- Enter test values
- Click predict
- Verify results from real API appear
- **Verification**: Website shows API predictions

**Step 6.3: Customize Website (Optional)** *(depends on: Step 4.2)*
- Edit title/subtitle in HTML
- Change colors in CSS variables
- Update disclaimer text
- Add hospital/clinic name
- **Verification**: Changes appear in browser

---

### **PHASE 7: Documentation & Sharing**

**Step 7.1: Document Your Process** *(no dependencies)*
- Create README.md explaining what each file does
- Add troubleshooting section
- Document model performance
- **Verification**: Documentation is clear and complete

**Step 7.2: Share Your Work** *(depends on: Step 5.4, 4.4)*
- Share Hugging Face Space URL with others
- Share website URL (can deploy via GitHub Pages/Netlify if desired)
- Share GitHub repository
- **Verification**: Others can access and use your system

---

## Relevant Files

All files will be created in your workspace at:
```
c:\Users\hp\OneDrive\Documents\machineLearning\Models\DiabetesPrediction\
```

### To Create (by you):

**Training**
- `training/train_model.py` — Main training script using Logistic Regression

**API**
- `huggingface-space/app.py` — Gradio app for inference

**Website**
- `website/index.html` — HTML structure with form
- `website/styles.css` — Responsive styling with CSS variables
- `website/script.js` — Frontend logic and API integration

**Config**
- `requirements.txt` — Python dependencies at project root

**Documentation** (optional but recommended)
- `README.md` — Project documentation
- `DEPLOYMENT.md` — Deployment instructions

---

## Verification

### Phase 1 Verification
- [ ] All 5 directories exist
- [ ] requirements.txt file created

### Phase 2 Verification
- [ ] Training script runs without errors
- [ ] Model metrics display (accuracy ~70-75%)
- [ ] 3 pickle files created in `/model`

### Phase 3 Verification
- [ ] Gradio app starts on localhost:7860
- [ ] Web interface appears with 8 input fields
- [ ] Can enter values and get predictions

### Phase 4 Verification
- [ ] Website loads in browser at localhost:8000
- [ ] Form has all 8 fields
- [ ] Mock predictions work
- [ ] Website is responsive on mobile

### Phase 5 Verification
- [ ] Hugging Face Space created
- [ ] Build completes successfully
- [ ] Space URL is accessible
- [ ] API responds to requests

### Phase 6 Verification
- [ ] Website connects to real Hugging Face API
- [ ] Enter values → real predictions appear
- [ ] No "API connection failed" errors

### Phase 7 Verification
- [ ] Documentation complete
- [ ] Can share links with others
- [ ] Others can use your system

---

## Decisions & Scope

### Included ✅
- Logistic Regression model (beginner-friendly)
- Gradio API (simple deployment)
- HTML/CSS/JavaScript website
- Hugging Face deployment
- 8 health features for prediction
- Binary classification (diabetes yes/no)

### Not Included ❌
- Advanced ML algorithms (Random Forest, Neural Networks)
- Database for storing predictions
- User authentication
- Email notifications
- Mobile app
- Real medical validation

### Key Assumptions
- You have Python 3.8+ installed
- You have Hugging Face account (free)
- You know basic Python and HTML/CSS/JS
- You want to implement each step yourself

---

## Timeline

| Phase | Steps | Estimated Time |
|-------|-------|-----------------|
| Setup | 1.1-1.2 | 10 min |
| ML Model | 2.1-2.2 | 30 min |
| Gradio API | 3.1-3.3 | 20 min |
| Website | 4.1-4.4 | 45 min |
| Deployment | 5.1-5.5 | 30 min |
| Integration | 6.1-6.3 | 15 min |
| Documentation | 7.1-7.2 | 20 min |
| **TOTAL** | | **~2.5 hours** |

---

## Critical Code References

### 8 Features Your Model Will Use (in order):
1. Age (0-120 years)
2. Number of Pregnancies (0-17)
3. Glucose Level (0-300 mg/dL)
4. Blood Pressure (0-200 mm Hg)
5. Skin Thickness (0-100 mm)
6. Insulin Level (0-900 mU/mL)
7. BMI (10-60 kg/m²)
8. Diabetes Pedigree Function (0-2.5)

### Pickle files to save from training:
- Model object (LogisticRegression)
- Scaler object (StandardScaler)
- Feature names list

### Gradio app requirements:
- Load pickle files
- Create function: predict_diabetes(age, pregnancies, glucose, bp, skin, insulin, bmi, pedigree)
- Return: prediction + probabilities

---

## Potential Blockers

1. **Pickle file path issues** — Ensure paths match when loading in different scripts
2. **CORS errors** — Hugging Face handles this, but verify in browser console
3. **API URL format** — Must be exact: `https://username-diabetes-prediction.hf.space` (no trailing slash)
4. **scaler.pkl mismatch** — Must use SAME scaler from training, not retrain it
5. **Feature order** — Must keep same order: age→pregnancies→glucose→bp→skin→insulin→bmi→pedigree

---

## Next Actions for You

1. ✅ Review this plan
2. ✅ Ask clarifying questions (if any)
3. ✅ Implement Phase 1: Project Setup
4. ✅ Then implement Phase 2: ML Model
5. ✅ Continue through phases sequentially
