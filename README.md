Kaya Identity Streamlit App

This repository contains a Streamlit application that explores the Kaya Identity using World Bank indicators.

Main app file
- streamlit_app/app.py

Requirements
- Python 3.9+ recommended
- Install dependencies with:

  pip install -r requirements.txt

Run locally

1. Clone the repo:

   git clone https://github.com/CiVil75/KAYA_APP.git
   cd KAYA_APP

2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run streamlit_app/app.py

Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Create a new app and point it to this repository and the main branch.
3. Set the main file path to `streamlit_app/app.py`.
4. Streamlit will install dependencies from requirements.txt automatically and deploy the app.

Notes & recommendations

- The app fetches data from the World Bank API. The current app uses the API URL with HTTP; consider switching to HTTPS in `streamlit_app/app.py` to avoid mixed-content/network issues.
- If you plan to use a narrow year range or countries with sparse data, the app may return "No data available". Consider expanding the year range or adding error handling.
- If you want, I can create a branch with these changes or add a GitHub Actions workflow for testing.
