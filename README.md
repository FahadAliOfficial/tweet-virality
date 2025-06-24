# Twitter Virality Prediction App

## Project Structure
```
twitter-virality/
├── data/                    # Dataset and processed data files
├── models/                  # Trained ML models
├── src/                     # Source code modules
├── assets/                  # Static files, images, etc.
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md               # This file
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   - Add your Gemini API key to `.env` file
   - Place your dataset in the `data/` folder

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Environment Usuage
    ```
     Navigate to project
    cd D:\Projects\twitter-virality

    # Activate environment
    D:/Projects/twitter-virality/.venv/Scripts/Activate.ps1

    # Now you can use shorter commands:
    python app.py
    streamlit run app.py
    pip install new_package
    ```
## Project Goals

Create a Twitter post virality prediction system that:
- Analyzes user input and predicts viral potential
- Suggests optimal posting times and hashtags
- Provides real-time statistical analysis
- Uses AI to generate relevant topics and tags

## Development Steps

See the step-by-step development plan below.
