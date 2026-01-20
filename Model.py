import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- SETTINGS ---
# We simulate 1 year of data for 5 different crews
NUM_DAYS = 365
CREW_LIST = ['Drywall-A', 'Drywall-B', 'Concrete-1', 'Framing-A', 'MEP-2']
START_DATE = datetime(2024, 1, 1)

# --- TEXT GENERATORS ---
# We need realistic notes that correlate with performance.
# Good days = clear notes. Bad days = confused/frustrated notes.

good_notes = [
    "Crew moving fast, materials arrived on time.",
    "Great weather, slab pour went perfectly.",
    "Ahead of schedule, inspection passed.",
    "Team is in good rhythm, no safety issues.",
    "Productive day, finished Zone A early."
]

bad_notes = [
    "Waiting on RFI response, crew standing around.",
    "Material delivery late, caused bottleneck.",
    "Confusion on drawings for HVAC, had to stop work.",
    "Crew exhausted from overtime, morale is low.",
    "Rain delay caused slip hazards, slowed down install.",
    "Argument with sub trade over laydown area.",
    "Rework needed on wall framing, incorrect specs."
]

neutral_notes = [
    "Standard progress, nothing to report.",
    "Routine install in Zone B.",
    "Weather overcast, work continuing.",
    "Crew size normal, steady pace.",
    "Safety talk held this morning, no incidents."
]

# --- DATA GENERATION LOOP ---
data = []

print("Generating Site Pulse Data...")

for crew in CREW_LIST:
    current_date = START_DATE
    
    # Track accumulated fatigue (simulated)
    fatigue_level = 0 
    
    for _ in range(NUM_DAYS):
        # 1. Simulate the "Scenario" for the day
        # Randomly decide if it's a Good (50%), Neutral (30%), or Bad (20%) day
        scenario_roll = random.random()
        
        if scenario_roll < 0.50:
            scenario = "Good"
        elif scenario_roll < 0.80:
            scenario = "Neutral"
        else:
            scenario = "Bad"
            
        # 2. Generate Data based on Scenario
        if scenario == "Good":
            hours_worked = random.randint(8, 9) # Normal hours
            units_installed = random.randint(100, 120) # High productivity
            note = random.choice(good_notes)
            safety_incident = 0
            rework_cost = 0
            fatigue_level = max(0, fatigue_level - 1) # Recovery
            
        elif scenario == "Neutral":
            hours_worked = random.randint(8, 10)
            units_installed = random.randint(80, 100)
            note = random.choice(neutral_notes)
            safety_incident = 0
            rework_cost = 0
            
        else: # BAD DAY (The interesting data!)
            hours_worked = random.randint(10, 14) # Overtime trying to catch up
            units_installed = random.randint(40, 70) # Low productivity despite high hours
            note = random.choice(bad_notes)
            
            # High chance of incident/rework on bad days
            safety_incident = 1 if random.random() < 0.3 else 0 
            rework_cost = random.randint(500, 5000) if random.random() < 0.5 else 0
            fatigue_level += 2 # Fatigue spikes

        # 3. Append to list
        data.append({
            "Date": current_date,
            "Crew_ID": crew,
            "Crew_Count": random.randint(5, 8),
            "Hours_Worked": hours_worked,
            "Units_Installed": units_installed,
            "Fatigue_Score": fatigue_level, # Hidden variable we want to model later
            "Superintendent_Notes": note,
            "Safety_Incident": safety_incident,
            "Rework_Cost": rework_cost
        })
        
        current_date += timedelta(days=1)

# --- EXPORT ---
df = pd.DataFrame(data)

# Let's peek at the "Messy" data
print(f"Generated {len(df)} rows of construction logs.")
print(df[['Date', 'Crew_ID', 'Superintendent_Notes', 'Safety_Incident']].head())

# Save to CSV for the next step
df.to_csv("DPR_Site_Pulse_Data_Raw.csv", index=False)
print("File 'DPR_Site_Pulse_Data_Raw.csv' saved successfully.")


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- SETUP ---
# Download the lexicon (dictionary) VADER uses to understand words
nltk.download('vader_lexicon')

# Load your raw data
df = pd.read_csv("DPR_Site_Pulse_Data_Raw.csv")

# Initialize the Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

print("Feature Engineering Started...")

# --- 1. NLP: QUANTIFYING THE "SITE PULSE" ---
# Function to get the 'compound' score (Overall sentiment from -1 to +1)
def get_sentiment(text):
    if pd.isna(text):
        return 0
    return sid.polarity_scores(text)['compound']

# Apply to the notes column
df['Sentiment_Score'] = df['Superintendent_Notes'].apply(get_sentiment)

# Interpretation: 
# If Score < -0.05, it's a "Stress" day.
# If Score > 0.05, it's a "Flow" day.

# --- 2. PHYSICAL: CALCULATING BURNOUT (Rolling Averages) ---
# In real life, one long day isn't bad. 7 long days in a row is dangerous.
# We need to calculate "Rolling 7-Day Overtime".

# Sort by Crew and Date to ensure rolling math works
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Crew_ID', 'Date'])

# Define "Overtime" as anything over 8 hours
df['Daily_Overtime'] = df['Hours_Worked'].apply(lambda x: max(0, x - 8))

# Calculate Rolling 7-Day Sum of Overtime for each crew
df['Rolling_Overtime_7d'] = df.groupby('Crew_ID')['Daily_Overtime'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())

# --- 3. METRICS: REALIZED PRODUCTIVITY ---
# Units Installed per Man-Hour (The gold standard metric)
# Avoid division by zero
df['Productivity_Rate'] = df.apply(lambda row: row['Units_Installed'] / (row['Hours_Worked'] * row['Crew_Count']) if row['Hours_Worked'] > 0 else 0, axis=1)

# --- 4. TARGET CREATION: THE "RISK" FLAG ---
# We want to predict if a day was "Risky" (Incident OR High Rework Cost)
# This is what our Machine Learning model will try to predict in Phase 3.
df['Risk_Event'] = df.apply(lambda row: 1 if (row['Safety_Incident'] == 1 or row['Rework_Cost'] > 1000) else 0, axis=1)

# --- EXPORT ---
# Let's see the correlation before we save
correlation = df[['Sentiment_Score', 'Rolling_Overtime_7d', 'Risk_Event']].corr()
print("\n--- Correlation Matrix (Does the data make sense?) ---")
print(correlation)

df.to_csv("DPR_Site_Pulse_Engineered.csv", index=False)
print("\nFile 'DPR_Site_Pulse_Engineered.csv' saved. Ready for Modeling.")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- LOAD DATA ---
df = pd.read_csv("DPR_Site_Pulse_Engineered.csv")

# --- DEFINE FEATURES ---
# We use the signals we created to predict the 'Risk_Event'
features = ['Hours_Worked', 'Units_Installed', 'Fatigue_Score', 
            'Rolling_Overtime_7d', 'Sentiment_Score', 'Productivity_Rate']

X = df[features]
y = df['Risk_Event']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training 'Site Pulse' Model...")

# --- TRAIN MODEL ---
# Using Random Forest: It builds multiple decision trees and averages them
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = rf_model.predict(X_test)
print("\n--- Model Performance Report ---")
# Precision = When we predict a risk, how often are we right?
# Recall = Of all real risks, how many did we catch?
print(classification_report(y_test, y_pred))

# --- FEATURE IMPORTANCE (The "Why") ---
# This tells DPR what actually causes the accidents
importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- What drives Risk at DPR? (Feature Importance) ---")
print(importances)

# --- SCORING THE FULL DATASET ---
# Now we apply the model to the whole dataset to generate probabilities for Power BI
# The model gives us a probability (0.00 to 1.00) that a Risk Event will occur
df['Predicted_Risk_Prob'] = rf_model.predict_proba(X)[:, 1]

# Categorize the probability for easy filtering in Power BI
df['Risk_Category'] = pd.cut(df['Predicted_Risk_Prob'], 
                             bins=[-1, 0.3, 0.7, 1.1], 
                             labels=['Low Risk', 'Medium Risk', 'High Risk'])

# --- EXPORT FINAL DATA ---
df.to_csv("DPR_Site_Pulse_Final_Scored.csv", index=False)
print("\nFile 'DPR_Site_Pulse_Final_Scored.csv' saved. Ready for Power BI.")

