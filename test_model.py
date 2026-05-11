import pickle
import pandas as pd

try:
    model = pickle.load(open('ipl_win_predictor.pkl', 'rb'))
    print("Model loaded successfully")
    # Test data based on model.py requirements
    # 'batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr'
    test_df = pd.DataFrame([['Royal Challengers Bangalore', 'Mumbai Indians', 'Mumbai', 100, 60, 8, 200, 10.0, 10.0]], 
                           columns=['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr'])
    result = model.predict_proba(test_df)
    print(f"Prediction result: {result}")
except Exception as e:
    print(f"Error: {e}")
