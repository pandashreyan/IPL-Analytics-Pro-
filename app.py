from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import pickle
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ipl-secret-key-123'

# Environment-aware database path
if os.environ.get('VERCEL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/ipl_predictor.db'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ipl_predictor.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    predictions = db.relationship('Prediction', backref='author', lazy=True)

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batting_team = db.Column(db.String(50), nullable=False)
    bowling_team = db.Column(db.String(50), nullable=False)
    win_prob = db.Column(db.Float, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='Pending') # Pending, Accurate, Inaccurate
    progression_json = db.Column(db.Text, nullable=True) # JSON string of win probability trend

    def __init__(self, **kwargs):
        super(Prediction, self).__init__(**kwargs)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML Model with absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'ipl_win_predictor.pkl')
model = pickle.load(open(model_path, 'rb'))

teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
    'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
    'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg',
    'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur',
    'Dharamsala', 'Kochi', 'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah'
]

matches_path = os.path.join(BASE_DIR, 'iplwin', 'matches.csv')
matches_df = pd.read_csv(matches_path)

def get_h2h_stats(team1, team2):
    h2h = matches_df[((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) | 
                     ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))]
    team1_wins = len(h2h[h2h['winner'] == team1])
    team2_wins = len(h2h[h2h['winner'] == team2])
    total = len(h2h)
    return {'team1_wins': team1_wins, 'team2_wins': team2_wins, 'total': total}

def get_venue_stats(city):
    venue_matches = matches_df[matches_df['city'] == city]
    if venue_matches.empty:
        return {
            'avg_score': 160, 
            'toss_win_match_win': 50.0,
            'radar': [70, 60, 50, 65, 55] # Pace, Spin, Batting, Death, Boundaries
        }
    
    # Mocking some realistic venue characteristics based on common knowledge
    characteristics = {
        'Mumbai': [85, 40, 90, 75, 60],
        'Bangalore': [40, 30, 95, 40, 90],
        'Chennai': [30, 95, 60, 80, 50],
        'Hyderabad': [70, 75, 75, 70, 65],
        'Kolkata': [65, 80, 70, 65, 70],
        'Delhi': [45, 90, 65, 60, 85]
    }
    radar = characteristics.get(city, [60, 60, 60, 60, 60])
    
    avg_score = 165
    toss_win_match_win = round((len(venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]) / len(venue_matches)) * 100, 1)
    return {'avg_score': avg_score, 'toss_win_match_win': toss_win_match_win, 'radar': radar}

def generate_ai_summary(batting_team, bowling_team, win, score, target, overs, wickets):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    
    if win > 80:
        return f"Dominant performance! {batting_team} has effectively sealed the deal. {bowling_team} needs a miracle or a catastrophic collapse to turn this around."
    elif win > 60:
        return f"{batting_team} is in the driver's seat, but the game is far from over. {bowling_team}'s death bowlers hold the key to a potential comeback."
    elif win > 40:
        return f"A nail-biter in the making! Both teams are neck-and-neck. Every run and wicket from here will be critical in deciding the winner."
    elif win > 20:
        return f"{bowling_team} has put {batting_team} under immense pressure. {batting_team} needs an extraordinary partnership to keep their hopes alive."
    else:
        return f"Total domination by {bowling_team}. {batting_team} is staring at a defeat unless a hero emerges immediately."

@app.route('/')
def index():
    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), win_probability=None)

@app.route('/predict', methods=['POST'])
def predict():
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']
    target = int(request.form['target'])
    score = int(request.form['score'])
    overs = float(request.form['overs'])
    wickets = int(request.form['wickets'])

    # Cricket notation conversion (e.g., 10.4 overs = 10*6 + 4 = 64 balls)
    overs_int = int(overs)
    balls_done = (overs_int * 6) + round((overs - overs_int) * 10)
    balls_left = 120 - balls_done
    
    h2h = get_h2h_stats(batting_team, bowling_team)
    venue = get_venue_stats(city)

    runs_left = target - score
    wickets_left = 10 - wickets
    crr = (score * 6) / balls_done if balls_done > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame([[batting_team, bowling_team, city, runs_left, balls_left, wickets_left, target, crr, rrr]],
                            columns=['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr'])

    result = model.predict_proba(input_df)
    win = round(result[0][1] * 100, 1)
    loss = round(result[0][0] * 100, 1)

    progression = [round(win + (i * (2 if i % 2 == 0 else -2)), 1) for i in range(1, 7)]
    progression = [min(100, max(0, p)) for p in progression]

    # Save to history if logged in
    if current_user.is_authenticated:
        try:
            import json
            prog_json = json.dumps(progression)
            pred = Prediction(
                batting_team=batting_team, 
                bowling_team=bowling_team, 
                win_prob=win, 
                user_id=current_user.id,
                progression_json=prog_json
            )
            db.session.add(pred)
            db.session.commit()
        except Exception as e:
            print(f"Error saving prediction: {e}")
            db.session.rollback()

    # Team Performance Stats (Mocked for professional comparison)
    team_stats = {
        batting_team: [80, 85, 70, 75, 90], # Batting, Bowling, Fielding, Death, Powerplay
        bowling_team: [75, 90, 85, 80, 70]
    }

    # Generate AI Summary
    ai_summary = generate_ai_summary(batting_team, bowling_team, win, score, target, overs, wickets)

    return render_template('index.html', 
                           teams=sorted(teams), 
                           cities=sorted(cities), 
                           win_probability=win, 
                           loss_probability=loss,
                           batting_team=batting_team,
                           bowling_team=bowling_team,
                           h2h=h2h,
                           venue=venue,
                           progression=progression,
                           ai_summary=ai_summary,
                           team_stats=team_stats)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_pw = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(username=request.form['username'], password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('auth.html', mode='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('index'))
    return render_template('auth.html', mode='login')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(author=current_user).order_by(Prediction.date_posted.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/update_status/<int:pred_id>/<string:status>')
@login_required
def update_status(pred_id, status):
    prediction = Prediction.query.get_or_404(pred_id)
    if prediction.author != current_user:
        return redirect(url_for('history'))
    prediction.status = status
    db.session.commit()
    return redirect(url_for('history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Simple auto-migration for status and progression_json columns
        try:
            from sqlalchemy import text
            with db.engine.connect() as conn:
                # Check for status column
                try:
                    conn.execute(text("ALTER TABLE prediction ADD COLUMN status VARCHAR(20) DEFAULT 'Pending'"))
                    conn.commit()
                except Exception:
                    pass # Column might already exist
                
                # Check for progression_json column
                try:
                    conn.execute(text("ALTER TABLE prediction ADD COLUMN progression_json TEXT"))
                    conn.commit()
                except Exception:
                    pass # Column might already exist
        except Exception as e:
            print(f"Migration notice: {e}")
            
    app.run(debug=True)
