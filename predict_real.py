import pandas as pd
import joblib

model = joblib.load("/home/claude/ipl_project/ipl_model.pkl")
encoders = joblib.load("/home/claude/ipl_project/encoders.pkl")

def predict_winner(team1, team2, venue, toss_winner, toss_decision):
    input_df = pd.DataFrame([{
        "team1": encoders["team1"].transform([team1])[0],
        "team2": encoders["team2"].transform([team2])[0],
        "venue": encoders["venue"].transform([venue])[0],
        "toss_winner": encoders["toss_winner"].transform([toss_winner])[0],
        "toss_decision": encoders["toss_decision"].transform([toss_decision])[0],
    }])

    pred_encoded = model.predict(input_df)[0]
    pred_team = encoders["winner"].inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(input_df)[0]
    classes = encoders["winner"].inverse_transform(model.classes_)
    prob_dict = dict(zip(classes, probs))
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    return pred_team, top3


if __name__ == "__main__":
    team1 = "Mumbai Indians"
    team2 = "Chennai Super Kings"
    venue = "Wankhede Stadium, Mumbai"
    toss_winner = "Mumbai Indians"
    toss_decision = "field"

    # Check venue exists, else pick a real one from the encoder
    if venue not in encoders["venue"].classes_:
        venue = encoders["venue"].classes_[0]

    winner, top3 = predict_winner(team1, team2, venue, toss_winner, toss_decision)

    print(f"Match: {team1} vs {team2}")
    print(f"Venue: {venue}")
    print(f"Toss: {toss_winner} chose to {toss_decision}")
    print(f"\nPredicted Winner: {winner}")
    print("\nTop probabilities:")
    for team, prob in top3:
        print(f"  {team}: {prob:.1%}")
