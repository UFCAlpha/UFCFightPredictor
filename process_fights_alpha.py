import csv
import random
import re
from flask import Flask

# seed so the red/blue corner swap (and therefore the generated dataset) is reproducible
random.seed(42)
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from models import db, Fighter, Fight
import os

# retrieve data from Flask database
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///detailedfighters.db"
db.init_app(app)

# ********** HELPER FUNCTIONS **********
def query_fighter(key, name):
    # key is the ufcstats id when the fight rows carry one; a name can belong to two fighters
    if key != name:
        fighter = Fighter.query.filter_by(ufcstats_id=key).first()
        if fighter:
            return fighter
    return Fighter.query.filter_by(name=name).first()

def fighter_key(fight, side):
    """Identity of one corner: its ufcstats id, or the name on rows without ids."""
    return (fight.get(f'{side} Fighter ID') or '').strip() or fight[f'{side} Fighter']

def row_names(fight):
    """fighter_key -> the name this fight row gives that corner."""
    return {fighter_key(fight, side): fight[f'{side} Fighter'] for side in ('Red', 'Blue')}

file_path = os.path.join('data', 'modified_fight_details.csv')

def reverse_csv_to_dict(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        # return list(csv_reader)
        reversed_rows = list(csv_reader)[::-1]
        return reversed_rows

def get_csv_headers(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        return headers
    
def calculate_k_factor(number_of_fights):
    # Adjust K-factor based on the number of fights
    if number_of_fights < 5:
        return 40
    elif 5 <= number_of_fights < 10:
        return 35
    elif 10 <= number_of_fights < 20:
        return 30
    else:
        return 25

def calculate_expected_win_probability(rating_a, rating_b):
    return 1 / (1 + pow(10, (rating_b - rating_a) / 400))

def update_elo_ratings(rating_a, rating_b, result, fights_a, fights_b):
    K_FACTOR_A = calculate_k_factor(fights_a)
    K_FACTOR_B = calculate_k_factor(fights_b)

    expected_win_probability = calculate_expected_win_probability(rating_a, rating_b)

    if result == "win":
        actual_score = 1
    elif result == "loss":
        actual_score = 0
    else:  # Draw
        actual_score = 0.5

    new_rating_a = rating_a + K_FACTOR_A * (actual_score - expected_win_probability)
    new_rating_b = rating_b + K_FACTOR_B * ((1 - actual_score) - (1 - expected_win_probability))

    return new_rating_a, new_rating_b

def split_at_first_space(text):
    # Split the text by the first space
    parts = text.split(" ", 1)
    # Return both parts, before and after the first space
    return parts[0], parts[1] if len(parts) > 1 else ""

def parse_height_inches(text):
    # ufcstats format: 5' 11"
    m = re.match(r"(\d+)'\s*(\d+)", text or "")
    return int(m.group(1)) * 12 + int(m.group(2)) if m else 0

def parse_reach_inches(text):
    # ufcstats format: 72"
    m = re.match(r'(\d+)"', text or "")
    return int(m.group(1)) if m else 0
    
fighter_stats = dict()

fights = reverse_csv_to_dict(file_path)

# per-fighter state is keyed by fighter_key(). names[key] is the fighter's latest name, used
# only to label the fighter-stat exports: each fight keeps the names stored on its own row,
# because ufcstats renames fighters ("Bobby Green" -> "King Green") and the Winner column,
# the odds file and the backtests all carry the name as it was at the time.
all_fighters = set()
names = {}
for fight in fights:
    for side in ('Red', 'Blue'):
        key = fighter_key(fight, side)
        all_fighters.add(key)
        names[key] = fight[f'{side} Fighter']

# retrieve all total strike numbers from head, body, leg, distance, clinch, ground, sig.str
# store it back into the dicts
headers=get_csv_headers(file_path)

hardcoded_features = ["dob","totalfights","elo","losestreak","winstreak","titlewins","height","reach","southpaw"]
hardcoded_features_divide = ["oppelo","wins","avg age"]
feature_list=[]
feature_list.extend(hardcoded_features)
feature_list.extend(hardcoded_features_divide)

header_features = []
for column in headers:
    s1,s2=split_at_first_space(column)
    if(s1=="Red" and s2 not in ("Fighter", "Fighter ID")):
        header_features.append(s2)

feature_list.extend(header_features)

with app.app_context():
    # sorted(): set iteration order varies with PYTHONHASHSEED, and this loop fixes the
    # insertion order of fighter_stats, which is the order both exports are written in.
    # Unsorted, every run rewrote data/detailed_fighter_stats.csv and
    # data/fighter_stats_readable.txt in a different order and left the repo dirty.
    for fighter in sorted(all_fighters, key=lambda k: (names[k], k)):
        fighter_stats[fighter] = {}
        for feature in feature_list:
            fighter_stats[fighter][feature]=0
            if feature in header_features:
                fighter_stats[fighter][f"{feature} differential"]=0
            if "%" in feature:
                fighter_stats[fighter][f"{feature} defense"]=0
        fighter_stats[fighter]["elo"]=1000
        # get DOB        
        fighter_object = query_fighter(fighter, names[fighter])
        if fighter_object:
            date_format = "%b %d, %Y"  # Format like "Oct 01, 1990"
            try:
                dob = datetime.strptime(fighter_object.DOB, date_format)
                fighter_stats[fighter]["dob"] = dob.year
            except ValueError:
                fighter_stats[fighter]["dob"] = 0
            #print(fighter_object.DOB)
            fighter_stats[fighter]["height"] = parse_height_inches(fighter_object.Height)
            # reach is unlisted for ~half of all fighters; height is the standard proxy
            reach = parse_reach_inches(fighter_object.Reach)
            fighter_stats[fighter]["reach"] = reach if reach else fighter_stats[fighter]["height"]
            # Switch/Open Stance/unknown count as 0: the flag encodes the
            # southpaw-vs-orthodox matchup via its oppdiff, not full stance
            fighter_stats[fighter]["southpaw"] = 1 if (fighter_object.Stance or "").strip() == "Southpaw" else 0

processed_fights=[]
count=0

def C2(x):
    return (x*(x+1))//2

def sqr(n):
    return n*n

def sqrSum(n):
    return n*(n+1)*(2*n+1)/6

def getTime(fight):
    return (float(fight['Round'])-1)*5 + float(fight['Time'])

def getDate(date_string, date_format):
    try:
        return datetime.strptime(date_string, date_format)
    except ValueError:
        return None
    
# PROCESS FIGHTS TO RED AND BLUE 
def processFight(fight, Red, Blue):
    name = row_names(fight)
    winner = fight['Winner']
    Result='draw'
    if winner == name[Red]:
        Result = 'win'
    elif winner == name[Blue]:
        Result = 'loss'
    if Result == 'draw':
        return
    switch = random.choice([True, False])
    if switch:
        Red, Blue = Blue, Red 
        if Result == 'win':
            Result = 'loss'
        elif Result == 'loss':
            Result = 'win'

    processed_fight = {"Result": Result}
    if fighter_stats[Red]["totalfights"] >= 2 and fighter_stats[Blue]["totalfights"] >= 2:
        processed_fight['Red Fighter'] = name[Red]
        processed_fight['Blue Fighter'] = name[Blue]
        processed_fight['Title'] = fight['Title']
        processed_fight['Date'] = fight['Date']
        fight_date=getDate(fight['Date'], "%B %d, %Y")
        # skip fights we can't date or where a fighter's DOB or height is unknown
        # (age/height/reach features would be garbage)
        if not fight_date or fighter_stats[Red]['dob'] == 0 or fighter_stats[Blue]['dob'] == 0:
            return
        if fighter_stats[Red]['height'] == 0 or fighter_stats[Blue]['height'] == 0:
            return
        processed_fight['Red age'] = fight_date.year - fighter_stats[Red]['dob']
        processed_fight['Blue age'] = fight_date.year - fighter_stats[Blue]['dob']
        processed_fight['age oppdiff'] = processed_fight['Red age'] - processed_fight['Blue age'] 
        processed_fight['Red last_fight'] = (fight_date-fighter_stats[Red]["last_fight"]).days
        processed_fight['Blue last_fight'] = (fight_date-fighter_stats[Blue]["last_fight"]).days
        processed_fight['last_fight oppdiff'] = processed_fight['Red last_fight'] - processed_fight['Blue last_fight']
        for feature in feature_list:
            if feature in fighter_stats[Red] and feature in fighter_stats[Blue]:
                processed_fight[f'Red {feature}'] = fighter_stats[Red][feature]
                processed_fight[f'Blue {feature}'] = fighter_stats[Blue][feature]
                if feature in header_features:
                    processed_fight[f'Red {feature} differential'] = fighter_stats[Red][f'{feature} differential']
                    processed_fight[f'Blue {feature} differential'] = fighter_stats[Blue][f'{feature} differential']
                    processed_fight[f'Red {feature}'] /= sqrSum(fighter_stats[Red]["totalfights"])
                    processed_fight[f'Blue {feature}'] /= sqrSum(fighter_stats[Blue]["totalfights"])
                    processed_fight[f'Red {feature} differential'] /= sqrSum(fighter_stats[Red]["totalfights"])
                    processed_fight[f'Blue {feature} differential'] /= sqrSum(fighter_stats[Blue]["totalfights"])
                    if "%" in feature:
                        processed_fight[f'Red {feature} defense'] = fighter_stats[Red][f"{feature} defense"] / sqrSum(fighter_stats[Red]["totalfights"])
                        processed_fight[f'Blue {feature} defense'] = fighter_stats[Blue][f"{feature} defense"] / sqrSum(fighter_stats[Blue]["totalfights"])
                if feature in hardcoded_features_divide:
                    processed_fight[f'Red {feature}'] /= fighter_stats[Red]["totalfights"]
                    processed_fight[f'Blue {feature}'] /= fighter_stats[Blue]["totalfights"]
        for feature in feature_list:
            # Basic feature difference
            red_key = f'Red {feature}'
            blue_key = f'Blue {feature}'
            if red_key in processed_fight and blue_key in processed_fight:
                processed_fight[f'{feature} oppdiff'] = processed_fight[red_key] - processed_fight[blue_key]

            # Differential feature difference
            red_diff_key = f'Red {feature} differential'
            blue_diff_key = f'Blue {feature} differential'
            if red_diff_key in processed_fight and blue_diff_key in processed_fight:
                processed_fight[f'{feature} differential oppdiff'] = processed_fight[red_diff_key] - processed_fight[blue_diff_key]

            # Defense feature difference
            red_defense_key = f'Red {feature} defense'
            blue_defense_key = f'Blue {feature} defense'
            if red_defense_key in processed_fight and blue_defense_key in processed_fight:
                processed_fight[f'{feature} defense oppdiff'] = processed_fight[red_defense_key] - processed_fight[blue_defense_key]

        processed_fights.append(processed_fight)

print(header_features)
for fight in fights:
    count+=1
    Red = fighter_key(fight, 'Red')
    Blue = fighter_key(fight, 'Blue')

    processFight(fight, Red, Blue)

    ### update stats
    fighter_stats[Red]["totalfights"] += 1
    fighter_stats[Blue]["totalfights"] += 1
    redfights = fighter_stats[Red]["totalfights"] 
    bluefights = fighter_stats[Blue]["totalfights"] 
    for feature in feature_list:
        if feature in hardcoded_features or feature in hardcoded_features_divide:
            continue
        red_feature_key = "Red " + feature
        blue_feature_key = "Blue " + feature
        # parse everything first so a bad value can't leave a fighter's stats half-updated
        try:
            red_value = float(fight[red_feature_key])
            blue_value = float(fight[blue_feature_key])
            fight_time = getTime(fight)
        except (KeyError, ValueError):
            continue
        if fight_time <= 0:
            continue

        fighter_stats[Red][f"{feature} differential"] += (red_value - blue_value) * sqr(redfights)
        fighter_stats[Blue][f"{feature} differential"] += (blue_value - red_value) * sqr(bluefights)
        if "%" in feature:
            # accuracy decimals are already rates; dividing by fight_time would make them per-minute
            fighter_stats[Red][f"{feature}"] += red_value * sqr(redfights)
            fighter_stats[Blue][f"{feature}"] += blue_value * sqr(bluefights)
            fighter_stats[Red][f"{feature} defense"] += (1 - blue_value) * sqr(redfights)
            fighter_stats[Blue][f"{feature} defense"] += (1 - red_value) * sqr(bluefights)
        else:
            fighter_stats[Red][f"{feature}"] += red_value * sqr(redfights) / fight_time
            fighter_stats[Blue][f"{feature}"] += blue_value * sqr(bluefights) / fight_time
    winner = fight['Winner']
    Result='draw'
    if winner == fight['Red Fighter']:
        Result = 'win'
    elif winner == fight['Blue Fighter']:
        Result = 'loss'

    title=False
    if "Title" in fight['Title']:
        title=True
    rating_a = fighter_stats[Red]["elo"]
    rating_b = fighter_stats[Blue]["elo"]
    fighter_stats[Red]["oppelo"]+=rating_b
    fighter_stats[Blue]["oppelo"]+=rating_a
    fight_date=getDate(fight['Date'], "%B %d, %Y")
    fighter_stats[Red]["last_fight"]=fight_date
    fighter_stats[Blue]["last_fight"]=fight_date
    fighter_stats[Red]["avg age"]+=fight_date.year-fighter_stats[Red]["dob"]
    fighter_stats[Blue]["avg age"]+=fight_date.year-fighter_stats[Blue]["dob"]
    if Result=='win':
        fighter_stats[Blue]["losestreak"]+=1
        fighter_stats[Red]["losestreak"]=0
        fighter_stats[Red]["winstreak"]+=1
        fighter_stats[Blue]["winstreak"]=0
        fighter_stats[Red]["wins"]+=1 
        if title:
            fighter_stats[Red]["titlewins"]+=1
    if Result=='loss':
        fighter_stats[Red]["losestreak"]+=1
        fighter_stats[Blue]["losestreak"]=0
        fighter_stats[Blue]["winstreak"]+=1
        fighter_stats[Red]["winstreak"]=0
        fighter_stats[Blue]["wins"]+=1 
        if title:
            fighter_stats[Blue]["titlewins"]+=1

    new_rating_a, new_rating_b = update_elo_ratings(rating_a, rating_b, Result, redfights, bluefights)
    fighter_stats[Red]["elo"]=new_rating_a
    fighter_stats[Blue]["elo"]=new_rating_b
    

def export_processed_fights(processed_fights, filename=os.path.join('data', 'detailed_fights.csv')):
    with open(filename, mode='w', newline='') as file:
        if processed_fights:  # check if the list is not empty
            headers = processed_fights[0].keys()  # Get the keys from the first dictionary as headers
            print(headers)
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for fight in processed_fights:
                writer.writerow(fight)

def export_fighter_stats(fighter_stats, filename=os.path.join('data', 'detailed_fighter_stats.csv')):
    with open(filename, mode='w', newline='') as file:
        if fighter_stats:  # check if the dictionary is not empty
            example_fighter = next(iter(fighter_stats.values()))  # Get an example of the inner dictionary
            headers = ['Fighter', 'ID'] + list(example_fighter.keys())  # name, ufcstats id, each stat
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()

            for fighter, stats in fighter_stats.items():
                row = {'Fighter': names[fighter], 'ID': fighter if fighter != names[fighter] else ''}
                row.update(stats)  # Add the stats
                writer.writerow(row)

# Assuming your processed_fights and fighter_stats are ready
export_processed_fights(processed_fights)
export_fighter_stats(fighter_stats)

def write_to_text_file(data, file_path, is_fighter_stats=False):
    with open(file_path, 'w') as file:
        if is_fighter_stats:
            for fighter, stats in data.items():
                file.write(f"Fighter: {names[fighter]}\n")
                if fighter != names[fighter]:
                    file.write(f"  id: {fighter}\n")
                for stat, value in stats.items():
                    file.write(f"  {stat}: {value}\n")
                file.write("\n")
        else:
            for fight in data:
                for key, value in fight.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")

# Paths for the output text files
processed_fights_txt_path = os.path.join('data', 'processed_fights_readable.txt')
fighter_stats_txt_path = os.path.join('data', 'fighter_stats_readable.txt')

# Write the processed fights and fighter stats to text files
write_to_text_file(processed_fights, processed_fights_txt_path)
write_to_text_file(fighter_stats, fighter_stats_txt_path, is_fighter_stats=True)

# Print paths to the generated files or handle further as needed
# print(f"Processed fights saved to {processed_fights_txt_path}")
# print(f"Fighter stats saved to {fighter_stats_txt_path}")

#TODO: STORE ALL THE PREVIOUS FIGHTS OF A FIGHTER. WHEN IT COMES TIME TO PROCESS, loop through the last 5 fights and compute the stats