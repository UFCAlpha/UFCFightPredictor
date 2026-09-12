# UFC Fight Predictor

# V4 Alpha Model

The new and improved UFC AI model is consistently averaging around $100 profit over the past year of UFC events in 2023, when starting with $1000 (10% profit). With a robust accuracy of 64%, it can accurately identify winning percentages of each fighter which is used in the Kelly Criterion to determine how much to bet on each fight. The best result we have simulated was turning $1000 on February 22, 2022 to $3042.19 by December 16, 2023, as well as averaging $1600 on this time period, which shows what could possibly happen in the long term if the model learns the fighting meta well. Previous research studies have found accuracies maxing out at around 63-64%. The website can be viewed at http://betufc.ca/.

# The Pipeline

## Data

### Web Scraping

All of the data was scraped from ufcstats.com. Previously, we had scraped data for each fighter as well as their fight histories and statistics, stored in an SQL database currently inside the instance folder. However, in order to get more detailed fight statistics, we had to revamp the scraper entirely. We have now collected stats such as strike accuracy, control time, and submissions attempts for every fight from 1994 to 2023. The newly collected statistics are stored inside the data folder.

### Data Processing and Cleaning

First, we need to clean the data, removing incomplete values, outdated fights, and duplicated data that was scraped. We also need to process all the data in a way which our machine learning model can understand, and retrieve all the raw numbers. This is done within modify_fights.py.

### Feature Engineering

With around 15 base stats for each fight, we now process the statistics in order to create expressive and predictive features. In order to predict a fight, we need the past fight histories of both fighters. The data scraped tells us which fighter is in the red corner and which fighter is in the blue corner. For simplicity in code, we call them Red and Blue.

By looping through all fights from past to present, we can dynamically compute stats from their past fights and then use those stats to predict a current fight. By doing this, we make sure that only data accessible before a fight has happened is used to predict a fight. Without doing this, our model accuracy would be up to 80+%, a mistake that some predictors with absurdly high accuracy may have.

Over 180+ features are engineering from these 15 base stats. For each base stat, such as significant strikes landed, we compute the number landed per minute, the accuracy of strikes, the difference between how many strikes you landed and how many strikes you took, the percentages of strikes you dodged, etc. Then, these stats are accumulated and processed into a weighted average which weighs recent fights much heavier than past, in order to get an accurate picture of each fighter's skills at different points in time. These in-depth statistics are then fed into the LightGBM ML model to be used to predict fights. This process is done within process_fights_alpha.py.

## Model

### LightGBM

We chose LightGBM (Light Gradient Boosting Machine), a tree-based learning algorithm, due to its exceptional speed, efficiency, and predicting power. LightGBM excels in capturing complex, non-linear relationships within UFC fight data, providing valuable insights into feature importance and tuning. With the new Alpha model, our accuracies were much more consistent than before. With much less features in our previous model, changing a few features could lead to dramatic changes in accuracy. Now that we have more robust data with a much wider selection of features, the reliability of the model was greatly increased. The code can be found within ml_ensemble.py.

### Feature Selection

With so many features, it's important to select those which actually help the model. We prune out features which have high correlation with each other, as well as features which have low impact on the model. This reduces noise and helps the model pick out the correct percentages.

### Hyperparameter Tuning and Trials

LightGBM allows you to fine-tune your model, setting parameters such as learning rate, number of leaves in a tree, etc. These can all have great impacts on the predictions that the model makes. We utilized the optuna library to run trails in order to select the best hyperparameters for the model. Our current model accuracy is 64%.

## Testing

Go to testing/testing_time_period.py. At the bottom, run process_dates with the desired start and end dates. This will automatically simulate our predictions on the chosen time period and write the results to test_results/testing_time_period_results.txt. Note that we only have reliable odds data from 2021 onwards, so testing will only work for 2021 and onwards. You may also choose to graph the bankroll over time, by using the matplotlib code at the bottom.

## Results

Testing on the fight cards in 2023, we ran 10 separate trials with different tuned hyperparameters, and averaged an $1100 final bankroll starting from $1000. We used a conservative 0.05 Kelly Criterion betting strategy, which helps us determine what fraction of our bankroll to wager on each bet. Testing on the fight cards from both 2022 and 2023 with a more risky 0.1 Kelly Criterion, we average a final bankroll of $1600, while maxing out at around $3000. We found that our model is consistently gaining money and can sometimes go up to $2000 or $3000 over a longer period of time, while only going down to around $750. Starting with $1000, for 73/86 or 85% of the trials, the model goes above $1000, going down only 13/86 or 15% of the trials.

## How to Bet on the Next Fight Card

To bet on the next fight card, we use predict_fights_alpha.py. After pasting the ufcstats.com url into events_url, you can run the file which scrapes all the names of the fighters on the card. Given we have enough data on both fighters (i.e. both fighters are not debuting) their stats will be processed together and put into predict_fights_alpha.csv in the data folder. Next we run ml_ensemble.py which trains the model ensemble and saves it, then load_ensemble.py reads from the csv to make predictions. These results take into account all the selected stats and output winning and losing probabilities for each fighter. These probablities are outputted to betting_predictions.csv. Finally, we use betting_alpha.py which reads the fighters and the odds of each fight from ufc.com to make bets. After getting the probabilities from predicted_data.json, based on the odds and kelly criterion we can automate which fights we will bet on. It should also be noted that we take the probability closer to the odds in order to reduce error in predicting.

## Other Studies

Stanford University: 62.6%
https://cs229.stanford.edu/proj2019aut/data/assignment_308875_raw/26426025.pdf

MMA AI: 64%
https://www.mma-ai.net/

Tilburg University: 62% (RandomForest, Neural Network)
http://arno.uvt.nl/show.cgi?fid=156304

### Friday email picks

[Email setup and operations](docs/email-picks.md) covers Gmail configuration,
Friday 9 PM Toronto scheduling, the weigh-in hold switch, previewing predictions,
and bankroll-percentage stakes. Start with `./run_email_scheduled.sh --dry-run`;
email submission requires `--send` or installation of the dedicated scheduler.
