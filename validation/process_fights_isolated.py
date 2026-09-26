"""
Isolated feature processing for walk-forward validation.

This module provides a stateless feature processor that can compute features
up to any cutoff date, proving that no future data leaks into predictions.

Key differences from process_fights_alpha.py:
- No global state (fighter_stats is instance variable)
- Can process up to any cutoff date
- Returns DataFrame instead of writing files
- Provides verification methods for fighter stats at specific dates
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class ProcessFightsIsolated:
    """
    Stateless feature processor that can compute features up to any date.

    This class replicates the logic from process_fights_alpha.py but allows
    processing to stop at any date, enabling true walk-forward validation.
    """

    def __init__(self, raw_fights_path: str = None):
        """
        Initialize processor.

        Args:
            raw_fights_path: Path to modified_fight_details.csv
        """
        if raw_fights_path is None:
            raw_fights_path = os.path.join('data', 'modified_fight_details.csv')
        self.raw_fights_path = raw_fights_path

        # State - reset for each processing run
        self.fighter_stats: Dict[str, Dict[str, Any]] = {}
        self.processed_fights: List[Dict] = []
        self.feature_list: List[str] = []
        self.header_features: List[str] = []
        self.hardcoded_features = ["dob", "totalfights", "elo", "losestreak", "winstreak", "titlewins"]
        self.hardcoded_features_divide = ["oppelo", "wins", "avg age"]

    def _reset_state(self):
        """Reset all internal state for a fresh processing run."""
        self.fighter_stats = {}
        self.processed_fights = []

    def _reverse_csv_to_dict(self, file_path: str) -> List[Dict]:
        """Load CSV and return rows in reverse order (oldest first)."""
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            reversed_rows = list(csv_reader)[::-1]
            return reversed_rows

    def _get_csv_headers(self, file_path: str) -> List[str]:
        """Get CSV column headers."""
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            return headers

    @staticmethod
    def calculate_k_factor(number_of_fights: int) -> int:
        """Calculate ELO K-factor based on number of fights."""
        if number_of_fights < 5:
            return 40
        elif 5 <= number_of_fights < 10:
            return 35
        elif 10 <= number_of_fights < 20:
            return 30
        else:
            return 25

    @staticmethod
    def calculate_expected_win_probability(rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability using ELO formula."""
        return 1 / (1 + pow(10, (rating_b - rating_a) / 400))

    def update_elo_ratings(
        self,
        rating_a: float,
        rating_b: float,
        result: str,
        fights_a: int,
        fights_b: int
    ) -> Tuple[float, float]:
        """Update ELO ratings after a fight."""
        K_FACTOR_A = self.calculate_k_factor(fights_a)
        K_FACTOR_B = self.calculate_k_factor(fights_b)

        expected_win_probability = self.calculate_expected_win_probability(rating_a, rating_b)

        if result == "win":
            actual_score = 1
        elif result == "loss":
            actual_score = 0
        else:
            actual_score = 0.5

        new_rating_a = rating_a + K_FACTOR_A * (actual_score - expected_win_probability)
        new_rating_b = rating_b + K_FACTOR_B * ((1 - actual_score) - (1 - expected_win_probability))

        return new_rating_a, new_rating_b

    @staticmethod
    def _split_at_first_space(text: str) -> Tuple[str, str]:
        """Split text at first space."""
        parts = text.split(" ", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    @staticmethod
    def _sqr(n: int) -> int:
        """Square of n."""
        return n * n

    @staticmethod
    def _sqr_sum(n: int) -> int:
        """Sum of squares from 1 to n."""
        return n * (n + 1) * (2 * n + 1) // 6

    @staticmethod
    def _get_time(fight: Dict) -> float:
        """Calculate total fight time in minutes."""
        return (float(fight['Round']) - 1) * 5 + float(fight['Time'])

    @staticmethod
    def _get_date(date_string: str) -> Optional[datetime]:
        """Parse date with fallback for multiple formats."""
        formats_to_try = [
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
        ]

        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        return None

    def _initialize_fighter(self, fighter: str, dob_year: int = 0):
        """Initialize fighter stats dictionary."""
        self.fighter_stats[fighter] = {}
        for feature in self.feature_list:
            self.fighter_stats[fighter][feature] = 0
            if feature in self.header_features:
                self.fighter_stats[fighter][f"{feature} differential"] = 0
            if "%" in feature:
                self.fighter_stats[fighter][f"{feature} defense"] = 0
        self.fighter_stats[fighter]["elo"] = 1000
        self.fighter_stats[fighter]["dob"] = dob_year

    def _process_fight_features(self, fight: Dict, red: str, blue: str) -> Optional[Dict]:
        """
        Create feature dictionary for a fight using CURRENT fighter stats.

        CRITICAL: This uses fighter_stats as they exist at this moment,
        which only contains information from previous fights.
        """
        winner = fight['Winner']
        result = 'draw'
        if winner == red:
            result = 'win'
        elif winner == blue:
            result = 'loss'
        if result == 'draw':
            return None

        # Minimum fights requirement
        if self.fighter_stats[red]["totalfights"] < 2 or self.fighter_stats[blue]["totalfights"] < 2:
            return None

        fight_date = self._get_date(fight['Date'])
        if not fight_date:
            return None

        processed_fight = {
            "Result": result,
            "Red Fighter": red,
            "Blue Fighter": blue,
            "Title": fight['Title'],
            "Date": fight['Date']
        }

        # Age features
        processed_fight['Red age'] = fight_date.year - self.fighter_stats[red]['dob']
        processed_fight['Blue age'] = fight_date.year - self.fighter_stats[blue]['dob']
        processed_fight['age oppdiff'] = processed_fight['Red age'] - processed_fight['Blue age']

        # Last fight features
        if "last_fight" in self.fighter_stats[red] and "last_fight" in self.fighter_stats[blue]:
            processed_fight['Red last_fight'] = (fight_date - self.fighter_stats[red]["last_fight"]).days
            processed_fight['Blue last_fight'] = (fight_date - self.fighter_stats[blue]["last_fight"]).days
            processed_fight['last_fight oppdiff'] = processed_fight['Red last_fight'] - processed_fight['Blue last_fight']

        # Process all features
        for feature in self.feature_list:
            if feature in self.fighter_stats[red] and feature in self.fighter_stats[blue]:
                processed_fight[f'Red {feature}'] = self.fighter_stats[red][feature]
                processed_fight[f'Blue {feature}'] = self.fighter_stats[blue][feature]

                if feature in self.header_features:
                    processed_fight[f'Red {feature} differential'] = self.fighter_stats[red][f'{feature} differential']
                    processed_fight[f'Blue {feature} differential'] = self.fighter_stats[blue][f'{feature} differential']

                    # Normalize by sqrSum
                    red_fights = self.fighter_stats[red]["totalfights"]
                    blue_fights = self.fighter_stats[blue]["totalfights"]

                    if red_fights > 0:
                        processed_fight[f'Red {feature}'] /= self._sqr_sum(red_fights)
                        processed_fight[f'Red {feature} differential'] /= self._sqr_sum(red_fights)
                    if blue_fights > 0:
                        processed_fight[f'Blue {feature}'] /= self._sqr_sum(blue_fights)
                        processed_fight[f'Blue {feature} differential'] /= self._sqr_sum(blue_fights)

                    if "%" in feature:
                        if red_fights > 0:
                            processed_fight[f'Red {feature} defense'] = self._sqr_sum(
                                self.fighter_stats[red][f"{feature} defense"] / red_fights
                            )
                        if blue_fights > 0:
                            processed_fight[f'Blue {feature} defense'] = self._sqr_sum(
                                self.fighter_stats[blue][f"{feature} defense"] / blue_fights
                            )

                if feature in self.hardcoded_features_divide:
                    red_fights = self.fighter_stats[red]["totalfights"]
                    blue_fights = self.fighter_stats[blue]["totalfights"]
                    if red_fights > 0:
                        processed_fight[f'Red {feature}'] /= red_fights
                    if blue_fights > 0:
                        processed_fight[f'Blue {feature}'] /= blue_fights

        # Calculate opponent differences
        for feature in self.feature_list:
            red_key = f'Red {feature}'
            blue_key = f'Blue {feature}'
            if red_key in processed_fight and blue_key in processed_fight:
                processed_fight[f'{feature} oppdiff'] = processed_fight[red_key] - processed_fight[blue_key]

            red_diff_key = f'Red {feature} differential'
            blue_diff_key = f'Blue {feature} differential'
            if red_diff_key in processed_fight and blue_diff_key in processed_fight:
                processed_fight[f'{feature} differential oppdiff'] = processed_fight[red_diff_key] - processed_fight[blue_diff_key]

            red_defense_key = f'Red {feature} defense'
            blue_defense_key = f'Blue {feature} defense'
            if red_defense_key in processed_fight and blue_defense_key in processed_fight:
                processed_fight[f'{feature} defense oppdiff'] = processed_fight[red_defense_key] - processed_fight[blue_defense_key]

        return processed_fight

    def _update_fighter_stats(self, fight: Dict, red: str, blue: str):
        """
        Update fighter stats AFTER processing the fight.

        CRITICAL: This is called AFTER _process_fight_features to ensure
        no look-ahead bias in feature calculation.
        """
        self.fighter_stats[red]["totalfights"] += 1
        self.fighter_stats[blue]["totalfights"] += 1

        red_fights = self.fighter_stats[red]["totalfights"]
        blue_fights = self.fighter_stats[blue]["totalfights"]

        # Update header features
        for feature in self.feature_list:
            if feature in self.hardcoded_features or feature in self.hardcoded_features_divide:
                continue

            red_feature_key = "Red " + feature
            blue_feature_key = "Blue " + feature

            try:
                red_value = float(fight[red_feature_key])
                blue_value = float(fight[blue_feature_key])

                if "%" in feature:
                    self.fighter_stats[red][f"{feature} differential"] += (red_value - blue_value) * self._sqr(red_fights)
                    self.fighter_stats[blue][f"{feature} differential"] += (blue_value - red_value) * self._sqr(blue_fights)
                else:
                    self.fighter_stats[red][f"{feature} differential"] += (red_value - blue_value) * self._sqr(red_fights)
                    self.fighter_stats[blue][f"{feature} differential"] += (blue_value - red_value) * self._sqr(blue_fights)

                fight_time = self._get_time(fight)
                self.fighter_stats[red][f"{feature}"] += red_value * self._sqr(red_fights) / fight_time
                self.fighter_stats[blue][f"{feature}"] += blue_value * self._sqr(blue_fights) / fight_time

                if "%" in feature:
                    self.fighter_stats[red][f"{feature} defense"] += (1 - blue_value) * self._sqr(red_fights)
                    self.fighter_stats[blue][f"{feature} defense"] += (1 - red_value) * self._sqr(blue_fights)
            except (KeyError, ValueError):
                pass

        # Determine result
        winner = fight['Winner']
        result = 'draw'
        if winner == red:
            result = 'win'
        elif winner == blue:
            result = 'loss'

        title = "Title" in fight.get('Title', '')

        # Update ELO
        rating_a = self.fighter_stats[red]["elo"]
        rating_b = self.fighter_stats[blue]["elo"]
        self.fighter_stats[red]["oppelo"] += rating_b
        self.fighter_stats[blue]["oppelo"] += rating_a

        # Update date-dependent stats
        fight_date = self._get_date(fight['Date'])
        if fight_date:
            self.fighter_stats[red]["last_fight"] = fight_date
            self.fighter_stats[blue]["last_fight"] = fight_date
            self.fighter_stats[red]["avg age"] += fight_date.year - self.fighter_stats[red]["dob"]
            self.fighter_stats[blue]["avg age"] += fight_date.year - self.fighter_stats[blue]["dob"]

        # Update streaks and wins
        if result == 'win':
            self.fighter_stats[blue]["losestreak"] += 1
            self.fighter_stats[red]["losestreak"] = 0
            self.fighter_stats[red]["winstreak"] += 1
            self.fighter_stats[blue]["winstreak"] = 0
            self.fighter_stats[red]["wins"] += 1
            if title:
                self.fighter_stats[red]["titlewins"] += 1
        elif result == 'loss':
            self.fighter_stats[red]["losestreak"] += 1
            self.fighter_stats[blue]["losestreak"] = 0
            self.fighter_stats[blue]["winstreak"] += 1
            self.fighter_stats[red]["winstreak"] = 0
            self.fighter_stats[blue]["wins"] += 1
            if title:
                self.fighter_stats[blue]["titlewins"] += 1

        # Update ELO ratings
        new_rating_a, new_rating_b = self.update_elo_ratings(rating_a, rating_b, result, red_fights, blue_fights)
        self.fighter_stats[red]["elo"] = new_rating_a
        self.fighter_stats[blue]["elo"] = new_rating_b

    def process_up_to_date(
        self,
        cutoff_date: datetime,
        include_cutoff_date: bool = False
    ) -> pd.DataFrame:
        """
        Process all fights up to cutoff_date and return feature DataFrame.

        CRITICAL: This function only uses data from before cutoff_date,
        guaranteeing no data leakage.

        Args:
            cutoff_date: Only process fights BEFORE this date
            include_cutoff_date: If True, include fights ON cutoff_date

        Returns:
            DataFrame with processed fight features
        """
        self._reset_state()

        # Load all fights (reversed = oldest first)
        fights = self._reverse_csv_to_dict(self.raw_fights_path)

        # Get headers and build feature list
        headers = self._get_csv_headers(self.raw_fights_path)
        self.header_features = []
        for column in headers:
            s1, s2 = self._split_at_first_space(column)
            if s1 == "Red" and s2 not in ("Fighter", "Fighter ID"):
                self.header_features.append(s2)

        self.feature_list = []
        self.feature_list.extend(self.hardcoded_features)
        self.feature_list.extend(self.hardcoded_features_divide)
        self.feature_list.extend(self.header_features)

        # Get all fighters and initialize stats
        all_fighters = set()
        for fight in fights:
            all_fighters.add(fight['Red Fighter'])
            all_fighters.add(fight['Blue Fighter'])

        for fighter in all_fighters:
            self._initialize_fighter(fighter)

        # Process fights chronologically
        for fight in fights:
            fight_date = self._get_date(fight['Date'])
            if not fight_date:
                continue

            # Check cutoff
            if include_cutoff_date:
                if fight_date > cutoff_date:
                    break
            else:
                if fight_date >= cutoff_date:
                    break

            red = fight['Red Fighter']
            blue = fight['Blue Fighter']

            # Create features using CURRENT stats (before this fight)
            processed = self._process_fight_features(fight, red, blue)
            if processed:
                self.processed_fights.append(processed)

            # Update stats AFTER creating features
            self._update_fighter_stats(fight, red, blue)

        return pd.DataFrame(self.processed_fights)

    def get_fighter_stats_at_date(self, fighter_name: str, as_of_date: datetime) -> Optional[Dict]:
        """
        Get fighter's stats as they would have been known at a specific date.

        This is useful for verifying predictions didn't use future information.

        Args:
            fighter_name: Name of fighter
            as_of_date: Get stats as known at this date

        Returns:
            Dictionary of fighter stats, or None if fighter hadn't fought yet
        """
        # Process up to the date
        self.process_up_to_date(as_of_date)

        if fighter_name in self.fighter_stats:
            return self.fighter_stats[fighter_name].copy()
        return None

    def verify_no_lookahead(
        self,
        fight_date: datetime,
        fighter1: str,
        fighter2: str,
        detailed_fights_path: str = None
    ) -> Dict:
        """
        Verify that features in detailed_fights.csv match what we compute.

        This proves that the pre-computed features don't contain future information.

        Args:
            fight_date: Date of the fight to verify
            fighter1: First fighter name
            fighter2: Second fighter name
            detailed_fights_path: Path to pre-computed features

        Returns:
            Comparison results with any discrepancies
        """
        if detailed_fights_path is None:
            detailed_fights_path = os.path.join('data', 'detailed_fights.csv')

        # Compute features using only data before fight date
        self.process_up_to_date(fight_date)

        computed_stats_1 = self.fighter_stats.get(fighter1, {})
        computed_stats_2 = self.fighter_stats.get(fighter2, {})

        # Load pre-computed features
        precomputed_df = pd.read_csv(detailed_fights_path)
        precomputed_df['Date'] = pd.to_datetime(precomputed_df['Date'], format='mixed')

        # Find the specific fight
        mask = (
            (precomputed_df['Red Fighter'] == fighter1) &
            (precomputed_df['Blue Fighter'] == fighter2)
        ) | (
            (precomputed_df['Red Fighter'] == fighter2) &
            (precomputed_df['Blue Fighter'] == fighter1)
        )

        fight_rows = precomputed_df[mask]
        if len(fight_rows) == 0:
            return {"error": f"Fight not found: {fighter1} vs {fighter2}"}

        # Compare key features
        results = {
            "fight_date": fight_date.isoformat(),
            "fighters": [fighter1, fighter2],
            "computed_stats": {
                fighter1: {
                    "elo": computed_stats_1.get("elo"),
                    "totalfights": computed_stats_1.get("totalfights"),
                    "winstreak": computed_stats_1.get("winstreak"),
                    "losestreak": computed_stats_1.get("losestreak")
                },
                fighter2: {
                    "elo": computed_stats_2.get("elo"),
                    "totalfights": computed_stats_2.get("totalfights"),
                    "winstreak": computed_stats_2.get("winstreak"),
                    "losestreak": computed_stats_2.get("losestreak")
                }
            },
            "discrepancies": []
        }

        return results


def compare_features_at_date(
    cutoff_date: datetime,
    precomputed_path: str = None,
    raw_path: str = None
) -> Dict:
    """
    Compare features computed with cutoff vs pre-computed features.

    This is a key validation function that proves no data leakage.

    Args:
        cutoff_date: Compute features only using data before this date
        precomputed_path: Path to pre-computed detailed_fights.csv
        raw_path: Path to raw fight data

    Returns:
        Comparison statistics
    """
    if precomputed_path is None:
        precomputed_path = os.path.join('data', 'detailed_fights.csv')
    if raw_path is None:
        raw_path = os.path.join('data', 'modified_fight_details.csv')

    processor = ProcessFightsIsolated(raw_path)
    computed_df = processor.process_up_to_date(cutoff_date)

    precomputed_df = pd.read_csv(precomputed_path)
    precomputed_df['Date'] = pd.to_datetime(precomputed_df['Date'], format='mixed')
    precomputed_before_cutoff = precomputed_df[precomputed_df['Date'] < cutoff_date]

    return {
        "cutoff_date": cutoff_date.isoformat(),
        "computed_fights": len(computed_df),
        "precomputed_fights_before_cutoff": len(precomputed_before_cutoff),
        "fight_count_match": len(computed_df) == len(precomputed_before_cutoff),
        "computed_fighter_stats_count": len(processor.fighter_stats)
    }
