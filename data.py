import csv

from constants import (
    ONEOFFS_TO_TOKEN,
    TOKEN_TO_STRENGTH,
)


class DataLoader:
    @classmethod
    def _map_to_strength(cls, strength_str):
        lowercase = strength_str.lower()
        if lowercase in ONEOFFS_TO_TOKEN:
            token = ONEOFFS_TO_TOKEN[lowercase]
        else:
            token = lowercase

        return TOKEN_TO_STRENGTH[token]

    @classmethod
    def _preprocess(cls, strengths):
        return [cls._map_to_strength(strength_str.strip()) for strength_str in strengths]

    @classmethod
    def load_user_strengths(cls, file):
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            user_strengths = {}
            for row in reader:
                if len(row) != 26:
                    print("{} has issue".format(row[0]))
                name = row[0]
                evaluator = row[1].lower().strip()
                strengths_in_order = row[2:]
                strengths = cls._preprocess(strengths_in_order)

                if evaluator == "self":
                    user_strengths[name] = strengths
        return user_strengths