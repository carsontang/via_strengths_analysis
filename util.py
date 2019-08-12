import csv
import numpy as np

from enum import Enum
from scipy import stats


class Strength(Enum):
    APPRECIATION = 1
    BRAVERY = 2
    CREATIVITY = 3
    CURIOSITY = 4
    FAIRNESS = 5
    FORGIVENESS = 6
    GRATITUDE = 7
    HONESTY = 8
    HOPE = 9
    HUMILITY = 10
    HUMOR = 11
    JUDGMENT = 12
    KINDNESS = 13
    LEADERSHIP = 14
    LOVE = 15
    LOVE_OF_LEARNING = 16
    PERSEVERANCE = 17
    PERSPECTIVE = 18
    PRUDENCE = 19
    SELF_REGULATION = 20
    SOCIAL_INTELLIGENCE = 21
    SPIRITUALITY = 22
    TEAMWORK = 23
    ZEST = 24


ONEOFFS_TO_TOKEN = {
    "appreciation of beauty & excellence": "appreciation",
    "love of learning": "love_of_learning",
    "self-regulation": "self_regulation",
    "social intelligence": "social_intelligence",
}

TOKEN_TO_STRENGTH = {
    "appreciation": Strength.APPRECIATION,
    "bravery": Strength.BRAVERY,
    "creativity": Strength.CREATIVITY,
    "curiosity": Strength.CURIOSITY,
    "fairness": Strength.FAIRNESS,
    "forgiveness": Strength.FORGIVENESS,
    "gratitude": Strength.GRATITUDE,
    "honesty": Strength.HONESTY,
    "hope": Strength.HOPE,
    "humility": Strength.HUMILITY,
    "humor": Strength.HUMOR,
    "judgment": Strength.JUDGMENT,
    "kindness": Strength.KINDNESS,
    "leadership": Strength.LEADERSHIP,
    "love": Strength.LOVE,
    "love_of_learning": Strength.LOVE_OF_LEARNING,
    "perseverance": Strength.PERSEVERANCE,
    "perspective": Strength.PERSPECTIVE,
    "prudence": Strength.PRUDENCE,
    "self_regulation": Strength.SELF_REGULATION,
    "social_intelligence": Strength.SOCIAL_INTELLIGENCE,
    "spirituality": Strength.SPIRITUALITY,
    "teamwork": Strength.TEAMWORK,
    "zest": Strength.ZEST
}


# class SurveyResultFactory:
#     @classmethod
#     def create_from_csv(cls, csv):
#         owner = csv[0]
#         rater = csv[1]
#         num_strengths = 24
#         rankings = np.empty(num_strengths)
#         for i, strength in enumerate(csv[2:]):
#             rankings[


class SurveyResult:
    def __init__(self, owner, rater):
        self.owner = owner
        self.rater = rater


def softmax(scores):
    e_scores = np.exp(scores)
    denominator = e_scores.sum()
    return e_scores / denominator


def cross_entropy(p, q):
    return stats.entropy(p, q)


def cognitive_dissonance_score(
    self_perception_scores,
    external_perception_scores
):
    self_softmax = softmax(self_perception_scores)
    external_softmax = softmax(external_perception_scores)
    return cross_entropy(self_softmax, external_softmax)