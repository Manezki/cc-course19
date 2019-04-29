import cmudict
import pickle
import os
import csv
from operator import add
import math

class Evaluator():
    def __init__(self):
        self.emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

        self.cmudict = cmudict.dict()

        self.title_bank = None
        self.folder = os.path.dirname(os.path.realpath(__file__))

        # Try reading content for the title_bank

        try:
            with open(os.path.join(self.folder, "data", "titles.pickle"), "rb") as f:
                self.title_bank = pickle.load(f)

        except FileNotFoundError:
            from title_scrape import download_gutenberg, gutenberg_preprocess

            download_gutenberg()
            gutenberg_preprocess()

            with open(os.path.join(self.folder, "data", "titles.pickle"), "rb") as f:
                self.title_bank = pickle.load(f)
        #Read content for the sentiment dictionary
        self.sentimentDictionary = {}
        with open(os.path.join(self.folder, "data", "EmotionLexicon.txt")) as emotionLexicon:
            lexicon = csv.reader(emotionLexicon, delimiter='\t')
            word = ""
            values = {}
            for row in lexicon:
                if word != row[0]:
                    self.sentimentDictionary[word] = values
                    word = row[0]
                    values = {}
                values[row[1]] = row[2]

    # Modified from https://www.python-course.eu/levenshtein_distance.php
    def __iterative_levenshtein(self, s, t, weights=(1, 1, 1)):
        """
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t

        weight_dict: keyword parameters setting the costs for characters,
                     the default value for a character will be 1
        """
        rows = len(s)+1
        cols = len(t)+1


        dist = [[0 for x in range(cols)] for x in range(rows)]
        # source prefixes can be transformed into empty strings
        # by deletions:
        for row in range(1, rows):
            dist[row][0] = dist[row-1][0] + weights[0]
        # target prefixes can be created from an empty source string
        # by inserting the characters
        for col in range(1, cols):
            dist[0][col] = dist[0][col-1] + weights[1]

        for col in range(1, cols):
            for row in range(1, rows):
                deletes = weights[0]
                inserts = weights[1]
                subs = max( (weights[2], weights[2]))
                if s[row-1] == t[col-1]:
                    subs = 0
                else:
                    subs = subs
                dist[row][col] = min(dist[row-1][col] + deletes,
                                     dist[row][col-1] + inserts,
                                     dist[row-1][col-1] + subs) # substitution

        return dist[row][col]


    def editDistance(self, phenotype, weights=(1, 1, 1)):
        """
        Calculate the shortest levenshtein distance between phenotype and known titles.

        Args:
            phenotype (str) : Candidate phenotype.
            title_bank (dict) : Known titles, needs to have dictionaries as values, and those disctionaries need to have
                                'title' key.
            weights (tuple of floats) : Weights for different operations. In order: Delete, Insert, Substitute

        Returns:
            int : Shortest edit distance.
        """

        # Checking for exact match from the dictionary is fast
        if phenotype in self.title_bank:
            return 0

        closest = 1000

        for _, b_info in self.title_bank.items():
            # Skip candidates using lower-bound of the levenshtein distance.
            # Does not take the weights into account
            if abs(len(phenotype.strip()) - len(b_info["title"].strip())) > closest:
                continue

            levenshtein = self.__iterative_levenshtein(phenotype.strip(), b_info["title"].strip(), weights)
            closest = min(closest, levenshtein)

        return closest

    def evaluate(self, title, emotion):
        """Runs the different evaluation schemes, which return values between 0 and 1, and returns an average over them.

        Args:
            title (list) : list of words forming the title when.

        Returns:
            float : Weighted average of the different evaluations.
        """
        val = 0
        val += self.eval_novelty(" ".join(title))
        val += self.eval_alliteration(title)
        val += self.eval_sentiment(title, emotion)
        return val / 3.0


    def eval_novelty(self, title):
        if self.title_bank is None:
            return 0.8
        else:
            dist = self.editDistance(title, (1, 1, 1))
            # Scale with the title length
            # Can be higher than 1 if weights are not all 1.
            dist = min(1.0, (dist/len(title))*(len(title)//5))
            return dist


    def eval_alliteration(self, title):
        unique_phonemes = []
        title_length = 1
        for word in title:
            try:
                phonemes = self.cmudict[word][0]
                title_length += len(phonemes)
                for phoneme in phonemes:
                    if phoneme not in unique_phonemes:
                        unique_phonemes.append(phoneme)
            except:
                #word was not in dict
                continue
        return len(unique_phonemes) / title_length


    def get_alliteration_score(self, ratio):
        """ A function that has it maximum = 1 when ratio is 1/2, meaning half of the phonemes in the
        title are non-unique, otherwise it grows close to 0
        """
        return (-4*(ratio-0.5))**2 + 1

    def eval_sentiment(self, title, emotion):
        """
        Builds a vector of emotions in the title and compares that vector to the emotion in the input
        Each word gets a weight of 1/n, where n is the number of words in the title
        """
        self.emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        goal_sentiment = list(map(lambda x: int(x == emotion), self.emotions))
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        title_sentiment = [0, 0, 0, 0, 0, 0]
        for word in title:
            sentiments = self.sentimentDictionary.get(word.lower(), None)
            if sentiments is not None:
                word_sentiment = list(map(lambda x: int(sentiments[x]), self.emotions))
                title_sentiment = list(map(add, title_sentiment, word_sentiment))
        title_sentiment = list(map(lambda x: x/len(title), title_sentiment))
        return self.get_sentiment_vector_diff(goal_sentiment, title_sentiment)

    def get_sentiment_vector_diff(self, goal, sentiment):
        """
        Take squared difference of vectors
        """
        diff = 0
        for i in range(len(self.emotions)):
            diff += (goal[i] - sentiment[i])**2
        #Normalize to range 0-1 and take complement, since small difference is good
        return 1 - math.sqrt(diff)/math.sqrt(6)
