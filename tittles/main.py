import os
import pickle
from pattern.en import pluralize, singularize

try:
    from .templates import TemplateBank, Title
except ModuleNotFoundError:
    from templates import TemplateBank, Title

try:
    from .evaluator import Evaluator
except ModuleNotFoundError:
    from evaluator import Evaluator

try:
    from .wordpicker import WordPicker
except ModuleNotFoundError:
    from wordpicker import WordPicker
try:
    from . import thesaurus
except ImportError:
    import thesaurus

class tittlesTitle():
    def __init__(self):
        self.threshold = 0.8
        self.domain = 'word'
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.evaluator = Evaluator()
        self.wordpicker = WordPicker()
        self.template_bank = TemplateBank(self.evaluator.title_bank)


    def generate(self, *args, **kwargs):
        return self.create("", {}, number_of_artifacts=1)

    def find_words(self, adjectives, activity, location, weather, slots):
        return self.wordpicker.find_pairs(adjectives, activity, location, weather, slots)

    def evaluate(self, title):
        """
        Evaluates given title to [0,1] range. 1 being best possible value.

        Args:
            Title (str) : title to be evaluated.

        Returns:
            Float [0, 1] : How good the title was - high being better.
        """
        return self.evaluator.evaluate(title.split(" "), self.emotion)

    def inject(self, title, word_pair):
        for i, cat in title.get_slots('NP'):
            if cat == 'plural':
                title.inject(pluralize(word_pair[0]).capitalize(), 'NP')
            else:
                title.inject(singularize(word_pair[0]).capitalize(), 'NP')
        for i, cat in title.get_slots('ADJ'):
            title.inject(word_pair[1].capitalize(), 'ADJ')

    def create(self, emotion, word_pairs, number_of_artifacts=10, **kwargs):
        """Create artifacts in the group's domain.

        The given inputs can be parsed and deciphered by the system using any methods available.

        The function should return a list in the form of:

            [
                (artifact1, {"evaluation": 0.76, 'foo': 'bar'}),
                (artifact2, {"evaluation": 0.89, 'foo': 'baz'}),
                # ...
                (artifactn, {"evaluation": 0.29, 'foo': 'bax'})
            ]

        :param str emotion:
            One of "the six basic emotions": anger, disgust, fear, happiness, sadness or surprise.
            The emotion should be perceivable in the output(s).
        :param list word_pairs:
            List of 2-tuples, the word pairs associated with the output(s). The word_pairs are (noun, property) pairings
            where each pair presents a noun and its property which may be visible in the output. (Think of more creative
            ways to present the pairings than literal meaning.)
        :param int number_of_artifacts:
            Number of artifacts returned
        :returns:
            List with *number_of_artifacts* elements. Each element should be (artifact, metadata) pair, where metadata
            should be a dictionary holding at least 'evaluation' keyword with float value.

        """

        self.emotion = emotion

        ret = []

        while len(ret) != number_of_artifacts:
            adjectives = (list(word_pair[1] for word_pair in word_pairs if word_pair[0] == 'animal'), list(word_pair[1] for word_pair in word_pairs if word_pair[0] == 'human'))
            weather = dict(word_pairs)['weather']
            activity = dict(word_pairs)['activity']
            location = dict(word_pairs)['location']

            template = self.template_bank.random_template()
            title = Title(template)
            word_pair = self.find_words(adjectives, activity, location, weather, title.slots)
            
            self.inject(title, word_pair)
            v = self.evaluate(str(title))
            if v >= self.threshold:
                ret.append((str(title), {"evaluation": v}))
                self.evaluator.add_title(str(title))

        # Comment out if you want to keep the original titles
        self.evaluator.dump_titles()
        return ret

if __name__ == "__main__":
    import pprint
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import inputs
    pp = pprint.PrettyPrinter(indent=2)
    emotion, word_pairs = inputs.get_input(False)
    T = tittlesTitle()
    print('INPUT')
    pp.pprint({'emotion': emotion, 'word_pairs': word_pairs})
    print('')
    print('OUTPUT')
    pp.pprint(T.create(emotion, word_pairs, number_of_artifacts=3))
