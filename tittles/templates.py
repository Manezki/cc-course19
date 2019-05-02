import random
import spacy
from markov import MarkovChain

nlp = spacy.load("en_core_web_sm")


class Title:
    def __init__(self, tokens):
        self.tokens = tokens

    def list_slots(self):
        for i, tok in enumerate(self.tokens):
            if tok == "[[ADJ]]":
                yield (i, "ADJ")
            elif tok == "[[NOUN]]":
                yield (i, "NOUN")
            elif tok == "[[NOUNS]]":
                yield (i, "NOUNS")
            elif tok == "[[PERSON]]":
                yield (i, "PERSON")
            elif tok == "[[LOC]]":
                yield (i, "LOC")

    def inject(self, token, tag, pos):
        """Inject the given token into the title."""
        assert pos >= -1
        assert pos < len(self.tokens)
        assert tag in ["ADJ", "NOUN", "NOUNS", "PERSON", "LOC"]
        assert self.tokens[pos] == "[[" + tag + "]]"
        self.tokens[pos] = token

    def __str__(self):
        return "".join(self.tokens)


class TemplateBank:
    def __init__(self, title_bank):
        self.markov = MarkovChain(3)
        for item in title_bank.values():
            self.markov.add(item['title'].replace('—', '-'))

    def _random_template(self):
        title = self.markov.generate()

        replacements = {}
        tokens = []
        doc = nlp(title)

        i = 0
        for token in doc:
            # Consider named entities as single token.
            if token.ent_type_ in ('PERSON', 'FAC', 'GPE', 'LOC'):
                if token.ent_iob == 1:
                    tokens[-2] += tokens[-1] + token.text
                    tokens[-1] = token.whitespace_
                else:
                    tokens.append(token.text)
                    tokens.append(token.whitespace_)
                    replacements[i] = '[[PERSON]]' if token.ent_type_ == 'PERSON' else '[[LOC]]'
                    i += 2
                continue

            tokens.append(token.text)
            tokens.append(token.whitespace_)
            if token.tag_ in ("NN", "NNP"):
                replacements[i] = "[[NOUN]]"
            elif token.tag_ in ("NNS", "NNPS"):
                replacements[i] = "[[NOUNS]]"
            elif token.pos_ == "ADJ":
                replacements[i] = "[[ADJ]]"
            i += 2

        if len(replacements) < 2:
            return None

        # Create a template by replacing two random tokens with POS tags
        for i, replacement in random.sample(replacements.items(), 2):
            tokens[i] = replacement

        return tokens

    def random_template(self):
        """Get random template from the bank."""
        for i in range(0, 25):
            template = self._random_template()
            if template is not None:
                return template
        raise RecursionError("Title generation was unable to find fitting template.")
