import spacy

nlp = spacy.load("pt_core_news_sm")

def definite_article(word):
    doc = nlp(word)
    for token in doc:
        if token.pos_ == "NOUN":
            return f"{'a' if token.morph.get('Gender') == ['Fem'] else 'o'}"
    return word

def generateSpeak(direction, object_name):
  direction_x = direction[0]
  direction_y = direction[1]
  article = definite_article(object_name)

  direction_x_article = definite_article(direction_x)

  return f"Tem um{'a' if article == "a" else ''} {object_name} {'na' if direction_x_article == "o" else 'no'} {direction_x} em {direction_y}"