def definite_article(word):
    # Definindo um conjunto básico de sufixos para determinar o gênero
    feminine_endings = ["a", "ção", "são", "dade", "tude", "gem"]
    masculine_endings = ["o", "or", "ma", "mento", "l", "r", "z"]

    # Identificar se a palavra parece feminina ou masculina
    word_lower = word.lower()
    if any(word_lower.endswith(suffix) for suffix in feminine_endings):
        return "a"
    elif any(word_lower.endswith(suffix) for suffix in masculine_endings):
        return "o"
    
    # Se não identificar, retornar "o" como padrão
    return "o"

def generateSpeak(direction, object_name):
    direction_x = direction[0]
    direction_y = direction[1]
    article = definite_article(object_name)
    direction_x_article = definite_article(direction_x)

    return f"Tem um{'a' if article == 'a' else ''} {object_name} {'na' if direction_x_article == 'o' else 'no'} {direction_x} em {direction_y}"
