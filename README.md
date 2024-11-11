# Vision IA Copilot

## Configuração do Ambiente Virtual (venv)

Para configurar o ambiente virtual (venv) no Python, execute o seguinte comando:

```bash
python3 -m venv env
```

> **Nota:** Sempre que o computador é reiniciado ou o ambiente virtual é trocado, o venv precisa ser reativado.

### Ativando o venv

- **No Windows:**

  ```bash
  .\env\Scripts\activate
  ```

- **No Linux:**

  ```bash
  source env/bin/activate
  ```

## Instalando as Dependências

Para instalar as dependências do projeto, execute o comando:

```bash
pip install ultralytics opencv-python pyttsx3 loguru spacy
```

> **Nota:** Certifique-se de que o `pip` está instalado.

### Baixando o modelo de idioma do spaCy

Para habilitar o suporte ao idioma português no spaCy, execute o comando abaixo:

```bash
python -m spacy download pt_core_news_sm
```
