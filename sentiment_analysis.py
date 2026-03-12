'''
Модуль сентиментального аналізу відгуків покупців (Рівень ІІ).
Використовує словниковий метод (lexicon-based approach):

  score_pos = кількість токенів відгуку що входять до словника P
  score_neg = кількість токенів відгуку що входять до словника N
  sentiment = score_pos - score_neg

Класифікація:
  sentiment > 0 → позитивний (+)
  sentiment < 0 → негативний  (-)
  sentiment = 0 → нейтральний (0)

Для NLP-обробки кожного відгуку використовується
той самий конвеєр що і в nlp_pipeline.py (Рівень І).
'''

import os
import re
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import RegexpTokenizer
from nltk.corpus   import stopwords

import spacy

try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False


# ============================== Словники тональності ==========================

# позитивні ключові слова
POSITIVE_WORDS = {
    'відмінний', 'відмінна', 'відмінне', 'чудовий', 'чудова', 'чудове',
    'якісний', 'якісна', 'якісне', 'якість', 'рекомендую', 'задоволений',
    'задоволена', 'задоволені', 'супер', 'топ', 'добрий', 'добра', 'добре',
    'гарний', 'гарна', 'гарне', 'класний', 'класна', 'зручний', 'зручна',
    'надійний', 'надійна', 'швидкий', 'швидка', 'потужний', 'потужна',
    'красивий', 'красива', 'приємний', 'приємна', 'хороший', 'хороша',
    'ідеальний', 'ідеальна', 'подобається', 'сподобалось', 'сподобалася',
    'задоволений', 'радий', 'рада', 'довго', 'працює', 'справно',
    'відповідає', 'опису', 'швидка', 'доставка', 'пакування', 'акуратне',
    'рекомендую', 'купив', 'задоволений', 'все', 'добре', 'відповідає',
    'очікуванням', 'якість', 'ціна', 'оптимальна', 'оптимальний',
}

# негативні ключові слова
NEGATIVE_WORDS = {
    'поганий', 'погана', 'погане', 'жахливий', 'жахлива', 'жахливе',
    'бракований', 'бракована', 'браковане', 'брак', 'зламався', 'зламалась',
    'зламалося', 'зламаний', 'не', 'розчарований', 'розчарована', 'повернув',
    'повернула', 'повернули', 'не рекомендую', 'відмовив', 'відмовила',
    'несправний', 'несправна', 'несправне', 'не працює', 'гудить', 'шумить',
    'скрипить', 'тріщина', 'подряпина', 'пошкоджений', 'пошкоджена',
    'дорого', 'дорогий', 'переплатив', 'обманули', 'розчарування',
    'незадоволений', 'незадоволена', 'проблема', 'проблеми', 'дефект',
    'неякісний', 'неякісна', 'повільний', 'повільна', 'гріється',
    'перегрівається', 'швидко', 'зламалось', 'не відповідає',
}


# ============================== Допоміжні функції ==============================

def _load_stopwords():
    '''Завантажує базові стоп-слова.'''
    stop = set()

    # --- завантаження україномовного словника стоп-слів з GitHub ---
    try:
        import requests as _requests
        url = ('https://raw.githubusercontent.com/'
               'olegdubetcky/Ukrainian-Stopwords/main/ukrainian')
        r = _requests.get(url, timeout=10)
        path = os.path.join(nltk.data.path[0], 'corpora', 'stopwords', 'ukrainian')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(r.content)
        stop |= set(stopwords.words('ukrainian'))
    except Exception:
        pass

    for lang in ('english',):
        try:
            stop |= set(stopwords.words(lang))
        except Exception:
            pass
    stop |= {
        # службові слова
        'це', 'та', 'що', 'як', 'але', 'або', 'якщо', 'тому', 'через',
        'після', 'перед', 'також', 'вже', 'ще', 'дуже', 'навіть',
        'тільки', 'адже', 'проте', 'однак', 'хоча', 'зараз',
        'він', 'вона', 'вони', 'воно', 'його', 'її', 'їх', 'бути',
        'цей', 'той', 'свій', 'який', 'такий', 'весь', 'все', 'всі',
        # HTML-мітки зі структури відгуків rozetka
        'перевага', 'переваги', 'недолік', 'недоліки',
        # назва магазину
        'розетка', 'rozetka',
        # займенники та прийменники
        'нас', 'вас', 'нам', 'вам', 'ним', 'ній', 'них', 'мені',
        'просто', 'тобто', 'саме', 'лише', 'поки', 'коли', 'бо',
    }
    return stop


def _preprocess_review(text, stop_words):
    '''
    NLP-конвеєр для одного відгуку:
    фільтрація → нормалізація → токенізація → стоп-слова.
    Повертає список токенів.
    '''
    # фільтрація та нормалізація
    text = text.replace('\n', ' ')
    text = re.sub(r'[0-9]+', '', text)
    text = text.lower()

    # regexp-токенізація (лише слова)
    tokenizer = RegexpTokenizer(r'[а-яёіїєґa-z]+')
    tokens    = tokenizer.tokenize(text)

    # видалення стоп-слів та коротких токенів
    tokens = [w for w in tokens
              if w not in stop_words and len(w) >= 3]
    return tokens


def _lemmatize_tokens(tokens, nlp):
    '''Лематизація списку токенів засобами spacy.'''
    if not tokens:
        return tokens
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]


def _classify(tokens, pos_words, neg_words):
    '''
    Підраховує score_pos та score_neg для одного відгуку.
    Повертає: score_pos, score_neg, label (+/-/0)
    '''
    score_pos = sum(1 for w in tokens if w in pos_words)
    score_neg = sum(1 for w in tokens if w in neg_words)

    if score_pos > score_neg:
        label = '+'
    elif score_neg > score_pos:
        label = '-'
    else:
        label = '0'

    return score_pos, score_neg, label


def _save_csv(records, filepath):
    '''Зберігає результати аналізу у CSV-файл.'''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(records, columns=[
        '№', 'Відгук (скорочено)', 'score_pos', 'score_neg', 'Тональність'
    ])
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f'Збережено: {filepath}')


def _plot_bar(stats, product_name, filepath):
    '''Будує стовпчасту діаграму розподілу тональності.'''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    labels = ['Позитивні (+)', 'Негативні (−)', 'Нейтральні (0)']
    values = [stats['+'], stats['-'], stats['0']]
    colors = ['#4caf50', '#f44336', '#9e9e9e']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)

    # підписи над стовпцями
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=12)

    total = sum(values)
    ax.set_title(f'Розподіл тональності відгуків - {product_name}\n'
                 f'(всього відгуків: {total})', fontsize=12)
    ax.set_ylabel('Кількість відгуків', fontsize=10)
    ax.set_ylim(0, max(values) * 1.2 + 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Збережено: {filepath}')


def _plot_wordcloud(tokens, title, filepath, colormap='Blues'):
    '''Будує хмару слів для набору токенів.'''
    if not WORDCLOUD_OK or not tokens:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    wc = WordCloud(
        width=900, height=420,
        background_color='white',
        max_words=50,
        colormap=colormap
    ).generate(' '.join(tokens))
    plt.figure(figsize=(11, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=13)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Збережено: {filepath}')


def _print_top10(label, top10):
    '''Виводить ТОП-10 ключових слів у консоль.'''
    print(f'\n  ТОП-10 слів ({label}):')
    for rank, (word, count) in enumerate(top10, start=1):
        print(f'    {rank:2}. {word:<25} - {count} разів')


# ============================== Головна функція ================================
def run_sentiment_analysis(reviews, product_name, results_dir='results'):
    '''
    Запускає повний сентиментальний аналіз для списку відгуків.

    Вхід:  reviews      - список рядків (тексти відгуків)
           product_name - назва товару (для підписів)
           results_dir  - папка збереження результатів
    '''
    print(f'\n------- Сентиментальний аналіз: {product_name} -------')

    os.makedirs(results_dir, exist_ok=True)

    # --- завантаження допоміжних інструментів ---
    stop_words = _load_stopwords()
    try:
        nlp = spacy.load('uk_core_news_sm')
        use_lemma = True
    except OSError:
        use_lemma = False
        nlp = None

    # --- аналіз кожного відгуку ---
    records    = []
    stats      = {'+': 0, '-': 0, '0': 0}
    pos_tokens = []   # усі токени з позитивних відгуків
    neg_tokens = []   # усі токени з негативних відгуків

    for idx, review in enumerate(reviews, start=1):

        # NLP-конвеєр
        tokens = _preprocess_review(review, stop_words)
        if use_lemma and tokens:
            tokens = _lemmatize_tokens(tokens, nlp)

        # класифікація
        score_pos, score_neg, label = _classify(
            tokens, POSITIVE_WORDS, NEGATIVE_WORDS
        )

        # накопичення токенів по тональності
        if label == '+':
            pos_tokens.extend(tokens)
        elif label == '-':
            neg_tokens.extend(tokens)

        stats[label] += 1

        # скорочений текст відгуку для таблиці
        short = review[:80].replace('\n', ' ') + ('...' if len(review) > 80 else '')
        records.append([idx, short, score_pos, score_neg, label])

    # --- виведення загальної статистики ---
    total = len(reviews)
    print(f'\n[Статистика] Всього відгуків:    {total}')
    print(f'  Позитивні (+): {stats["+"]:4}  ({stats["+"] / total * 100:.1f}%)')
    print(f'  Негативні (-): {stats["-"]:4}  ({stats["-"] / total * 100:.1f}%)')
    print(f'  Нейтральні (0):{stats["0"]:4}  ({stats["0"] / total * 100:.1f}%)')

    # --- ТОП-10 ключових слів ---
    top10_pos = Counter(pos_tokens).most_common(10)
    top10_neg = Counter(neg_tokens).most_common(10)
    _print_top10('позитивні відгуки', top10_pos)
    _print_top10('негативні відгуки', top10_neg)

    # --- збереження результатів ---
    safe_name = product_name.replace(' ', '_')[:30]

    _save_csv(records,
              f'{results_dir}/sentiment_results_{safe_name}.csv')

    _plot_bar(stats, product_name,
              f'{results_dir}/sentiment_chart_{safe_name}.png')

    _plot_wordcloud(pos_tokens,
                   f'Хмара слів - позитивні відгуки ({product_name})',
                   f'{results_dir}/wordcloud_pos_{safe_name}.png',
                   colormap='Greens')

    _plot_wordcloud(neg_tokens,
                   f'Хмара слів - негативні відгуки ({product_name})',
                   f'{results_dir}/wordcloud_neg_{safe_name}.png',
                   colormap='Reds')

    print(f'\n------- Sentiment аналіз завершено: {product_name} -------')