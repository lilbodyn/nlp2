'''
Модуль реалізації повного NLP-конвеєра (Рівень І).

Конвеєр обробки:
  1. Фільтрація шуму та нормалізація
  2. Токенізація (3 типи): word / sent / regexp
  3. Видалення стоп-слів (nltk ukrainian + власний словник)
  4. Лематизація (spacy uk_core_news_sm)
  5. Стемінг (SnowballStemmer + PorterStemmer)
  6. Частотний аналіз TF, ТОП-10
  7. Збереження кожного етапу у окремий файл results
'''

import re
import os
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

import nltk
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.tokenize   import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus     import stopwords
from nltk.stem       import SnowballStemmer, PorterStemmer

import spacy

try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False


# ============================== Стоп-слова =====================================

def _load_ukrainian_stopwords():
    '''
    Завантажує україномовний словник стоп-слів з репозиторію GitHub
    та підключає його до nltk. Повертає множину стоп-слів.
    '''
    # --- завантажуємо всі потрібні корпуси nltk ---
    for corpus in ('stopwords',):
        try:
            nltk.download(corpus, quiet=True)
        except Exception:
            pass

    # --- україномовний словник з GitHub ---
    try:
        url = ('https://raw.githubusercontent.com/'
               'olegdubetcky/Ukrainian-Stopwords/main/ukrainian')
        r = requests.get(url, timeout=10)
        path = os.path.join(nltk.data.path[0], 'corpora', 'stopwords', 'ukrainian')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(r.content)
        stop_ua = set(stopwords.words('ukrainian'))
        print(f'Завантажено українських стоп-слів: {len(stop_ua)}')
    except Exception as e:
        print(f'Не вдалось завантажити укр. стоп-слова: {e}')
        stop_ua = set()

    # --- англійські стоп-слова ---
    try:
        stop_en = set(stopwords.words('english'))
    except Exception:
        stop_en = set()

    # --- власний розширений словник ---
    stop_extra = {
        'це', 'та', 'що', 'як', 'але', 'або', 'якщо', 'тому', 'через',
        'після', 'перед', 'також', 'вже', 'ще', 'дуже', 'навіть',
        'тільки', 'адже', 'проте', 'однак', 'хоча', 'зараз', 'тут',
        'там', 'нас', 'вас', 'нам', 'вам', 'ним', 'ній', 'них',
    }

    all_stops = stop_ua | stop_extra | stop_en
    print(f'Загальна кількість стоп-слів: {len(all_stops)}')
    return all_stops


# ============================== Крок 1: Фільтрація та нормалізація =============
def step_filter_normalize(text):
    '''
    Фільтрація шуму та нормалізація вхідного тексту.
    Вхід:  text         - сирий текст
    Вихід: кортеж (clean_text, original_for_sent)
           clean_text        - нормалізований текст для word/regexp токенізації
           original_for_sent - текст для sent_tokenize (зі збереженою пунктуацією)
    '''
    # зберігаємо копію для sent_tokenize до видалення пунктуації
    original_for_sent = text.replace('\n', ' ').replace('\r', ' ')

    # заміна переносів рядка на пробіл
    text = text.replace('\n', ' ').replace('\r', ' ')
    # заміна апострофа на порожній рядок
    text = text.replace("'", '').replace('ʼ', '').replace('\u2019', '')
    # видалення знаків пунктуації
    text = text.replace(',', '').replace('.', '').replace('?', '')
    text = text.replace('!', '').replace('-', '').replace(':', '')
    text = text.replace(';', '').replace('(', '').replace(')', '')
    text = text.replace('«', '').replace('»', '').replace('–', '')
    text = text.replace('"', '').replace('…', '')
    # видалення цифр
    text = re.sub(r'[0-9]+', '', text)
    # нижній регістр
    clean_text = text.lower()
    print(f'[Крок 1] Нормалізовано. Символів: {len(clean_text)}')
    return clean_text, original_for_sent


# ============================== Крок 2: Токенізація ===========================
def step_tokenize(clean_text, original_for_sent):
    '''
    Токенізація трьома методами nltk.
    Вхід:  clean_text        - нормалізований текст (для word/regexp)
           original_for_sent - текст зі збереженою пунктуацією (для sent)
    Вихід: кортеж (words, sentences, regexp_words)
    '''
    # тип 1: пословна токенізація
    words = word_tokenize(clean_text)
    print(f'[Крок 2] Пословна токенізація:    {len(words)} токенів')

    # тип 2: поречення токенізація - використовуємо текст до видалення пунктуації
    sentences = sent_tokenize(original_for_sent)
    print(f'[Крок 2] Речення-токенізація:    {len(sentences)} речень')

    # тип 3: regexp токенізація (тільки слова з кириличних та латинських літер)
    tokenizer    = RegexpTokenizer(r'[а-яёіїєґa-z]+')
    regexp_words = tokenizer.tokenize(clean_text)
    print(f'[Крок 2] Regexp-токенізація:  {len(regexp_words)} токенів')

    return words, sentences, regexp_words


# ============================== Крок 3: Стоп-слова ============================
def step_remove_stopwords(tokens, stop_words):
    '''
    Видалення стоп-слів та коротких токенів (менше 3 символів).
    Вхід:  tokens     - список токенів
           stop_words - множина стоп-слів
    Вихід: filtered   - очищений список токенів
    '''
    filtered = [w for w in tokens
                if w.casefold() not in stop_words and len(w) >= 3]
    print(f'[Крок 3] Після видалення стоп-слів: {len(filtered)} токенів '
          f'(було {len(tokens)})')
    return filtered


# ============================== Крок 4: Лематизація ===========================
def step_lemmatize(tokens):
    '''
    Лематизація токенів засобами spacy (uk_core_news_sm).
    Для кожного токена: якщо token.text != token.lemma_ - фіксуємо зміну.
    Вхід:  tokens       - список рядків
    Вихід: lemma_list   - список лем
           changes      - список пар (оригінал, лема) де є зміна
    '''
    try:
        nlp = spacy.load('uk_core_news_sm')
    except OSError:
        return tokens, []

    # spacy обробляє текст - об'єднуємо токени
    text_joined = ' '.join(tokens)
    doc = nlp(text_joined)

    lemma_list = []
    changes    = []
    for token in doc:
        lemma = token.lemma_
        lemma_list.append(lemma)
        if token.text != lemma:
            changes.append((token.text, lemma))

    print(f'[Крок 4] Лематизовано: {len(lemma_list)} лем, '
          f'змінено форм: {len(changes)}')
    return lemma_list, changes


# ============================== Крок 5: Стемінг ================================
def step_stem(tokens):
    '''
    Стемінг двома алгоритмами:
      - SnowballStemmer
      - PorterStemmer
    Вхід:  tokens         - список рядків
    Вихід: snowball_stems - список стемів (Snowball)
           porter_stems   - список стемів (Porter)
    '''
    snowball = SnowballStemmer(language='russian')
    porter   = PorterStemmer()

    snowball_stems = [snowball.stem(w) for w in tokens]
    porter_stems   = [porter.stem(w)   for w in tokens]

    print(f'[Крок 5] Стемінг виконано: {len(snowball_stems)} стемів (Snowball), '
          f'{len(porter_stems)} стемів (Porter)')
    return snowball_stems, porter_stems


# ============================== Крок 6: ТОП-10 ================================
def step_top10(tokens):
    '''
    Частотний аналіз TF - визначення ТОП-10 найчастотніших токенів.
    Вхід:  tokens  - список токенів
    Вихід: top10   - список [(слово, частота)]
    '''
    freq  = Counter(tokens)
    top10 = freq.most_common(10)
    print(f'\n[Крок 6] ТОП-10 слів:')
    for rank, (word, count) in enumerate(top10, start=1):
        print(f'  {rank:2}. {word:<25} - {count} разів')
    return top10


# ============================== Збереження у файли ============================
def _save(filepath, content):
    '''Зберігає рядок або список у текстовий файл UTF-8.'''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(content, list):
            f.write('\n'.join(str(x) for x in content))
        else:
            f.write(str(content))
    print(f'Збережено: {filepath}')


def _plot_top10(top10, filename):
    '''Будує горизонтальну стовпчасту діаграму ТОП-10.'''
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    words  = [w for w, _ in top10][::-1]
    counts = [c for _, c in top10][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(words, counts, color='steelblue')
    ax.set_xlabel('Частота (кількість входжень)', fontsize=10)
    ax.set_title('ТОП-10 найчастотніших слів тексту', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Збережено: {filename}')


def _plot_wordcloud(tokens, filename):
    '''Будує хмару слів.'''
    if not WORDCLOUD_OK or not tokens:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    wc = WordCloud(
        width=900, height=450,
        background_color='white',
        max_words=60,
        colormap='Blues'
    ).generate(' '.join(tokens))
    plt.figure(figsize=(11, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Хмара слів - ТОП термінів тексту', fontsize=13)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Збережено: {filename}')


# ============================== Головна функція конвеєра ======================
def run_nlp_pipeline(raw_text, results_dir='results'):
    '''
    Запускає повний NLP-конвеєр для вхідного тексту.
    Зберігає результати кожного кроку у окремі файли.
    '''
    print('\n------- NLP-конвеєр розпочато -------')

    # --- Крок 1: фільтрація та нормалізація ---
    clean_text, original_for_sent = step_filter_normalize(raw_text)
    _save(f'{results_dir}/output_filtered.txt', clean_text)

    # --- Крок 2: токенізація ---
    words, sentences, regexp_words = step_tokenize(clean_text, original_for_sent)
    content_tokens = (
        f'=== Пословна токенізація ({len(words)} токенів) ===\n' +
        ' '.join(words) +
        f'\n\n=== Речення-токенізація ({len(sentences)} речень) ===\n' +
        '\n'.join(sentences) +
        f'\n\n=== Regexp-токенізація ({len(regexp_words)} токенів) ===\n' +
        ' '.join(regexp_words)
    )
    _save(f'{results_dir}/output_tokens.txt', content_tokens)

    # далі використовуємо regexp_words як основний потік
    base_tokens = regexp_words

    # --- Крок 3: видалення стоп-слів ---
    stop_words = _load_ukrainian_stopwords()
    filtered   = step_remove_stopwords(base_tokens, stop_words)
    _save(f'{results_dir}/output_stopwords.txt', filtered)

    # --- Крок 4: лематизація ---
    lemmas, changes = step_lemmatize(filtered)
    content_lemma = (
        '=== Леми ===\n' + ' '.join(lemmas) +
        '\n\n=== Змінені форми (оригінал → лема) ===\n' +
        '\n'.join(f'{orig:<25} → {lemma}' for orig, lemma in changes)
    )
    _save(f'{results_dir}/output_lemma.txt', content_lemma)

    # --- Крок 5: стемінг ---
    snowball_stems, porter_stems = step_stem(filtered)
    content_stem = (
        '=== Snowball Stemmer ===\n' + ' '.join(snowball_stems) +
        '\n\n=== Porter Stemmer ===\n' + ' '.join(porter_stems) +
        '\n\n=== Порівняння (слово | Snowball | Porter) ===\n' +
        '\n'.join(
            f'{filtered[i]:<25} | {snowball_stems[i]:<20} | {porter_stems[i]}'
            for i in range(min(50, len(filtered)))
        )
    )
    _save(f'{results_dir}/output_stem.txt', content_stem)

    # --- Крок 6: ТОП-10 ---
    top10 = step_top10(filtered)
    content_top10 = (
        '=== ТОП-10 найчастотніших слів ===\n' +
        '\n'.join(f'{rank}. {word:<25} - {count} разів'
                  for rank, (word, count) in enumerate(top10, start=1))
    )
    _save(f'{results_dir}/output_top10.txt', content_top10)

    # --- Графіки ---
    _plot_top10(top10,  f'{results_dir}/top10_chart.png')
    _plot_wordcloud(filtered, f'{results_dir}/wordcloud.png')

    print('\n------- NLP-конвеєр завершено -------')
    print(f'Збережені файли:')
    print(f'  {results_dir}/output_filtered.txt')
    print(f'  {results_dir}/output_tokens.txt')
    print(f'  {results_dir}/output_stopwords.txt')
    print(f'  {results_dir}/output_lemma.txt')
    print(f'  {results_dir}/output_stem.txt')
    print(f'  {results_dir}/output_top10.txt')
    print(f'  {results_dir}/top10_chart.png')
    print(f'  {results_dir}/wordcloud.png')