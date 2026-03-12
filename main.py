# ======================== Лабораторна робота №2 ===========================
'''
Дисципліна: Аналіз та обробка природної мови (NLP)
Тема: Токенізація та нормалізація в NLP
Рівень складності: ІІ
Виконав: Омельченко Богдан

Рівень І - NLP-конвеєр:
  Джерело вхідного тексту: https://www.ukrlib.com.ua
  Обробка: фільтрація, нормалізація, токенізація (3 типи),
           видалення стоп-слів, лематизація, стемінг, ТОП-10

Рівень ІІ - Sentiment аналіз:
  Джерело відгуків: https://www.rozetka.com.ua
  Обробка: NLP-конвеєр + оцінка тональності відгуків
           за позитивними/негативними ключовими словами

Використані бібліотеки:
Package             Version
------------------- -------
requests            2.32.5
beautifulsoup4      4.14.3
lxml                6.0.2
nltk                3.9.3
spacy               3.8.11
wordcloud           1.9.6
matplotlib          3.10.8
pandas              3.0.1
'''

import os
from text_loader      import load_text_from_ukrlib
from nlp_pipeline     import run_nlp_pipeline
from sentiment_parser import load_reviews_from_rozetka
from sentiment_analysis import run_sentiment_analysis

# ----------------------------- Налаштування ------------------------------------
RESULTS_DIR = 'results'


def init_results_dir():
    '''Створює папку results якщо вона ще не існує.'''
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f'Папка результатів: {RESULTS_DIR}')


# ============================== РЕЖИМ 1 - NLP-конвеєр ==========================
def run_level_1():
    '''
    Рівень І: повний NLP-конвеєр для тексту з ukrlib.com.ua.
    Кроки: завантаження → фільтрація → нормалізація →
           токенізація (3 типи) → стоп-слова → лематизація →
           стемінг → ТОП-10 → збереження у файли.
    '''
    print('\n======= Рівень І: NLP-конвеєр =======\n')

    # --- Завантаження тексту ---
    print('Введіть URL твору з ukrlib.com.ua')
    print('Приклад: https://www.ukrlib.com.ua/books/printit.php?tid=907')
    url = input('URL: ').strip()

    if not url:
        # URL за замовчуванням - "Кайдашева сім'я" Івана Нечуя-Левицького
        url = 'https://www.ukrlib.com.ua/books/printit.php?tid=907'
        print(f'За замовчуванням: {url}')

    raw_text = load_text_from_ukrlib(url)

    if not raw_text.strip():
        print('Не вдалося отримати текст. Перевірте URL.')
        return

    run_nlp_pipeline(raw_text, results_dir=RESULTS_DIR)

    print('\n=== Рівень І завершено. Результати збережено у папці results ===')


# ============================== РЕЖИМ 2 - Sentiment ============================
def run_level_2():
    '''
    Рівень ІІ: сентиментальний аналіз відгуків з rozetka.com.ua.
    Кроки: скрапінг відгуків → NLP-конвеєр → оцінка тональності →
           статистика → ТОП-10 ключових слів → графіки → CSV.
    '''
    print('\n======= Рівень ІІ: Sentiment аналіз (rozetka.com.ua) =======\n')

    print('Введіть URL сторінки товару на rozetka.com.ua')
    print('Приклад: https://rozetka.com.ua/ua/apple-iphone-17-pro-max-256gb-cosmic-orange-mfyn4af-a/p543550585/')
    url = input('URL товару: ').strip()

    if not url:
        print('URL не введено.')
        return

    product_name = input('Введіть назву товару: ').strip()
    if not product_name:
        product_name = 'Товар'

    # --- опція локального HTML файлу ---
    print('Вкажіть шлях до файлу або натисніть Enter щоб спробувати автоматично:')
    html_file = input('Шлях до HTML файлу (або Enter): ').strip()
    if not html_file:
        html_file = None
    elif not os.path.exists(html_file):
        print(f'Файл не знайдено: {html_file}. Спробую автоматично.')
        html_file = None

    # --- збір відгуків ---
    reviews = load_reviews_from_rozetka(url, html_file=html_file)

    if not reviews:
        print('Відгуки не знайдено.')
        return

    print(f'Зібрано відгуків: {len(reviews)}')

    # --- Sentiment аналіз ---
    run_sentiment_analysis(
        reviews      = reviews,
        product_name = product_name,
        results_dir  = RESULTS_DIR
    )

    print('\n=== Рівень ІІ завершено. Результати збережено у папці results ===')


# ============================== ГОЛОВНИЙ ВИКЛИК ================================
if __name__ == '__main__':

    print('=' * 60)
    print('  Лабораторна робота №2. Токенізація та нормалізація в NLP')
    print('  Дисципліна: Аналіз та обробка природної мови (NLP)')
    print('=' * 60)

    init_results_dir()

    print('\nОберіть режим виконання:')
    print('  1 - Рівень І:  NLP-конвеєр (текст з ukrlib.com.ua)')
    print('  2 - Рівень ІІ: Sentiment аналіз (відгуки rozetka.com.ua)')
    print('  3 - Обидва рівні послідовно')

    mode = input('\nРежим: ').strip()

    if mode == '1':
        run_level_1()
    elif mode == '2':
        run_level_2()
    elif mode == '3':
        run_level_1()
        run_level_2()
    else:
        print('Введіть 1, 2 або 3.')

    print('\nВсі результати збережено у папці results')