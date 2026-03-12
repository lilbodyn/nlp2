'''
Модуль завантаження вхідного тексту з сайту ukrlib.com.ua.

Підтримувані формати URL:
  https://www.ukrlib.com.ua/books/printit.php?tid=<id>  - сторінка друку
  https://www.ukrlib.com.ua/books/read.php?tid=<id>     - сторінка читання
'''

import requests
from bs4 import BeautifulSoup

# ----------------------------- Загальні налаштування ---------------------------
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/143.0.0.0 Safari/537.36'
}
REQUEST_TIMEOUT = 15


# ============================== Допоміжна функція GET ==========================
def _get_html(url):
    '''Отримує HTML-вміст сторінки. Повертає рядок або порожній рядок при помилці.'''
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        # ukrlib.com.ua використовує кодування windows-1251
        # декодуємо вміст байтів вручну щоб уникнути помилок автовизначення
        try:
            text = r.content.decode('windows-1251')
        except Exception:
            try:
                text = r.content.decode('utf-8')
            except Exception:
                text = r.text
        return text
    except Exception as e:
        print(f'  Помилка запиту: {url} → {e}')
        return ''


# ============================== Парсер ukrlib.com.ua ===========================
def _parse_ukrlib(html):
    '''
    Витягує лише текст твору з HTML-сторінки ukrlib.com.ua,
    без навігаційних елементів сайту.

    Для сторінки printit.php структура така:
      - Весь корисний текст знаходиться між тегами <br> після заголовку
      - Навігація йде до і після тексту твору
      - Найнадійніший спосіб: взяти весь текст body і відрізати
        все до першого абзацу тексту та після "завантажити цей текст"
    '''
    soup = BeautifulSoup(html, 'lxml')

    # --- видаляємо зайві теги: скрипти, стилі, nav, footer ---
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'head']):
        tag.decompose()

    # --- div з class що містить 'text' ---
    container = soup.find('div', class_='text')

    # --- div#text ---
    if not container:
        container = soup.find('div', id='text')

    # --- div.printable ---
    if not container:
        container = soup.find('div', class_='printable')

    if container:
        text = container.get_text(separator=' ', strip=True)
    else:
        # --- весь body ---
        body = soup.find('body')
        text = body.get_text(separator=' ', strip=True) if body else ''

    # --- відрізаємо навігацію сайту ---
    # Текст твору на ukrlib завжди закінчується перед "Завантажити цей текст"
    cutoff_phrases = [
        'завантажити цей текст',
        'надіслати розповісти',
        'інші твори цього автора',
        'дивіться також',
        'головна контакти реклама',
    ]
    text_lower = text.lower()
    for phrase in cutoff_phrases:
        idx = text_lower.find(phrase)
        if idx != -1:
            text = text[:idx]
            text_lower = text_lower
            break

    # --- відрізаємо навігацію на початку ---
    # Шукаємо перше речення що починається з великої літери і є частиною тексту
    start_phrases = [
        'недалеко', 'був', 'жила', 'жив', 'одного', 'давно', 'колись',
        'стояла', 'стояв', 'у селі', 'в селі', 'одного разу',
    ]
    text_lower2 = text.lower()
    best_start = -1
    for phrase in start_phrases:
        idx = text_lower2.find(phrase)
        if idx > 100 and (best_start == -1 or idx < best_start):
            best_start = idx

    if best_start > 200:
        text = text[best_start:]

    return text.strip()


# ============================== Головна функція ================================
def load_text_from_ukrlib(url):
    '''
    Завантажує та повертає текст літературного твору з ukrlib.com.ua.

    Вхід:  url     - посилання на твір
    Вихід: text    - рядок з повним текстом твору
    '''
    print(f'Завантаження тексту з: {url}')
    html = _get_html(url)

    if not html:
        print('HTML не отримано.')
        return ''

    text = _parse_ukrlib(html)

    if not text.strip():
        print('Текст не знайдено на сторінці.')
        return ''

    print(f'Отримано символів: {len(text)}')
    print(f'Перші 200 символів: {text[:200]}')
    return text