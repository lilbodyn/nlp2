'''
Модуль web-скрапінгу відгуків покупців з платформи rozetka.com.ua.

Реалізовано два режими отримання відгуків:
  1. ОНЛАЙН - спроба запиту до API Rozetka з розширеними заголовками (Але є нюанси, воно не працює в цій версії)
  2. ОФЛАЙН - парсинг локально збереженого HTML-файлу сторінки відгуків
             (File → Save As → Webpage, HTML Only у браузері)
'''

import re
import os
import time
import requests
from bs4 import BeautifulSoup

# ----------------------------- Загальні налаштування --------------------------
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/143.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'uk-UA,uk;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Origin': 'https://rozetka.com.ua',
    'Referer': 'https://rozetka.com.ua/',
    'Connection': 'keep-alive',
    'sec-ch-ua': '"Chromium";v="143"',
    'sec-ch-ua-platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
}
REQUEST_TIMEOUT = 15
MAX_PAGES       = 10
DELAY_SEC       = 1.5


# ============================== Визначення product_id =========================
def _extract_product_id(url):
    '''Витягує числовий product_id з URL сторінки товару.'''
    match = re.search(r'/p(\d+)/?', url)
    if match:
        pid = match.group(1)
        print(f'Product ID: {pid}')
        return pid
    print('Не вдалось визначити product_id з URL.')
    return None


# ============================== Парсинг локального HTML =======================
def _parse_html_reviews(html):
    '''
    Парсить відгуки з HTML-вмісту сторінки rozetka.com.ua.
    Витягує текст відгуку, переваги та недоліки.
    '''
    soup    = BeautifulSoup(html, 'lxml')
    reviews = []

    # --- знаходимо всі блоки відгуків ---
    comment_blocks = soup.find_all('div', class_='comment__body-wrapper')

    for block in comment_blocks:
        parts = []

        # текст відгуку (основний абзац)
        p_tag = block.find('p')
        if p_tag and p_tag.get_text(strip=True):
            parts.append(p_tag.get_text(strip=True))

        # переваги та недоліки з dl.comment__essentials
        essentials = block.find('dl', class_='comment__essentials')
        if essentials:
            for div in essentials.find_all('div'):
                dt = div.find('dt')
                dd = div.find('dd')
                if dt and dd:
                    label = dt.get_text(strip=True).replace(':', '')
                    value = dd.get_text(strip=True)
                    if value and value.lower() not in ('немає', 'нема', '-', ''):
                        parts.append(f'{label}: {value}')

        full_text = ' '.join(parts).strip()
        if full_text and len(full_text) >= 3:
            reviews.append(full_text)

    return reviews


# ============================== Режим 1: API Rozetka ==========================
def _fetch_via_api(product_id, max_pages):
    '''
    Спроба отримати відгуки через API Rozetka.
    Повертає список відгуків або порожній список при помилці 403.
    '''
    session = requests.Session()
    session.headers.update(HEADERS)

    # спочатку "відвідуємо" головну сторінку щоб отримати cookies
    try:
        session.get('https://rozetka.com.ua/', timeout=REQUEST_TIMEOUT)
        time.sleep(1)
    except Exception:
        pass

    all_reviews = []

    for page in range(1, max_pages + 1):
        api_url = (
            f'https://rozetka.com.ua/api/review/v1/list'
            f'?product_id={product_id}&page={page}&lang=uk'
        )
        try:
            r = session.get(api_url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 403:
                print(f'  Отримано 403 Forbidden - сайт блокує прямі запити.')
                return []
            r.raise_for_status()
            data    = r.json()
            items   = data.get('data', {}).get('items', [])
            total   = data.get('data', {}).get('total', 0)
            reviews = []

            for item in items:
                text = (item.get('comment') or item.get('text') or '').strip()
                pros = (item.get('advantages') or '').strip()
                cons = (item.get('disadvantages') or '').strip()
                full = ' '.join(filter(None, [text, pros, cons]))
                if full:
                    reviews.append(full)

            all_reviews.extend(reviews)
            print(f'  [Сторінка {page}] Отримано: {len(reviews)} відгуків')

            if not reviews or len(all_reviews) >= total:
                break
            time.sleep(DELAY_SEC)

        except Exception as e:
            print(f'  Помилка: {e}')
            return []

    return all_reviews


# ============================== Режим 2: Локальний HTML =======================
def load_reviews_from_html_file(html_filepath):
    '''
    Завантажує відгуки з локально збереженого HTML-файлу.
    Як зберегти HTML: відкрий сторінку відгуків у браузері →
    File → Save As → Webpage, HTML Only (.html)

    Вхід:  html_filepath - шлях до .html файлу
    Вихід: reviews       - список рядків
    '''
    if not os.path.exists(html_filepath):
        print(f'Файл не знайдено: {html_filepath}')
        return []

    print(f'Парсинг локального HTML: {html_filepath}')
    with open(html_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()

    reviews = _parse_html_reviews(html)
    print(f'Знайдено відгуків у файлі: {len(reviews)}')
    return reviews


# ============================== Головна функція ================================
def load_reviews_from_rozetka(url, max_pages=MAX_PAGES, html_file=None):
    '''
    Збирає відгуки з сторінки товару rozetka.com.ua.

    Порядок спроб:
      1. Якщо передано html_file - парсинг локального HTML
      2. Спроба через API Rozetka
      3. Якщо обидва способи не дали результату - повідомлення користувачу

    Вхід:  url       - посилання на сторінку товару
           max_pages - максимальна кількість сторінок
           html_file - (необов'язково) шлях до локального .html файлу
    Вихід: reviews   - список рядків (тексти відгуків)
    '''
    print(f'\nЗбір відгуків для: {url}')

    # --- локальний HTML файл ---
    if html_file:
        reviews = load_reviews_from_html_file(html_file)
        if reviews:
            return reviews

    # --- API ---
    product_id = _extract_product_id(url)
    if product_id:
        print(f'  Спроба API: product_id={product_id}...')
        reviews = _fetch_via_api(product_id, max_pages)
        if reviews:
            print(f'Зібрано через API: {len(reviews)} відгуків')
            return reviews

    # --- не вдалось отримати відгуки ---
    print('\nНе вдалось отримати відгуки автоматично.')
    return []