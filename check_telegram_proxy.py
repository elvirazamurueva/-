"""
check_telegram_proxy.py — безопасная диагностика подключения к Telegram через прокси.

Запуск:  python check_telegram_proxy.py

Скрипт читает .env сам и НИКОГДА не выводит токены, логины и пароли.
"""

import sys

import requests

from config import load_config


def main() -> None:
    config = load_config()

    if not config.telegram_bot_token:
        print("[ОШИБКА] TELEGRAM_BOT_TOKEN не задан")
        sys.exit(1)

    proxy = config.get_telegram_proxy_url()
    if not proxy:
        print("[ОШИБКА] PROXY_URL не задан. В РФ прямое подключение к")
        print("api.telegram.org недоступно — настройте прокси в .env")
        sys.exit(1)

    # Маскируем учётные данные
    masked = "***:***@" + proxy.rsplit("@", 1)[1] if "@" in proxy else proxy
    print(f"Проверяю прокси: {masked}")

    api = f"https://api.telegram.org/bot{config.telegram_bot_token}"
    proxies = {"http": proxy, "https": proxy}

    # 1. getMe — проверка токена и связности прокси
    try:
        r = requests.get(f"{api}/getMe", proxies=proxies, timeout=20)
        if r.status_code == 200:
            ok = r.json().get("result", {}).get("username", "?")
            print(f"[1/2] getMe: OK — бот @{ok}")
        else:
            print(f"[1/2] getMe: HTTP {r.status_code} — {r.text[:120]}")
            sys.exit(1)
    except requests.exceptions.ProxyError:
        print("[1/2] getMe: прокси отклонил подключение (проверьте PROXY_URL)")
        sys.exit(1)
    except Exception as exc:
        print(f"[1/2] getMe: {type(exc).__name__}")
        sys.exit(1)

    # 2. getUpdates — проверка polling
    try:
        r = requests.get(f"{api}/getUpdates", proxies=proxies, timeout=20)
        if r.status_code == 200:
            n = len(r.json().get("result", []))
            print(f"[2/2] getUpdates: OK — {n} необработанных обновлений")
        else:
            print(f"[2/2] getUpdates: HTTP {r.status_code}")
            sys.exit(1)
    except Exception as exc:
        print(f"[2/2] getUpdates: {type(exc).__name__}")
        sys.exit(1)

    print("\nПрокси для Telegram работает. Polling должен получать сообщения.")


if __name__ == "__main__":
    main()
