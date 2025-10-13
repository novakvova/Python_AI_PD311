import asyncio
from spotdl import Spotdl

CLIENT_ID = "cc03cdeac64c42d2914d26a50730e84f"
CLIENT_SECRET = "24b080e4106b4bc680567d5d952d65b7"

async def main():
    # 🔹 Ініціалізація Spotdl з твоїми ключами Spotify API
    spotdl = Spotdl(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    # 🔹 Посилання на трек, альбом або плейлист
    url = "https://open.spotify.com/track/2CKbBBLXXWqTopO1vvS8y1"

    # 🔹 Завантаження
    await spotdl.download([url])

    print("✅ Завантаження завершено!")

# 🔹 Запуск асинхронної події
asyncio.run(main())
