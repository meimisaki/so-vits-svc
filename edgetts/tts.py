import asyncio
import random
import sys

import edge_tts
from edge_tts import VoicesManager
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

async def _main() -> None:
    TEXT = sys.argv[1]
    LANG = sys.argv[2]
    RATE = sys.argv[3]
    VOLUME = sys.argv[4]
    GENDER = sys.argv[5] if 5 < len(sys.argv) else None
    OUTPUT_FILE = sys.argv[6] if 6 < len(sys.argv) else "tts.wav"

    print("Running TTS...")
    print(f"Text: {TEXT}, Language: {LANG}, Gender: {GENDER}, Rate: {RATE}, Volume: {VOLUME}")

    voices = await VoicesManager.create()
    if LANG == "Auto":
        LANG = detect(TEXT)
        # From "zh-cn" to "zh-CN" etc.
        if LANG == "zh-cn" or LANG == "zh-tw":
            LOCALE = LANG[:-2] + LANG[-2:].upper()
            voice = voices.find(Gender=GENDER, Locale=LOCALE)
        else:
            voice = voices.find(Gender=GENDER, Language=LANG)
        VOICE = random.choice(voice)["Name"]
        print(f"Using random {LANG} voice: {VOICE}")
    else:
        VOICE = LANG
        
    communicate = edge_tts.Communicate(text = TEXT, voice = VOICE, rate = RATE, volume = VOLUME)
    await communicate.save(OUTPUT_FILE)

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_main())
    else:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        try:
            loop.run_until_complete(_main())
        finally:
            loop.close()
