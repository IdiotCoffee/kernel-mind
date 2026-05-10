import os

from dotenv import load_dotenv

from generation.providers.sarvam_provider import SarvamProvider

load_dotenv()


provider = SarvamProvider(api_key=os.getenv("SARVAM_API_KEY", ""))

prompt = """
Question:
How does login work?

"""


for token in provider.generate(prompt, stream=True):
    print(token, end="", flush=True)
