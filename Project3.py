import openai
API_KEY = "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"
openai.api_key = API_KEY
def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.5,
    )
    return response.choices[0].text.strip()
prompt = input("What is NLP \n")
response = generate_text(prompt)
print(response)




























# import openai
# import os
# from dotenv import load_dotenv
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.environ["OPENAI_API_KEY"]

# # Send a text completion request to the API
# response = openai.Completion.create(
#   engine="davinci",
#   prompt="Hello, my name is ChatGPT. What can I help you with today?",
#   max_tokens=50
# )

# # Extract the completed text from the API response
# text = response.choices[0].text

# # Print the completed text
# print(text)
# # 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












