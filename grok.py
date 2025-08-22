from groq import Groq

client = Groq(api_key="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")

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why does the Earth orbit the Sun?"}
    ],
    model="llama3-70b-8192",  # or "mixtral-8x7b-32768"
)

print(chat_completion.choices[0].message.content)
