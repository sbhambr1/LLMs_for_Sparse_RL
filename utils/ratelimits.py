import os
from openai import OpenAI
from ratelimit import limits, sleep_and_retry

api_key = os.environ["OPENAI_API_KEY"]

if api_key is None:
    raise Exception("Please insert your OpenAI API key in conversation.py")

client = OpenAI(
  api_key=api_key,
)

# Define rate limits (e.g., 1000 requests per hour)
RATE_LIMIT = 1000
RATE_PERIOD = 3600  # 3600 seconds = 1 hour

# Define monetary usage limits (e.g., $100)
MONETARY_LIMIT = 100.0
COST_PER_REQUEST = 0.1  # Cost per API request

# Track total cost
total_cost = 0.0

# Decorator to enforce rate limits
@sleep_and_retry
@limits(calls=RATE_LIMIT, period=RATE_PERIOD)
def limited_request(*args, **kwargs):
    global total_cost
    cost = COST_PER_REQUEST  # Assume fixed cost per request for simplicity
    if total_cost + cost > MONETARY_LIMIT:
        raise ValueError("Monetary limit exceeded")
    total_cost += cost
    return client.chat.completions.create(*args, **kwargs)  # Example API call

# Example usage
try:
    response = limited_request(
        engine="davinci",
        prompt="Once upon a time",
        max_tokens=50
    )
    print(response.choices[0].text.strip())
except ValueError as e:
    print("Error:", e)
