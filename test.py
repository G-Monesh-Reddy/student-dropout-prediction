import google.generativeai as genai

API_KEY = "AIzaSyBTNHP7PGDnzp6NkwhuyWXJ_CuIXWhiA08"


genai.configure(api_key=API_KEY)

try:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        "Explain how AI helps students in two simple sentences."
    )

    print("✅ Gemini API is working")
    print(response.text)

except Exception as e:
    print("❌ Error")
    print(e)