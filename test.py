import streamlit as st
from groq import Groq
import json


client = Groq(api_key="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")
products = [
    {"id": 1, "name": "iPhone 13", "price": 799},
    {"id": 2, "name": "Samsung Galaxy A52", "price": 499},
    {"id": 3, "name": "OnePlus Nord", "price": 399},
    {"id": 4, "name": "Google Pixel 6", "price": 599},
    {"id": 5, "name": "Xiaomi Redmi Note 10", "price": 299},
    {"id": 6, "name": "iPhone SE (2022)", "price": 429},
    {"id": 7, "name": "Samsung Galaxy S21", "price": 699},
    {"id": 8, "name": "OnePlus 10 Pro", "price": 899},
    {"id": 9, "name": "Google Pixel 7a", "price": 499},
    {"id": 10, "name": "Xiaomi Mi 11 Lite", "price": 349},
    {"id": 11, "name": "Realme GT Neo 3", "price": 450},
    {"id": 12, "name": "Poco X3 Pro", "price": 299},
    {"id": 13, "name": "iPhone 14 Pro", "price": 999},
    {"id": 14, "name": "Samsung Galaxy Z Flip 4", "price": 999},
    {"id": 15, "name": "Motorola Edge 30", "price": 399}
]


st.title("📱 AI-Powered Product Recommendation System")

user_input = st.text_input("Enter your preference (e.g., phone under $500):")

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        prompt = f"""
        You are a recommendation system.
        Here is a list of products: {json.dumps(products)}.
        Based on the user preference: "{user_input}",
        recommend the best matching products.

        ⚠️ IMPORTANT: Only return a valid JSON list, like this:
        [
          {{"name": "Product A", "price": 123}},
          {{"name": "Product B", "price": 456}}
        ]
        No explanations, no extra text.
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON
            recommendations = json.loads(content)

            if recommendations:
                st.subheader("✅ Recommended Products:")
                for r in recommendations:
                    st.write(f"- **{r['name']}** — ${r['price']}")
            else:
                st.warning("No valid recommendations received. Try again.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
