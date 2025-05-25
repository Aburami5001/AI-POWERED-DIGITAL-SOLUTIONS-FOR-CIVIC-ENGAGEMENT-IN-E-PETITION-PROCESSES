def classify_complaint(complaint):
    categories = {
        "Transportation and Traffic": ["vehicle", "commute", "route", "highway",
                                       "signal", "accident", "road safety", "transportation"],
        "Public Health Related": ["healthcare", "wellness", "health policy", "mental health",
                                  "infection", "treatment", "hospital", "health insurance"],
        "Municipality": ["government", "governance", "municipal law", "waste management",
                         "fire department", "police department", "housing", "social services"],
        "Urban Development": ["land use", "smart city", "sustainable development", "green spaces",
                              "building permits", "urbanization", "environmental impact"],
        "Public Work Department": ["road maintenance", "water supply", "sewage system",
                                   "drainage", "street lighting"]
    }

    complaint_lower = complaint.lower()

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in complaint_lower:  # Match substrings including phrases
                return category
    return "Uncategorized"

info = "road safety"
print(classify_complaint(info))