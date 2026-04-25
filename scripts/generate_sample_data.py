"""Generate a synthetic customer dataset for testing the churn pipeline."""

import random
import csv
import os

random.seed(42)

NUM_CUSTOMERS = 500

GENDERS = ["Male", "Female"]
LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Credit card", "Bank transfer", "Electronic check", "Mailed check"]

rows = []
for i in range(NUM_CUSTOMERS):
    tenure = random.randint(1, 72)
    monthly = round(random.uniform(20, 120), 2)
    total = round(monthly * tenure * random.uniform(0.8, 1.1), 2)
    tickets = random.randint(0, 10)
    age = random.randint(18, 75)
    contract = random.choice(CONTRACT_TYPES)
    annual_rev = round(monthly * 12 * random.uniform(0.9, 1.1), 2)

    # Churn logic: higher monthly charges + short tenure + many tickets = more likely to churn
    churn_score = (monthly / 120) * 0.4 + (1 - tenure / 72) * 0.3 + (tickets / 10) * 0.3
    churn = 1 if random.random() < churn_score else 0

    rows.append({
        "customer_id": f"CUST_{i+1:04d}",
        "gender": random.choice(GENDERS),
        "age": age,
        "location": random.choice(LOCATIONS),
        "tenure_months": tenure,
        "contract_type": contract,
        "payment_method": random.choice(PAYMENT_METHODS),
        "monthly_charges": monthly,
        "total_charges": total,
        "num_support_tickets": tickets,
        "annual_revenue": annual_rev,
        "churn": churn,
    })

os.makedirs("data", exist_ok=True)
output_path = "data/customers.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {NUM_CUSTOMERS} customer records at {output_path}")
churned = sum(r["churn"] for r in rows)
print(f"Churn rate: {churned}/{NUM_CUSTOMERS} ({churned/NUM_CUSTOMERS*100:.1f}%)")
