import pandas as pd
import sys
import io
from cleverminer import *

# Načtení dat
df = pd.read_csv('TimeDeposit_10K_clean.csv', sep=';')

# Výběr pouze potřebných sloupců
df = df[[
    'Gender',
    'Marital_Status',
    'Occupation_Category',
    'Business_Loans_Flag',
    'Investment_Products_Flag',
    'Insurance_Products_Flag',
    'Saving_Current_Accounts_Flag',
    'Credit_Cards_Flag',
    'Housing_Loans_Flag',
    'Consumer_Loans_Flag'
]]

# CleverMiner analýza
clm = cleverminer(
    df=df, 
    proc='4ftMiner',
    quantifiers={'Base': 300, 'aad': 1},
    ante={
        'attributes': [
            {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Occupation_Category', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Business_Loans_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Investment_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Insurance_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Saving_Current_Accounts_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Credit_Cards_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ], 
        'minlen': 2, 
        'maxlen': 4, 
        'type': 'con'
    },
    succ={
        'attributes': [
            {'name': 'Housing_Loans_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Consumer_Loans_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ], 
        'minlen': 1, 
        'maxlen': 1, 
        'type': 'con'
    }
)

# Zachycení výstupu
output = io.StringIO()
sys.stdout = output

print("=== HOUSING & CONSUMER LOANS ===\n")
clm.print_summary()

print("\n=== Seznam pravidel (seřazeno podle BASE) ===")

# Seřazení podle BASE
rules_data = []
for i in range(1, len(clm.rulelist) + 1):
    quants = clm.get_quantifiers(i)
    rules_data.append({'id': i, 'base': quants.get('base', 0)})

rules_sorted = sorted(rules_data, key=lambda x: x['base'], reverse=True)

print(f"{'ID':<5} {'BASE':<6} Rule")
for r in rules_sorted:
    print(f"{r['id']:<5} {r['base']:<6} {clm.get_ruletext(r['id'])}")

print("\n=== TOP 20 pravidel ===")
for i, r in enumerate(rules_sorted[:20], 1):
    print(f"\n=== Rule {r['id']} (BASE={r['base']}) ===\n")
    clm.print_rule(r['id'])

sys.stdout = sys.__stdout__

# Uložení
with open("uloha1_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("✓ Výstup uložen: uloha1_output.txt")