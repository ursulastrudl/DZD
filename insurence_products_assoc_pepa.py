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
    'Saving_Current_Accounts_Flag',
    'Investment_Products_Flag',
    'Credit_Cards_Flag',
    'Housing_Loans_Flag',
    'Consumer_Loans_Flag',
    'Insurance_Products_Flag'
]]

# CleverMiner analýza
clm = cleverminer(
    df=df,
    proc='4ftMiner',
    quantifiers={'Base': 250, 'aad': 1.2},
    ante={
        'attributes': [
            {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Occupation_Category', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Saving_Current_Accounts_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Investment_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Credit_Cards_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Housing_Loans_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Consumer_Loans_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 2,
        'maxlen': 4,
        'type': 'con'
    },
    succ={
        'attributes': [
            {'name': 'Insurance_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)

# Zachycení výstupu
output = io.StringIO()
sys.stdout = output

print("=" * 80)
print("ANALÝZA: POJIŠŤOVACÍ PRODUKTY (INSURANCE)")
print("=" * 80)
print("\nBusiness otázka: Které skupiny klientů si pořizují pojištění?")
print("Použití: Cross-selling pojištění, risk management, ochrana majetku\n")

clm.print_summary()

# Seřazení podle BASE
rules_data = []
for i in range(1, len(clm.rulelist) + 1):
    quants = clm.get_quantifiers(i)
    rules_data.append({'id': i, 'base': quants.get('base', 0), 
                       'conf': quants.get('conf', 0), 'aad': quants.get('aad', 0)})

rules_sorted = sorted(rules_data, key=lambda x: x['base'], reverse=True)

print("\n=== Seznam pravidel (seřazeno podle BASE) ===")
print(f"{'RULEID':<7} {'BASE':<6} {'CONF':<6} {'AAD':<7} Rule")

for r in rules_sorted:
    print(f"{r['id']:<7} {r['base']:<6} {r['conf']:<6.3f} {r['aad']:+<7.3f} {clm.get_ruletext(r['id'])}")

print("\n=== TOP 20 pravidel ===")
for i, r in enumerate(rules_sorted[:20], 1):
    print(f"\n{'='*80}")
    print(f"PRAVIDLO #{i} (Rule ID: {r['id']}, BASE={r['base']})")
    print(f"{'='*80}")
    clm.print_rule(r['id'])

sys.stdout = sys.__stdout__

# Uložení
with open("insurance_products_assoc_pepa.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("✓ Analýza Insurance dokončena: insurance_products_assoc_pepa.txt")