import pandas as pd
import sys
import io
from cleverminer import *

df = pd.read_csv('TimeDeposit_10K_clean.csv', sep=';')
clm = cleverminer(
    df=df,
    proc='4ftMiner',
    quantifiers={
        'Base': 300,      
        'aad': 0.75  
    },
    ante={
        'attributes': [
            {'name': 'Age_Group', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Income_Group', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        ],
        'minlen': 1,
        'maxlen': 4,   
        'type': 'con'
    },
    succ={
        'attributes': [
            {'name': 'Business_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)

# Zachycení výstupu
output = io.StringIO()
sys.stdout = output


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
with open("uloha3_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("✓ Analýza Insurance dokončena: uloha3_output.txt")