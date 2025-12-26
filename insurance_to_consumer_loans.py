import pandas as pd
from cleverminer import cleverminer
import io
import sys
df = pd.read_csv('TimeDeposit_10K_clean.csv', sep=';')
clm = cleverminer(df=df, proc='4ftMiner',
                   quantifiers={'Base': 500, 'aad': 0.4},
                   ante={
                       'attributes': [
                           {'name': 'Insurance_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Age_Group', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                           {'name': 'Income_Group', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                           {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Children_Num', 'type': 'seq', 'minlen': 1, 'maxlen': 2}
                       ], 'minlen': 1, 'maxlen': 3, 'type': 'con'},
                   succ={
                       'attributes': [
                           {'name': 'Consumer_Loans_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                       ], 'minlen': 1, 'maxlen': 1, 'type': 'con'}
                   )

# Zachycení výstupu
output1 = io.StringIO()
sys.stdout = output1

print("=== Shrnutí ===")
clm.print_summary()

# Získání všech pravidel a jejich BASE hodnot
rules_data = []
for i in range(1, len(clm.rulelist) + 1):
    quants = clm.get_quantifiers(i)
    rules_data.append({
        'rule_id': i,
        'base': quants.get('base', 0),
        'conf': quants.get('conf', 0),
        'aad': quants.get('aad', 0)
    })

# Seřazení podle BASE (sestupně)
rules_data_sorted = sorted(rules_data, key=lambda x: x['base'], reverse=True)

print("\n=== Seznam pravidel (seřazeno podle BASE) ===")
print(f"{'RULEID':<7} {'BASE':<6} {'CONF':<6} {'AAD':<7} Rule")

for rule_info in rules_data_sorted:
    rule_id = rule_info['rule_id']
    rule_text = clm.get_ruletext(rule_id)
    print(f"{rule_id:<7} {rule_info['base']:<6} {rule_info['conf']:<6.3f} {rule_info['aad']:+<7.3f} {rule_text}")

print("\n=== Jednotlivá pravidla (seřazeno podle BASE) ===")
for rule_info in rules_data_sorted:
    rule_id = rule_info['rule_id']
    print(f"\n=== Rule {rule_id} (BASE={rule_info['base']}) ===\n")
    clm.print_rule(rule_id)

sys.stdout = sys.__stdout__

# Uložení výstupu
with open("insurance_to_consumer_loans.txt", "w", encoding="utf-8") as f:
    f.write(output1.getvalue())

print("✓ Analýza 1 dokončena a uložena: insurance_to_consumer_loans.txt")