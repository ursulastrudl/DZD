import pandas as pd
import sys
import io
from sklearn.impute import SimpleImputer
from cleverminer import *

df = pd.read_csv('TimeDeposit_10K_clean.csv', sep=';')
df = df[[
    # Demografické atributy
    'Gender',
    'Marital_Status',
    'Children_Num',
    'Occupation_Category',
    'Age_Calculated',
    'Age_Group',
    'Income_Group',
    
    # Produktové flagy (vlastnictví)
    'Payroll_Flag',
    'Business_Flag',
    'Saving_Current_Accounts_Flag',
    'Investment_Products_Flag',
    'Insurance_Products_Flag',
    'Business_Loans_Flag',
    'Housing_Loans_Flag',
    'Consumer_Loans_Flag',
    'Credit_Cards_Flag',
    'Time_Deposits_Flag',
    
    # Balance proměnné (vyberte jen ty nejdůležitější!)
    'Saving_Current_Balance',
    'Credit_Cards_Balance',
    
    # Transakční chování - počty
    'Branch_Trans_Num',
    'ATM_Trans_Num',
    'Internet_Trans_Num',
    
    # Transakční chování - částky
    'Deposit_Trans_Amount',
    'Payment_Trans_Amount',
    
    # Kreditní karty - detaily
    'Credit_Cards_Payments_Num',
    'Credit_Cards_Purchases_Num',
    
    # Rizikovost
    'Arrears_Months_Max'
]]
clm = cleverminer(df=df, proc='4ftMiner',
                   quantifiers={'Base': 500, 'aad': 0.4},  # Mírnější podmínky
                   ante={
                       'attributes': [
                           {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Age_Group', 'type': 'seq', 'minlen': 1, 'maxlen': 3},  # Až 3 věkové skupiny
                           {'name': 'Income_Group', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                           {'name': 'Children_Num', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                           {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 2},  # Až 2 stavy
                           {'name': 'Occupation_Category', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Payroll_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Business_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Saving_Current_Accounts_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Credit_Cards_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                       ], 'minlen': 3, 'maxlen': 5, 'type': 'con'},  # 3-5 atributů!
                   succ={
                       'attributes': [
                           {'name': 'Housing_Loans_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Investment_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Insurance_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                       ], 'minlen': 1, 'maxlen': 1, 'type': 'con'}
                   )

# 8. Zachycení výstupu
output = io.StringIO()
sys.stdout = output

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

# 9. Uložení výstupu
with open("4ftMiner_mega_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako 4ftMiner_mega_output.txt (seřazeno podle BASE)")