import pandas as pd
from sklearn.impute import SimpleImputer
from cleverminer import cleverminer
import sys
import io 

# ============================================================================
# NAČTENÍ DAT
# ============================================================================

df = pd.read_csv('TimeDeposit_10K_clean.csv', sep=';', encoding='utf-8')

# Vytvoření kategoriálních atributů
cleverminer_df = pd.DataFrame()
cleverminer_df['Gender'] = df['Gender']
cleverminer_df['Marital_Status'] = df['Marital_Status']
cleverminer_df['Occupation'] = df['Occupation_Category']

flag_columns = {
    'Payroll': 'Payroll_Flag',
    'Business': 'Business_Flag',
    'Saving_Account': 'Saving_Current_Accounts_Flag',
    'Investment': 'Investment_Products_Flag',
    'Insurance': 'Insurance_Products_Flag',
    'Business_Loan': 'Business_Loans_Flag',
    'Housing_Loan': 'Housing_Loans_Flag',
    'Consumer_Loan': 'Consumer_Loans_Flag',
    'Credit_Card': 'Credit_Cards_Flag'
}

for new_name, old_name in flag_columns.items():
    cleverminer_df[new_name] = df[old_name].apply(
        lambda x: 'Yes' if x == 1.0 else 'No'
    )

cleverminer_df['Time_Deposit'] = df['Time_Deposits_Flag'].apply(
    lambda x: 'Yes' if x == 'T' else 'No'
)

imputer = SimpleImputer(strategy="most_frequent")
cleverminer_df = pd.DataFrame(imputer.fit_transform(cleverminer_df), 
                               columns=cleverminer_df.columns)

# ============================================================================
# ASOCIAČNÍ PRAVIDLA: INVESTMENT PRODUCTS
# ============================================================================
# Business otázka: Kdo investuje? (wealth management, premium služby)

clm = cleverminer(
    df=cleverminer_df,
    proc='4ftMiner',
    
    quantifiers={
        'Base': 200,        # Střední BASE
        'aad': 1.5,         # Vyšší AAD pro silnější vztahy (15 p.b.)
    },
    
    ante={
        'attributes': [
            # Demografické
            {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Occupation', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            
            # Produkty (bez Investment!)
            {'name': 'Saving_Account', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Insurance', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Credit_Card', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Business_Loan', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Time_Deposit', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        ],
        'minlen': 2,
        'maxlen': 4,
        'type': 'con'
    },
    
    succ={
        'attributes': [
            {'name': 'Investment', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)

# ============================================================================
# VÝSTUP
# ============================================================================

output1 = io.StringIO()
sys.stdout = output1

print("=" * 80)
print("ANALÝZA: INVESTIČNÍ PRODUKTY (INVESTMENT PRODUCTS)")
print("=" * 80)
print("\nBusiness otázka: Které skupiny klientů investují?")
print("Použití: Wealth management, premium banking služby, finanční poradenství\n")

print("=== Shrnutí ===")
clm.print_summary()

# Získání pravidel
rules_data = []
for i in range(1, len(clm.rulelist) + 1):
    quants = clm.get_quantifiers(i)
    rules_data.append({
        'rule_id': i,
        'base': quants.get('base', 0),
        'conf': quants.get('conf', 0),
        'aad': quants.get('aad', 0)
    })

rules_data_sorted = sorted(rules_data, key=lambda x: x['base'], reverse=True)

print("\n=== Seznam pravidel (seřazeno podle BASE) ===")
print(f"{'RULEID':<7} {'BASE':<6} {'CONF':<6} {'AAD':<7} Rule")

for rule_info in rules_data_sorted:
    rule_id = rule_info['rule_id']
    rule_text = clm.get_ruletext(rule_id)
    print(f"{rule_id:<7} {rule_info['base']:<6} {rule_info['conf']:<6.3f} {rule_info['aad']:+<7.3f} {rule_text}")

print("\n=== TOP 20 pravidel (seřazeno podle BASE) ===")
for i, rule_info in enumerate(rules_data_sorted[:20], 1):
    rule_id = rule_info['rule_id']
    print(f"\n{'='*80}")
    print(f"PRAVIDLO #{i} (Rule ID: {rule_id}, BASE={rule_info['base']})")
    print(f"{'='*80}")
    clm.print_rule(rule_id)

sys.stdout = sys.__stdout__

with open("investment_products_assoc_pepa.txt", "w", encoding="utf-8") as f:
    f.write(output1.getvalue())

print("✓ Analýza Investment Products dokončena: investment_products_assoc_pepa.txt")