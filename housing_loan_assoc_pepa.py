import pandas as pd
from sklearn.impute import SimpleImputer
from cleverminer import cleverminer
import sys
import io 
# PŘESMĚROVÁNÍ VÝSTUPU DO SOUBORU
sys.stdout = open('housing_consumer_loan_assoc_pepa.txt', 'w', encoding='utf-8')

# ============================================================================
# NAČTENÍ DAT
# ============================================================================

df = pd.read_csv('TimeDeposit_10K_clean.csv', sep=';', encoding='utf-8')

print("=" * 80)
print("CLEVERMINER - EXPLORATIVNÍ ANALÝZA")
print("=" * 80)
print(f"\nNačteno: {len(df)} řádků, {len(df.columns)} sloupců")

# ============================================================================
# VÝBĚR A PŘEVOD KATEGORIÁLNÍCH ATRIBUTŮ
# ============================================================================

print("\n📊 Připravuji kategoriální atributy...")

# Vytvoření nového DataFrame POUZE s kategoriálními atributy
cleverminer_df = pd.DataFrame()

# 1. KATEGORIÁLNÍ ATRIBUTY (již existující)
cleverminer_df['Gender'] = df['Gender']
cleverminer_df['Marital_Status'] = df['Marital_Status']
cleverminer_df['Occupation'] = df['Occupation_Category']

# 2. FLAG ATRIBUTY → převod na Yes/No
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

print(f"✓ Připraveno {len(cleverminer_df.columns)} kategoriálních atributů:")
for col in cleverminer_df.columns:
    unique_count = cleverminer_df[col].nunique()
    print(f"  - {col:25s}: {unique_count} kategorií")

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
cleverminer_df = pd.DataFrame(imputer.fit_transform(cleverminer_df), 
                               columns=cleverminer_df.columns)

# ============================================================================
# NASTAVENÍ CLEVERMINER
# ============================================================================

print("\n" + "=" * 80)
print("KONFIGURACE CLEVERMINER:")
print("=" * 80)
print("Quantifiers:")
print("  - Min support (Base): 100 řádků (1%)")
print("  - Min confidence (aad): 60%")
print("  - Min lift: 1.5")
print("  - Max délka antecedentu: 3 atributy")

clm = cleverminer(
    df=cleverminer_df,
    proc='4ftMiner',
    
    # METRIKY
    quantifiers={
        'Base': 300,        # Minimální support = 100 řádků (1%)
        'aad': 1,         
    },
    
    # ANTECEDENT (levá strana: IF ...)
ante={
    'attributes': [
        # Demografické
        {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Occupation', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        
        # Produkty - VŠECHNY kromě cílových
        {'name': 'Business_Loan', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Investment', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Insurance', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Saving_Account', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Credit_Card', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
    ],
    'minlen': 2,
    'maxlen': 4,
    'type': 'con'
},

# CONSEQUENT (pravá strana: THEN ...)
# POUZE cílové proměnné, které NEJSOU v antecedent
succ={
    'attributes': [
        {'name': 'Housing_Loan', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Consumer_Loan', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
    ],
    'minlen': 1,
    'maxlen': 2,
    'type': 'con'
}
)

# ============================================================================
# SPUŠTĚNÍ A VÝSLEDKY
# ============================================================================

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
with open("housing_consumer_loan_assoc_pepa.txt", "w", encoding="utf-8") as f:
    f.write(output1.getvalue())

print("✓ Analýza 1 dokončena a uložena: housing_consumer_loan_assoc_pepa.txt")