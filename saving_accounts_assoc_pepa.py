import pandas as pd
from sklearn.impute import SimpleImputer
from cleverminer import cleverminer
import sys

# PŘESMĚROVÁNÍ VÝSTUPU DO SOUBORU
sys.stdout = open('saving_accounts_assoc_pepa.txt', 'w', encoding='utf-8')

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

# 3. CÍLOVÁ PROMĚNNÁ
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
        'Base': 100,        # Minimální support = 100 řádků (1%)
        'aad': 2,         # Minimální confidence = 70%
    },
    
    # ANTECEDENT (levá strana: IF ...)
ante={
    'attributes': [
        # Demografické
        {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Occupation', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        
        # Produkty - VŠECHNY kromě cílových
        {'name': 'Investment', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Insurance', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Business_Loan', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Housing_Loan', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Consumer_Loan', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        {'name': 'Credit_Card', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
    ],
    'minlen': 1,
    'maxlen': 3,
    'type': 'con'
},

# CONSEQUENT (pravá strana: THEN ...)
# POUZE cílové proměnné, které NEJSOU v antecedent
succ={
    'attributes': [
        {'name': 'Saving_Account', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
    ],
    'minlen': 1,
    'maxlen': 1,
    'type': 'con'
}
)

# ============================================================================
# SPUŠTĚNÍ A VÝSLEDKY
# ============================================================================

print("\n" + "=" * 80)
print("SPOUŠTÍM CLEVERMINER...")
print("=" * 80)
print()

# Souhrn
""" print("=" * 80)
print("SOUHRN ANALÝZY:")
print("=" * 80)
clm.print_summary() """

# Seznam všech pravidel
print("\n" + "=" * 80)
print("SEZNAM VŠECH PRAVIDEL (seřazeno podle Lift):")
print("=" * 80)
clm.print_rulelist()

# Detailní výpis TOP 20 pravidel
print("\n" + "=" * 80)
print("TOP 20 NEJZAJÍMAVĚJŠÍCH PRAVIDEL:")
print("=" * 80)

num_rules = min(20, len(clm.rulelist))
for i in range(num_rules):
    print("\n" + "=" * 80)
    print(f"PRAVIDLO #{i+1}")
    print("=" * 80)
    clm.print_rule(i)

# Export do CSV
""" try:
    results_df = pd.DataFrame(clm.rulelist)
    results_df.to_csv('CleverMiner_Results_Exploratory.csv', 
                      index=False, encoding='utf-8', sep=';')
    print("\n" + "=" * 80)
    print("✓ EXPORT ÚSPĚŠNÝ")
    print("=" * 80)
    print(f"Soubor: CleverMiner_Results_Exploratory.csv")
    print(f"Počet pravidel: {len(results_df)}")
except Exception as e:
    print(f"\n⚠ Export se nezdařil: {e}")
 """
# Závěrečné statistiky
print("\n" + "=" * 80)
print("ANALÝZA DOKONČENA!")
print("=" * 80)
print(f"\n Celkem nalezeno pravidel: {len(clm.rulelist)}")
print(f" Kategoriálních atributů použito: {len(cleverminer_df.columns)}")

sys.stdout.close()