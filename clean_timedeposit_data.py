"""
Skript pro čištění datasetu TimeDeposit_10K.csv
Opravuje chybné hodnoty a přidává nové sloupce (Věk, Věková skupina, Příjmová skupina)

Autor: Analýza dat
Datum: 26.12.2024
"""

import pandas as pd
import re
import numpy as np

# ============================================================================
# NASTAVENÍ
# ============================================================================

input_file = 'TimeDeposit_10K.csv'
output_file = 'TimeDeposit_10K_clean.csv' # Přejmenováno na enriched

# ============================================================================
# SLOVNÍK PRO PŘEVOD ŘÍMSKÝCH ČÍSLIC
# ============================================================================

roman_to_int = {
    'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6',
    'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10', 'XI': '11', 'XII': '12'
}

# ============================================================================
# FUNKCE
# ============================================================================

def clean_value(val):
    """Funkce pro čištění jedné hodnoty (oprava římských číslic)."""
    if pd.isna(val): return val
    val_str = str(val)
    
    # Vzory pro detekci římských číslic
    pattern1 = re.match(r'^(\d+)\.([IVX]+)$', val_str) # 02.V
    pattern2 = re.match(r'^([IVX]+)\.(\d+)$', val_str) # VI.74
    
    if pattern1:
        num, roman = pattern1.group(1), pattern1.group(2)
        if roman in roman_to_int:
            return float(f"{int(num)}.{roman_to_int[roman]}")
    elif pattern2:
        roman, num = pattern2.group(1), pattern2.group(2)
        if roman in roman_to_int:
            return float(f"{roman_to_int[roman]}.{num}")
    
    return val

def get_age_group(age):
    """Přiřadí věkovou skupinu."""
    if pd.isna(age): return "Unknown"
    if age < 35: return "Young"
    if age < 55: return "Middle-Age"
    return "Senior"

def get_income_group(income):
    """Přiřadí příjmovou skupinu."""
    if pd.isna(income): return "Unknown"
    if income <= 100: return "No Income"
    if income <= 15000: return "Low Income"
    if income <= 45000: return "Medium Income"
    return "High Income"

# ============================================================================
# SEZNAMY SLOUPCŮ
# ============================================================================

columns_to_clean = [
    'Branch_Trans_Num', 'ATM_Trans_Num', 'APS_Trans_Num', 'Phone_Trans_Num', 
    'Internet_Trans_Num', 'Deposit_Trans_Num', 'Withdrawl_Trans_Num', 
    'Payment_Trans_Num', 'Transfer_Trans_Num', 'Deposit_Trans_Amount', 
    'Withdrawl_Trans_Amount', 'Payment_Trans_Amount', 'Transfer_Trans_Amount', 
    'Saving_Current_Balance', 'Investment_Products_Balance', 'Insurance_Balances', 
    'Business_Loans_Balance', 'Consumer_Loans_Balance', 'Credit_Cards_Balance', 
    'Credit_Cards_Installments', 'Credit_Cards_Payments_Num', 
    'Credit_Cards_Purchases_Num', 'Credit_Cards_Witrhdrawals_Num', 
    'Credit_Cards_Payments_Amount', 'Credit_Cards_Purchases_Amount', 
    'Credit_Cards_Witrhdrawals_Amount'
]

# ============================================================================
# HLAVNÍ KÓD
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("ZPRACOVÁNÍ DAT - START")
    print("=" * 80)
    
    # 1. Načtení dat
    print(f"\n📂 Načítám data ze souboru: {input_file}")
    df = pd.read_csv(input_file, sep=';', encoding='utf-8')
    print(f"✓ Načteno {len(df)} řádků")
    
    # ---------------------------------------------------------
    # 2. ČIŠTĚNÍ DAT (Původní logika)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("FÁZE 1: ČIŠTĚNÍ CHYBNÝCH HODNOT")
    print("=" * 80)
    
    total_cleaned = 0
    for col in columns_to_clean:
        if col in df.columns:
            pattern = re.compile(r'^\d+\.[IVX]+$|^[IVX]+\.\d+$')
            errors_before = df[col].astype(str).str.match(pattern, na=False).sum()
            
            if errors_before > 0:
                df[col] = df[col].apply(clean_value)
                errors_after = df[col].astype(str).str.match(pattern, na=False).sum()
                cleaned = errors_before - errors_after
                total_cleaned += cleaned
                print(f"✓ {col}: opraveno {cleaned} hodnot")

    print(f"Celkem opraveno chyb: {total_cleaned}")

    # ---------------------------------------------------------
    # 3. NOVÉ SLOUPCE
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("FÁZE 2: TVORBA NOVÝCH SLOUPCŮ (Feature Engineering)")
    print("=" * 80)

    # A) VÝPOČET VĚKU (Age_Calculated)
    print("⚙️  Převádím sloupce s daty na datetime...")
    df['Birth_Date_dt'] = pd.to_datetime(df['Birth_Date'], format='%d.%m.%Y', errors='coerce')
    df['Ref_Date_dt'] = pd.to_datetime(df['Ref_Date'], format='%d.%m.%Y', errors='coerce')
    
    print("⚙️  Počítám nový věk (Age_Calculated)...")
    # Logika: Rok reference - Rok narození, minus 1 pokud ještě letos neměl narozeniny
    df['Age_Calculated'] = df.apply(
        lambda row: row['Ref_Date_dt'].year - row['Birth_Date_dt'].year - 
        ((row['Ref_Date_dt'].month, row['Ref_Date_dt'].day) < (row['Birth_Date_dt'].month, row['Birth_Date_dt'].day))
        if pd.notnull(row['Ref_Date_dt']) and pd.notnull(row['Birth_Date_dt']) else np.nan, 
        axis=1
    )
    # Převedeme na integer (tam kde nejsou NaN)
    df['Age_Calculated'] = df['Age_Calculated'].astype('Int64')

    # B) VĚKOVÁ SKUPINA (Age_Group)
    print("⚙️  Vytvářím sloupec Age_Group...")
    # Použijeme buď původní sloupec 'Age' nebo náš nově vypočtený 'Age_Calculated'. 
    # Zde použiji nový výpočet pro přesnost.
    df['Age_Group'] = df['Age_Calculated'].apply(get_age_group)

    # C) PŘÍJMOVÁ SKUPINA (Income_Group)
    # Předpokládám, že existuje sloupec 'Total_Income'. Pokud ne, skript by spadl, 
    # takže přidám kontrolu.
    if 'Total_Income' in df.columns:
        print("⚙️  Vytvářím sloupec Income_Group...")
        # Nejdřív se ujistíme, že Total_Income je číslo
        df['Total_Income'] = pd.to_numeric(df['Total_Income'], errors='coerce')
        df['Income_Group'] = df['Total_Income'].apply(get_income_group)
    else:
        print("⚠  VAROVÁNÍ: Sloupec 'Total_Income' nebyl nalezen! Income_Group nelze vytvořit.")

    # Úklid pomocných sloupců (dt)
    df.drop(columns=['Birth_Date_dt', 'Ref_Date_dt'], inplace=True)

    # ---------------------------------------------------------
    # 4. ULOŽENÍ
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("UKÁZKA NOVÝCH DAT:")
    print("=" * 80)
    
    # Ukázka nových sloupců
    preview_cols = ['Birth_Date', 'Ref_Date', 'Age_Calculated', 'Age_Group']
    if 'Income_Group' in df.columns:
        preview_cols.append('Income_Group')
        
    print(df[preview_cols].head())

    print(f"\n💾 Ukládám data do: {output_file}")
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    print("✓ Hotovo!")