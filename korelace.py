import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns  # Knihovna pro hezčí grafy (Heatmapy)

# ============================================================================
# 1. NAČTENÍ A ČIŠTĚNÍ
# ============================================================================
try:
    df = pd.read_csv('TimeDeposit_10K_clean.csv', sep=';')
except FileNotFoundError:
    print("Soubor nenalezen, generuji náhodná data.")
    df = pd.DataFrame(np.random.rand(50, 15), columns=[f'Var_{i}' for i in range(15)])

# DŮLEŽITÉ: Odstranění ID a Datumů, které kazí výsledek
# Zkusíme různé varianty názvů, kdyby se lišila velikost písmen
cols_to_drop = ['customer_id', 'Customer_ID', 'Ref_Date', 'Birth_Date', 'Birth_Date_dt', 'Ref_Date_dt']
df = df.drop(columns=cols_to_drop, errors='ignore')

# Mapování textů na čísla
gender_map = {'F': 0, 'M': 1, 'Female': 0, 'Male': 1}
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map(gender_map).fillna(0)

income_map = {'No Income': 0, 'Low Income': 1, 'Medium Income': 2, 'High Income': 3}
if 'Income_Group' in df.columns:
    df['Income_Group'] = df['Income_Group'].map(income_map).fillna(0)

age_group_map = {'Young': 0, 'Middle-Age': 1, 'Senior': 2}
if 'Age_Group' in df.columns:
    df['Age_Group'] = df['Age_Group'].map(age_group_map).fillna(0)

tf_map = {'F': 0, 'T': 1, 'False': 0, 'True': 1}
if 'Time_Deposits_Flag' in df.columns:
    df['Time_Deposits_Flag'] = df['Time_Deposits_Flag'].map(tf_map).fillna(0)

marital_map = {
    'Single': 0,    # Svobodný
    'Married': 1,   # Ženatý/Vdaná
    'Divorced': 2,  # Rozvedený
    'Widow': 3      # Vdova/Vdovec
}
if 'Marital_Status' in df.columns:
    df['Marital_Status'] = df['Marital_Status'].map(marital_map).fillna(0)

    
# Zbytek textů na čísla
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# ============================================================================
# 2. VÝPOČET KORELACE
# ============================================================================
corr_matrix = df.corr(method='pearson') # Necháme i záporné hodnoty pro Heatmapu
abs_corr_matrix = corr_matrix.abs()     # Absolutní pro Síťový graf

# ============================================================================
# 3. VIZUALIZACE A: HEATMAPA (Nejlepší pro přehled)
# ============================================================================
plt.figure(figsize=(16, 12))
plt.title("Heatmapa korelací (Červená = Silná vazba)", fontsize=18)

# Vykreslení heatmapy
# cmap='coolwarm' -> Modrá (záporná korelace) ... Bílá (nula) ... Červená (kladná)
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=False) # annot=True by vypsalo čísla, ale u velkých dat je to nečitelné

plt.tight_layout()
plt.show()

