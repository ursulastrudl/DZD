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
# VARIANTA 1: Predikce vlastnictví produktů


clm = cleverminer(df=df, proc='4ftMiner',
                   quantifiers={'Base': 200, 'aad': 0.5},
                   ante={
                       'attributes': [
                           {'name': 'Gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Age_Group', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                           {'name': 'Income_Group', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                           {'name': 'Marital_Status', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Occupation_Category', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                           {'name': 'Children_Num', 'type': 'seq', 'minlen': 1, 'maxlen': 2}
                       ], 'minlen': 1, 'maxlen': 5, 'type': 'con'},
                   succ={
                       'attributes': [
                           {'name': 'Investment_Products_Flag', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                      
                       ], 'minlen': 1, 'maxlen': 1, 'type': 'con'}
                   )


# 8. Zachycení výstupu
output = io.StringIO()
sys.stdout = output

print("=== Shrnutí ===")
clm.print_summary()

print("\n=== Seznam pravidel ===")
clm.print_rulelist()

print("\n=== Jednotlivá pravidla ===")
for i in range(1, len(clm.rulelist) + 1):
    print(f"\n=== Rule {i} ===\n")
    clm.print_rule(i)

sys.stdout = sys.__stdout__


sys.stdout = sys.__stdout__

# 9. Uložení výstupu
with open("4ftMiner_investment_products_flag_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako 4ftMiner_investment_products_flag_output.txt")