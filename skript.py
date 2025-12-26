import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

# 1. NAČTENÍ DAT Z CSV
# 'data.csv' nahraď cestou k tvému souboru.
# Pokud máš data oddělená středníkem (časté v českém Excelu), přidej sep=';'
try:
    df = pd.read_csv('TimeDeposit_10K_cleaned.csv', sep=';')
    
    # Pokud tvůj soubor obsahuje i textové sloupce (např. ID, Jméno), které nechceš korelovat,
    df = df.select_dtypes(include=[np.number])
    
except FileNotFoundError:
    print("Chyba: Soubor 'data.csv' nebyl nalezen. Ujisti se, že je ve stejné složce jako skript.")
    # Pro ukázku si zde vygenerujeme data, abys viděla výsledek i bez souboru:
    df = pd.DataFrame(np.random.rand(10, 10), columns=[f'Proměnná_{i}' for i in range(10)])

# 2. VÝPOČET KORELACE
# Vypočítáme korelaci (vazbu) každého s každým.
# Používáme absolutní hodnotu, protože nás zajímá síla vazby, ne jestli je přímá/nepřímá.
corr_matrix = df.corr().abs()

# 3. NASTAVENÍ GRAFU
G = nx.Graph()

# Přidání uzlů (názvy sloupců z CSV)
for col in corr_matrix.columns:
    G.add_node(col)

# 4. FILTRACE A TVORBA HRAN
# Změň tuto hodnotu podle toho, jak moc "hustý" graf chceš.
# 0.5 znamená, že se ukáží jen vazby se silou nad 50 %.
min_threshold = 0.5

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        # Název proměnných
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        
        # Síla vazby (číslo 0 až 1)
        strength = corr_matrix.iloc[i, j]
        
        if strength > min_threshold:
            # Přidáme hranu. Váhu (weight) uložíme pro tloušťku čáry.
            G.add_edge(col1, col2, weight=strength)

# 5. VYKRESLENÍ
plt.figure(figsize=(12, 12)) # Větší plátno, aby se popisky nepřekrývaly

# Rozmístění do kruhu
pos = nx.circular_layout(G)

# Příprava tloušťky čar podle síly vazby
edges = G.edges()
# Hodnotu násobíme (např. * 5), aby byly rozdíly v tloušťce lépe vidět
weights = [G[u][v]['weight'] * 5 for u, v in edges]

# Kreslení
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='navy')
nx.draw_networkx_edges(G, pos, width=weights, edge_color='blue', alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=11, font_family="sans-serif", font_weight="bold")

plt.title("Vizualizace vazeb z CSV souboru", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()