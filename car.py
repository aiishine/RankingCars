import pandas as pd
import numpy as np
from pyDecision.algorithm import electre_i, promethee_i, promethee_ii, promethee_gaia, topsis_method
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv('./cars.csv')


print("Aperçu des données chargées :")
print(df.head())

if 'Marque' in df.columns:
    df.set_index('Marque', inplace=True)


# Encodage de la colonne 'Transmission'
label_encoder = LabelEncoder()
df['Transmission'] = label_encoder.fit_transform(df['Transmission'])


# Définir les critères et les poids
criteria = ['Prix', 'Consommation', 'Performances', 'Fiabilité', 'Confort', 'Design', 'Espace Intérieur', 'Technologie', 'Sécurité', 'Transmission']
weights = [0.15, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
criterion_type = ['min', 'min', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max']

df.drop(columns=['ID'], inplace=True)

# print(df.columns)


# Vérification de la forme des données et des poids
assert len(criteria) == df.shape[1], "Le nombre de critères doit correspondre au nombre de colonnes (hors 'Marque')."
assert len(weights) == df.shape[1], "Le nombre de poids doit correspondre au nombre de colonnes (hors 'Marque')."

# Normalisation des données
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
print('Normalisation des données:')
print(normalized_df.head())


# TOPSIS
topsis_result = topsis_method(normalized_df.values, weights, criterion_type)
df['TOPSIS_Score'] = topsis_result
df['TOPSIS_Rank'] = df['TOPSIS_Score'].rank(ascending=False)

# ELECTRE I
concordance_threshold = 0.6
discordance_threshold = 0.4
electre_result = electre_i(normalized_df.values, weights, criterion_type, concordance_threshold, discordance_threshold)

# PROMETHEE I et II
Q = [0.1] * len(criteria)  # preference_thresholds
S = [0.2] * len(criteria)  # indifference_thresholds
P = [0.3] * len(criteria)  # preference_inverse_thresholds
F = [0.0] * len(criteria)  # veto_thresholds

promethee_i_result = promethee_i(normalized_df.values, weights, criterion_type, Q, S, P, F)
promethee_ii_result = promethee_ii(normalized_df.values, weights, criterion_type, Q, S, P, F, graph=True, verbose=True)
promethee_ii_res = [int(sub_array[0]) for sub_array in promethee_ii_result]

# Affichage des résultats
print("TOPSIS Scores and Ranks:")
print(df[['TOPSIS_Score', 'TOPSIS_Rank']])
print("\nELECTRE I Result:")
print(electre_result)
print("\nPROMETHEE I Result:")
print(promethee_i_result)
print("\nPROMETHEE II Result:")
print(promethee_ii_result)

# GAIA
gaia_result = promethee_gaia(normalized_df.values, weights, Q, S, P, F)


# Combiner les résultats avec les données originales
df['PROMETHEE II Rank'] = promethee_ii_res

# Afficher la marque avec le meilleur classement PROMETHEE II
print("Marques avec le meilleur classement selon PROMETHEE II:")
print(df.sort_values(by='PROMETHEE II Rank')[['PROMETHEE II Rank']].head(10))


# Afficher la marque avec le meilleur classement TOPSIS
print("Marques avec le meilleur classement selon TOPSIS:")
print(df.sort_values(by='TOPSIS_Rank')[['TOPSIS_Rank']].head(10))
