import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Configuration de la page
st.set_page_config(
    page_title="Classification des Données Bancaires",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Barre latérale

if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'a_propos'  # Page par défaut

# Fonction pour mettre à jour page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('🏦 Classification des Données Bancaires')

    # Navigation par boutons
    st.subheader("Sections")
    if st.button("À Propos", use_container_width=True, on_click=set_page_selection, args=('a_propos',), key="btn_a_propos"):
        pass
    if st.button("Jeu de Données", use_container_width=True, on_click=set_page_selection, args=('jeu_de_donnees',), key="btn_jeu_de_donnees"):
        pass
    if st.button("Analyse Exploratoire", use_container_width=True, on_click=set_page_selection, args=('analyse_exploratoire',), key="btn_analyse_exploratoire"):
        pass
    if st.button("Prédiction", use_container_width=True, on_click=set_page_selection, args=('prediction',), key="btn_prediction"):
        pass
    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',), key="btn_conclusion"):
        pass

    # Détails du projet
    st.subheader("Résumé")
    st.markdown("""
        Un tableau de bord interactif pour explorer et classifier les données d'une campagne marketing bancaire.

        - [Jeu de Données](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
        - [Notebook Google Colab](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)
        - [Dépôt GitHub](https://github.com/teguegni/bank-additionnal-full/Streamlit-Bank-Classification-Dashboard)

        **Auteur :** [`Kenfack Teguegni Junior`](https://jcdiamante.com)
    """)

# -------------------------

# Charger les données
try:
    df = pd.read_csv('bank-additional-full.csv', delimiter=';')
except FileNotFoundError:
    st.error("Le fichier 'bank-additional-full.csv' est introuvable. Veuillez vérifier son emplacement.")
    st.stop()

# Page principale
if st.session_state.page_selection == 'a_propos':
    # Page À Propos
    st.title("️ À Propos")
    st.markdown("""
        Cette application explore le jeu de données **Bank Marketing** et propose :

        - Une exploration visuelle des données.
        - Un prétraitement et nettoyage des données.
        - La construction et l'évaluation de modèles d'apprentissage automatique.
        - Une interface interactive pour prédire si un client souscrira à un produit.

        **Technologies utilisées :**
        - Python (Streamlit, Altair, Pandas)
        - Machine Learning (Scikit-learn)

        **Auteur : Kenfack Teguegni Junior**

        ✉️ Contact : kenfackteguegni@gmail.com
    """)

elif st.session_state.page_selection == 'conclusion':
    # Page Conclusion
    st.title("️ Conclusion")
    st.markdown("""
        Un traitement minutieux et réfléchi du DataFrame `bank-additional-full.csv` est fondamental pour maximiser la précision
        et la fiabilité du modèle de prédiction. En combinant explorations, prétraitements adéquats, et évaluations rigoureuses,
        un modèle robuste peut être développé pour mieux prédire les comportements des clients envers la souscription à un produit.
    """)

elif st.session_state.page_selection == 'jeu_de_donnees':
    # Page Jeu de Données
    st.title(" Jeu de Données")

    # Afficher les premières lignes du DataFrame
    if st.checkbox("Afficher le DataFrame", key="chk_afficher_df"):
        nb_rows = st.slider("Nombre de lignes à afficher :", min_value=1, max_value=len(df), value=10, key="slider_nb_rows")
        st.write(df.head(nb_rows))

    # Afficher les statistiques descriptives
    if st.checkbox("Afficher les statistiques descriptives", key="chk_stats_desc"):
        st.write(df.describe())

elif st.session_state.page_selection == 'analyse_exploratoire':
    st.title(" Analyse Exploratoire")

    # Vérification des duplications
    duplicates = df.duplicated()
    st.write(f"Nombre de duplications : {duplicates.sum()}")

    # Suppression des duplications
    df = df.drop_duplicates()

    # Remplacement des valeurs 'unknown'
    for column in df.columns:
        if df[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column] = df[column].replace('unknown', mode_value)

    # Vérification des valeurs manquantes
    st.subheader("Vérification des valeurs manquantes")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Graphique Altair
    st.subheader("Relation entre l'âge et le métier")
    df['job'] = df['job'].astype('category')
    age_job_chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X('age:Q', title='Âge'),
            y=alt.Y('job:O', title='Métier', sort=None),
            color='y:N',
            tooltip=['age:Q', 'job:N', 'y:N']
        )
        .properties(
            title='Relation entre l\'âge et le métier',
            width=600,
            height=400
        )
        .interactive()
    )
    st.altair_chart(age_job_chart, use_container_width=True)

elif st.session_state.page_selection == 'apprentissage_automatique':
    # Sélection des colonnes numériques
    colonne_numerique = df[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]

    # Fonction pour détecter et remplacer les valeurs aberrantes
    def replace_outliers(df):
        for col in df.select_dtypes(include=['number']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    replace_outliers(df)

    # Encodage des variables catégorielles
    for column in ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']:
        fe = df.groupby(column).size() / len(df)
        df[f'{column}_freq_encode'] = df[column].map(fe)

    # Encodage des colonnes catégorielles binaires
    le = LabelEncoder()
    for column in ['default', 'housing', 'loan', 'contact']:
        df[column] = le.fit_transform(df[column])

    # Encodage de la colonne cible
    encoder = LabelEncoder()
    df['y_encoded'] = encoder.fit_transform(df['y'])

    # Suppression des colonnes catégorielles originales
    colonnes_a_supprimer = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
    df_propre = df.drop(colonnes_a_supprimer, axis=1)

    # Suréchantillonnage
    y_encoded = df_propre['y_encoded'].values
    X = df_propre.drop(columns=['y_encoded']).values
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y_encoded)
    df_resampled = pd.DataFrame(X_resampled, columns=df_propre.columns[:-1])
    df_resampled['y_encoded'] = y_resampled

    # Division des données
    X = df_resampled.drop(columns=['y_encoded'])
    y = df_resampled['y_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Évaluation du modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f'Accuracy: {accuracy:.2f}')
    st.write('Classification Report:')
    st.write(report)

elif st.session_state.page_selection == 'prediction':
    # Page Prédiction
    st.title("🔮 Prédiction")

    # Création des champs de saisie
    age = st.number_input("Âge", min_value=0, max_value=120, value=30, key="input_age")
    duration = st.number_input("Durée (en secondes)", min_value=0, value=100, key="input_duration")
    campaign = st.number_input("Nombre de contacts lors de cette campagne", min_value=0, value=1, key="input_campaign")
    pdays = st.number_input("Nombre de jours depuis le dernier contact", min_value=-1, value=-1, key="input_pdays")
    previous = st.number_input("Nombre de contacts précédents", min_value=0, value=0, key="input_previous")
    emp_var_rate = st.number_input("Taux de variation de l'emploi (%)", value=0.0, key="input_emp_var_rate")
    cons_price_idx = st.number_input("Indice des prix à la consommation", value=92.0, key="input_cons_price_idx")
    cons_conf_idx = st.number_input("Indice de confiance des consommateurs", value=-50.0, key="input_cons_conf_idx")
    euribor3m = st.number_input("Taux d'Euribor à 3 mois (%)", value=0.0, key="input_euribor3m")
    nr_employed = st.number_input("Nombre d'employés", value=5000, key="input_nr_employed")
    marital_freq_encode = st.number_input("Code marital", min_value=0, value=0, key="input_marital_freq_encode")
    job_freq_encode = st.number_input("Code emploi", min_value=0, value=0, key="input_job_freq_encode")
    education_freq_encode = st.number_input("Code éducation", min_value=0, value=0, key="input_education_freq_encode")
    month_freq_encode = st.number_input("Code mois", min_value=0, value=0, key="input_month_freq_encode")
    day_freq_encode = st.number_input("Code jour", min_value=0, value=0, key="input_day_freq_encode")
    poutcome_freq_encode = st.number_input("Code résultat de la campagne précédente", min_value=0, value=0, key="input_poutcome_freq_encode")

    # Bouton pour soumettre le formulaire
    if st.button("Soumettre", key="btn_soumettre"):
        input_data = np.array([[age, duration, campaign, pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, marital_freq_encode, job_freq_encode, education_freq_encode, month_freq_encode, day_freq_encode, poutcome_freq_encode]])
        prediction = model.predict(input_data)
        result = "Prêt accordé." if prediction[0] == 1 else "Prêt non accordé."
        st.write(result)


   
