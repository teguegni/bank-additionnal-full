import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Non utilisé directement ici
# import seaborn as sns # Non utilisé directement ici
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pickle
import requests # Nécessaire pour charger depuis GitHub
import io # Utile avec requests et pickle

# --- Configuration de la page ---
st.set_page_config(
    page_title="Classification des Données Bancaires",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fonction pour ajouter l'arrière-plan depuis URL GitHub ---
def add_bg_from_url(url):
    """Ajoute une image d'arrière-plan à partir d'une URL brute GitHub."""
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-attachment: fixed;
             background-size: cover;
             background-repeat: no-repeat;
         }}
         /* Optionnel: Ajouter un peu de transparence pour mieux voir le fond */
         /* [data-testid="stSidebar"], .main .block-container {{
             background-color: rgba(255, 255, 255, 0.85); /* Blanc avec 85% opacité */
         /* }} */
         </style>
         """,
         unsafe_allow_html=True
     )

# --- URL de l'image brute sur GitHub ---
image_url_raw = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/image1.jpg" # URL Corrigée
add_bg_from_url(image_url_raw)

# --- Thème Altair ---
# alt.themes.enable("dark") # À ajuster selon la lisibilité avec le fond

# --- Barre latérale ---
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'a_propos'

def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('🏦 Classification Données Bancaires')
    st.subheader("Sections")
    # Utilisation de st.radio pour une navigation peut-être plus claire
    pages = ["À Propos", "Jeu de Données", "Analyse Exploratoire", "Apprentissage Automatique", "Prédiction", "Conclusion"]
    page_keys = ['a_propos', 'jeu_de_donnees', 'analyse_exploratoire', 'apprentissage_automatique', 'prediction', 'conclusion']
    default_index = page_keys.index(st.session_state.page_selection)

    selected_page_title = st.radio("Navigation", pages, index=default_index, key="nav_radio")
    # Mettre à jour l'état si la sélection radio change
    if selected_page_title:
        st.session_state.page_selection = page_keys[pages.index(selected_page_title)]


    st.subheader("Résumé")
    st.markdown("""
        Tableau de bord interactif pour explorer et classifier les données marketing bancaires.
        - [Dépôt GitHub](https://github.com/teguegni/bank-additionnal-full) Auteur : Kenfack Teguegni Junior
    """)

# --- Chargement des données (avec fallback GitHub) ---
# Mettre en cache le chargement des données pour la performance
@st.cache_data
def load_data():
    data_url_local = 'bank-additional-full.csv'
    github_data_url = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/bank-additional-full.csv"
    try:
        df = pd.read_csv(data_url_local, delimiter=';')
        st.success(f"Données chargées depuis '{data_url_local}'.")
        return df
    except FileNotFoundError:
        st.warning(f"Fichier '{data_url_local}' non trouvé. Tentative de chargement depuis GitHub...")
        try:
            df = pd.read_csv(github_data_url, delimiter=';')
            st.success("Données chargées depuis GitHub.")
            return df
        except Exception as e:
            st.error(f"Échec du chargement des données depuis le fichier local et GitHub. Erreur : {e}")
            return None # Retourner None en cas d'échec total

df_original = load_data()

# Arrêter l'application si les données n'ont pas pu être chargées
if df_original is None:
    st.stop()

# --- Fonction de chargement du modèle depuis GitHub ---
# Mettre en cache le chargement du modèle
@st.cache_resource
def load_model_from_github(url):
    """Charge un objet pickle (modèle, etc.) depuis une URL brute GitHub."""
    try:
        st.info(f"Tentative de chargement du modèle depuis GitHub...")
        response = requests.get(url, timeout=30)
        response.raise_for_status() # Vérifie les erreurs HTTP

        # Utiliser io.BytesIO pour que pickle.load fonctionne comme avec un fichier
        model_data = pickle.load(io.BytesIO(response.content))

        # Vérifier si les clés attendues sont présentes
        required_keys = ['model', 'encoder_y', 'features']
        if not all(key in model_data for key in required_keys):
            st.error(f"Le fichier pickle chargé depuis {url} ne contient pas les clés requises: {required_keys}")
            return None, None, None

        st.success("Modèle, encodeur cible et liste des features chargés avec succès depuis GitHub.")
        return model_data['model'], model_data['encoder_y'], model_data['features']

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur réseau lors du téléchargement du modèle : {e}")
        return None, None, None
    except pickle.UnpicklingError as e:
        st.error(f"Erreur lors du désérialisage du fichier pickle : {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue lors du chargement du modèle : {e}")
        return None, None, None

# --- URL du fichier modèle sur GitHub ---
MODEL_URL_GITHUB = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/model_classification_bank.pkl" # Vérifiez ce chemin!

# --- Fonction principale et logique des pages ---
def main():
    global df_original # Utiliser le df chargé globalement

    # -------------------- SECTION À PROPOS --------------------
    if st.session_state.page_selection == 'a_propos':
        st.title("️ À Propos")
        st.markdown("""
            Cette application explore le jeu de données Bank Marketing et propose :
            - Une exploration visuelle des données.
            - Un prétraitement et nettoyage des données.
            - La construction et l'évaluation de modèles d'apprentissage automatique.
            - Une interface interactive pour prédire si un client souscrira à un produit.

            **Technologies utilisées :** Python (Streamlit, Pandas, Scikit-learn), Machine Learning.

            **Auteur :** Kenfack Teguegni Junior | ✉️ **Contact :** kenfackteguegni@gmail.com
            """)

    # -------------------- SECTION JEU DE DONNÉES --------------------
    elif st.session_state.page_selection == 'jeu_de_donnees':
        st.title(" Jeu de Données")
        st.markdown("Aperçu et statistiques descriptives du jeu de données brut.")
        st.dataframe(df_original.head())
        st.write(f"Dimensions : {df_original.shape[0]} lignes, {df_original.shape[1]} colonnes")

        if st.checkbox("Afficher les statistiques descriptives", key="chk_stats_desc"):
            st.subheader("Statistiques Numériques")
            st.write(df_original.describe(include=np.number))
            st.subheader("Statistiques Catégorielles")
            st.write(df_original.describe(include='object'))

    # -------------------- SECTION ANALYSE EXPLORATOIRE --------------------
    elif st.session_state.page_selection == 'analyse_exploratoire':
        st.title(" Analyse Exploratoire et Prétraitement Simple")

        # Travailler sur une copie pour ne pas affecter df_original
        df_processed = df_original.copy()

        st.subheader("1. Gestion des Duplications")
        duplicates_before = df_processed.duplicated().sum()
        if duplicates_before > 0:
            df_processed = df_processed.drop_duplicates()
            st.write(f"{duplicates_before} lignes dupliquées trouvées et supprimées.")
            st.write(f"Nouvelles dimensions : {df_processed.shape}")
        else:
            st.write("Aucune ligne dupliquée trouvée.")

        st.subheader("2. Gestion des Valeurs 'unknown'")
        st.write("Remplacement des 'unknown' par le mode de chaque colonne (si applicable).")
        unknown_replaced_count = 0
        for column in df_processed.select_dtypes(include='object').columns:
             if 'unknown' in df_processed[column].unique():
                mode_val = df_processed[column].mode()[0]
                # Gérer le cas où le mode lui-même est 'unknown'
                if mode_val == 'unknown':
                    modes = df_processed[column].mode()
                    if len(modes) > 1:
                        mode_val = modes[1]
                    else: # Si 'unknown' est la seule valeur ou le seul mode
                        st.warning(f"Impossible de remplacer 'unknown' dans '{column}' car c'est la seule valeur/mode.")
                        continue
                count = (df_processed[column] == 'unknown').sum()
                if count > 0:
                    df_processed[column] = df_processed[column].replace('unknown', mode_val)
                    unknown_replaced_count += count
        st.write(f"{unknown_replaced_count} valeurs 'unknown' remplacées au total.")


        st.subheader("3. Vérification des Valeurs Manquantes (Null)")
        missing_values = df_processed.isnull().sum()
        missing_data = missing_values[missing_values > 0]
        if not missing_data.empty:
            st.write("Valeurs manquantes (Null) restantes :")
            st.write(missing_data)
        else:
            st.write("Aucune valeur manquante (Null) détectée après le traitement.")

        st.subheader("4. Visualisation : Relation Âge / Métier / Souscription")
        # Préparer pour Altair
        df_chart = df_processed.copy()
        df_chart['Souscription'] = df_chart['y'].map({'yes': 'Oui', 'no': 'Non'})
        df_chart['job'] = df_chart['job'].astype(str) # S'assurer que c'est une chaîne pour Altair

        age_job_chart = (
            alt.Chart(df_chart)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X('age', title='Âge'),
                y=alt.Y('job', title='Métier', sort='-x'),
                color=alt.Color('Souscription', title='Souscrit ?'),
                tooltip=[
                    alt.Tooltip('age', title='Âge'),
                    alt.Tooltip('job', title='Métier'),
                    alt.Tooltip('Souscription', title='Souscrit ?')
                    ]
            )
            .properties(title="Relation Âge / Métier colorée par la Souscription", height=500)
            .interactive()
        )
        st.altair_chart(age_job_chart, use_container_width=True)

    # -------------------- SECTION APPRENTISSAGE AUTOMATIQUE --------------------
    elif st.session_state.page_selection == 'apprentissage_automatique':
        st.title("⚙️ Apprentissage Automatique (Entraînement)")
        st.warning("Cette section entraîne un modèle RandomForest et sauvegarde un fichier .pkl localement.")
        st.info("Assurez-vous d'avoir exécuté cette section si vous voulez utiliser le fichier 'model_classification_bank.pkl' local pour la prédiction.")

        # Utiliser une copie pour l'entraînement
        df_ml = df_original.copy()

        # Appliquer les prétraitements initiaux (cohérence avec exploration)
        df_ml = df_ml.drop_duplicates()
        for column in df_ml.select_dtypes(include='object').columns:
             if 'unknown' in df_ml[column].unique():
                mode_val = df_ml[column].mode()[0]
                if mode_val == 'unknown':
                   modes = df_ml[column].mode(); mode_val = modes[1] if len(modes) > 1 else None
                if mode_val: df_ml[column] = df_ml[column].replace('unknown', mode_val)

        st.subheader("1. Traitement des Valeurs Aberrantes (Outliers)")
        numerics = df_ml.select_dtypes(include=np.number).columns.tolist()
        if st.checkbox("Remplacer les outliers par les limites IQR", key="cb_outliers_train", value=False):
            df_ml_out = df_ml.copy()
            for col in numerics:
                Q1 = df_ml_out[col].quantile(0.25); Q3 = df_ml_out[col].quantile(0.75); IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
                df_ml_out[col] = np.clip(df_ml_out[col], lower, upper)
            df_ml = df_ml_out
            st.write("Outliers remplacés.")
        else:
            st.write("Traitement des outliers désactivé.")


        st.subheader("2. Encodage des Variables")
        df_encoded = df_ml.copy()
        categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
        categorical_cols_label = ['default', 'housing', 'loan', 'contact']

        # Encodage par fréquence
        freq_maps = {} # Stocker les mappings pour une éventuelle sauvegarde/utilisation en prédiction
        for column in categorical_cols_freq:
            fe = df_encoded.groupby(column).size() / len(df_encoded)
            df_encoded[f'{column}_freq_encode'] = df_encoded[column].map(fe)
            freq_maps[column] = fe # Sauvegarder le mapping

        # Encodage Label
        label_encoders = {} # Stocker les encodeurs fittés
        for column in categorical_cols_label:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])
            label_encoders[column] = le # Sauvegarder l'encodeur fitté

        # Encodage Cible
        encoder_y = LabelEncoder()
        df_encoded['y_encoded'] = encoder_y.fit_transform(df_encoded['y'])


        st.subheader("3. Préparation Finale et Suréchantillonnage")
        # Suppression des colonnes originales + cible 'y'
        cols_to_drop = categorical_cols_freq + categorical_cols_label + ['y']
        df_final = df_encoded.drop(columns=cols_to_drop)
        expected_features = df_final.drop(columns=['y_encoded']).columns.tolist() # Liste finale des features

        # Suréchantillonnage (Upsampling)
        if st.checkbox("Appliquer le suréchantillonnage (Upsampling)", key="cb_upsample_train", value=True):
             df_majority = df_final[df_final.y_encoded == 0]
             df_minority = df_final[df_final.y_encoded == 1]
             df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
             data_ready = pd.concat([df_majority, df_minority_upsampled])
             st.write("Suréchantillonnage appliqué.")
        else:
             data_ready = df_final
             st.write("Suréchantillonnage désactivé.")

        st.write("Distribution de la cible après préparation :")
        st.write(data_ready['y_encoded'].value_counts(normalize=True))

        X = data_ready[expected_features].values
        y = data_ready['y_encoded'].values


        st.subheader("4. Entraînement et Évaluation")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        with st.spinner("Entraînement du modèle RandomForest..."):
             model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
             model.fit(X_train, y_train)
        st.success("Modèle entraîné.")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=encoder_y.classes_)

        st.metric(label="Accuracy (Test Set)", value=f"{accuracy:.2%}")
        st.text('Rapport de Classification (Test Set):')
        st.text(report)


        st.subheader("5. Sauvegarde du Modèle (Localement)")
        save_filename = 'model_classification_bank.pkl'
        if st.button(f"Sauvegarder le modèle sous '{save_filename}'", key="btn_save_model"):
             try:
                # Sauvegarder le modèle, l'encodeur cible et la liste des features
                model_data_to_save = {
                    'model': model,
                    'encoder_y': encoder_y,
                    'features': expected_features
                    # Optionnel mais recommandé: sauvegarder aussi freq_maps et label_encoders
                    # 'freq_maps': freq_maps,
                    # 'label_encoders': label_encoders
                }
                with open(save_filename, 'wb') as f:
                    pickle.dump(model_data_to_save, f)
                st.success(f"Modèle sauvegardé avec succès dans '{save_filename}'.")
                st.info("Vous pouvez maintenant utiliser ce fichier (ou le téléverser sur GitHub) pour la section Prédiction.")
             except Exception as e:
                st.error(f"Erreur lors de la sauvegarde du modèle : {e}")


    # -------------------- SECTION PRÉDICTION --------------------
    elif st.session_state.page_selection == 'prediction':
        st.title("🔮 Prédiction de Souscription Client")

        # --- Charger le modèle, encodeur_y, expected_features ---
        model, encoder_y, expected_features = None, None, None

        # Option 1: Charger depuis GitHub (méthode recommandée si le modèle est stable)
        if st.toggle("Charger le modèle depuis GitHub", value=True, key="load_github"):
             model, encoder_y, expected_features = load_model_from_github(MODEL_URL_GITHUB)
        else:
        # Option 2: Charger depuis le fichier local (si créé par la section entraînement)
             local_filename = 'model_classification_bank.pkl'
             try:
                 with open(local_filename, 'rb') as f:
                     loaded_data = pickle.load(f)
                     model = loaded_data.get('model')
                     encoder_y = loaded_data.get('encoder_y')
                     expected_features = loaded_data.get('features')
                     if not all([model, encoder_y, expected_features]):
                          st.error(f"Le fichier local '{local_filename}' ne contient pas les clés requises.")
                          model, encoder_y, expected_features = None, None, None
                     else:
                          st.success(f"Modèle chargé depuis le fichier local '{local_filename}'.")
             except FileNotFoundError:
                 st.error(f"Fichier local '{local_filename}' introuvable. Entraînez le modèle ou chargez depuis GitHub.")
             except Exception as e:
                 st.error(f"Erreur lors du chargement depuis le fichier local: {e}")

        # Arrêter si le chargement a échoué
        if model is None or encoder_y is None or expected_features is None:
            st.warning("Le modèle n'a pas pu être chargé. Impossible de faire des prédictions.")
            st.stop()

        # --- Formulaire de saisie utilisateur ---
        st.markdown("Entrez les informations du client :")
        with st.form(key='prediction_form'):
            st.subheader("Infos Client")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Âge", 18, 100, 40, 1, key="pred_age")
                job = st.selectbox("Métier", options=df_original['job'].unique(), key="pred_job")
                marital = st.selectbox("Statut Marital", options=df_original['marital'].unique(), key="pred_marital")
                education = st.selectbox("Éducation", options=df_original['education'].unique(), key="pred_education")
                default = st.selectbox("Défaut Crédit?", options=df_original['default'].unique(), format_func=lambda x: 'Oui' if x=='yes' else ('Non' if x=='no' else x) , key="pred_default")
                housing = st.selectbox("Prêt Immo?", options=df_original['housing'].unique(), format_func=lambda x: 'Oui' if x=='yes' else ('Non' if x=='no' else x), key="pred_housing")
                loan = st.selectbox("Prêt Perso?", options=df_original['loan'].unique(), format_func=lambda x: 'Oui' if x=='yes' else ('Non' if x=='no' else x), key="pred_loan")

            with col2:
                contact = st.selectbox("Type Contact", options=df_original['contact'].unique(), key="pred_contact")
                month = st.selectbox("Mois Dernier Contact", options=df_original['month'].unique(), key="pred_month")
                day_of_week = st.selectbox("Jour Semaine Dernier Contact", options=df_original['day_of_week'].unique(), key="pred_day")
                duration = st.number_input("Durée Dernier Contact (sec)", 0, 6000, 120, 10, key="pred_duration", help="Attention: Valeur connue après l'appel.")
                campaign = st.number_input("Nb Contacts Campagne", 1, 100, 2, 1, key="pred_campaign")
                pdays = st.number_input("Jours Depuis Dernier Contact (Préc.)", -1, 999, 999, 1, key="pred_pdays", help="-1 ou 999 = Jamais contacté")
                previous = st.number_input("Nb Contacts Avant Campagne", 0, 100, 0, 1, key="pred_previous")
                poutcome = st.selectbox("Résultat Préc. Campagne", options=df_original['poutcome'].unique(), key="pred_poutcome")

            st.subheader("Indicateurs Économiques")
            col_eco1, col_eco2, col_eco3 = st.columns(3)
            with col_eco1: emp_var_rate = st.number_input("Taux Var. Emploi", -5.0, 5.0, 1.1, 0.1, format="%.1f", key="pred_emp_var")
            with col_eco2: cons_price_idx = st.number_input("Indice Prix Conso.", 90.0, 95.0, 93.9, 0.1, format="%.1f", key="pred_cons_price")
            with col_eco2: cons_conf_idx = st.number_input("Indice Conf. Conso.", -55.0, -25.0, -40.0, 0.1, format="%.1f", key="pred_cons_conf")
            with col_eco3: euribor3m = st.number_input("Taux Euribor 3 Mois", 0.5, 5.5, 4.8, 0.1, format="%.3f", key="pred_euribor")
            with col_eco3: nr_employed = st.number_input("Nb Employés (milliers)", 4800.0, 5300.0, 5190.0, 10.0, format="%.1f", key="pred_nr_emp")

            submitted = st.form_submit_button("🔮 Obtenir la Prédiction")

            if submitted:
                # --- Créer DataFrame pour l'entrée utilisateur ---
                input_data = {
                    'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
                    'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
                    'duration': duration, 'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
                    'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                    'euribor3m': euribor3m, 'nr.employed': nr_employed
                }
                input_df = pd.DataFrame([input_data])
                st.write("Données saisies :", input_df) # Debug

                # --- Appliquer le PRÉTRAITEMENT EXACTEMENT comme à l'entraînement ---
                df_processed_input = input_df.copy()

                # 1. Encodage par fréquence (en utilisant df_original pour les fréquences)
                categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
                try:
                    if df_original is None: raise ValueError("df_original n'est pas chargé pour l'encodage par fréquence.")
                    for col in categorical_cols_freq:
                        freq_map = df_original.groupby(col).size() / len(df_original)
                        df_processed_input[f'{col}_freq_encode'] = df_processed_input[col].map(freq_map).fillna(0)
                except Exception as e:
                    st.error(f"Erreur pendant l'encodage par fréquence: {e}")
                    st.stop()

                # 2. Encodage Label (en fittant sur les valeurs connues de df_original)
                categorical_cols_label = ['default', 'housing', 'loan', 'contact']
                try:
                    if df_original is None: raise ValueError("df_original n'est pas chargé pour l'encodage Label.")
                    for col in categorical_cols_label:
                        le = LabelEncoder()
                        le.fit(df_original[col].unique()) # Fit sur les valeurs possibles
                        df_processed_input[col] = le.transform(df_processed_input[col])
                except Exception as e:
                    st.error(f"Erreur pendant l'encodage Label: {e}")
                    st.stop()

                # 3. Supprimer les colonnes catégorielles originales
                cols_to_drop_input = categorical_cols_freq + categorical_cols_label
                df_processed_input = df_processed_input.drop(columns=cols_to_drop_input, errors='ignore')

                # 4. S'assurer que les colonnes sont dans le bon ordre et complètes
                try:
                    input_final = df_processed_input.reindex(columns=expected_features, fill_value=0)
                    st.write("Données après prétraitement (prêtes pour modèle) :", input_final) # Debug
                except Exception as e:
                    st.error(f"Erreur lors de l'alignement des colonnes finales: {e}")
                    st.stop()

                # --- Faire la Prédiction ---
                try:
                    prediction_proba = model.predict_proba(input_final)
                    prediction = model.predict(input_final)
                    probability_yes = prediction_proba[0][1]
                    result_label = encoder_y.inverse_transform(prediction)[0] # Utiliser l'encodeur chargé

                    # --- Afficher le résultat ---
                    st.subheader("Résultat")
                    if result_label == 'yes':
                        st.success(f"✅ Prédiction : Souscription Probable (Confiance: {probability_yes:.1%})")
                        st.balloons()
                    else:
                        st.warning(f"❌ Prédiction : Souscription Improbable (Confiance souscription: {probability_yes:.1%})")

                    #st.write(f"(Probabilité 'Non': {prediction_proba[0][0]:.1%}, Probabilité 'Oui': {probability_yes:.1%})")

                except Exception as e:
                    st.error(f"Erreur lors de l'exécution de la prédiction sur le modèle : {e}")
                    st.dataframe(input_final) # Afficher les données qui ont causé l'erreur


    # -------------------- SECTION CONCLUSION --------------------
    elif st.session_state.page_selection == 'conclusion':
        st.title("️ Conclusion")
        st.markdown("""
            Un traitement minutieux et un prétraitement cohérent des données sont essentiels pour construire un modèle de prédiction fiable.
            Cette application démontre les étapes clés, de l'exploration à la prédiction interactive, en utilisant Streamlit, Pandas et Scikit-learn.
            L'hébergement du modèle sur GitHub et son chargement dynamique rendent l'application plus portable.
            """)

# --- Exécution de l'application ---
if __name__ == '__main__':
    main()
