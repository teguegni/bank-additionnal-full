import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Pas directement utilisé pour les graphiques Streamlit ici
# import seaborn as sns # Pas directement utilisé pour les graphiques Streamlit ici
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pickle
import requests # Pour charger le modèle depuis GitHub
import io       # Pour lire les bytes du modèle chargé

# ==============================================================================
# Configuration de la Page Streamlit
# ==============================================================================
st.set_page_config(
    page_title="Classification Données Bancaires",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Fonctions Utilitaires (Mise en Cache)
# ==============================================================================

# --- Fonction pour ajouter l'arrière-plan ---
def add_bg_from_url(url):
    """Ajoute une image d'arrière-plan à partir d'une URL brute GitHub."""
    # Correction: Utiliser le sélecteur .stApp qui est plus fiable
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-attachment: fixed;
             background-size: cover;
             background-repeat: no-repeat;
         }}
         /* Optionnel: Améliorer la lisibilité */
         /* [data-testid="stSidebar"], .main .block-container {{
             background-color: rgba(255, 255, 255, 0.85);
         }} */
         </style>
         """,
         unsafe_allow_html=True
     )

# --- Fonction pour charger les données (avec cache) ---
@st.cache_data
def load_data():
    """Charge les données depuis un fichier local avec fallback sur GitHub."""
    data_url_local = 'bank-additional-full.csv'
    github_data_url = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/bank-additional-full.csv"
    try:
        df = pd.read_csv(data_url_local, delimiter=';')
        # st.success(f"Données chargées depuis '{data_url_local}'.") # Optionnel
        return df
    except FileNotFoundError:
        st.warning(f"Fichier '{data_url_local}' non trouvé. Tentative de chargement depuis GitHub...")
        try:
            df = pd.read_csv(github_data_url, delimiter=';')
            # st.success("Données chargées depuis GitHub.") # Optionnel
            return df
        except Exception as e:
            st.error(f"Échec du chargement des données depuis le fichier local et GitHub. Erreur : {e}")
            return None

# --- Fonction pour charger le modèle depuis GitHub (avec cache) ---
@st.cache_resource # Utiliser cache_resource pour les modèles/objets lourds
def load_model_from_github(url):
    """Charge un dictionnaire contenant modèle, encodeur, features depuis une URL GitHub."""
    try:
        st.info(f"Chargement du modèle depuis GitHub...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        # Utiliser io.BytesIO pour lire les bytes avec pickle.load
        model_data = pickle.load(io.BytesIO(response.content))
        required_keys = ['model', 'encoder_y', 'features']
        if not all(key in model_data for key in required_keys):
            st.error(f"Le fichier pickle chargé ne contient pas les clés requises: {required_keys}")
            return None, None, None
        st.success("Modèle chargé avec succès depuis GitHub.")
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

# ==============================================================================
# Initialisation Globale
# ==============================================================================

# --- Appliquer l'arrière-plan ---
# Correction: Utiliser l'URL brute de l'image
IMAGE_URL_RAW = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/image1.jpg"
add_bg_from_url(IMAGE_URL_RAW)

# --- Charger les données ---
df_original = load_data()

# --- URL du modèle sur GitHub (Vérifiez ce chemin!) ---
MODEL_URL_GITHUB = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/model_classification_bank.pkl"

# --- Définir le thème Altair ---
# alt.themes.enable("dark") # Ajustez si besoin

# ==============================================================================
# Barre Latérale et Navigation
# ==============================================================================

# --- Initialiser l'état de la session pour la navigation ---
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'a_propos' # Page par défaut

# --- Barre latérale ---
with st.sidebar:
    st.title('🏦 Classification Données Bancaires')
    st.subheader("Navigation")

    # Options de page
    pages = ["À Propos", "Jeu de Données", "Analyse Exploratoire", "Apprentissage Automatique", "Prédiction", "Conclusion"]
    page_keys = ['a_propos', 'jeu_de_donnees', 'analyse_exploratoire', 'apprentissage_automatique', 'prediction', 'conclusion']

    # Trouver l'index de la page actuelle pour st.radio
    try:
        current_index = page_keys.index(st.session_state.page_selection)
    except ValueError:
        current_index = 0 # Par défaut à "À Propos" si l'état est invalide

    # Widget Radio pour la navigation
    selected_page_title = st.radio(
        "Choisissez une section :",
        pages,
        index=current_index,
        key="nav_radio"
    )

    # Mettre à jour l'état de la session en fonction de la sélection radio
    if selected_page_title:
        st.session_state.page_selection = page_keys[pages.index(selected_page_title)]

    # Informations supplémentaires
    st.divider()
    st.subheader("Résumé")
    st.markdown("""
        Tableau de bord interactif pour explorer et classifier les données marketing bancaires.
        - [Dépôt GitHub](https://github.com/teguegni/bank-additionnal-full)
        Auteur : Kenfack Teguegni Junior
    """)

# ==============================================================================
# Fonction Principale (Contenu des Pages)
# ==============================================================================
def main():
    # Arrêter si les données n'ont pas pu être chargées
    if df_original is None:
        st.error("Impossible d'afficher les pages car les données n'ont pas pu être chargées.")
        st.stop()

    # -------------------- SECTION À PROPOS --------------------
    if st.session_state.page_selection == 'a_propos':
        st.title("️ À Propos")
        # Conteneur pour améliorer la lisibilité sur l'image de fond
        with st.container():
             st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px;">
                Cette application explore le jeu de données Bank Marketing et propose :
                <ul>
                    <li>Une exploration visuelle des données.</li>
                    <li>Un prétraitement et nettoyage des données.</li>
                    <li>La construction et l'évaluation de modèles d'apprentissage automatique.</li>
                    <li>Une interface interactive pour prédire si un client souscrira à un produit.</li>
                </ul>
                <p><strong>Technologies utilisées :</strong> Python (Streamlit, Pandas, Scikit-learn), Machine Learning.</p>
                <p><strong>Auteur :</strong> Kenfack Teguegni Junior | ✉️ <strong>Contact :</strong> kenfackteguegni@gmail.com</p>
                </div>
                """, unsafe_allow_html=True)


    # -------------------- SECTION JEU DE DONNÉES --------------------
    elif st.session_state.page_selection == 'jeu_de_donnees':
        st.title("📊 Jeu de Données")
        with st.container(): # Améliore la lisibilité
             st.markdown("""
                 <div style="background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px;">
                 Aperçu et statistiques descriptives du jeu de données brut chargé.
                 </div>
                 """, unsafe_allow_html=True)
             st.dataframe(df_original.head())
             st.write(f"**Dimensions :** {df_original.shape[0]} lignes, {df_original.shape[1]} colonnes")

             if st.checkbox("Afficher les statistiques descriptives", key="chk_stats_desc"):
                st.subheader("Statistiques Numériques")
                st.write(df_original.describe(include=np.number))
                st.subheader("Statistiques Catégorielles")
                st.write(df_original.describe(include='object'))


    # -------------------- SECTION ANALYSE EXPLORATOIRE --------------------
    elif st.session_state.page_selection == 'analyse_exploratoire':
        st.title("🔍 Analyse Exploratoire et Prétraitement Simple")
        with st.container():
             st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px;">
                Analyse initiale des données, incluant la gestion des doublons et des valeurs 'unknown'.
                </div>
                """, unsafe_allow_html=True)

             # Travailler sur une copie
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
             st.write("Remplacement des 'unknown' par le mode (valeur la plus fréquente) de chaque colonne.")
             unknown_replaced_count = 0
             cols_with_unknown = []
             for column in df_processed.select_dtypes(include='object').columns:
                 if 'unknown' in df_processed[column].unique():
                    mode_val = df_processed[column].mode()[0]
                    if mode_val == 'unknown':
                       modes = df_processed[column].mode(); mode_val = modes[1] if len(modes) > 1 else None
                    if mode_val:
                       count = (df_processed[column] == 'unknown').sum()
                       if count > 0:
                           df_processed[column] = df_processed[column].replace('unknown', mode_val)
                           unknown_replaced_count += count
                           cols_with_unknown.append(f"{column} ({count})")
             if unknown_replaced_count > 0:
                 st.write(f"{unknown_replaced_count} valeurs 'unknown' remplacées dans les colonnes : {', '.join(cols_with_unknown)}")
             else:
                 st.write("Aucune valeur 'unknown' trouvée ou remplacée.")

             st.subheader("3. Vérification des Valeurs Manquantes (Null)")
             missing_values = df_processed.isnull().sum()
             missing_data = missing_values[missing_values > 0]
             if not missing_data.empty: st.write("Valeurs manquantes (Null) restantes :", missing_data)
             else: st.write("Aucune valeur manquante (Null) détectée.")

             st.subheader("4. Visualisation : Relation Âge / Métier / Souscription")
             df_chart = df_processed.copy()
             df_chart['Souscription'] = df_chart['y'].map({'yes': 'Oui', 'no': 'Non'})
             df_chart['job'] = df_chart['job'].astype(str)

             age_job_chart = alt.Chart(df_chart).mark_circle(size=60, opacity=0.7).encode(
                    x=alt.X('age', title='Âge', scale=alt.Scale(zero=False)),
                    y=alt.Y('job', title='Métier', sort='-x'),
                    color=alt.Color('Souscription', title='Souscrit ?'),
                    tooltip=['age', 'job', 'Souscription']
                ).properties(title="Relation Âge / Métier colorée par la Souscription", height=500).interactive()
             st.altair_chart(age_job_chart, use_container_width=True)


    # -------------------- SECTION APPRENTISSAGE AUTOMATIQUE --------------------
    elif st.session_state.page_selection == 'apprentissage_automatique':
        st.title("⚙️ Apprentissage Automatique (Entraînement)")
        with st.container():
            st.markdown("""
               <div style="background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px;">
               Cette section effectue le prétraitement complet, entraîne un modèle RandomForest et sauvegarde un fichier <code>model_classification_bank.pkl</code> localement.
               <br><strong>Note:</strong> Exécutez cette section si vous souhaitez utiliser le fichier local pour la prédiction.
               </div>
               """, unsafe_allow_html=True)

            # Utiliser une copie pour l'entraînement
            df_ml = df_original.copy()

            # Prétraitements initiaux (cohérence)
            df_ml = df_ml.drop_duplicates()
            for column in df_ml.select_dtypes(include='object').columns:
                 if 'unknown' in df_ml[column].unique():
                    mode_val = df_ml[column].mode()[0]
                    if mode_val == 'unknown': modes = df_ml[column].mode(); mode_val = modes[1] if len(modes) > 1 else None
                    if mode_val: df_ml[column] = df_ml[column].replace('unknown', mode_val)

            st.subheader("1. Traitement Optionnel des Outliers")
            numerics = df_ml.select_dtypes(include=np.number).columns.tolist()
            if st.checkbox("Remplacer les outliers par les limites IQR", key="cb_outliers_train", value=False):
                df_ml_out = df_ml.copy()
                for col in numerics:
                    Q1 = df_ml_out[col].quantile(0.25); Q3 = df_ml_out[col].quantile(0.75); IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
                    df_ml_out[col] = np.clip(df_ml_out[col], lower, upper)
                df_ml = df_ml_out
                st.write("Outliers traités.")

            st.subheader("2. Encodage des Variables")
            df_encoded = df_ml.copy()
            # Définir les listes de colonnes ici
            categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
            categorical_cols_label = ['default', 'housing', 'loan', 'contact']

            freq_maps = {} # Stocker les mappings
            label_encoders = {} # Stocker les encodeurs fittés

            with st.spinner("Encodage en cours..."):
                 # Encodage par fréquence
                 for column in categorical_cols_freq:
                     fe = df_encoded.groupby(column).size() / len(df_encoded)
                     df_encoded[f'{column}_freq_encode'] = df_encoded[column].map(fe)
                     freq_maps[column] = fe

                 # Encodage Label
                 for column in categorical_cols_label:
                     le = LabelEncoder()
                     df_encoded[column] = le.fit_transform(df_encoded[column])
                     label_encoders[column] = le

                 # Encodage Cible
                 encoder_y = LabelEncoder()
                 df_encoded['y_encoded'] = encoder_y.fit_transform(df_encoded['y'])
            st.success("Encodage terminé.")


            st.subheader("3. Préparation Finale et Suréchantillonnage Optionnel")
            cols_to_drop = categorical_cols_freq + categorical_cols_label + ['y']
            df_final = df_encoded.drop(columns=cols_to_drop, errors='ignore') # errors='ignore' si une colonne manque déjà
            expected_features = df_final.drop(columns=['y_encoded']).columns.tolist()

            if st.checkbox("Appliquer le suréchantillonnage (Upsampling)", key="cb_upsample_train", value=True):
                 df_majority = df_final[df_final.y_encoded == 0]
                 df_minority = df_final[df_final.y_encoded == 1]
                 if len(df_minority) > 0: # Vérifier qu'il y a une classe minoritaire
                     df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
                     data_ready = pd.concat([df_majority, df_minority_upsampled])
                     st.write("Suréchantillonnage appliqué.")
                 else:
                     st.warning("Classe minoritaire vide, impossible de suréchantillonner.")
                     data_ready = df_final
            else:
                 data_ready = df_final
                 st.write("Suréchantillonnage désactivé.")

            st.write("**Distribution de la cible pour l'entraînement :**")
            st.write(data_ready['y_encoded'].value_counts(normalize=True))

            # Préparer les données pour Sklearn
            X = data_ready[expected_features].values
            y = data_ready['y_encoded'].values


            st.subheader("4. Entraînement et Évaluation du Modèle")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # test_size=0.25

            # Paramètres du modèle (peuvent être rendus interactifs)
            n_estimators = st.slider("Nombre d'arbres (n_estimators)", 50, 300, 100, 50, key="train_n_estimators")
            max_depth_options = [None, 10, 20, 30]
            max_depth = st.select_slider("Profondeur max des arbres (max_depth)", options=max_depth_options, value=None, key="train_max_depth")


            with st.spinner(f"Entraînement du RandomForest ({n_estimators} arbres)..."):
                 model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1, class_weight='balanced_subsample') # Ajouter poids
                 model.fit(X_train, y_train)
            st.success("Modèle entraîné.")

            # Évaluation
            y_pred_test = model.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            report_test = classification_report(y_test, y_pred_test, target_names=encoder_y.classes_)

            st.metric(label="Accuracy (Test Set)", value=f"{accuracy_test:.2%}")
            st.text('Rapport de Classification (Test Set):')
            st.text(report_test)


            st.subheader("5. Sauvegarde du Modèle Complet (Localement)")
            save_filename = 'model_classification_bank.pkl'
            model_data_to_save = {
                    'model': model,
                    'encoder_y': encoder_y,
                    'features': expected_features,
                    # Optionnel: Sauvegarder les mappings pour un encodage plus robuste en prédiction
                    'freq_maps': freq_maps,
                    'label_encoders': label_encoders
                }
            if st.button(f"💾 Sauvegarder le Modèle sous '{save_filename}'", key="btn_save_model"):
                 try:
                    with open(save_filename, 'wb') as f:
                        pickle.dump(model_data_to_save, f)
                    st.success(f"Modèle sauvegardé avec succès dans '{save_filename}'.")
                 except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde du modèle : {e}")


    # -------------------- SECTION PRÉDICTION --------------------
    elif st.session_state.page_selection == 'prediction':
        st.title("🔮 Prédiction de Souscription Client")
        with st.container():
            st.markdown("""
               <div style="background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px;">
               Utilisez le formulaire ci-dessous pour obtenir une prédiction basée sur le modèle entraîné. Le modèle peut être chargé depuis GitHub ou un fichier local.
               </div>
               """, unsafe_allow_html=True)

            # --- Charger le modèle, encodeur_y, expected_features ---
            model, encoder_y, expected_features = None, None, None
            # Optionnel: Charger les mappings/encodeurs s'ils ont été sauvegardés
            freq_maps_loaded = None
            label_encoders_loaded = None

            if st.toggle("Charger le modèle depuis GitHub", value=True, key="load_github_pred"):
                 model, encoder_y, expected_features = load_model_from_github(MODEL_URL_GITHUB)
                 # Essayer de charger les mappings aussi s'ils sont dans le pickle (nécessite modification de load_model_from_github)
            else:
                 local_filename = 'model_classification_bank.pkl'
                 try:
                     with open(local_filename, 'rb') as f:
                         loaded_data = pickle.load(f)
                         model = loaded_data.get('model')
                         encoder_y = loaded_data.get('encoder_y')
                         expected_features = loaded_data.get('features')
                         freq_maps_loaded = loaded_data.get('freq_maps') # Optionnel
                         label_encoders_loaded = loaded_data.get('label_encoders') # Optionnel

                         if not all([model, encoder_y, expected_features]):
                              st.error(f"Le fichier local '{local_filename}' ne contient pas les clés requises.")
                              model, encoder_y, expected_features = None, None, None
                         else:
                              st.success(f"Modèle chargé depuis '{local_filename}'.")
                 except FileNotFoundError: st.error(f"'{local_filename}' introuvable.")
                 except Exception as e: st.error(f"Erreur chargement local: {e}")

            # Arrêter si le chargement a échoué
            if model is None: st.stop()

            # Vérifier la disponibilité de df_original (nécessaire pour l'encodage si mappings non chargés)
            if (freq_maps_loaded is None or label_encoders_loaded is None) and df_original is None:
                 st.error("Le DataFrame original est nécessaire pour l'encodage car les mappings/encodeurs n'ont pas été chargés depuis le fichier modèle. Impossible de continuer.")
                 st.stop()


            # --- Formulaire de saisie utilisateur ---
            with st.form(key='prediction_form'):
                st.subheader("Entrez les Informations")
                col1, col2 = st.columns(2)
                # Utilisation de df_original pour peupler les options des selectbox
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
                    duration = st.number_input("Durée Dernier Contact (sec)", 0, 6000, 120, 10, key="pred_duration")
                    campaign = st.number_input("Nb Contacts Campagne", 1, 100, 2, 1, key="pred_campaign")
                    pdays = st.number_input("Jours Depuis Dernier Contact (Préc.)", -1, 999, 999, 1, key="pred_pdays")
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
                    # --- Préparer l'entrée pour le prétraitement ---
                    input_data = pd.DataFrame([{
                        'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
                        'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
                        'duration': duration, 'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
                        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                        'euribor3m': euribor3m, 'nr.employed': nr_employed
                    }])
                    df_processed_input = input_data.copy()

                    # --- Appliquer le Prétraitement ---
                    # Définir les listes de colonnes (doit correspondre à l'entraînement)
                    categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
                    categorical_cols_label = ['default', 'housing', 'loan', 'contact']

                    try:
                        # 1. Encodage par fréquence
                        # Utiliser les mappings chargés si disponibles, sinon recalculer depuis df_original
                        st.write("Encodage Fréquence...")# Debug
                        for col in categorical_cols_freq:
                            if freq_maps_loaded and col in freq_maps_loaded:
                                freq_map = freq_maps_loaded[col]
                            else: # Recalculer si non chargé
                                if df_original is None: raise ValueError("df_original nécessaire pour recalculer les fréquences.")
                                freq_map = df_original.groupby(col).size() / len(df_original)
                            df_processed_input[f'{col}_freq_encode'] = df_processed_input[col].map(freq_map).fillna(0)

                        # 2. Encodage Label
                        # Utiliser les encodeurs chargés si disponibles, sinon recréer et fitter depuis df_original
                        st.write("Encodage Label...")# Debug
                        for col in categorical_cols_label:
                            if label_encoders_loaded and col in label_encoders_loaded:
                                le = label_encoders_loaded[col]
                            else: # Recréer si non chargé
                                if df_original is None: raise ValueError("df_original nécessaire pour re-fitter LabelEncoder.")
                                le = LabelEncoder()
                                le.fit(df_original[col].unique())
                            # Utiliser try-except pour gérer les valeurs inconnues pendant transform
                            try:
                                df_processed_input[col] = le.transform(df_processed_input[col])
                            except ValueError as ve:
                                st.warning(f"Valeur inconnue '{df_processed_input[col].iloc[0]}' pour la colonne '{col}' lors de l'encodage Label. Assignation de -1 (ou autre stratégie). Erreur: {ve}")
                                # Gérer la valeur inconnue, ex: assigner une valeur spéciale ou la plus fréquente encodée
                                # Pour l'instant, on peut assigner 0 ou -1 si l'encodeur n'accepte pas les inconnues
                                df_processed_input[col] = -1 # Ou une autre valeur par défaut gérée par le modèle

                        # 3. Supprimer colonnes originales
                        cols_to_drop_input = categorical_cols_freq + categorical_cols_label
                        df_processed_input = df_processed_input.drop(columns=cols_to_drop_input, errors='ignore')

                        # 4. Réindexer pour correspondre aux features attendues
                        st.write("Alignement des features...")# Debug
                        input_final = df_processed_input.reindex(columns=expected_features, fill_value=0)
                        # st.dataframe(input_final) # Debug: Vérifier les données finales

                    except Exception as e:
                        st.error(f"Erreur lors du prétraitement des données d'entrée: {e}")
                        st.stop()


                    # --- Faire la Prédiction ---
                    try:
                        st.write("Prédiction...")# Debug
                        prediction_proba = model.predict_proba(input_final)
                        prediction = model.predict(input_final)
                        probability_yes = prediction_proba[0][1]
                        result_label = encoder_y.inverse_transform(prediction)[0]

                        # --- Afficher le résultat ---
                        st.subheader("Résultat de la Prédiction")
                        if result_label == 'yes':
                            st.success(f"✅ **Souscription Probable** (Confiance: {probability_yes:.1%})")
                            st.balloons()
                        else:
                            st.warning(f"❌ **Souscription Improbable** (Confiance souscription: {probability_yes:.1%})")

                    except Exception as e:
                        st.error(f"Erreur lors de l'exécution de la prédiction sur le modèle : {e}")
                        st.dataframe(input_final) # Afficher les données en cas d'erreur

    # -------------------- SECTION CONCLUSION --------------------
    elif st.session_state.page_selection == 'conclusion':
        st.title("🏁 Conclusion")
        with st.container():
             st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px;">
                Un traitement et un prétraitement cohérents sont cruciaux pour des prédictions fiables.
                Cette application illustre un flux complet, de l'analyse à la prédiction interactive.
                L'utilisation de Streamlit facilite la création d'interfaces pour les modèles de Machine Learning.
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# Point d'Entrée de l'Application
# ==============================================================================
if __name__ == '__main__':
    main()
