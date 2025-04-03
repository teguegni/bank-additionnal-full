import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Non utilisé directement dans ce code, peut être enlevé si inutile ailleurs
# import seaborn as sns # Non utilisé directement dans ce code, peut être enlevé si inutile ailleurs
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pickle
import base64 # Utile pour une alternative d'encodage si l'URL pose problème

# --- Configuration de la page ---
st.set_page_config(
    page_title="Classification des Données Bancaires",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fonction pour ajouter l'arrière-plan ---
def add_bg_from_url(url):
    """Ajoute une image d'arrière-plan à partir d'une URL."""
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-attachment: fixed;
             background-size: cover;
             background-repeat: no-repeat;
         }}
         /* Ajout d'un peu de transparence aux éléments principaux pour mieux voir le fond */
         /* [data-testid="stSidebar"], [data-testid="stHeader"], .main .block-container {{
             background-color: rgba(255, 255, 255, 0.8); /* Blanc avec 80% opacité */
             /* Vous pouvez ajuster la couleur et l'opacité */
         /* }} */
         /* Optionnel: Style pour rendre le texte plus lisible sur l'image */
         /* body, .stMarkdown, .stButton > button, .stTextInput > div > div > input, .stNumberInput > div > div > input {{
             color: #FFFFFF; /* Texte blanc */
             /* text-shadow: 1px 1px 2px black; /* Ombre portée pour lisibilité */
         /* }} */

         </style>
         """,
         unsafe_allow_html=True
     )

# --- URL de l'image brute sur GitHub ---
image_url = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/image1.jpg"
add_bg_from_url(image_url)


# --- Thème Altair ---
# alt.themes.enable("dark") # Peut entrer en conflit avec un fond clair, à ajuster si besoin

# --- Barre latérale ---
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'a_propos'  # Page par défaut

# Fonction pour mettre à jour page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:
    st.title('🏦 Classification Données Bancaires')
    # Navigation par boutons
    st.subheader("Sections")
    st.button("À Propos", use_container_width=True, on_click=set_page_selection, args=('a_propos',), key="btn_a_propos")
    st.button("Jeu de Données", use_container_width=True, on_click=set_page_selection, args=('jeu_de_donnees',), key="btn_jeu_de_donnees")
    st.button("Analyse Exploratoire", use_container_width=True, on_click=set_page_selection, args=('analyse_exploratoire',), key="btn_analyse_exploratoire")
    # Correction: Ajout du bouton pour la section 'apprentissage_automatique' si elle existe
    st.button("Apprentissage Automatique", use_container_width=True, on_click=set_page_selection, args=('apprentissage_automatique',), key="btn_apprentissage")
    st.button("Prédiction", use_container_width=True, on_click=set_page_selection, args=('prediction',), key="btn_prediction")
    st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',), key="btn_conclusion")

    # Détails du projet
    st.subheader("Résumé")
    st.markdown("""
        Un tableau de bord interactif pour explorer et classifier les données d'une campagne marketing bancaire.
        - [Jeu de Données](URL_VERS_VOTRE_JEU_DE_DONNEES) - [Notebook Google Colab](URL_VERS_VOTRE_NOTEBOOK) - [Dépôt GitHub](https://github.com/teguegni/bank-additionnal-full) Auteur : Kenfack Teguegni Junior
    """)

# --- Chargement des données ---
# Utiliser un chemin relatif ou absolu correct, ou s'assurer que le fichier est dans le même dossier
DATA_URL = 'bank-additional-full.csv'
try:
    df_original = pd.read_csv(DATA_URL, delimiter=';')
    # Créer une copie pour les manipulations afin de ne pas modifier l'original à chaque rechargement
    df = df_original.copy()
except FileNotFoundError:
    st.error(f"Le fichier '{DATA_URL}' est introuvable. Assurez-vous qu'il se trouve dans le bon répertoire.")
    # Essayer de charger depuis l'URL brute du dépôt GitHub comme alternative
    try:
        st.warning("Tentative de chargement du fichier depuis GitHub...")
        github_data_url = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/bank-additional-full.csv"
        df_original = pd.read_csv(github_data_url, delimiter=';')
        df = df_original.copy()
        st.success("Chargement depuis GitHub réussi.")
    except Exception as e:
        st.error(f"Échec du chargement depuis GitHub également. Erreur : {e}")
        st.stop() # Arrêter l'exécution si les données ne peuvent pas être chargées


# --- Fonction principale et logique des pages ---
def main():
    # Page principale
    if st.session_state.page_selection == 'a_propos':
        st.title("️ À Propos")
        st.markdown("""
            Cette application explore le jeu de données Bank Marketing et propose :
            - Une exploration visuelle des données.
            - Un prétraitement et nettoyage des données.
            - La construction et l'évaluation de modèles d'apprentissage automatique.
            - Une interface interactive pour prédire si un client souscrira à un produit.

            **Technologies utilisées :**
            - Python (Streamlit, Altair, Pandas, Scikit-learn)
            - Machine Learning (Random Forest)

            **Auteur :** Kenfack Teguegni Junior
            ✉️ **Contact :** kenfackteguegni@gmail.com
            """)

    elif st.session_state.page_selection == 'conclusion':
        st.title("️ Conclusion")
        st.markdown("""
            Un traitement minutieux et réfléchi du DataFrame `bank-additional-full.csv` est fondamental pour maximiser la précision
            et la fiabilité du modèle de prédiction. En combinant explorations, prétraitements adéquats (gestion des 'unknown', encodage, traitement des outliers, rééquilibrage), et évaluations rigoureuses,
            un modèle robuste peut être développé pour mieux prédire les comportements des clients envers la souscription à un produit.

            Ce tableau de bord Streamlit fournit une interface interactive pour visualiser les données, comprendre les étapes de prétraitement, et tester le modèle de prédiction final.
            """)

    elif st.session_state.page_selection == 'jeu_de_donnees':
        st.title(" Jeu de Données")
        st.markdown("Aperçu du jeu de données brut chargé.")
        # Afficher les premières lignes du DataFrame original
        if st.checkbox("Afficher le DataFrame brut", key="chk_afficher_df_brut", value=True):
            nb_rows = st.slider("Nombre de lignes à afficher :", min_value=5, max_value=50, value=10, key="slider_nb_rows_brut")
            st.dataframe(df_original.head(nb_rows)) # Utiliser df_original ici

        st.markdown("---")
        st.markdown("Informations sur le DataFrame :")
        st.write(f"Nombre de lignes : {df_original.shape[0]}")
        st.write(f"Nombre de colonnes : {df_original.shape[1]}")

        # Afficher les statistiques descriptives
        if st.checkbox("Afficher les statistiques descriptives (Numériques)", key="chk_stats_desc_num"):
            st.write(df_original.describe(include=np.number))
        if st.checkbox("Afficher les statistiques descriptives (Catégorielles)", key="chk_stats_desc_cat"):
             st.write(df_original.describe(include='object'))


    elif st.session_state.page_selection == 'analyse_exploratoire':
        st.title(" Analyse Exploratoire et Prétraitement")

        # Utiliser la copie df pour les modifications
        df_processed = df.copy()

        st.subheader("1. Gestion des Duplications")
        duplicates_before = df_processed.duplicated().sum()
        st.write(f"Nombre de lignes dupliquées avant suppression : {duplicates_before}")
        if duplicates_before > 0:
            df_processed = df_processed.drop_duplicates()
            st.write(f"Nombre de lignes après suppression des duplications : {df_processed.shape[0]}")
        else:
            st.write("Aucune ligne dupliquée trouvée.")

        st.subheader("2. Gestion des Valeurs 'unknown'")
        st.write("Remplacement des valeurs 'unknown' par le mode (valeur la plus fréquente) de chaque colonne catégorielle.")
        unknown_counts = {}
        for column in df_processed.select_dtypes(include='object').columns:
            if 'unknown' in df_processed[column].unique():
                unknown_counts[column] = df_processed[column].value_counts().get('unknown', 0)
                mode_value = df_processed[column].mode()[0]
                if mode_value == 'unknown': # Si le mode lui-même est 'unknown', choisir le second mode si possible
                   modes = df_processed[column].mode()
                   if len(modes) > 1:
                       mode_value = modes[1]
                   else: # Cas très improbable où 'unknown' est la seule valeur
                       st.warning(f"La seule valeur dans '{column}' est 'unknown'. Impossible de remplacer.")
                       continue # Passer à la colonne suivante
                df_processed[column] = df_processed[column].replace('unknown', mode_value)
        if unknown_counts:
             st.write("Nombre de valeurs 'unknown' remplacées par colonne :")
             st.write(unknown_counts)
        else:
             st.write("Aucune valeur 'unknown' trouvée dans les colonnes objectives.")


        st.subheader("3. Vérification des Valeurs Manquantes (après remplacement des 'unknown')")
        missing_values = df_processed.isnull().sum()
        missing_data = missing_values[missing_values > 0]
        if not missing_data.empty:
            st.write("Colonnes avec des valeurs manquantes restantes :")
            st.write(missing_data)
            # Ici, vous pourriez ajouter une stratégie pour imputer ces valeurs si nécessaire
        else:
            st.write("Aucune valeur manquante (null) détectée après le traitement des 'unknown'.")


        st.subheader("4. Visualisation Exploratoire")
        st.write("Relation entre l'âge, le métier et la souscription (variable cible 'y').")

        # S'assurer que 'job' est de type object ou category pour le graphique
        df_processed['job'] = df_processed['job'].astype('category')

        # Utiliser une copie pour éviter les avertissements de modification
        df_chart = df_processed.copy()
        df_chart['Souscription'] = df_chart['y'].map({'yes': 'Oui', 'no': 'Non'}) # Rendre la légende plus claire


        age_job_chart = (
            alt.Chart(df_chart)
            .mark_circle(size=60, opacity=0.7) # Ajout d'opacité pour les points superposés
            .encode(
                x=alt.X('age', title='Âge'),
                y=alt.Y('job', title='Métier', sort='-x'), # Trier les métiers par exemple
                color=alt.Color('Souscription', title='Souscrit ?', scale=alt.Scale(scheme='category10')), # Utiliser la colonne renommée
                tooltip=[
                    alt.Tooltip('age', title='Âge'),
                    alt.Tooltip('job', title='Métier'),
                    alt.Tooltip('Souscription', title='Souscrit ?')
                    ]
            )
            .properties(
                title="Relation Âge / Métier colorée par la Souscription",
                # width=700, # Laisser Streamlit gérer la largeur avec use_container_width
                height=500
            )
            .interactive() # Permet le zoom et le déplacement
        )
        st.altair_chart(age_job_chart, use_container_width=True)

        # Sauvegarder l'état traité pour la page suivante si nécessaire
        # st.session_state.df_processed = df_processed # Décommenter si besoin


    elif st.session_state.page_selection == 'apprentissage_automatique':
        st.title("⚙️ Apprentissage Automatique")
        st.write("Préparation des données, entraînement et évaluation du modèle Random Forest.")

        # Récupérer le df traité de l'étape précédente ou le recalculer
        # Pour la simplicité ici, on refait les étapes de prétraitement minimales nécessaires
        # Assurez-vous que les étapes ici correspondent à celles de 'analyse_exploratoire'
        df_ml = df_original.copy() # Partir de l'original pour cette section isolée

        # 1. Doublons
        df_ml = df_ml.drop_duplicates()

        # 2. 'unknown'
        for column in df_ml.select_dtypes(include='object').columns:
             if 'unknown' in df_ml[column].unique():
                mode_value = df_ml[column].mode()[0]
                if mode_value == 'unknown':
                   modes = df_ml[column].mode()
                   if len(modes) > 1: mode_value = modes[1]
                   else: continue
                df_ml[column] = df_ml[column].replace('unknown', mode_value)

        # 3. Outliers (Optionnel, mais présent dans le code original)
        st.subheader("Traitement des Valeurs Aberrantes (Outliers)")
        st.write("Remplacement des outliers par les limites basées sur l'IQR (Interquartile Range).")
        numerics = df_ml.select_dtypes(include=np.number).columns.tolist()
        if st.checkbox("Activer le remplacement des outliers", key="cb_outliers", value=True):
            df_ml_outliers = df_ml.copy() # Travailler sur une copie
            outliers_replaced_count = {col: 0 for col in numerics}
            for col in numerics:
                Q1 = df_ml_outliers[col].quantile(0.25)
                Q3 = df_ml_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                original_values = df_ml_outliers[col].copy()
                df_ml_outliers[col] = np.where(df_ml_outliers[col] < lower_bound, lower_bound, df_ml_outliers[col])
                df_ml_outliers[col] = np.where(df_ml_outliers[col] > upper_bound, upper_bound, df_ml_outliers[col])

                # Compter combien de valeurs ont été modifiées
                outliers_replaced_count[col] = (original_values != df_ml_outliers[col]).sum()

            st.write("Nombre d'outliers remplacés par colonne numérique :")
            st.write({k: v for k, v in outliers_replaced_count.items() if v > 0})
            df_ml = df_ml_outliers # Utiliser la version traitée pour la suite
        else:
            st.write("Remplacement des outliers désactivé.")


        st.subheader("Encodage des Variables Catégorielles")
        # Encodage par fréquence pour certaines colonnes (comme dans le code original)
        st.write("Encodage par fréquence pour : 'marital', 'job', 'education', 'month', 'day_of_week', 'poutcome'")
        df_encoded = df_ml.copy()
        categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
        for column in categorical_cols_freq:
            fe = df_encoded.groupby(column).size() / len(df_encoded)
            df_encoded[f'{column}_freq_encode'] = df_encoded[column].map(fe)

        # Encodage Label pour les binaires/ordinales simples (comme dans le code original)
        st.write("Encodage Label (0/1) pour : 'default', 'housing', 'loan', 'contact'")
        le = LabelEncoder()
        categorical_cols_label = ['default', 'housing', 'loan', 'contact']
        for column in categorical_cols_label:
            # Gérer le cas où 'yes'/'no' sont présents mais peut-être pas d'autres valeurs
            try:
                 df_encoded[column] = le.fit_transform(df_encoded[column])
            except Exception as e:
                 st.error(f"Erreur d'encodage pour {column}: {e}")
                 st.write(f"Valeurs uniques dans {column}: {df_encoded[column].unique()}")


        # Encodage de la colonne cible 'y'
        st.write("Encodage Label (0/1) pour la variable cible 'y' ('no'=0, 'yes'=1)")
        encoder_y = LabelEncoder()
        # S'assurer que 'y' existe avant d'encoder
        if 'y' in df_encoded.columns:
            df_encoded['y_encoded'] = encoder_y.fit_transform(df_encoded['y'])
            target_mapping = dict(zip(encoder_y.classes_, encoder_y.transform(encoder_y.classes_)))
            st.write(f"Mapping de la cible : {target_mapping}")
        else:
            st.error("La colonne cible 'y' est manquante.")
            st.stop()

        # Suppression des colonnes catégorielles originales et de 'y'
        cols_to_drop = categorical_cols_freq + categorical_cols_label + ['y']
        # Vérifier que les colonnes à supprimer existent bien
        cols_to_drop_existing = [col for col in cols_to_drop if col in df_encoded.columns]
        df_final = df_encoded.drop(columns=cols_to_drop_existing)

        st.subheader("Préparation Finale des Données")
        st.write("Dimensions du DataFrame prêt pour le modèle : ", df_final.shape)
        if st.checkbox("Afficher les premières lignes des données encodées", key="chk_df_final"):
             st.dataframe(df_final.head())


        st.subheader("Gestion du Déséquilibre des Classes")
        target_counts = df_final['y_encoded'].value_counts(normalize=True)
        st.write("Distribution de la classe cible avant rééquilibrage :")
        st.write(target_counts)

        if st.checkbox("Activer le suréchantillonnage (Upsampling) de la classe minoritaire", key="cb_upsampling", value=True):
             st.write("Application du suréchantillonnage pour équilibrer les classes...")
             df_majority = df_final[df_final.y_encoded == 0]
             df_minority = df_final[df_final.y_encoded == 1]

             if len(df_minority) == 0 or len(df_majority) == 0:
                 st.error("Impossible de suréchantillonner : une des classes est vide.")
                 st.stop()

             df_minority_upsampled = resample(df_minority,
                                              replace=True,     # échantillonnage avec remplacement
                                              n_samples=len(df_majority), # pour correspondre à la classe majoritaire
                                              random_state=42) # pour la reproductibilité

             df_upsampled = pd.concat([df_majority, df_minority_upsampled])
             st.write("Distribution de la classe cible après suréchantillonnage :")
             st.write(df_upsampled['y_encoded'].value_counts(normalize=True))
             data_ready = df_upsampled
        else:
             st.write("Entraînement sur les données déséquilibrées.")
             data_ready = df_final


        st.subheader("Division Entraînement / Test")
        X = data_ready.drop(columns=['y_encoded']).values
        y = data_ready['y_encoded'].values
        test_size = st.slider("Pourcentage pour l'ensemble de test", 0.1, 0.5, 0.2, 0.05, key="slider_test_size")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y) # stratify est important, surtout si non rééquilibré
        st.write(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} échantillons")
        st.write(f"Taille de l'ensemble de test : {X_test.shape[0]} échantillons")


        st.subheader("Entraînement du Modèle Random Forest")
        n_estimators = st.slider("Nombre d'arbres (n_estimators)", 50, 500, 100, 50, key="slider_n_estimators")
        max_depth = st.select_slider("Profondeur maximale (max_depth)", options=[None, 5, 10, 15, 20, 30], value=None, key="slider_max_depth") # None = pas de limite

        with st.spinner("Entraînement du modèle en cours..."):
             model = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           random_state=42,
                                           n_jobs=-1) # Utiliser tous les cœurs CPU
             model.fit(X_train, y_train)
        st.success("Modèle entraîné avec succès !")


        st.subheader("Évaluation du Modèle")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=encoder_y.classes_) # Utiliser les noms originaux

        st.metric(label="Accuracy sur l'ensemble de test", value=f"{accuracy:.2%}")

        st.text('Rapport de Classification :')
        st.text(report) # Utiliser st.text pour préserver le formatage


        st.subheader("Sauvegarder le Modèle Entraîné")
        model_filename = 'model_bank_data.pkl'
        if st.button("Sauvegarder le modèle", key="btn_save_model"):
             try:
                with open(model_filename, 'wb') as model_file:
                    # Sauvegarder le modèle, l'encodeur de la cible, et les colonnes attendues
                    pickle.dump({
                        'model': model,
                        'encoder_y': encoder_y,
                        'features': data_ready.drop(columns=['y_encoded']).columns.tolist() # Important pour la prédiction
                    }, model_file)
                st.success(f"Modèle sauvegardé sous le nom '{model_filename}'")
             except Exception as e:
                st.error(f"Erreur lors de la sauvegarde du modèle : {e}")


    elif st.session_state.page_selection == 'prediction':
    st.title("🔮 Prédiction de Souscription Client")

    # --- Chargement du modèle depuis GitHub (comme avant) ---
    MODEL_URL = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/model_classification_bank.pkl"
    model = None
    encoder_y = None
    expected_features = None

    @st.cache_resource
    def load_model_from_github(url):
        # ... (code de la fonction load_model_from_github comme avant) ...
        try:
            st.info(f"Téléchargement et chargement du modèle depuis {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            saved_data = pickle.loads(response.content)
            model_loaded = saved_data.get('model')
            encoder_loaded = saved_data.get('encoder_y')
            features_loaded = saved_data.get('features')
            if not all([model_loaded, encoder_loaded, features_loaded]):
                 st.error("Le fichier pickle chargé ne contient pas toutes les clés attendues ('model', 'encoder_y', 'features').")
                 return None, None, None
            st.success("Modèle chargé avec succès depuis GitHub.")
            return model_loaded, encoder_loaded, features_loaded
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur réseau lors du téléchargement du modèle : {e}")
            return None, None, None
        except pickle.UnpicklingError as e:
            st.error(f"Erreur lors du désérialisage du fichier pickle : {e}")
            return None, None, None
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue lors du chargement du modèle : {e}")
            return None, None, None

    model, encoder_y, expected_features = load_model_from_github(MODEL_URL)

    if model is None or encoder_y is None or expected_features is None:
        st.error("Impossible de continuer sans le modèle chargé.")
        st.stop()

    # Essayer de s'assurer que df_original est disponible ici si nécessaire pour les options des selectbox
    # Cela suppose que df_original est chargé plus tôt dans le script globalement ou dans la session state
    # Si ce n'est pas le cas, vous devrez gérer le chargement de df_original ici ou utiliser des listes statiques
    df_original = None # Initialiser
    try:
        # Supposons que df est chargé plus tôt et est accessible (sinon, il faut le charger ici)
        # Remplacer 'df' par la variable réelle contenant le dataframe original si elle a un autre nom
        if 'df' in locals() or 'df' in globals():
            # Assurez-vous que 'df' est bien le dataframe *original* non modifié
            df_original = df.copy() # Utiliser une copie pour être sûr
        else:
            # Tentative de rechargement si non trouvé (ajuster le chemin/URL si nécessaire)
            st.warning("df_original non trouvé, tentative de rechargement...")
            DATA_URL = 'bank-additional-full.csv' # Ou l'URL GitHub
            try:
                 df_original = pd.read_csv(DATA_URL, delimiter=';')
            except FileNotFoundError:
                 github_data_url = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/bank-additional-full.csv"
                 df_original = pd.read_csv(github_data_url, delimiter=';')
            st.success("df_original rechargé.")

    except Exception as e:
        st.error(f"Impossible de charger df_original pour les options du formulaire : {e}")
        # On pourrait continuer avec des listes par défaut, mais l'encodage échouera probablement plus tard

    st.markdown("Entrez les informations du client...")

    with st.form(key='prediction_form'):
        # ... (définition du formulaire comme avant, utilisant df_original pour les options si possible) ...
        st.subheader("Informations du Client")
        col1, col2, col3 = st.columns(3)
        # --- Champs de saisie (comme avant, mais s'assurer que les listes d'options sont définies) ---
        # (Exemple pour job)
        try: job_options = df_original['job'].unique().tolist() if df_original is not None else ['admin.', 'technician', 'blue-collar'] # Fallback
        except: job_options = ['admin.', 'technician', 'blue-collar'] # Fallback plus sûr
        job = st.selectbox("Métier", options=job_options, index=0, key="input_job")
        # ... (Répéter pour marital, education, default, housing, loan, contact, month, day_of_week, poutcome en utilisant df_original si possible avec fallback)

        # ... Autres champs (age, duration, etc.) ...

        st.subheader("Indicateurs Économiques")
        # ... (Champs indicateurs économiques comme avant) ...

        submitted = st.form_submit_button("🔮 Lancer la Prédiction")

        if submitted:
            # --- Création du DataFrame d'entrée (comme avant) ---
            input_dict = { ... } # Remplir avec les valeurs du formulaire
            input_df = pd.DataFrame([input_dict])

            # --- Définition des listes de colonnes (comme avant) ---
            categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
            categorical_cols_label = ['default', 'housing', 'loan', 'contact']

            # --- Prétraitement ---
            # Encodage par fréquence
            try:
                # >>> Vérification critique de df_original <<<
                if df_original is None:
                    st.error("Erreur critique: df_original non disponible pour calculer les fréquences d'encodage.")
                    st.stop() # <<<=== AJOUT CRUCIAL DE st.stop()

                st.write("Application de l'encodage par fréquence...")
                for column in categorical_cols_freq:
                     fe = df_original.groupby(column).size() / len(df_original)
                     input_df[f'{column}_freq_encode'] = input_df[column].map(fe).fillna(0)

            except Exception as e:
                 st.error(f"Erreur lors de l'encodage par fréquence : {e}")
                 st.stop() # Arrêter si l'encodage échoue

            # Encodage Label
            try:
                st.write("Application de l'encodage Label...")
                le_pred = LabelEncoder()
                for column in categorical_cols_label:
                     all_options = df_original[column].unique() if df_original is not None else locals().get(f"{column}_options", ['yes','no']) # Utiliser df_original ou fallback
                     if not list(all_options): all_options = ['yes', 'no'] # Fallback
                     le_pred.fit(all_options)
                     input_df[column] = le_pred.transform(input_df[column])

            except Exception as e:
                 st.error(f"Erreur lors de l'encodage Label pour {column} : {e}.")
                 st.stop() # Arrêter si l'encodage échoue

            # Supprimer les colonnes originales catégorielles (comme avant)
            cols_to_drop_pred = categorical_cols_freq + categorical_cols_label
            input_df_encoded = input_df.drop(columns=cols_to_drop_pred, errors='ignore')

            # Alignement des colonnes (comme avant)
            try:
                st.write("Alignement des colonnes...")
                input_final = input_df_encoded.reindex(columns=expected_features, fill_value=0)

            except Exception as e:
                 st.error(f"Erreur lors de l'alignement final des colonnes d'entrée : {e}")
                 st.stop() # Arrêter si l'alignement échoue

            # --- Prédiction ---
            # Ce bloc ne devrait être atteint que si toutes les étapes précédentes ont réussi
            try:
                st.write("Lancement de la prédiction...")
                prediction_proba = model.predict_proba(input_final) # Assignation de prediction_proba
                prediction = model.predict(input_final) # Assignation de prediction

                # Maintenant, prediction et encoder_y devraient être définis
                result_label = encoder_y.inverse_transform(prediction)[0]
                probability_yes = prediction_proba[0][1]

                # --- Affichage du résultat ---
                st.subheader("Résultat de la Prédiction")
                if result_label == 'yes':
                    st.success(f"Le client va probablement souscrire ! (Probabilité: {probability_yes:.2%})")
                    st.balloons()
                else:
                    st.warning(f"Le client ne va probablement pas souscrire. (Probabilité de souscription: {probability_yes:.2%})")

                st.write("Probabilités prédites :")
                st.write(f"- Non ('no'): {prediction_proba[0][0]:.2%}")
                st.write(f"- Oui ('yes'): {prediction_proba[0][1]:.2%}")

            except Exception as e:
                st.error(f"Erreur lors de l'exécution de la prédiction : {e}")
                st.write("Données finales envoyées au modèle :")
                st.dataframe(input_final)


# --- Reste du code ---
# if __name__ == '__main__':
#     main()
