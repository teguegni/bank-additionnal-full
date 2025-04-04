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

        # Charger le modèle et les informations associées
        model_filename = 'model_bank_data.pkl'
        try:
            with open(model_filename, 'rb') as model_file:
                saved_data = pickle.load(model_file)
                model = saved_data['model']
                encoder_y = saved_data['encoder_y']
                expected_features = saved_data['features'] # Récupérer les noms des features attendues
            st.success(f"Modèle '{model_filename}' chargé.")
        except FileNotFoundError:
            st.error(f"Le fichier modèle '{model_filename}' n'a pas été trouvé. Veuillez entraîner et sauvegarder le modèle dans la section 'Apprentissage Automatique'.")
            st.stop()
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            st.stop()

        st.markdown("Entrez les informations du client pour prédire s'il va souscrire à un dépôt à terme.")

        # Utiliser st.form pour regrouper les champs et le bouton
        with st.form(key='prediction_form'):
            st.subheader("Informations du Client")
            # Utiliser des colonnes pour mieux organiser
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Âge", min_value=18, max_value=100, value=40, step=1, key="input_age")
                job_options = df_original['job'].unique().tolist() # Utiliser les options du df original
                job = st.selectbox("Métier", options=job_options, index=job_options.index('admin.') if 'admin.' in job_options else 0 , key="input_job") # Exemple de valeur par défaut
                marital_options = df_original['marital'].unique().tolist()
                marital = st.selectbox("Statut Marital", options=marital_options, index=marital_options.index('married') if 'married' in marital_options else 0, key="input_marital")
                education_options = df_original['education'].unique().tolist()
                education = st.selectbox("Niveau d'Éducation", options=education_options, index=education_options.index('university.degree') if 'university.degree' in education_options else 0, key="input_education")
                default_options = df_original['default'].unique().tolist()
                default = st.selectbox("Défaut de Crédit ?", options=default_options, index=default_options.index('no') if 'no' in default_options else 0, key="input_default")

            with col2:
                housing_options = df_original['housing'].unique().tolist()
                housing = st.selectbox("Prêt Immobilier ?", options=housing_options, index=housing_options.index('yes') if 'yes' in housing_options else 0, key="input_housing")
                loan_options = df_original['loan'].unique().tolist()
                loan = st.selectbox("Prêt Personnel ?", options=loan_options, index=loan_options.index('no') if 'no' in loan_options else 0, key="input_loan")
                contact_options = df_original['contact'].unique().tolist()
                contact = st.selectbox("Type de Contact", options=contact_options, index=contact_options.index('cellular') if 'cellular' in contact_options else 0, key="input_contact")
                month_options = df_original['month'].unique().tolist()
                month = st.selectbox("Mois du Dernier Contact", options=month_options, index=month_options.index('may') if 'may' in month_options else 0, key="input_month")
                day_of_week_options = df_original['day_of_week'].unique().tolist()
                day_of_week = st.selectbox("Jour de la Semaine", options=day_of_week_options, index=day_of_week_options.index('thu') if 'thu' in day_of_week_options else 0, key="input_day_of_week")

            with col3:
                 duration = st.number_input("Durée Dernier Contact (secondes)", min_value=0, value=150, step=10, key="input_duration", help="Important: Ce champ influence fortement la prédiction mais n'est connu qu'après l'appel. Pour une prédiction réaliste *avant* l'appel, il devrait être estimé ou non utilisé.")
                 campaign = st.number_input("Nb Contacts Campagne Actuelle", min_value=1, value=2, step=1, key="input_campaign")
                 pdays = st.number_input("Jours Depuis Dernier Contact (Préc. Campagne)", min_value=-1, value=999, step=1, key="input_pdays", help="-1 ou 999 signifie jamais contacté précédemment")
                 previous = st.number_input("Nb Contacts Avant Campagne Actuelle", min_value=0, value=0, step=1, key="input_previous")
                 poutcome_options = df_original['poutcome'].unique().tolist()
                 poutcome = st.selectbox("Résultat Préc. Campagne", options=poutcome_options, index=poutcome_options.index('nonexistent') if 'nonexistent' in poutcome_options else 0, key="input_poutcome")

            st.subheader("Indicateurs Économiques (valeurs récentes typiques)")
            col_eco1, col_eco2, col_eco3 = st.columns(3)
            with col_eco1:
                emp_var_rate = st.number_input("Taux Variation Emploi", value=-0.1, step=0.1, format="%.1f", key="input_emp_var_rate")
            with col_eco2:
                cons_price_idx = st.number_input("Indice Prix Consommation", value=93.2, step=0.1, format="%.1f", key="input_cons_price_idx")
                cons_conf_idx = st.number_input("Indice Confiance Consommateur", value=-42.0, step=0.1, format="%.1f", key="input_cons_conf_idx")
            with col_eco3:
                euribor3m = st.number_input("Taux Euribor 3 Mois", value=1.3, step=0.1, format="%.3f", key="input_euribor3m")
                nr_employed = st.number_input("Nombre d'Employés (milliers)", value=5100.0, step=10.0, format="%.1f", key="input_nr_employed")


            # Bouton pour soumettre le formulaire
            submitted = st.form_submit_button("🔮 Lancer la Prédiction")

            elif st.session_state.page_selection == 'prediction':
        st.title("🔮 Prédiction de Souscription Client")

        # --- Charger le modèle, encodeur_y, expected_features ---
        model, encoder_y, expected_features = None, None, None

        # Option 1: Charger depuis GitHub
        if st.toggle("Charger le modèle depuis GitHub", value=True, key="load_github"):
             # Assurez-vous que la fonction load_model_from_github est définie comme dans le code précédent
             model, encoder_y, expected_features = load_model_from_github(MODEL_URL_GITHUB)
        else:
        # Option 2: Charger depuis le fichier local
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

        # Assurer la disponibilité de df_original pour les options et l'encodage
        # (Ce bloc peut être simplifié si df_original est garanti d'être chargé au début)
        df_original_for_prediction = None
        if 'df_original' in globals() and df_original is not None:
            df_original_for_prediction = df_original # Utiliser le df global chargé
        else:
             st.error("Le DataFrame original n'a pas pu être chargé initialement. L'encodage échouera.")
             st.stop()


        # --- Formulaire de saisie utilisateur ---
        st.markdown("Entrez les informations du client :")
        with st.form(key='prediction_form'):
            # ... (Définition des colonnes et des champs de formulaire comme dans le code précédent)
            st.subheader("Infos Client")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Âge", 18, 100, 40, 1, key="pred_age")
                job = st.selectbox("Métier", options=df_original_for_prediction['job'].unique(), key="pred_job")
                marital = st.selectbox("Statut Marital", options=df_original_for_prediction['marital'].unique(), key="pred_marital")
                education = st.selectbox("Éducation", options=df_original_for_prediction['education'].unique(), key="pred_education")
                default = st.selectbox("Défaut Crédit?", options=df_original_for_prediction['default'].unique(), format_func=lambda x: 'Oui' if x=='yes' else ('Non' if x=='no' else x) , key="pred_default")
                housing = st.selectbox("Prêt Immo?", options=df_original_for_prediction['housing'].unique(), format_func=lambda x: 'Oui' if x=='yes' else ('Non' if x=='no' else x), key="pred_housing")
                loan = st.selectbox("Prêt Perso?", options=df_original_for_prediction['loan'].unique(), format_func=lambda x: 'Oui' if x=='yes' else ('Non' if x=='no' else x), key="pred_loan")

            with col2:
                contact = st.selectbox("Type Contact", options=df_original_for_prediction['contact'].unique(), key="pred_contact")
                month = st.selectbox("Mois Dernier Contact", options=df_original_for_prediction['month'].unique(), key="pred_month")
                day_of_week = st.selectbox("Jour Semaine Dernier Contact", options=df_original_for_prediction['day_of_week'].unique(), key="pred_day")
                duration = st.number_input("Durée Dernier Contact (sec)", 0, 6000, 120, 10, key="pred_duration", help="Attention: Valeur connue après l'appel.")
                campaign = st.number_input("Nb Contacts Campagne", 1, 100, 2, 1, key="pred_campaign")
                pdays = st.number_input("Jours Depuis Dernier Contact (Préc.)", -1, 999, 999, 1, key="pred_pdays", help="-1 ou 999 = Jamais contacté")
                previous = st.number_input("Nb Contacts Avant Campagne", 0, 100, 0, 1, key="pred_previous")
                poutcome = st.selectbox("Résultat Préc. Campagne", options=df_original_for_prediction['poutcome'].unique(), key="pred_poutcome")

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

                 # --- >>> AJOUT/CORRECTION : Définition des listes ICI <<< ---
                categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
                categorical_cols_label = ['default', 'housing', 'loan', 'contact']
                # --- >>> FIN DE L'AJOUT/CORRECTION <<< ---

                # 1. Encodage par fréquence
                try:
                    # Utiliser df_original_for_prediction qui a été vérifié plus haut
                    if df_original_for_prediction is None: raise ValueError("df_original_for_prediction n'est pas disponible.")
                    st.write("Application de l'encodage par fréquence...") # Debug
                    for col in categorical_cols_freq: # Utilise la liste définie ci-dessus
                        freq_map = df_original_for_prediction.groupby(col).size() / len(df_original_for_prediction)
                        df_processed_input[f'{col}_freq_encode'] = df_processed_input[col].map(freq_map).fillna(0)
                        st.write(f"FreqEnc - Colonne: {col}, Valeur encodée: {df_processed_input[f'{col}_freq_encode'].iloc[0]}") # Debug
                except Exception as e:
                    st.error(f"Erreur pendant l'encodage par fréquence: {e}")
                    st.stop()

                # 2. Encodage Label
                try:
                    if df_original_for_prediction is None: raise ValueError("df_original_for_prediction n'est pas disponible.")
                    st.write("Application de l'encodage Label...") # Debug
                    for col in categorical_cols_label: # Utilise la liste définie ci-dessus
                        le = LabelEncoder()
                        le.fit(df_original_for_prediction[col].unique()) # Fit sur les valeurs possibles
                        df_processed_input[col] = le.transform(df_processed_input[col])
                        st.write(f"LabelEnc - Colonne: {col}, Valeur encodée: {df_processed_input[col].iloc[0]}") # Debug
                except Exception as e:
                    st.error(f"Erreur pendant l'encodage Label pour {col}: {e}")
                    st.stop()

                # 3. Supprimer les colonnes catégorielles originales
                cols_to_drop_input = categorical_cols_freq + categorical_cols_label
                df_processed_input = df_processed_input.drop(columns=cols_to_drop_input, errors='ignore')

                # 4. S'assurer que les colonnes sont dans le bon ordre et complètes
                try:
                    st.write("Alignement des colonnes finales...") # Debug
                    input_final = df_processed_input.reindex(columns=expected_features, fill_value=0)
                    st.write("Données après prétraitement (prêtes pour modèle) :", input_final) # Debug
                except Exception as e:
                    st.error(f"Erreur lors de l'alignement des colonnes finales: {e}")
                    st.stop()

                # --- Faire la Prédiction ---
                try:
                    st.write("Lancement de la prédiction sur le modèle...") # Debug
                    prediction_proba = model.predict_proba(input_final)
                    prediction = model.predict(input_final)
                    probability_yes = prediction_proba[0][1]
                    result_label = encoder_y.inverse_transform(prediction)[0] # Utiliser l'encodeur chargé

                    # --- Afficher le résultat ---
                    st.subheader("Résultat")
                    # ... (affichage succès/warning comme avant) ...
                    if result_label == 'yes':
                        st.success(f"✅ Prédiction : Souscription Probable (Confiance: {probability_yes:.1%})")
                        st.balloons()
                    else:
                        st.warning(f"❌ Prédiction : Souscription Improbable (Confiance souscription: {probability_yes:.1%})")

                except Exception as e:
                    st.error(f"Erreur lors de l'exécution de la prédiction sur le modèle : {e}")
                    st.dataframe(input_final) # Afficher les données qui ont causé l'erreur

# --- Exécution de l'application ---
if __name__ == '__main__':
    main()

# --- Fonctions utilitaires (si non définies ailleurs) ---
# Assurez-vous que load_model_from_github est définie
@st.cache_resource
def load_model_from_github(url):
    """Charge un objet pickle (modèle, etc.) depuis une URL brute GitHub."""
    try:
        st.info(f"Tentative de chargement du modèle depuis GitHub ({url})...")
        response = requests.get(url, timeout=30)
        response.raise_for_status() # Vérifie les erreurs HTTP
        model_data = pickle.load(io.BytesIO(response.content))
        required_keys = ['model', 'encoder_y', 'features']
        if not all(key in model_data for key in required_keys):
            st.error(f"Le fichier pickle chargé ne contient pas les clés requises: {required_keys}")
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

# Répétez le code pour add_bg_from_url si nécessaire ou assurez-vous qu'il est défini avant l'appel
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
         </style>
         """,
         unsafe_allow_html=True
     )
image_url_raw = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/image1.jpg" # URL Corrigée
add_bg_from_url(image_url_raw) # Appel de la fonction


# --- Point d'entrée de l'application ---
if __name__ == '__main__': # Correction: utiliser __name__ et __main__
    main()
