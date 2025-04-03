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
    df_ml = df_original.copy()

    # 1. Liste des colonnes à encoder
    categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']  # Définition explicite

    # 2. Encodage par fréquence
    st.subheader("Encodage des Variables Catégorielles")
    df_encoded = df_ml.copy()
    for column in categorical_cols_freq:  # Pas d'erreur ici
        fe = df_encoded.groupby(column).size() / len(df_encoded)
        df_encoded[f'{column}_freq_encode'] = df_encoded[column].map(fe)

    # 3. Encodage Label pour les autres colonnes
    categorical_cols_label = ['contact', 'housing', 'loan', 'default', 'y']
    label_encoder = LabelEncoder()
    for column in categorical_cols_label:
        df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

    # ... (suite du code)

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

            if submitted:
                # Créer un DataFrame avec les entrées utilisateur
                input_dict = {
                    'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
                    'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
                    'duration': duration, 'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
                    'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                    'euribor3m': euribor3m, 'nr.employed': nr_employed
                }
                input_df = pd.DataFrame([input_dict])

                # --- Appliquer le MÊME prétraitement que pour l'entraînement ---
                # 1. Gérer 'unknown' (ici on suppose que les selectbox n'ont pas 'unknown', sinon il faudrait le gérer)
                # Normalement, les selectbox empêchent l'entrée 'unknown' si elle n'est pas une option valide.

                # 2. Traitement Outliers (si activé pendant l'entraînement - important d'être cohérent)
                # NOTE : Appliquer le clipping basé sur les bornes calculées sur le *train set* serait l'idéal.
                # Ici, pour simplifier, on suppose que les valeurs entrées sont raisonnables ou que le clipping n'était pas critique.
                # Si le clipping était activé et important, il faudrait sauvegarder les bornes (lower/upper) pour chaque variable
                # numérique et les appliquer ici.

                # 3. Encodage
                # Freq encoding
                for column in categorical_cols_freq: # Utiliser les mêmes listes que pour l'entraînement
                    # Charger ou recalculer les fréquences du jeu d'entraînement est le plus sûr
                    # Ici, on recalcule sur df_original pour la démo, MAIS C'EST MOINS ROBUSTE
                    # L'idéal: Sauvegarder les mappings de fréquence avec le modèle.
                    fe = df_original.groupby(column).size() / len(df_original)
                    input_df[f'{column}_freq_encode'] = input_df[column].map(fe).fillna(0) # fillna(0) pour les catégories inconnues

                # Label encoding
                le_pred = LabelEncoder() # Recréer un encodeur
                for column in categorical_cols_label:
                    # Adapter aux valeurs possibles ('yes', 'no', etc.) comme lors de l'entraînement
                     try:
                        # Il faut s'assurer que l'encodeur est fitté sur les mêmes valeurs que pendant l'entraînement
                        # Le plus simple est de le fitter sur les options possibles du df original
                        le_pred.fit(df_original[column].unique())
                        input_df[column] = le_pred.transform(input_df[column])
                     except Exception as e:
                         st.error(f"Erreur d'encodage prédiction pour {column}: {e}. Valeur entrée: {input_df[column].iloc[0]}")
                         st.stop()


                # 4. Supprimer les colonnes originales non nécessaires
                cols_to_drop_pred = categorical_cols_freq + categorical_cols_label
                input_df_encoded = input_df.drop(columns=cols_to_drop_pred)

                # 5. S'assurer que l'ordre des colonnes est le même que celui attendu par le modèle
                try:
                    input_final = input_df_encoded[expected_features] # Réorganiser/sélectionner selon l'ordre attendu
                except KeyError as e:
                    st.error(f"Erreur: Colonne manquante ou incorrecte pour la prédiction : {e}")
                    st.error(f"Colonnes attendues: {expected_features}")
                    st.error(f"Colonnes après encodage: {input_df_encoded.columns.tolist()}")
                    st.stop()


                # --- Prédiction ---
                prediction_proba = model.predict_proba(input_final)
                prediction = model.predict(input_final)

                probability_yes = prediction_proba[0][1] # Probabilité de la classe '1' (yes)
                result_label = encoder_y.inverse_transform(prediction)[0] # Revenir à 'yes' ou 'no'

                # --- Afficher le résultat ---
                st.subheader("Résultat de la Prédiction")
                if result_label == 'yes':
                    st.success(f"Le client va probablement souscrire ! (Probabilité: {probability_yes:.2%})")
                    st.balloons()
                else:
                    st.warning(f"Le client ne va probablement pas souscrire. (Probabilité de souscription: {probability_yes:.2%})")

                # Afficher les probabilités détaillées
                st.write("Probabilités prédites :")
                st.write(f"- Non ('no'): {prediction_proba[0][0]:.2%}")
                st.write(f"- Oui ('yes'): {prediction_proba[0][1]:.2%}")


# --- Point d'entrée de l'application ---
if __name__ == '__main__': # Correction: utiliser __name__ et __main__
    main()
