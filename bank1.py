# --- Assurez-vous que ces imports sont présents au début de votre fichier ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import LabelEncoder
# ... (autres imports nécessaires)

# --- Début de la fonction main() ou de la section de page ---
# ... (code des autres pages)

elif st.session_state.page_selection == 'prediction':
    st.title("🔮 Prédiction de Souscription Client")

    # --- Chargement du modèle depuis GitHub ---
    MODEL_URL = "https://raw.githubusercontent.com/teguegni/bank-additionnal-full/main/model_classification_bank.pkl"
    model = None
    encoder_y = None
    expected_features = None

    @st.cache_resource
    def load_model_from_github(url):
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

    st.markdown("Entrez les informations du client pour prédire s'il va souscrire à un dépôt à terme.")

    with st.form(key='prediction_form'):
        st.subheader("Informations du Client")
        col1, col2, col3 = st.columns(3)

        # --- Champs de saisie (comme avant) ---
        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=40, step=1, key="input_age")
            try:
                job_options = df_original['job'].unique().tolist()
                marital_options = df_original['marital'].unique().tolist()
                education_options = df_original['education'].unique().tolist()
                default_options = df_original['default'].unique().tolist()
            except NameError:
                 st.warning("df_original non trouvé, utilisation de listes d'options par défaut.")
                 # Ajoutez ici des listes par défaut si df_original peut ne pas être défini
                 job_options = ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student']
                 marital_options = ['married', 'single', 'divorced']
                 education_options = ['university.degree', 'high.school', 'basic.9y', 'professional.course', 'basic.4y', 'basic.6y', 'illiterate']
                 default_options = ['no', 'yes']

            job = st.selectbox("Métier", options=job_options, index=job_options.index('admin.') if 'admin.' in job_options else 0 , key="input_job")
            marital = st.selectbox("Statut Marital", options=marital_options, index=marital_options.index('married') if 'married' in marital_options else 0, key="input_marital")
            education = st.selectbox("Niveau d'Éducation", options=education_options, index=education_options.index('university.degree') if 'university.degree' in education_options else 0, key="input_education")
            default = st.selectbox("Défaut de Crédit ?", options=default_options, index=default_options.index('no') if 'no' in default_options else 0, key="input_default")

        with col2:
             try:
                 housing_options = df_original['housing'].unique().tolist()
                 loan_options = df_original['loan'].unique().tolist()
                 contact_options = df_original['contact'].unique().tolist()
                 month_options = df_original['month'].unique().tolist()
                 day_of_week_options = df_original['day_of_week'].unique().tolist()
             except NameError:
                 st.warning("df_original non trouvé, utilisation de listes d'options par défaut.")
                 housing_options = ['yes', 'no']
                 loan_options = ['no', 'yes']
                 contact_options = ['cellular', 'telephone']
                 month_options = ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep']
                 day_of_week_options = ['thu', 'mon', 'wed', 'tue', 'fri']

             housing = st.selectbox("Prêt Immobilier ?", options=housing_options, index=housing_options.index('yes') if 'yes' in housing_options else 0, key="input_housing")
             loan = st.selectbox("Prêt Personnel ?", options=loan_options, index=loan_options.index('no') if 'no' in loan_options else 0, key="input_loan")
             contact = st.selectbox("Type de Contact", options=contact_options, index=contact_options.index('cellular') if 'cellular' in contact_options else 0, key="input_contact")
             month = st.selectbox("Mois du Dernier Contact", options=month_options, index=month_options.index('may') if 'may' in month_options else 0, key="input_month")
             day_of_week = st.selectbox("Jour de la Semaine", options=day_of_week_options, index=day_of_week_options.index('thu') if 'thu' in day_of_week_options else 0, key="input_day_of_week")

        with col3:
             duration = st.number_input("Durée Dernier Contact (secondes)", min_value=0, value=150, step=10, key="input_duration")
             campaign = st.number_input("Nb Contacts Campagne Actuelle", min_value=1, value=2, step=1, key="input_campaign")
             pdays = st.number_input("Jours Depuis Dernier Contact (Préc. Campagne)", min_value=-1, value=999, step=1, key="input_pdays")
             previous = st.number_input("Nb Contacts Avant Campagne Actuelle", min_value=0, value=0, step=1, key="input_previous")
             try:
                 poutcome_options = df_original['poutcome'].unique().tolist()
             except NameError:
                 st.warning("df_original non trouvé, utilisation de listes d'options par défaut.")
                 poutcome_options = ['nonexistent', 'failure', 'success']
             poutcome = st.selectbox("Résultat Préc. Campagne", options=poutcome_options, index=poutcome_options.index('nonexistent') if 'nonexistent' in poutcome_options else 0, key="input_poutcome")

        st.subheader("Indicateurs Économiques")
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
            # --- Création du DataFrame d'entrée ---
            input_dict = {
                'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
                'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
                'duration': duration, 'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
                'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
                'euribor3m': euribor3m, 'nr.employed': nr_employed
            }
            input_df = pd.DataFrame([input_dict])

            # --- Définition des listes de colonnes ICI ---
            # !!! ASSUREZ-VOUS QUE CES LISTES CORRESPONDENT EXACTEMENT A L'ENTRAINEMENT !!!
            categorical_cols_freq = ['marital', 'job', 'education', 'month', 'day_of_week', 'poutcome']
            categorical_cols_label = ['default', 'housing', 'loan', 'contact']
            # --- FIN de la définition des listes ---

            # --- Prétraitement du DataFrame d'entrée ---
            # Encodage par fréquence
            try:
                if 'df_original' not in locals() or df_original is None:
                    # Si df_original n'existe pas, on ne peut pas calculer les fréquences.
                    # Il faudrait une autre stratégie (ex: charger les fréquences sauvegardées)
                    st.error("Erreur critique: df_original non disponible pour calculer les fréquences d'encodage.")
                    st.stop() # Arrêter car l'encodage sera incorrect

                st.write("Application de l'encodage par fréquence...") # Debug
                for column in categorical_cols_freq:
                     fe = df_original.groupby(column).size() / len(df_original)
                     input_df[f'{column}_freq_encode'] = input_df[column].map(fe).fillna(0) # fillna(0) pour catégories inconnues
                     st.write(f"Colonne: {column}, Valeur encodée: {input_df[f'{column}_freq_encode'].iloc[0]}") # Debug

            except Exception as e:
                 st.error(f"Erreur lors de l'encodage par fréquence : {e}")
                 st.stop()

            # Encodage Label
            try:
                st.write("Application de l'encodage Label...") # Debug
                le_pred = LabelEncoder()
                for column in categorical_cols_label:
                     # Fitter sur les options connues
                     all_options = df_original[column].unique() if 'df_original' in locals() and df_original is not None else locals().get(f"{column}_options", [])
                     if not list(all_options):
                         st.error(f"Options non trouvées pour l'encodage de '{column}'. Utilisation de ['yes', 'no'] par défaut.")
                         all_options = ['yes', 'no'] # Fallback très basique

                     le_pred.fit(all_options) # Fitter sur les options possibles
                     input_df[column] = le_pred.transform(input_df[column]) # Transformer la valeur entrée
                     st.write(f"Colonne: {column}, Valeur encodée: {input_df[column].iloc[0]}") # Debug

            except Exception as e:
                 st.error(f"Erreur lors de l'encodage Label pour {column} : {e}. Valeur entrée: {input_df[column].iloc[0]}")
                 st.stop()

            # Supprimer les colonnes originales catégorielles
            cols_to_drop_pred = categorical_cols_freq + categorical_cols_label
            input_df_encoded = input_df.drop(columns=cols_to_drop_pred, errors='ignore') # errors='ignore' au cas où une colonne n'existe pas

            # S'assurer que l'ordre/présence des colonnes est correct
            try:
                st.write("Alignement des colonnes avec celles attendues par le modèle...") # Debug
                st.write(f"Colonnes attendues: {expected_features}") # Debug
                st.write(f"Colonnes présentes avant alignement: {input_df_encoded.columns.tolist()}") # Debug
                input_final = input_df_encoded.reindex(columns=expected_features, fill_value=0) # Utiliser reindex pour garantir toutes les colonnes attendues
                st.write(f"Colonnes finales pour la prédiction: {input_final.columns.tolist()}") # Debug

                # Vérifier les types (parfois nécessaire)
                # for col in input_final.columns:
                #     input_final[col] = pd.to_numeric(input_final[col], errors='ignore')

            except Exception as e:
                 st.error(f"Erreur lors de l'alignement final des colonnes d'entrée : {e}")
                 st.stop()

            # --- Prédiction ---
            try:
                st.write("Lancement de la prédiction...") # Debug
                prediction_proba = model.predict_proba(input_final)
                prediction = model.predict(input_final)
                probability_yes = prediction_proba[0][1]
                result_label = encoder_y.inverse_transform(prediction)[0]

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
                st.dataframe(input_final) # Afficher les données en cas d'erreur de prédiction


# --- Reste du code ---
# if __name__ == '__main__':
#     main()
