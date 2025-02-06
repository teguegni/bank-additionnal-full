# importation des bibiotheques
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import streamlit as st
import altair as alt 


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
    st.title(' 🏦 Classification des Données Bancaires')

    # Navigation par boutons
    st.subheader("Sections")
    if st.button("À Propos", use_container_width=True, on_click=set_page_selection, args=('a_propos',)):
        pass
    if st.button("Jeu de Données", use_container_width=True, on_click=set_page_selection, args=('jeu_de_donnees',)):
        pass
    if st.button("Analyse Exploratoire", use_container_width=True, on_click=set_page_selection, args=('analyse_exploratoire',)):
        pass
    if st.button("Prédiction", use_container_width=True, on_click=set_page_selection, args=('prediction',)):
        pass
    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        pass

    # Détails du projet
    st.subheader("Résumé")
    st.markdown("""
        Un tableau de bord interactif pour explorer et classifier les données d'une campagne marketing bancaire.

        -  [Jeu de Données](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
        -  [Notebook Google Colab](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)
        -  [Dépôt GitHub](https://github.com/teguegni/bank-additionnal-full/Streamlit-Bank-Classification-Dashboard)

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
     # Page À Propos
    st.title("️ conclusion")
    st.markdown("""
        Un traitement minutieux et réfléchi du DataFrame bank-additional-full est fondamental pour maximiser la précision 
        et la fiabilité du modèle de prédiction. En combinant explorations, prétraitements adéquats, et évaluations rigoureuses,
         on peut développer un modèle robuste qui est mieux équipé pour prédire les comportements des clients envers la souscription à un produit.
    """)

elif st.session_state.page_selection == 'jeu_de_donnees':
    # Page Jeu de Données
    st.title(" Jeu de Données")

    # Afficher les premières lignes du DataFrame
    if st.checkbox("Afficher le DataFrame"):
        nb_rows = st.slider("Nombre de lignes à afficher :", min_value=105, max_value=41189, value=10)
        st.write(df.head(nb_rows))

    # Afficher les statistiques descriptives
    if st.checkbox("Afficher les statistiques descriptives"):
        st.write(df.describe())

elif st.session_state.page_selection == 'analyse_exploratoire':
    st.title(" Analyse Exploratoire")
    #  verifies s'il ya des duplications dans la dataframe et affiche le nombre de duplications
    duplicates = df.duplicated()
    duplicates.value_counts()
    # supprimations des duplications 
    df.drop_duplicates()
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
    # selection des colonnes numerique 
    colonne_numerique = df[['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
    print(colonne_numerique)
    # Fonction pour détecter et remplacer les valeurs aberrantes avec les Bounds
    def replace_outliers(df):
        for col in df.select_dtypes(include=['number']).columns:  # Only process numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound,df[col])
    # Vérifiez que les colonnes nécessaires existent
    print(df.columns)
    replace_outliers(df)
    sns.boxplot(x='age',data=df)
    # faire la matrice de correlation
    sns.heatmap(df.select_dtypes(include=['number']).corr(),annot =True)
    # encodage categorielle avec plusieurs valeurs differentes pour les features 
    # grouper les categories 
    df.groupby('marital').size()
    # calcul de la frequence par categories
    fe = df.groupby('marital').size()/len(df)
    # insertion dans le dataframe
    df.loc[:,'marital_freq_encode'] = df['marital'].map(fe)
    df
    # encodage categorielle avec plusieurs valeurs differentes pour les features 
    # grouper les categories 
    df.groupby('job').size()
    # calcul de la frequence par categories
    fe = df.groupby('job').size()/len(df)
    # insertion dans le dataframe
    df.loc[:,'job_freq_encode'] = df['job'].map(fe)
    df
    # encodage categorielle avec plusieurs valeurs differentes pour les features 
    # grouper les categories 
    df.groupby('education').size()
    # calcul de la frequence par categories
    fe = df.groupby('education').size()/len(df)
    # insertion dans le dataframe
    df.loc[:,'education_freq_encode'] = df['education'].map(fe)
    df
    # encodage colonne categorielle de type binaire x
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df.default = le.fit_transform(df.default)
    df.housing = le.fit_transform(df.housing)
    df.loan = le.fit_transform(df.loan)
    df.contact = le.fit_transform(df.contact)
    df.head()
    # encodage categorielle avec plusieurs valeurs differentes pour les features 
    # grouper les categories 
    df.groupby('month').size()
    # calcul de la frequence par categories
    fe = df.groupby('month').size()/len(df)
    # insertion dans le dataframe
    df.loc[:,'month_freq_encode'] = df['month'].map(fe)
    df
    # encodage categorielle avec plusieurs valeurs differentes pour les features 
    # grouper les categories 
    df.groupby('day_of_week').size()
    # calcul de la frequence par categories
    fe = df.groupby('day_of_week').size()/len(df)
    # insertion dans le dataframe
    df.loc[:,'day_freq_encode'] = df['day_of_week'].map(fe)
    df
    # encodage categorielle avec plusieurs valeurs differentes pour les features 
    # grouper les categories 
    df.groupby('poutcome').size()
    # calcul de la frequence par categories
    fe = df.groupby('poutcome').size()/len(df)
    # insertion dans le dataframe
    df.loc[:,'poutcome_freq_encode'] = df['poutcome'].map(fe)
    df
    from sklearn.preprocessing import LabelEncoder  
    # Création de l'encodeur  
    encoder = LabelEncoder()  
    df['y_encoded'] = encoder.fit_transform(df['y'])  
    # Suppression de plusieurs colonnes  
    colonnes_a_supprimer = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']  
    df_propre = df.drop(colonnes_a_supprimer, axis=1)  

    # Afficher le DataFrame résultant  
    df_propre  
    from imblearn.over_sampling import RandomOverSampler  
    y_encoded = df_propre['y_encoded'].values  # Remplacez 'target' par le nom de votre colonne cible dans df_propre si nécessaire  

    # Définir le suréchantillonneur  
    ros = RandomOverSampler(random_state=42)  

    # Créer une matrice X factice (ou utilisez vos features réelles à partir de df_propre)  
    X = df_propre.drop(columns=['y_encoded']).values  # Assurez-vous de retirer la colonne cible  

    # Appliquer le suréchantillonnage  
    X_resampled, y_resampled = ros.fit_resample(X, y_encoded)  

    # Convertir les résultats en DataFrame pour une manipulation plus facile si nécessaire  
    df_resampled = pd.DataFrame(X_resampled, columns=df_propre.columns[:-1])  # Retirez la dernière colonne correspondant à 'target'  
    df_resampled['y_encoded'] = y_resampled  # Ajoutez la colonne cible équilibrée  

    # Afficher les résultats  
    print("Distribution originale :")  
    print(pd.Series(y_encoded).value_counts())  
    print("\nDistribution après suréchantillonnage :")  
    print(pd.Series(y_resampled).value_counts())  
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    # definition des donnees d'entrainement et de la variable cible
    X = df_propre[['age', 'duration', 'campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','marital_freq_encode','job_freq_encode','education_freq_encode','month_freq_encode','day_freq_encode','poutcome_freq_encode']]
    y = df_propre['y_encoded']  
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner un modèle simple pour tester
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score, classification_report 
    # Faire des prédictions sur l'ensemble de test  
    y_pred = model.predict(X_test)  

    # Évaluer le modèle  
    accuracy = accuracy_score(y_test, y_pred)  
    report = classification_report(y_test, y_pred)  

    # Afficher les résultats  
    print(f'Accuracy: {accuracy:.2f}')  
    print('Classification Report:')  
    print(report)   
elif st.session_state.page_selection == 'prediction':
        # ... (votre code pour la page de prédiction)
        # Page Prédiction  
    st.title("🔮 Prédiction")  
    from sklearn.ensemble import RandomForestClassifier  
    import streamlit as st  

    # Formulaire pour saisir les caractéristiques  
    age = st.number_input("Âge du client", min_value=18, max_value=120, value=30)  
    duration = st.number_input("Durée du contact (seconds)", min_value=0, value=60)  
    campaign = st.number_input("Nombre de contacts lors de la campagne", min_value=1, value=1)  
    pdays = st.number_input("Nombre de jours depuis le dernier contact", min_value=-1, value=-1)  # -1 si jamais contacté  
    previous = st.number_input("Nombre de contacts avant cette campagne", min_value=0, value=0)  
    emp_var_rate = st.number_input("Taux de variation de l'emploi (%)", value=0.0)  
    cons_price_idx = st.number_input("Indice des prix à la consommation", value=93.0)  
    cons_conf_idx = st.number_input("Indice de confiance des consommateurs", value=-40.0)  
    euribor3m = st.number_input("Taux Euribor à 3 mois (%)", value=0.0)  
    nr_employed = st.number_input("Nombre d'employés (calculé)", value=5191)  

    # Variables catégorielles (encodées)  
    marital_freq_encode = st.selectbox("Statut marital", ["célibataire", "marié", "divorcé"])  
    job_freq_encode = st.selectbox("Profession", ["ouvrier", "employé", "professionnel", "autre"])  
    education_freq_encode = st.selectbox("Niveau d'éducation", ["primaire", "secondaire", "tertiaire", "autre"])  
    month_freq_encode = st.selectbox("Mois de contact", ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"])  
    day_freq_encode = st.selectbox("Jour de contact", list(range(1, 32)))  
    poutcome_freq_encode = st.selectbox("Résultat de la campagne précédente", ["succès", "échec", "autre"])  

    # Vous pouvez également ajouter un bouton pour soumettre le formulaire  
if st.button('Soumettre'):  
    # Traitement des données ou affichage des résultats ici  
    st.write("Données soumises avec succès!")
    
   
