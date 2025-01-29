# Importer les bibliothèques nécessaires  
import streamlit as st  
import pandas as pd  
import altair as alt  
#import seaborn as sns  
#import matplotlib.pyplot as plt  

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
    if st.button("À Propos", use_container_width=True, on_click=set_page_selection, args=('a_propos',)):  
        pass  
    if st.button("Jeu de Données", use_container_width=True, on_click=set_page_selection, args=('jeu_de_donnees',)):  
        pass  
    if st.button("Analyse Exploratoire", use_container_width=True, on_click=set_page_selection, args=('analyse_exploratoire',)):  
        pass  
    if st.button("Nettoyage / Prétraitement des Données", use_container_width=True, on_click=set_page_selection, args=('nettoyage_donnees',)):  
        pass  
    if st.button("Apprentissage Automatique", use_container_width=True, on_click=set_page_selection, args=('apprentissage_automatique',)):   
        pass  
    if st.button("Prédiction", use_container_width=True, on_click=set_page_selection, args=('prediction',)):   
        pass  
    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):  
        pass  
         # Détails du projet  
    st.subheader("Résumé")  
    st.markdown("""  
        Un tableau de bord interactif pour explorer et classifier les données d'une campagne marketing bancaire.  
        
        - 📊 [Jeu de Données](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)  
        - 📗 [Notebook Google Colab](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)  
        - 🐙 [Dépôt GitHub](https://github.com/teguegni/bank-additionnal-full/Streamlit-Bank-Classification-Dashboard)  
        
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
    st.title("🏷️ À Propos")  
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

elif st.session_state.page_selection == 'jeu_de_donnees':  
    # Page Jeu de Données  
    st.title("📊 Jeu de Données")  
    
    # Afficher les premières lignes du DataFrame  
    if st.checkbox("Afficher le DataFrame"):  
        nb_rows = st.slider("Nombre de lignes à afficher :", min_value=105, max_value=41189, value=10)  
        st.write(df.head(nb_rows))  
    
    # Afficher les statistiques descriptives  
    if st.checkbox("Afficher les statistiques descriptives"):  
        st.write(df.describe())  

elif st.session_state.page_selection == 'analyse_exploratoire': 
    import pandas as pd  
    #import seaborn as sns  
    #import matplotlib.pyplot as plt   
    # Page Analyse Exploratoire  
    st.title("🔍nettoyage_donnees ") 
     # Traitement des variables catégorielles  
    data = pd.get_dummies(df, drop_first=True)  
      
    # Remplacer 'unknown' par le mode de chaque colonne  
    for column in df.columns:  
        if df[column].dtype == 'object':  # Vérifie si la colonne est de type object (catégorielle)  
            mode_value = df[column].mode()[0]  
            df[column] = df[column].replace('unknown', mode_value)  

    # Afficher le résultat des tables croisées pour chaque colonne d'intérêt  
    for column in df.columns:  
        if df[column].dtype == 'object' and column != 'y':  # Ignore la colonne cible 'y'   
            print(f"Table croisée pour {column}:")  
            print(df.groupby(['y', column])[column].size().unstack(level=0))  

            # Afficher le countplot  
            plt.figure(figsize=(10, 6))  
            sns.countplot(x=df["y"], hue=df[column])  
            plt.title(f'Countplot pour {column}')  
            plt.show()
    # Page Analyse Exploratoire  
    st.title("🔍 Analyse Exploratoire")  
    
    # Vérification des valeurs manquantes  
    st.subheader("Vérification des valeurs manquantes")  
    missing_values = df.isnull().sum()  
    st.write(missing_values[missing_values > 0])  
    # Visualisation de la relation entre l'âge et le métier  
      # Convertir 'job' en catégorie pour un meilleur affichage  
    df['job'] = df['job'].astype('category')  

    # Créer le graphique d'Altair  
    age_job_chart = (  
    alt.Chart(df)  
    .mark_circle(size=60)  
    .encode(  
        x=alt.X('age:Q', title='Âge'),  # Assurez-vous que l'âge est traité comme une mesure  
        y=alt.Y('job:O', title='Métier', sort=None),  # Utiliser Order pour classer les métiers  
        color='y:N',  # Une couleur par catégorie 'y'  
        tooltip=['age:Q', 'job:N', 'y:N']  # Améliorer les info-bulles  
    )  
    .properties(  
        title='Relation entre l\'âge et le métier',  
        width=600,  
        height=400  
        )  
        .interactive()  
    )  

    # Afficher le graphique dans Streamlit  
    st.subheader("Relation entre l'âge et le métier")  
    st.altair_chart(age_job_chart, use_container_width=True)      
  

elif st.session_state.page_selection == 'apprentissage_automatique':
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier  
    from sklearn.metrics import classification_report, confusion_matrix  
    # Séparation des données en ensembles d'entraînement et de test  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    # Création et entraînement du modèle  
    model = RandomForestClassifier(n_estimators=100, random_state=42)  
    model.fit(X_train, y_train)  

elif st.session_state.page_selection == 'prediction':  
    # Page Prédiction  
    st.title("🔮 Prédiction")  
    
    # Formulaire pour saisir les caractéristiques  
    age = st.number_input("Âge du client", min_value=18, max_value=120, value=30)  
    duration = st.number_input("Durée du contact (seconds)", min_value=0, value=60)  
    campaign = st.number_input("Nombre de contacts lors de la campagne", min_value=1, value=1)  
    
    if st.button("Prédire"): 
        from sklearn.model_selection import train_test_split  
        from sklearn.preprocessing import StandardScaler  
        from sklearn.ensemble import RandomForestClassifier
        import seaborn as sns  
        import matplotlib.pyplot as plt  
    try:  
            # Prétraitement potentiel des données d'entrée et des caractéristiques  
            # (Assurez-vous que le modèle est déjà formé au préalable et chargé ici)  
            X = df[['age', 'duration', 'campaign']]  # Ajustez selon vos colonnes de caractéristiques.  
            y = df['y']  # Cible à prédire  
            
            # Splitting and training d'un modèle d'exemple  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  
            model = RandomForestClassifier()  
            model.fit(X_train, y_train)  
            
            prediction = model.predict([[age, duration, campaign]])  
            subscription_status = "Oui" if prediction[0] == 'yes' else "Non"  
            st.success(f"Le client va-t-il souscrire au produit ? : **{subscription_status}**") 
            # Prédictions  
            y_pred = model.predict(X_test)  

            # Évaluation  
            print(confusion_matrix(y_test, y_pred))  
            print(classification_report(y_test, y_pred))
    except Exception as e:  
            st.error(f"Une erreur est survenue : {e}")
