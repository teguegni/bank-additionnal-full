import streamlit as st  
import pandas as pd  
#import matplotlib.pyplot as plt  
#import seaborn as sns  
#from sklearn.model_selection import train_test_split  
#from sklearn.ensemble import RandomForestClassifier  

# Chargement des données  
@st.cache_data  
def load_data():  
    # Remplacez le chemin ci-dessous par le chemin de votre fichier .csv  
    return pd.read_csv('bank-additional-full.csv', sep=';')  

data = load_data()  

# Affichage des données  
st.title("Prédiction de la souscription d'un dépôt à terme")  
st.write("Voici un aperçu des données :")  
st.dataframe(data.head())  

# Feature engineering ou sélection des caractéristiques, selon votre modèle  
features = data.drop('y', axis=1)  # 'y' est la cible  
target = data['y']  

# Conversion des variables catégorielles  
features = pd.get_dummies(features)  

# Séparation des données pour l'entraînement et le test  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)  

# Entraînement du modèle  
model = RandomForestClassifier()  
model.fit(X_train, y_train)  

# Prédictions  
if st.button('Faire une prédiction'):  
    # Ici, récupérez les entrées de l'utilisateur, par exemple via des champs de texte ou des sélectionneurs  
    inputs = {}  
    # Exemple : inputs['age'] = st.slider("Age", 18, 100)  
    # Ajoutez d'autres champs en fonction de vos données  
    input_data = pd.DataFrame([inputs])  
    prediction = model.predict(input_data)  
    
    if prediction[0] == 'yes':  
        st.success("Le client est susceptible de souscrire un dépôt à terme.")  
    else:  
        st.error("Le client n'est pas susceptible de souscrire.")  
    
# Visualisations  
st.write("Visualisation des données")  
sns.countplot(x='y', data=data)  
plt.title('Souscription de dépôts à terme')  
st.pyplot(plt)  

sns.boxplot(x='y', y='age', data=data)  
plt.title('Âge par souscription')  
st.pyplot(plt)
