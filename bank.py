import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
import base64
import streamlit as st  

# Initialisation de la clé si elle n'existe pas  
if 'page_selection' not in st.session_state:  
    st.session_state.page_selection = 'valeur_par_defaut'  # Remplacez par une valeur par défaut appropriée
# Configuration de la page avec image de fond
def set_bg_hack(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

#set_bg_hack("background.png")  # Mettez votre image dans le même dossier et décommentez cette ligne

# -------------------------
# Configuration de base
st.set_page_config(
    page_title="Classification des Données Bancaires",
    page_icon="🏦",  
    layout="wide", 
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# -------------------------
# Barre latérale (identique à votre version originale)
# ... [Le reste de votre code de sidebar reste inchangé] ...

# -------------------------
# Contenu principal avec améliorations

if st.session_state.page_selection == 'analyse_exploratoire':
    st.title("🔍 Analyse Exploratoire")
    
    # Section interactive avec boutons
    with st.expander("Contrôles des Visualisations"):
        col1, col2 = st.columns(2)
        with col1:
            show_trend = st.checkbox("Afficher la tendance sur le graphique Âge/Métier")
        with col2:
            plot_type = st.radio("Type de visualisation", ["Nuage de points", "Histogramme"])

    # Graphique interactif
    if plot_type == "Nuage de points":
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x='age',
            y='job',
            color='y',
            tooltip=['age', 'job', 'y']
        )
        
        if show_trend:
            trend = chart.transform_regression('age', 'y').mark_line()
            chart = (chart + trend).interactive()
            
        st.altair_chart(chart, use_container_width=True)

    elif plot_type == "Histogramme":
        hist_chart = alt.Chart(df).mark_bar().encode(
            alt.X("age:Q", bin=True),
            y='count()',
            color='y'
        ).interactive()
        st.altair_chart(hist_chart, use_container_width=True)

    # Section matrice de corrélation avec bouton d'export
    st.subheader("Matrice de Corrélation Interactive")
    if st.button("Générer la matrice de corrélation"):
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        if st.download_button("Télécharger la matrice",
                             data=fig.to_image(format="png"),
                             file_name="correlation_matrix.png",
                             mime="image/png"):
            st.success("Matrice téléchargée avec succès!")

# ... [Le reste de vos pages reste inchangé] ...

# Page de prédiction améliorée
elif st.session_state.page_selection == 'prediction':
    st.title("🔮 Prédiction Interactive")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Âge du client", 18, 100, 30)
            duration = st.number_input("Durée du contact (secondes)", 0, 3600, 60)
        with col2:
            campaign = st.selectbox("Nombre de contacts", options=range(1, 50))
            previous = st.selectbox("Contacts précédents", options=range(0, 50))
        
        submitted = st.form_submit_button("Effectuer la prédiction")
        
        if submitted:
            try:
                # Prétraitement des données
                X = df[['age', 'duration', 'campaign', 'previous']]
                y = df['y'].map({'yes': 1, 'no': 0})
                
                # Entraînement du modèle
                model = RandomForestClassifier()
                model.fit(X, y)
                
                # Prédiction
                prediction = model.predict([[age, duration, campaign, previous]])
                result = "✅ Client intéressé" if prediction[0] == 1 else "❌ Client non intéressé"
                
                # Affichage stylisé
                st.markdown(f"""
                <div style='padding: 20px; 
                            border-radius: 10px; 
                            background-color: {'#d4edda' if prediction[0] == 1 else '#f8d7da'};
                            text-align: center;
                            margin: 20px 0;'>
                    <h3 style='color: {'#155724' if prediction[0] == 1 else '#721c24'};'>
                        {result}
                    </h3>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {str(e)}")

# ... [Le reste de votre code] ...
