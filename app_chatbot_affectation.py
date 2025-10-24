import streamlit as st
import pickle
import pandas as pd
import io
import os

# --- Configuration de la page ---
st.set_page_config(
    page_title="Assistant d’Affectation IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chargement du modèle ---
@st.cache_resource
def load_model():
    with open("modele_affectation_SMOTE.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()
model = data["model"]
encoders = data["encoders"]
feature_names = data["feature_names"]
target_encoder = data["target_encoder"]

# --- Initialisation ---
if "step" not in st.session_state:
    st.session_state.step = 0
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

def next_step():
    st.session_state.step += 1

def restart():
    st.session_state.step = 0
    st.session_state.user_info = {}
    st.rerun()

# --- Navigation ---
menu = st.sidebar.radio("📋 Menu", ["🧩 Assistant d'affectation", "📊 Tableau des affectations"])

# =========================================================================================
# 🧩 ASSISTANT D'AFFECTATION
# =========================================================================================
if menu == "🧩 Assistant d'affectation":
    st.title("🤖 Assistant d’affectation intelligente")
    st.write("Réponds aux questions ci-dessous pour découvrir le **service le plus adapté** à ton profil.")
    progress = st.progress(st.session_state.step / 7)

    # --- Étape 0 : Nom ---
    if st.session_state.step == 0:
        st.subheader("1️⃣ Ton nom")
        nom = st.text_input("Quel est ton **nom** ?")
        if st.button("➡️ Suivant", type="primary") and nom:
            st.session_state.user_info["Nom"] = nom
            next_step()
            st.rerun()

    # --- Étape 1 : Prénom ---
    elif st.session_state.step == 1:
        st.subheader("2️⃣ Ton prénom")
        prenom = st.text_input("Quel est ton **prénom** ?")
        if st.button("➡️ Suivant", type="primary") and prenom:
            st.session_state.user_info["Prénom"] = prenom
            next_step()
            st.rerun()

    # --- Étape 2 : Âge ---
    elif st.session_state.step == 2:
        st.subheader("3️⃣ Ton âge")
        age = st.number_input("Quel est ton **âge** ?", min_value=1, max_value=120, step=1)
        if st.button("➡️ Suivant", type="primary") and age > 0:
            st.session_state.user_info["Âge"] = age
            next_step()
            st.rerun()

    # --- Étape 3 : Motif ---
    elif st.session_state.step == 3:
        st.subheader("4️⃣ Motif de la demande")
        motifs = sorted(list(encoders["PATIENT Motif Demande"].classes_))
        motif = st.selectbox("Sélectionne le **motif de ta demande** :", ["— Sélectionner —"] + motifs)
        if st.button("➡️ Suivant", type="primary") and motif != "— Sélectionner —":
            st.session_state.user_info["PATIENT Motif Demande"] = motif
            next_step()
            st.rerun()

    # --- Étape 4 : Diagnostic ---
    elif st.session_state.step == 4:
        st.subheader("5️⃣ Diagnostic")
        diagnostics = sorted(list(encoders["PATIENT Diagnostique"].classes_))
        diag = st.multiselect("Sélectionne un ou plusieurs **diagnostics** :", diagnostics)
        if st.button("➡️ Continuer", type="primary") and diag:
            st.session_state.user_info["PATIENT Diagnostique"] = diag
            next_step()
            st.rerun()

    # --- Étape 5 : Type de logement ---
    elif st.session_state.step == 5:
        st.subheader("6️⃣ Type de logement")
        logements = sorted(list(encoders["PATIENT Type Logement"].classes_))
        logement = st.selectbox("Quel est ton **type de logement** ?", ["— Sélectionner —"] + logements)
        if st.button("➡️ Suivant", type="primary") and logement != "— Sélectionner —":
            st.session_state.user_info["PATIENT Type Logement"] = logement
            next_step()
            st.rerun()

    # --- Étape 6 : Tranche de revenu ---
    elif st.session_state.step == 6:
        st.subheader("7️⃣ Tranche de revenu")
        revenus = sorted(list(encoders["PATIENT Tranche Revenue"].classes_))
        revenu = st.selectbox("Quelle est ta **tranche de revenu** ?", ["— Sélectionner —"] + revenus)
        if st.button("Voir le résultat 🔍", type="primary") and revenu != "— Sélectionner —":
            st.session_state.user_info["PATIENT Tranche Revenue"] = revenu
            next_step()
            st.rerun()

    # --- Étape 7 : Résultat final ---
    elif st.session_state.step == 7:
        st.subheader("📊 Résultat de ton évaluation")

        diag_final = st.session_state.user_info["PATIENT Diagnostique"][0]
        user_data = pd.DataFrame({
            "PATIENT Motif Demande": [st.session_state.user_info["PATIENT Motif Demande"]],
            "PATIENT Diagnostique": [diag_final],
            "PATIENT Type Logement": [st.session_state.user_info["PATIENT Type Logement"]],
            "PATIENT Tranche Revenue": [st.session_state.user_info["PATIENT Tranche Revenue"]]
        })

        # Encodage
        for col in feature_names:
            user_data[col] = encoders[col].transform(user_data[col])

        # Prédiction
        prediction_encoded = model.predict(user_data)[0]
        prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

        # Résumé clair
        st.success(f"🎯 {st.session_state.user_info['Prénom']} {st.session_state.user_info['Nom']} "
                   f"({st.session_state.user_info['Âge']} ans) est orienté vers le service **{prediction_label}**.")
        st.info("⏳ Un intervenant confirmera ta demande par courriel sous peu.")

        # --- Sauvegarde dans le fichier Excel commun ---
        result_df = pd.DataFrame([st.session_state.user_info])
        result_df["Service Affecté"] = prediction_label
        file_path = "historique_affectations.xlsx"

        if os.path.exists(file_path):
            old_df = pd.read_excel(file_path)
            combined_df = pd.concat([old_df, result_df], ignore_index=True)
            combined_df.to_excel(file_path, index=False, engine="openpyxl")
        else:
            result_df.to_excel(file_path, index=False, engine="openpyxl")

        st.success("✅ Résultat enregistré dans le fichier **historique_affectations.xlsx**.")
        st.button("🔄 Recommencer", on_click=restart)

# =========================================================================================
# 📊 TABLEAU DE BORD DES AFFECTATIONS
# =========================================================================================
elif menu == "📊 Tableau des affectations":
    st.title("📊 Tableau de bord des affectations enregistrées")

    file_path = "historique_affectations.xlsx"

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)

        st.subheader("📈 Données enregistrées")
        st.dataframe(df, use_container_width=True)

        # Filtres dynamiques
        with st.expander("🔎 Filtres"):
            col1, col2, col3 = st.columns(3)
            with col1:
                service = st.selectbox("Filtrer par service :", ["Tous"] + sorted(df["Service Affecté"].unique().tolist()))
            with col2:
                motif = st.selectbox("Filtrer par motif :", ["Tous"] + sorted(df["PATIENT Motif Demande"].unique().tolist()))
            with col3:
                logement = st.selectbox("Filtrer par logement :", ["Tous"] + sorted(df["PATIENT Type Logement"].unique().tolist()))

        filtered_df = df.copy()
        if service != "Tous":
            filtered_df = filtered_df[filtered_df["Service Affecté"] == service]
        if motif != "Tous":
            filtered_df = filtered_df[filtered_df["PATIENT Motif Demande"] == motif]
        if logement != "Tous":
            filtered_df = filtered_df[filtered_df["PATIENT Type Logement"] == logement]

        st.subheader("📋 Résultats filtrés")
        st.dataframe(filtered_df, use_container_width=True)

        # Statistiques simples
        st.markdown("### 📊 Statistiques globales")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total bénéficiaires", len(df))
        with col2:
            st.metric("Nombre de services", len(df["Service Affecté"].unique()))
        with col3:
            top_service = df["Service Affecté"].mode()[0]
            st.metric("Service le plus attribué", top_service)

    else:
        st.warning("⚠️ Aucun enregistrement trouvé. Le fichier `historique_affectations.xlsx` sera créé automatiquement après la première affectation.")
