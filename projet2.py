import streamlit as st 
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
# pip install Pillow 
from PIL import Image # manipuler des images en Python et qui inclut le module Image
#pip install streamlit-tags
from streamlit_tags import st_tags
import os # pour le chemin de stockage des fichiers de travail - en standart dans pandas
# pip install plotly
import plotly.express as px # affichage graph plotly
import csv # enregistre les données du formulaire dans un fichier csv  - en standart dans pandas
import time # pour datetime dans le formualaire
from datetime import datetime # pour datetime dans le formualaire
import re
import requests
import time
import random



# Config de la page.
st.set_page_config(
    page_title = "Projet 2",
    page_icon = ":movie_camera:",
    layout = "wide",
    )


# cacher les menus de streamlit
hide_streamlit_style = """
                <style> div[data-testid="stToolbar"] {visibility: hidden; height: 0%; position: fixed;}
                div[data-testid="stDecoration"] {visibility: hidden; height: 0%; position: fixed;}
                div[data-testid="stStatusWidget"] {visibility: hidden; height: 0%; position: fixed;}
                #MainMenu {visibility: hidden; height: 0%;}
                header {visibility: hidden; height: 0%;}
                footer {visibility: hidden; height: 0%;}
                </style> """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True
)


# Changer le répertoire de travail pour qu'il corresponde à votre situation en local
#os.chdir(r"C:\Users\cbrun\Desktop\WCS\Streamlit")

image = Image.open('cinema.jpg.webp')
st.sidebar.image(image, use_column_width=True) #caption="Description de l'image"

# barre de navigation vertical
st.sidebar.title("Sommaire")

# liste des pages 
pages = ["Projet 2", "Accueil", "Statistiques de films", "Moteur de recommandations de films", "Alerte"]

#page = st.sidebar.radio("Aller vers la page :", pages)
page = st.sidebar.radio("", pages)

#<nav class="navbar navbar-expand-lg navbar-dark" style="background-color: #3498DB;">




# -----------------------------------------------------------
# -----------------------------------------------------------
############################################
#### menu Projet 2
############################################

if page == pages[0]:


    st.markdown('<h1 id="section_projet" style="color: #BD4BFF; text-align: center;">Projet 2</h1>', unsafe_allow_html=True)
    st.markdown('<h1 style="color: #BD4BFF; text-align: center;">Système de recommandation de films</h1>', unsafe_allow_html=True)

    st.markdown("---")

    # declaration 2 colonnes
    col1, col2 = st.columns(2)
    col1.subheader("Résumé du projet")
    col2.subheader("Equipe ayant travaillé sur le projet")


    # Création de la colonne principale
    main_col = st.columns(2)

    # Contenu de la première sous-colonne de la première colonne principale
    sub_col1_main_col1 = main_col[0].columns(1)[0]

    sub_col1_main_col1.write("Durée : du 18/10/2023 au 1/12/2023 - 7 semaines")
    sub_col1_main_col1.write("Budget : 0 €")

    # Contenu de la deuxième sous-colonne de la première colonne principale
    sub_col2_main_col1, sub_col3_main_col1 = main_col[0].columns(2)

    # ...

    # Diviser la deuxième sous-colonne en 3 sous-colonnes de la deuxième colonne principale
    sub_col1_main_col2, sub_col2_main_col2, sub_col3_main_col2 = main_col[1].columns(3)

    # Sous-colonne 1
    sub_col1_main_col2.write("Claudette")
    image = Image.open('claudette.png')
    sub_col1_main_col2.image(image, use_column_width=False) # , caption='Claudette'

    # Sous-colonne 2
    sub_col2_main_col2.write("Florence")
    image = Image.open('florence.png')
    sub_col2_main_col2.image(image, use_column_width=False) # , caption='Florence'

    # Sous-colonne 3
    sub_col3_main_col2.write("Julien")
    image = Image.open('julien.png')
    sub_col3_main_col2.image(image, use_column_width=False) # , caption='Julien'


    st.markdown("---")

    st.subheader("Contexte du projet")

    # declaration 2 colonnes
    col1, col2 = st.columns([4,1])

    # col1, affiche le texte avec un espace réduit entre chaque phrase
    col1.write(
        "- Un cinéma en perte de vitesse situé dans la Creuse nous a contacté.\n"
        "- Il a décidé de passer le cap du digital en créant un site Internet taillé pour les locaux.\n"
        "- Pour aller encore plus loin, il demande de créer un moteur de recommandations de films qui à terme, enverra des notifications aux clients via Internet.\n"
        "- Pour l’instant, aucun client n’a renseigné ses préférences, Nous sommes dans une situation de **cold start**."
    )


    # Définir la largeur souhaitée de l'image gif
    #image_width = 300

    # Afficher l'image avec la largeur spécifiée
    gif = "https://www.forum.arassocies.com/uploads/monthly_2022_01/178048511_Travoltasalledecine.gif.049a18b2b123c0170d165f2ee2a9499d.gif"
    col2.image(gif, caption="Un cinema dans la Creuse", use_column_width=True)

    st.markdown("---")

    # declaration 2 colonnes
    col1, col2 = st.columns([1, 4])

    # col2, affiche l'image
    image = Image.open('creuse.png')
    col1.image(image, caption='carte de france avec focus sur la creuse', use_column_width=True) 

    col2.subheader("RESSOURCES")
    col2.write(
        "- Le client nous as donné une base de données de films basée sur la plateforme IMDb.\n"
        "- elle est composé de 6 datasets (title.akas.tsv.gz, title.basics.tsv.gz, title.crew.tsv.gz, title.episode.tsv.gz, title.principals.tsv.gzv, title.ratings.tsv.gz, name.basics.tsv.gz \n"
        "- Une base de données complémentaires venant de TMDB, contenant des données sur les pays des sociétés de production, le budget, les recettes et également un chemin vers les posters des films.\n"
        "- les datasets sont très volumineux, il y a plus de 7M films et 10M acteurs référencés.\n"
        "- La documentation expliquant brièvement la définition de chaque colonne et de chaque table\n"
        "- Le client nous demande de récupérer les images des films pour les afficher dans l'interface de recommandation\n"
        )

    st.markdown("---")

    # declaration 2 colonnes
    col1, col2 = st.columns(2)

    col1.subheader("ORGANISATION ET PLANNING")
    col1.write("- Semaine 1 et 2 : Appropriation et première exploration des données \n" 
               "- Semaine 3 et 4 : Jointures, filtres, nettoyage, recherche de corrélation \n"
               "- Semaine 5 et 6 : Machine learning, recommandations \n"
               "- Semaine 7 : Affinage, interface, présentation et Demo Day"
               )

    col2.subheader("PROCESS DE PRODUCTION")
    col2.write(
        "- Analyse exploratoire et nettoyage des BDD \n" 
        "- Developpement d'un algorithme de machine learning pour la recommandation \n" 
        "- Utilisation de l'API pour le moteur de recommandation \n" 
        "- Developpement du site avec Streamit"
        )

    st.markdown("---")

    # declaration 2 colonnes
    col1, col2 = st.columns(2)
    col1.subheader("OUTILS & TECHNOLOGIES UTILISES")
    col1.write("VS Code, Pandas, Numpy, Plotly, Scikit-learn, Streamlit, API Tmdb")



# -----------------------------------------------------------
# -----------------------------------------------------------
############################################
#### menu page Accueil 
############################################

elif page == pages[1] :



    st.markdown(
    '<h1 id="section_accueil" style="color: #BD4BFF; text-align: center;">Bienvenue sur notre moteur de recommandations de films</h1>',
    unsafe_allow_html=True) 

    #st.write("### Accueil")

    st.markdown("---")

    st.subheader("Vous trouverez : ")
    st.write(":bar_chart: Une rubrique de **Statistiques de films** qui contient des statistiques sur les films (type, durée), acteurs (nombre de films, type de films).")
    st.write(":clapper: Une rubrique de **Recommandations de films** qui contient des un moteur de recommandation de films en fonction de plusieurs critères")
    st.write(":email: Une rubrique d'**alerte** qui contient un formulaire pour vous inscrire à nos alertes")


    #image = Image.open('image_accueil.png')
    #st.image(image, caption='mosaique d\'affiche de fims')


    # Define the API key globally
    api_key = "684824421ee197f34e3394e5eb0d58b5"


    ### Affiche de film random

    # Function to fetch a random movie poster path from TMDb API
    def get_random_movie_poster_path():
        discover_url = "https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": api_key,
            "language": "en",
            "sort_by": "popularity.desc",
            "include_adult": False,
            "include_video": False,
            "page": 1
        }

        response = requests.get(discover_url, params=params)
        data = response.json()

        # Check if there are results
        if 'results' in data and data['results']:
            # Get random movies from the results
            random_movies = random.sample(data['results'], 12)  # Select 8 random movies
            return [movie['poster_path'] for movie in random_movies]
        else:
            return [None] * 12


    while True:
        # Get random movie poster paths
        random_poster_paths = get_random_movie_poster_path()

        # Display the posters in a 4x2 grid
        for i in range(0, 12, 6):
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            for j, col in enumerate([col1, col2, col3, col4, col5, col6]):
                poster_path = random_poster_paths[i + j]
                if poster_path:
                    col.image(f"https://image.tmdb.org/t/p/original{poster_path}", use_column_width=True) #, caption=f"Movie {i + j + 1}"
                else:
                    st.warning(f"Aucun poster trouvé {i + j + 1}")

        # Pause execution for 5 seconds
        time.sleep(5)
        st.text("")

        # Clear the output for the next iteration
        st.rerun()



# -----------------------------------------------------------
# -----------------------------------------------------------
############################################
#### menu page Statistiques de films 
############################################

elif page == pages[2]:


    st.markdown(
    '<h1 id="section_statistiques" style="color: #BD4BFF; text-align: center;">Statistiques de la Base de données</h1>',
    unsafe_allow_html=True)

    st.markdown("---")

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # bdd films unique
    link1 = "dataset/df_imdb_filmUnik.parquet"
    #link1 = "https://drive.google.com/file/d/1EZOyDpi_o7OXDQvpUjC9d9rEU_sl7HHi/view?usp=sharing"
    df_imdb_filmUnik = pd.read_parquet(link1)

    st.write("Nous avons à notre disposition le dataset de films (imdb) qui contient des données sur les films. ")
    st.write("\n")  # Ajoute un saut de ligne manuel

    st.subheader('Aperçu du dataframe')
    st.write("Chaque observation en ligne correspond à un film. Chaque variable en colonne est une caractéristique des films ")
    st.dataframe(df_imdb_filmUnik.head())
    #st.write("Dimensions du dataframe")
    #st.write(df_imdb_filmUnik.shape)
    st.write("\n")  # Ajoute un saut de ligne manuel

    st.write("Dans un premier temps : Appropriation des données et exploration. construction du Schéma de la BDD. Jointure, Filtre, nettoyage")
    st.write("Puis nous analyserons visuellement pour en extraire des informations selon certains axes d'étude.")

    st.subheader("**Graphiques & Tableaux**")

    # -----------------------------------------------------------

    # Options pour le menu déroulant
    options = ["", #option vide pour ne rien afficher
            "Nombre ce films par genre (type Movie)",
            "Durée moyenne Durée moyenne des films en fonction de leur date de sortie",
            "Nombre de films sortis par année de sortie",
            "Top 10 des réalisateurs les mieux cotés",
            "Top 10 des réalisateurs les plus productifs",
            "Meilleur film par décennie",
            "Top 10 des films par genre",
            "Top 10 des genres les plus présents par décennie",
            "Acteurs, Actrices & Directeurs les plus présents"
            ]

    selected_chart = st.selectbox("Sélectionnez le graphique à afficher", options)


    if selected_chart == "Nombre ce films par genre (type Movie)":

        # ********************************************************
        # Nombre ce films par genre (type Movie) - Claudette
        # ********************************************************

            # Comptez le nombre de films par genre
            genre_counts = df_imdb_filmUnik['imdb_genres'].value_counts().sort_values(ascending=False)

            # Créer un graphique à barres avec Plotly Express
            fig1 = px.bar(x=genre_counts.index, y=genre_counts.values, labels={'x': 'Genre', 'y': 'Nombre de Films'},
                        title="Nombre de Films par Genre (Type: 'movie') - 1950 - 2023", 
                        text=genre_counts.values, orientation='v')

            # Personnaliser le graphique
            fig1.update_layout(xaxis=dict(tickangle=-45), showlegend=False)

            # Afficher le graphique
            st.plotly_chart(fig1)


        # ********************************************************
        # Durée moyenne Durée moyenne des films en fonction de leur date de sortie - Claudette
        # ********************************************************

    elif selected_chart == "Durée moyenne Durée moyenne des films en fonction de leur date de sortie":


            # Calculer la moyenne de la durée des films par année
            average_runtimes = df_imdb_filmUnik.groupby('imdb_startYear')['imdb_runtimeMinutes'].mean().reset_index()

            # Créer un graphique en ligne avec Plotly Express
            fig2 = px.line(average_runtimes, x='imdb_startYear', y='imdb_runtimeMinutes', 
                        labels={'imdb_startYear': 'Année de sortie', 'imdb_runtimeMinutes': 'Moyenne de la durée (en min)'},
                        title="Durée moyenne des films en fonction de leur date de sortie - 1950 - 2023")

            # Personnaliser le graphique
            fig2.update_layout(xaxis=dict(tickangle=-45))

            # Afficher le graphique
            st.plotly_chart(fig2)


        # ********************************************************
        # Nombre de films sortis par année de sortie - Claudette
        # ********************************************************

    elif selected_chart == "Nombre de films sortis par année de sortie":

            # Utilise Plotly Express pour créer un graphique en barres
            fig3 = px.histogram(df_imdb_filmUnik, x='imdb_startYear', title="Nombre de films sortis par année de sortie - 1950 - 2023",
                                labels={'imdb_startYear': 'Année de sortie', 'count': 'Nombre de films'},
                                color_discrete_sequence=['skyblue'])

            # Personnalisation des axes
            fig3.update_layout(xaxis=dict(tickangle=-45))

            # Afficher le graphique
            st.plotly_chart(fig3)


        # ********************************************************
        # Top 10 des réalisateurs les mieux cotés - Claudette
        # ********************************************************

    elif selected_chart == "Top 10 des réalisateurs les mieux cotés":

            # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
            # bdd directeur unique
            link2 = "dataset/df_imdb_DirUnik.parquet"
            #link2 ="https://drive.google.com/file/d/13cZwpuJpoeLPIImg7zQ3TbSqcn7zHqtZ/view?usp=sharing"
            df_imdb_DirUnik = pd.read_parquet(link2)


            # Créez un DataFrame pour les réalisateurs les mieux cotés par décennie
            top_directors_by_decade = (
                df_imdb_DirUnik.groupby(['imdb_primaryName', 'imdb_decade'])
                .agg({'imdb_averageRating': 'mean', 'imdb_numVotes': 'sum'})
                .reset_index()
            )

            # Sélectionnez les décennies uniques et triez-les par ordre croissant
            unique_decades = sorted(top_directors_by_decade['imdb_decade'].unique())

            # Sélectionnez la décennie avec un widget selectbox
            selected_decade = st.selectbox("Sélectionnez une décennie", unique_decades)

            # Filtrer le DataFrame pour la décennie sélectionnée
            if selected_decade:
                selected_decade_data = top_directors_by_decade[top_directors_by_decade['imdb_decade'] == selected_decade]
                # Afficher les 10 réalisateurs les mieux cotés par décennie
                top_directors_decade = selected_decade_data.sort_values(by='imdb_averageRating', ascending=False).head(10)
            else:
                top_directors_decade = pd.DataFrame()

            # Utilisez Plotly Express pour créer un graphique à barres
            fig4 = px.bar(top_directors_decade, x='imdb_primaryName', y='imdb_averageRating',
                        title=f'Top 10 des réalisateurs les mieux cotés dans les années {selected_decade}' if selected_decade else 'Sélectionnez une décennie',
                        labels={'imdb_primaryName': 'Réalisateurs', 'imdb_averageRating': 'Note moyenne'})

            # Personnalisez les axes
            fig4.update_layout(xaxis=dict(tickangle=-45))

            # Affichez le graphique
            st.plotly_chart(fig4)


        # ********************************************************
        # Top 10 des réalisateurs les plus productifs - Claudette
        # ********************************************************

    elif selected_chart == "Top 10 des réalisateurs les plus productifs":

                # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
                # bdd directeur unique
                link2 = "dataset/df_imdb_DirUnik.parquet"
                #link2 ="https://drive.google.com/file/d/13cZwpuJpoeLPIImg7zQ3TbSqcn7zHqtZ/view?usp=sharing"
                df_imdb_DirUnik = pd.read_parquet(link2)

                top_directors_by_decade = df_imdb_DirUnik

                # Créez un DataFrame pour le nombre de films par réalisateur et décennie
                top_directors_by_decade = (
                    df_imdb_DirUnik.groupby(['imdb_primaryName', 'imdb_decade'])
                    .size()
                    .reset_index(name='Nombre de films')
                )

                # Sélectionnez les décennies uniques et triez-les par ordre croissant
                unique_decades = sorted(top_directors_by_decade['imdb_decade'].unique())

                # Sélectionnez la décennie avec un widget selectbox
                selected_decade = st.selectbox("Sélectionnez une décennie", unique_decades)

                # Filtrer le DataFrame pour la décennie sélectionnée
                if selected_decade:
                    selected_decade_data = top_directors_by_decade[top_directors_by_decade['imdb_decade'] == selected_decade]
                    # Afficher le top 10 par décennie
                    top_directors_decade = selected_decade_data.sort_values(by='Nombre de films', ascending=False).head(10)
                else:
                    top_directors_decade = pd.DataFrame()

                # Utilisez Plotly Express pour créer un graphique à barres
                fig5 = px.bar(top_directors_decade, x='imdb_primaryName', y='Nombre de films',
                            title=f'Top 10 des réalisateurs avec le plus grand nombre de films dans les années {selected_decade}' if selected_decade else 'Sélectionnez une décennie',
                            labels={'imdb_primaryName': 'Réalisateurs', 'Nombre de films': 'Nombre de films'})

                # Personnalisez les axes
                fig5.update_layout(xaxis=dict(tickangle=-45))

                # Affichez le graphique
                st.plotly_chart(fig5)




        # ********************************************************
        # Best_movie_decade - Meilleur film par décennie - Julien
        # ********************************************************

    elif selected_chart == "Meilleur film par décennie":

        # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # bdd Best_movie_decade
            link3 = "dataset/Best_movie_decade.parquet"
            #link3 ="https://drive.google.com/file/d/1rAAo4UnSGfz0RwbuEePZMH5VELxjTljL/view?usp=sharing"
            Best_movie_decade = pd.read_parquet(link3)

            # Convertir la colonne 'startYear' en chaînes de caractères (str) avant d'utiliser .str.extract()
            Best_movie_decade['startYear'] = Best_movie_decade['startYear'].astype(str)

            # Extraire l'année du format 'YYYY-01-01' dans la colonne 'startYear'
            Best_movie_decade['startYear'] = Best_movie_decade['startYear'].str.extract(r'(\d{4})')

            # Convertir la colonne 'startYear' en type entier
            Best_movie_decade['startYear'] = Best_movie_decade['startYear'].astype(int)

            # Créer la colonne 'decade' en extrayant la décennie à partir de 'startYear'
            Best_movie_decade['decade'] = (Best_movie_decade['startYear'] // 10) * 10

            # Filtrer les films avec un minimum de votes
            min_votes_threshold = 1000
            filtered_movies = Best_movie_decade[Best_movie_decade['numVotes'] >= min_votes_threshold]

            # Trouver le meilleur film par décennie basé sur la note moyenne
            best_movies_by_decade = filtered_movies.groupby('decade').apply(lambda x: x.loc[x['averageRating'].idxmax()])

            # Afficher les meilleurs films par décennie sous forme de DataFrame
            result_df = best_movies_by_decade[['title', 'startYear', 'averageRating', 'numVotes', 'decade']].reset_index(drop=True)

            # Renomer les titres des colonnes
            # Renommer les colonnes
            result_df = result_df.rename(columns={
                'title': 'Titre',
                'startYear': 'Année de sortie',
                'averageRating': 'Note moyenne',
                'numVotes': 'Nombre de votes',
                'decade': 'Décennie'
            })

            # Convertir la colonne 'Année de sortie' en type int
            result_df['Année de sortie'] = result_df['Année de sortie'].astype(int)

            # Formater la colonne 'Année de sortie' au format YYYY
            result_df['Année de sortie'] = result_df['Année de sortie'].apply(lambda x: f"{x:04d}")

            # Formater la colonne 'Nombre de votes' à la française avec un espace comme séparateur des milliers
            result_df['Nombre de votes'] = result_df['Nombre de votes'].apply(lambda x: '{:,.0f}'.format(x).replace(',', ' '))

            st.dataframe(result_df)




        # ********************************************************
        # Top_10_genres_global - Top 10 des films par genre - Julien
        # ********************************************************

    elif selected_chart == "Top 10 des films par genre":

        # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # bdd Top_10_genres_global
            link4 = "dataset/Top_10_genres_global.parquet"
            #link3 ="https://drive.google.com/file/d/1faBYvqfN8O_HBteOKUxZ-EEafx1Dv9io/view?usp=sharing"
            Top_10_genres_global = pd.read_parquet(link4)

            # Compter les éléments uniques dans chaque colonne 'genre1', 'genre2', 'genre3'
            genre1_counts = Top_10_genres_global['genre1'].value_counts()
            genre2_counts = Top_10_genres_global['genre2'].value_counts()
            genre3_counts = Top_10_genres_global['genre3'].value_counts()

            # Additionner le compte de chaque genre de chaque colonne
            total_genre_counts = genre1_counts.add(genre2_counts, fill_value=0).add(genre3_counts, fill_value=0)
            # Classer les genres par ordre décroissant de fréquence
            sorted_genre_counts = total_genre_counts.sort_values(ascending=False)

            # Afficher les genres triés par ordre décroissant
            top_10_genres = sorted_genre_counts.head(10)

            # Créer un DataFrame à partir des 10 genres les plus présents
            top_10_genres_df = pd.DataFrame({'Genre': top_10_genres.index, 'Count': top_10_genres.values})

            # Créer un graphique à barres avec Plotly
            fig6 = px.bar(top_10_genres_df, x='Genre', y='Count', title='Top 10 Genres', labels={'Count': 'Nombre de Films', 'Genre': 'Genre'})

            # Ajouter le texte au-dessus des barres avec les counts
            fig6.update_traces(texttemplate='%{y}', textposition='outside')

            # Personnaliser le layout
            fig6.update_layout(yaxis_title='Nombre de Films', xaxis_title='Genre', showlegend=False)

            # Affiche le graphique
            st.plotly_chart(fig6)



        # ********************************************************
        # top_10_genres_decade - Top 10 des genres les plus présents par décennie - Julien
        # ********************************************************

    elif selected_chart == "Top 10 des genres les plus présents par décennie":

        # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # bdd top_10_genres_decade
            link4 = "dataset/Top_10_genres_global.parquet"
            #link4 ="https://drive.google.com/file/d/1faBYvqfN8O_HBteOKUxZ-EEafx1Dv9io/view?usp=sharing"
            top10decade = pd.read_parquet(link4)

            # Convertir la colonne 'startYear' en type entier (si elle n'est pas déjà de ce type)
            top10decade['startYear'] = top10decade['startYear'].astype(str).str.extract(r'(\d{4})')

            # Compter les genres par décennie
            genre_counts_by_decade = top10decade.groupby(['decade', 'genres']).size().reset_index(name='count')

            # Obtenir les 10 genres les plus présents par décennie
            top_10_genres_by_decade = genre_counts_by_decade.groupby('decade').apply(lambda x: x.nlargest(10, 'count')).reset_index(drop=True)

            # Compter les genres les plus présents par décennie dans la colonne 'genre1'
            top_genres_genre1 = top10decade.groupby(['decade', 'genre1']).size().reset_index(name='count_genre1')
            top_genres_genre1 = top_genres_genre1.groupby('decade').apply(lambda x: x.nlargest(10, 'count_genre1')).reset_index(drop=True)

            # Compter les genres les plus présents par décennie dans la colonne 'genre2'
            top_genres_genre2 = top10decade.groupby(['decade', 'genre2']).size().reset_index(name='count_genre2')
            top_genres_genre2 = top_genres_genre2.groupby('decade').apply(lambda x: x.nlargest(10, 'count_genre2')).reset_index(drop=True)

            # Compter les genres les plus présents par décennie dans la colonne 'genre3'
            top_genres_genre3 = top10decade.groupby(['decade', 'genre3']).size().reset_index(name='count_genre3')
            top_genres_genre3 = top_genres_genre3.groupby('decade').apply(lambda x: x.nlargest(10, 'count_genre3')).reset_index(drop=True)

            # Fusionner les DataFrames des genres par décennie
            top_genres_by_decade = pd.merge(top_genres_genre1, top_genres_genre2, on='decade', how='outer', suffixes=('_genre1', '_genre2'))
            top_genres_by_decade = pd.merge(top_genres_by_decade, top_genres_genre3, on='decade', how='outer')

            # Additionner les comptes de chaque genre pour chaque décennie
            top_genres_by_decade['total_count'] = top_genres_by_decade['count_genre1'] + top_genres_by_decade['count_genre2'] + top_genres_by_decade['count_genre3']

            # Sélectionner les colonnes pertinentes
            top_genres_by_decade = top_genres_by_decade[['decade', 'genre1', 'genre2', 'genre3', 'total_count']]

            # Créer une liste de colonnes de genre
            genre_columns = ['genre1', 'genre2', 'genre3']

            # Initialiser un dictionnaire pour stocker les comptes globaux de chaque genre par décennie
            genre_counts_by_decade = {}

            # Boucle à travers chaque colonne de genre
            for genre_column in genre_columns:
                # Grouper par décennie et le genre de la colonne actuelle, puis sommer les occurrences
                genre_counts = top10decade.groupby(['decade', genre_column]).size().reset_index(name='count')
                # Ajouter les comptes au dictionnaire
                for index, row in genre_counts.iterrows():
                    decade = row['decade']
                    genre = row[genre_column]
                    count = row['count']
                    if (decade, genre) not in genre_counts_by_decade:
                        genre_counts_by_decade[(decade, genre)] = count
                    else:
                        genre_counts_by_decade[(decade, genre)] += count

            # Afficher les comptes globaux de chaque genre par décennie
            for key, value in genre_counts_by_decade.items():
                decade, genre = key
                print(f"{decade} {genre}: {value} occurences")


            # Créer une liste pour stocker les données
            genre_counts_data = []

            # Boucle à travers chaque élément du dictionnaire genre_counts_by_decade
            for key, value in genre_counts_by_decade.items():
                decade, genre = key
                genre_counts_data.append((decade, genre, value))

            # Créer un DataFrame à partir de la liste de données
            genre_counts_df = pd.DataFrame(genre_counts_data, columns=['Decade', 'Genre', 'Occurrences'])

            # Trier le DataFrame par décennie de manière ascendante et par occurrences de manière descendante
            genre_counts_df_sorted = genre_counts_df.sort_values(by=['Decade', 'Occurrences'], ascending=[True, False])

             # Sélectionner les décennies uniques
            unique_decades = genre_counts_df_sorted['Decade'].unique()

            # Sélectionner la décennie avec un widget selectbox
            selected_decade = st.selectbox("Sélectionnez une décennie", unique_decades)

            # Filtrer le DataFrame pour la décennie sélectionnée
            selected_decade_data = genre_counts_df_sorted[genre_counts_df_sorted['Decade'] == selected_decade]

            # Sélectionner les 10 genres les plus présents par décennie
            top_10_genres = selected_decade_data.head(10)[::-1]

            # Créer un graphique à barres horizontales avec Plotly Express
            fig7 = px.bar(top_10_genres, x='Occurrences', y='Genre', orientation='h', 
                        title=f'Top 10 des genres les plus présents dans les années {selected_decade}',
                        labels={'Occurrences': 'Occurrences', 'Genre': 'Genre'})

            # Affiche le graphique
            st.plotly_chart(fig7)


        # ********************************************************
        # Acteurs les plus présents - Florence
        # ********************************************************


    elif selected_chart == "Acteurs, Actrices & Directeurs les plus présents":

        # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
        # bdd Acteurs les plus présents
        link5 = "dataset/gbV6Actors.parquet"
        # link5 ="https://drive.google.com/file/d/128Si_11-jsECEYZsopYDcJaJMG4eH6tx/view?usp=sharing"
        gbV6Actors = pd.read_parquet(link5)

        # Interface utilisateur avec Streamlit
        st.subheader("**Analyse de Période pour les Acteurs, Actrices & Directeurs**")

        # Saisie des valeurs par l'utilisateur avec des widgets interactifs
        annee1 = st.slider("Saisir l'année de début", min_value=1950, max_value=2022, value=1950)
        annee2 = st.slider("Saisir l'année de fin", min_value=1950, max_value=2022, value=2022)
        genre = st.selectbox("Saisir la catégorie", [''] + list(gbV6Actors['category'].unique()))


        # Fonction pour delimiter les périodes
        def periode_delimiter(annee1, annee2, genre):
            # Filtrer les données en fonction des critères
            periode = gbV6Actors.loc[
                (gbV6Actors['decade'] >= annee1)
                & (gbV6Actors['decade'] < annee2)
                & (gbV6Actors['category'] == genre),
                'primaryName'
            ]

            # Créer un DataFrame avec le décompte des valeurs et convertir 'primaryName' en chaînes de caractères
            df = pd.DataFrame({'primaryName': periode.value_counts(sort=True).head(5).index, 'nb_de_film': periode.value_counts(sort=True).head(5).values})

            # Essayer de convertir la colonne 'primaryName' en chaînes de caractères
            try:
                df['primaryName'] = df['primaryName'].astype(str)
            except Exception as e:
                # Afficher l'erreur et les valeurs problématiques
                st.error(f"Erreur lors de la conversion primaryName en str : {e}")
                st.write("Valeurs problématiques dans la colonne 'primaryName' lors de la conversion primaryName en str :", df[df['primaryName'].apply(lambda x: not isinstance(x, str))])

            # Fusionner avec les données additionnelles
            actors_KFT = gbV6Actors.loc[gbV6Actors.KFT1 == gbV6Actors.originalTitle, ['KFT1', 'originalTitle', 'primaryName', 'age']]

            # Remplir les valeurs manquantes
            actors_KFT['KFT1'] = actors_KFT['KFT1'].fillna('Inconnu')
            actors_KFT['age'] = actors_KFT['age'].fillna('Inconnu')

            # Renommer les colonnes pour éviter les conflits lors de la fusion
            actors_KFT = actors_KFT.rename(columns={'primaryName': 'primaryName_KFT', 'age': 'age_KFT', 'KFT1': 'KFT1'})

            # Effectuer la fusion en spécifiant un suffixe pour les colonnes en conflit
            df_final = pd.merge(df, actors_KFT, how='left', left_on=['primaryName'], right_on=['primaryName_KFT'], suffixes=('_df', '_actors_KFT'))

            # Renommer les colonnes finales
            df_final = df_final[['primaryName', 'nb_de_film', 'age_KFT', 'KFT1']].rename(columns={'primaryName': 'Name', 'age_KFT' : 'Age', 'KFT1': 'Connu_pour_ces_films'})

            return df_final


        # Appeler la fonction avec les paramètres sélectionnés par l'utilisateur
        resultat = periode_delimiter(annee1, annee2, genre)

        # Afficher le résultat final
        st.dataframe(resultat)




# -----------------------------------------------------------
# -----------------------------------------------------------
############################################
#### menu page recommandations de films
############################################

elif page == pages[3]:


    # Stocker la valeur initiale de title_query
    initial_title_query = ""

    # Initialiser la session avec un dictionnaire vide si nécessaire
    if 'session_state' not in st.session_state:
        st.session_state.session_state = {}

        # Initialiser title_query avec la valeur initiale
        st.session_state.title_query = initial_title_query

    # Ajoutez un print ici
    #print("Initial title_query:", st.session_state.title_query)

    #st.header('Moteur de recommandations de films')
    st.markdown('<h1 id="section_recommandations" style="color: #BD4BFF; text-align: center;">Moteur de recommandations de films</h1>',
                unsafe_allow_html=True)

    st.markdown("---")

    st.subheader('Votre choix de film')

    # Define the API key globally
    api_key = "684824421ee197f34e3394e5eb0d58b5"

    #### Les fonctions ####

    # Fonction pour obtenir les détails d'un film de TMDb API
    def get_movie_details(title, language='fr-FR'):
        base_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": api_key,
            "query": title,
            "language": language  # Specify the language parameter
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        # Vérifier si la requête a réussi
        if 'results' in data and data['results']:
            # Obtenir les détails pour le premier résultat
            movie_details = data['results'][0]
            return movie_details
        else:
            st.warning("La requête pour obtenir les détails du film a échoué.")
            return None

    # Fonction pour obtenir la clé de la bande-annonce
    def get_trailer_key(movie_id):
        trailer_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        params = {"api_key": api_key}
        trailer_response = requests.get(trailer_url, params=params).json()

        # Vérifier si la clé 'results' existe avant d'y accéder
        if 'results' in trailer_response and trailer_response['results']:
            return trailer_response['results'][0]['key']
        else:
            return None

    # Fonction pour obtenir les noms de genre
    def get_genre_names(genre_ids):
        genre_url = "https://api.themoviedb.org/3/genre/movie/list"
        params = {
            "api_key": api_key,
            "language": "fr-FR"  # Specify the language for genre names
        }

        genre_response = requests.get(genre_url, params=params).json()

        # Créer un dictionnaire qui fait correspondre les IDs de genre à leurs noms
        genre_mapping = {genre['id']: genre['name'] for genre in genre_response['genres']}

        # Convertir les IDs de genre en noms
        genre_names = [genre_mapping.get(genre_id, f"Unknown Genre {genre_id}") for genre_id in genre_ids]

        return genre_names

    # Fonction pour obtenir les recommandations de films
    def get_movie_recommendations(movie_id):
        recommendations_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations"
        params = {"api_key": api_key, "language": "fr-FR", "page": 1}

        recommendations_response = requests.get(recommendations_url, params=params).json()

        # Vérifier si la clé 'results' existe avant d'y accéder
        if 'results' in recommendations_response and recommendations_response['results']:
            return recommendations_response['results']
        else:
            return None

    # Fonction pour obtenir les titres de films en fonction de la requête de l'utilisateur
    def get_movie_titles(query):
        base_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": api_key,
            "query": query,
            "language": "fr-FR"  # Spécifiez la langue pour les résultats en français
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        # Vérifiez si la clé 'results' existe avant d'y accéder
        if 'results' in data and data['results']:
            # Renvoyer une liste de titres de films
            return [result['original_title'] for result in data['results']]
        else:
            return []

    #### Fin des fonctions ####

    # Utilisation de l'autocomplétion pour saisir le titre du film
    title_query = st.text_input("Entrer le titre du film puis appuyer sur entrée :", key="movie_title_query", value=st.session_state.title_query)

    # Ajoutez un print ici
    #print("Current title_query:", title_query)

    # Autocomplétion basée sur la requête de l'utilisateur
    if title_query:

        # Obtenez les options d'autocomplétion en utilisant la fonction
        movie_titles = get_movie_titles(title_query)

        # Affichez les options d'autocomplétion
        title = st.selectbox("Sélectionnez un titre de film :", options=movie_titles, key="movie_title")

    else:
        # Si title_query est vide, définissez title sur None
        title = None

    # Vérifiez si le titre est vide ou pas
    if title:

        # Obtenez les détails du film en français (code de langue 'fr-FR')
        movie_details = get_movie_details(title, language='fr-FR')


        # Affichez les informations sur le film
        if movie_details:
            # Disposition côte à côte à l'aide de colonnes
            col1, col2, col3 = st.columns([1, 2, 1])

            # Afficher l'affiche dans la première colonne
            col1.image("https://image.tmdb.org/t/p/w500" + movie_details['poster_path'], caption=movie_details['original_title'], use_column_width=True)

            # Afficher les détails dans la deuxième colonne
            col2.text("Titre: " + movie_details['original_title'])
            col2.text_area("Synopsis:", movie_details['overview'], height=160)

            # Format et affichage de la date de sortie en français
            release_date_parts = movie_details['release_date'].split('-')
            formatted_release_date = f"{release_date_parts[2]}/{release_date_parts[1]}/{release_date_parts[0]}"
            col2.text("Date de sortie: " + formatted_release_date)

            # Obtenir les noms de genre
            genre_names = get_genre_names(movie_details['genre_ids'])
            col2.text("Genre: " + ", ".join(genre_names))

            col2.text("Note: " + str(movie_details['vote_average']))

            col2.text("Bande Annonce")
            # Obtenir la clé de la bande-annonce
            trailer_key = get_trailer_key(movie_details['id'])

            # Vérifier si la clé de la bande-annonce est disponible
            if trailer_key:
                col2.write(f'<iframe width="320" height="160" src="https://www.youtube.com/embed/{trailer_key}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
            else:
                col2.warning("Bande annonce non disponible.")

            st.markdown("---")


            # Afficher les recommandations de films
            st.subheader("Recommandations de films:")
            recommendations = get_movie_recommendations(movie_details['id'])

            if recommendations:
                for recommendation in recommendations[:5]:  # Afficher les 5 premières recommandations
                    col1, col2, col3 = st.columns([1, 2, 1])

                    # col 1 : affiche du film
                    col1.image("https://image.tmdb.org/t/p/w500" + recommendation['poster_path'],
                            caption=recommendation['original_title'], use_column_width=True)

                    # col2 : titre, synopsis, date de sortie, genre, note, bouton de sélection
                    col2.text("Titre: " + recommendation['original_title'])
                    col2.text_area("Synopsis:", recommendation['overview'], height=200)

                    # Format et affichage de la date de sortie en français
                    release_date_parts = recommendation['release_date'].split('-')
                    formatted_release_date = f"{release_date_parts[2]}/{release_date_parts[1]}/{release_date_parts[0]}"
                    col2.text("Date de sortie: " + formatted_release_date)

                    # Obtenir les noms de genre pour la recommandation
                    genre_names = get_genre_names(recommendation['genre_ids'])
                    col2.text("Genre: " + ", ".join(genre_names))

                    col2.text("Note: " + str(recommendation['vote_average']))

                    # Ajouter le bouton pour sélectionner le film
                    button_key = f"select_button_{recommendation['original_title']}"

                    if col2.button(f"Sélectionner {recommendation['original_title']}", key=button_key):
                        # Stocker le titre sélectionné dans la session comme le film principal
                        st.session_state.selected_recommendation = recommendation['original_title']
                        st.session_state.title_query = recommendation['original_title']

                        # Ajoutez des prints ici
                        #print("Selected recommendation:", st.session_state.selected_recommendation)
                        #print("Updated title_query:", st.session_state.title_query)

                        # Relancer l'application
                        st.rerun()


                    st.markdown("---")

                else:
                    st.warning("Aucune recommandation disponible.")

            else:
                st.warning("Film non trouvé.")
        else:
            st.warning("Film non trouvé.")
    else:
        # Ne rien afficher quand le titre est vide
        pass



# -----------------------------------------------------------
# -----------------------------------------------------------
############################################
#### menu page Alerte
############################################

elif page == pages[4]:


    st.markdown('<h1 id="section_alerte" style="color: #BD4BFF; text-align: center;">Pour rester informé des nouveautés</h1>', unsafe_allow_html=True)

    st.markdown("---")

    st.write("""### Inscrivez-vous à nos alertes """)


    def main():
        # Create a form with a unique key
        with st.form(key='user_form'):
            # Initialize variables
            nom, prenom, date_naissance, adresse_mail, genres, accept = "", "", None, "", [], ""

            # Request user information
            nom = st.text_input("Nom", value=nom)
            prenom = st.text_input("Prénom", value=prenom)

            # Request birthdate
            date_naissance = st.date_input("Date de naissance", format="DD/MM/YYYY", value=None)

            # Display formatted birthdate
            if date_naissance:
                formatted_date = date_naissance.strftime('%d/%m/%Y')
                #st.text(f"Date sélectionnée : {formatted_date}")

            adresse_mail = st.text_input("Adresse e-mail", value=adresse_mail)


            # Select movie genres
            genres = st.multiselect("Sélectionnez les genres de films (jusqu'à 5)",
                                        ["Adult", "Adventure", "Action", "Animation", "Biography", "Comedy",
                                        "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir",
                                        "History", "Horror", "Mystery", "Music", "Musical", "Romance",
                                        "Sci-Fi", "Sport", "Thriller", "War", "Western"])

            # Add checkbox for accepting terms
            accept = st.checkbox("Acceptez-vous de recevoir des alertes de notre site et nos conditions d'utilisation.", value=False)

            # Add a submit button
            submit_button = st.form_submit_button(label="Soumettre")

        # Process form submission
        if submit_button:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not nom and not prenom and not date_naissance and not adresse_mail and not genres and not accept:
                st.warning("Veuillez remplir tous les champs et accepter les conditions.")
            else : 
                if not nom:
                    st.warning("Veuillez saisir votre nom")
                elif not prenom:
                    st.warning("Veuillez saisir votre prénom")
                elif not date_naissance:
                    st.warning("Veuillez saisir votre date de naissance")
                elif not adresse_mail:
                    st.warning("Veuillez saisir votre adresse mail")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", adresse_mail):
                    st.warning("Veuillez saisir une adresse e-mail valide.")
                elif not genres:
                    st.warning("Veuillez sélectionner le genre de film que vous aimez")
                elif not accept:
                    st.warning("Veuillez accepter nos conditions en cochant la case")

                else:
                    # Vérifier si le fichier existe
                    file_exists = os.path.isfile("donnees.csv")

                    # Si le fichier n'existe pas, créer le fichier avec les entêtes
                    if not file_exists:
                        with open("donnees.csv", "w", newline='') as csvfile:
                            # Create a CSV writer object
                            writer = csv.writer(csvfile)

                            # Écrire les entêtes
                            writer.writerow(["Date", "Nom", "Prénom", "Date de naissance", "Adresse e-mail", "Genres sélectionnés", "Accepté"])

                    # Ajouter les données au fichier
                    with open("donnees.csv", "a", newline='') as csvfile:
                        # Create a CSV writer object
                        writer = csv.writer(csvfile)

                        # Écrire les données
                        writer.writerow([current_datetime, nom, prenom, str(date_naissance), adresse_mail, ", ".join(genres), accept])

                    # Afficher un message de réussite
                    st.success("Merci d'avoir soumis le formulaire !")

    if __name__ == "__main__":
        main()

