# Présentation du projet

Ce projet est le compte rendu d'un groupe de lecture sur l'**Intelligence Artificielle (IA)** dirigé par Aurélien Garivier. Il se concentre sur les **Generative Adversarial Networks (GANs)**, en s'appuyant sur le papier de recherche fondateur de 2014. L'objectif est de rendre les concepts des GANs accessibles tout en documentant les explorations pratiques sur les jeux de données **MNIST** et **Olivetti Faces**, ainsi qu'une étude théorique de leur fonctionnement.  


# Structure du projet

Le projet est organisé pour faciliter la gestion du code source, des données et du rapport final. Voici la structure des fichiers :
```Tree
├── README.md                  # Le document que vous êtes en train de lire
├── resume.tex                 # Le code source LaTeX du rapport final.  
├── gan_2d_fusion.py           # Script Python qui génère les GANs avec bruit et déterministe avec 2 paramètres  
├── gan_on_mnist.py            # Script Python pour l'entraînement d'un GAN sur le jeu de données *MNIST*.  
├── gan_on_olivetti_faces.py   # Script Python pour l'entraînement d'un GAN sur le jeu de données *Olivetti Faces*.  
├── images_GAN_MNIST/          # Dossier contenant les images de chiffres générées par le GAN du code *gan_on_mnist.py*  
│   ├── epoch_1.png  
│   ├── epoch_2.png  
│   └── ... (jusqu'à epoch_10.png)  
├── images_GAN_Olivetti_faces/ # Dossier contenant les images de visages générées par le GAN du code *gan_on_olivetti_faces.py*  
│   ├── epoch_0.png  
│   ├── epoch_25.png  
│   └── ... (jusqu'à epoch_300.png)  
├── travail_session3/          # Contient les images utilisées pour l'étude de la fusion de visages.  
    ├── 2_images_donnees.png  
    ├── 2_images_generees.png  
    ├── 3_images_donnees.png  
    └── 3_images_generees.png  
└── dataset/                   # Dossier contenant les jeux de données utilisés  
    └── olivetti_faces.npy     # Jeu de données des visage de *Olivetti Faces*
```
