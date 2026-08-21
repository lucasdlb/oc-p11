# Mission - Pilote RL Eagle-1

## Objectif

Developper un pilote automatique par reinforcement learning pour le module d'atterrissage spatial `Eagle-1`.

L'objectif principal est d'obtenir un atterrissage doux, precis et fiable sur une zone cible simulee avec l'environnement `LunarLander-v3` de Gymnasium.

Le critere de reussite attendu est une recompense moyenne stable superieure a `200`, mesuree sur plusieurs episodes d'evaluation.

## Contexte

L'entreprise est un leader du developpement de systemes autonomes pour l'exploration spatiale. Elle souhaite automatiser les procedures d'atterrissage de ses modules pour ameliorer la securite et optimiser la consommation de carburant.

Le projet consiste a construire, entrainer, evaluer et exposer un agent RL capable de piloter automatiquement le module `Eagle-1`.

## Environnement Technique

Le repo utilise Python `>=3.12` avec les dependances principales suivantes :

- `gymnasium[box2d]` pour l'environnement `LunarLander-v3`.
- `stable-baselines3` pour les algorithmes RL prets a l'emploi.
- `torch` comme backend deep learning.
- `tensorboard` pour le suivi des entrainements.
- `matplotlib` pour les visualisations.
- `opencv-python` pour la generation ou manipulation de videos.

Notebooks deja presents :

- `notebooks/01_hands_on.ipynb` : prise en main des environnements Gymnasium.
- `notebooks/02_q_table.ipynb` : approche tabulaire pedagogique.
- `notebooks/03_dqn.ipynb` : entrainement DQN.

## Livrables Attendus

Les livrables demandes pour la mission sont :

- Un notebook Colab propre, commente et reproductible au format `.ipynb`.
- Une video `.mp4` de 20 a 30 secondes montrant une performance reussie du pilote automatique.
- Une API permettant d'interagir avec le modele RL entraine.
- Une interface graphique permettant de visualiser une partie jouee par l'agent.
- Un dashboard interactif de suivi des performances, par exemple avec Streamlit, Gradio ou Looker Studio.

Le notebook doit presenter toute la demarche :

- Exploration de l'environnement.
- Choix de l'algorithme.
- Entrainement du modele.
- Optimisation des hyperparametres.
- Evaluation finale.
- Interpretation des resultats.

## Critere De Performance

Le score cible est une recompense moyenne superieure a `200`.

Cette moyenne doit etre stable et mesuree sur un nombre suffisant d'episodes. Une performance ponctuelle sur un seul episode ne suffit pas.

Evaluation recommandee :

- Utiliser `evaluate_policy` de Stable-Baselines3.
- Evaluer sur au moins `50` episodes pour une baseline.
- Evaluer idealement sur `100` episodes pour valider le modele final.
- Documenter la moyenne et l'ecart-type.

## Plan De Realisation

### Etape 1 - Explorer Et Etablir Une Baseline

Objectifs :

- Comprendre l'environnement `LunarLander-v3`.
- Inspecter l'espace d'observation et l'espace d'action.
- Tester un agent aleatoire ou un premier agent avec parametres par defaut.
- Obtenir une premiere mesure de performance.

Actions recommandees :

- Creer l'environnement avec `gymnasium.make("LunarLander-v3")`.
- Afficher `observation_space` et `action_space`.
- Verifier si l'espace d'action est discret ou continu.
- Tester quelques episodes manuels ou aleatoires.
- Entrainer un premier modele simple.
- Mesurer la recompense moyenne avec `evaluate_policy`.

Resultat attendu :

- Une baseline documentee dans le notebook.
- Une premiere recompense moyenne calculee.
- Une comprehension claire des variables d'observation et des actions disponibles.

### Etape 2 - Optimiser L'Agent

Objectifs :

- Ameliorer la performance du modele.
- Depasser une recompense moyenne stable de `200`.
- Comparer les experiences de maniere structuree.

Guidelines :

- Ne modifier qu'un hyperparametre a la fois pour isoler son effet.
- Documenter chaque experience dans le notebook.
- Utiliser TensorBoard ou un tableau de suivi equivalent.
- Sauvegarder les modeles prometteurs.
- Eviter de juger une configuration sur un seul episode.

Hyperparametres candidats :

- `learning_rate`.
- `gamma`.
- `batch_size`.
- `buffer_size` pour DQN.
- `exploration_fraction` et `exploration_final_eps` pour DQN.
- `n_steps`, `gae_lambda`, `clip_range` pour PPO.
- Architecture reseau via `policy_kwargs`.
- Nombre total de timesteps.

Resultat attendu :

- Un agent atteignant une recompense moyenne superieure a `200` sur `100` episodes d'evaluation.
- Un modele sauvegarde.
- Une justification du choix final.

### Etape 3 - Deployer API, GUI Et Dashboard

Objectifs :

- Exposer le modele entraine via une API.
- Visualiser une partie jouee par l'agent.
- Suivre les performances avec un dashboard.

API recommandee :

- Utiliser FastAPI ou Flask.
- Prevoir un endpoint simple, par exemple `POST /predict` ou `GET /play`.
- Charger le modele sauvegarde au demarrage.
- Retourner l'action predite ou les donnees d'un episode.

GUI recommandee :

- Utiliser Streamlit ou Gradio.
- Afficher une animation ou une sequence d'images d'un episode joue par l'agent.
- Permettre de lancer une evaluation ou une partie.

Dashboard recommande :

- Afficher les courbes de recompense.
- Afficher moyenne, ecart-type, nombre d'episodes et hyperparametres.
- Comparer les meilleures experiences.
- S'appuyer sur TensorBoard, Streamlit, Gradio ou Looker Studio.

Points de vigilance :

- Garder la logique RL cote backend/API.
- Eviter de melanger entrainement, inference et interface dans un seul bloc monolithique.
- Utiliser des episodes ou batchs plus petits si la RAM est limitee.

### Etape 4 - Verifier Et Preparer La Restitution

Objectifs :

- Verifier que tous les livrables sont presents.
- S'assurer que le projet est reproductible.
- Preparer une explication claire de la demarche.

Checklist finale :

- Notebook propre, commente et executable.
- Baseline documentee.
- Choix d'algorithme justifie.
- Hyperparametres testes et resultats compares.
- Modele final sauvegarde.
- Evaluation finale sur au moins `100` episodes.
- Score moyen final superieur a `200`.
- Video `.mp4` de 20 a 30 secondes.
- API fonctionnelle.
- GUI fonctionnelle.
- Dashboard ou suivi interactif disponible.
- Instructions d'utilisation documentees.

## Guidelines RL

Choix d'algorithme :

- Utiliser DQN si l'espace d'action est discret.
- Utiliser PPO si l'espace d'action est continu ou si PPO donne une meilleure stabilite.
- Commencer simple avant d'augmenter la complexite.

Evaluation :

- Toujours distinguer recompense d'entrainement et recompense d'evaluation.
- Utiliser une seed fixe pour certains essais afin de faciliter la comparaison.
- Repeter l'evaluation sur plusieurs episodes.
- Conserver moyenne et ecart-type.

Tracking :

- Nommer chaque experience clairement.
- Sauvegarder les logs TensorBoard.
- Garder une table de comparaison : algorithme, timesteps, hyperparametres, score moyen, ecart-type, commentaire.

Robustesse :

- Ne pas conclure sur un episode unique.
- Verifier que l'agent n'exploite pas un comportement accidentel.
- Tester plusieurs seeds si possible.
- Verifier visuellement plusieurs atterrissages.

## Organisation Recommandee Du Repo

Structure cible possible :

```text
.
|-- MISSION.md
|-- README.md
|-- pyproject.toml
|-- pretty_printer.py
|-- notebooks/
|   |-- 01_hands_on.ipynb
|   |-- 02_q_table.ipynb
|   |-- 03_dqn.ipynb
|   `-- 04_lunarlander_final.ipynb
|-- models/
|   `-- lunarlander_best.zip
|-- reports/
|   |-- evaluation_results.csv
|   `-- demo_lunarlander.mp4
|-- src/
|   |-- train.py
|   |-- evaluate.py
|   |-- record_video.py
|   |-- api.py
|   `-- app.py
`-- runs/
    `-- tensorboard_logs/
```

Cette structure est indicative. Il vaut mieux garder une organisation simple et coherente plutot que multiplier les fichiers inutilement.

## Workflow Semi-Agentique

Le projet peut etre decoupe en taches confiees a un assistant de coding ou a plusieurs agents specialises.

### Agent 1 - Exploration Et Baseline

Mission :

- Lire les notebooks existants.
- Identifier le style de travail deja utilise.
- Creer ou completer un notebook final pour `LunarLander-v3`.
- Documenter l'environnement, les observations, les actions et la baseline.

Sortie attendue :

- Notebook avec exploration et baseline reproductible.
- Score moyen initial.
- Notes sur l'espace d'action et l'algorithme recommande.

### Agent 2 - Entrainement Et Optimisation

Mission :

- Entrainer DQN ou PPO selon l'environnement choisi.
- Lancer plusieurs experiences controlees.
- Suivre les performances avec TensorBoard.
- Sauvegarder les meilleurs modeles.

Sortie attendue :

- Modele final sauvegarde.
- Tableau comparatif des experiences.
- Score final moyen et ecart-type.

### Agent 3 - Evaluation Et Video

Mission :

- Charger le meilleur modele.
- Evaluer sur `100` episodes.
- Generer une video `.mp4` de 20 a 30 secondes.
- Verifier visuellement que l'atterrissage est reussi.

Sortie attendue :

- Fichier video.
- Rapport d'evaluation.
- Cellules notebook de demonstration.

### Agent 4 - API Et Interface

Mission :

- Creer une API minimale pour charger le modele et predire une action.
- Creer une interface Streamlit ou Gradio pour visualiser une partie.
- Garder l'interface separee de la logique RL.

Sortie attendue :

- API executable localement.
- GUI executable localement.
- Instructions d'utilisation.

### Agent 5 - Documentation Et Restitution

Mission :

- Nettoyer le notebook final.
- Verifier que les livrables sont relies et expliques.
- Rediger les instructions dans le README.
- Preparer les points de restitution pour le mentor.

Sortie attendue :

- Documentation claire.
- Checklist livrables complete.
- Explication concise des choix techniques.

## Definition Of Done

Le projet est considere termine quand :

- Le modele final depasse une recompense moyenne de `200` de facon stable.
- Le notebook raconte toute la demarche sans cellules inutiles ou non executees.
- Les experiences sont documentees et comparables.
- Les livrables demandes existent et peuvent etre ouverts ou executes.
- Une personne externe peut comprendre comment entrainer, evaluer et utiliser le modele.

## Ressources Mentionnees

- Gymnasium.
- Documentation `LunarLander-v3`.
- Stable-Baselines3.
- TensorBoard.
- Article sur l'optimisation des hyperparametres en reinforcement learning.
- FastAPI.
- Streamlit.
- Gradio.
- Looker Studio.
