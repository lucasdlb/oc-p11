# OC P11 - LunarLander RL

Outils CLI config-driven pour explorer, entrainer et evaluer des agents `LunarLander-v3`.

## Commandes

Inspecter l'environnement :

```bash
uv run python scripts/inspect.py --env configs/envs/default.toml
```

Evaluer une baseline random et generer une video sample :

```bash
uv run python scripts/evaluate.py --config configs/experiments/random_default.toml
```

Entrainer SB3 DQN :

```bash
uv run python scripts/train.py --config configs/experiments/sb3_dqn_default.toml
```

Evaluer SB3 DQN et generer une video sample :

```bash
uv run python scripts/evaluate.py --config configs/experiments/sb3_dqn_default.toml
```

Lancer l'API du pilote Eagle-1 :

```bash
uv run uvicorn src.api:app --reload
```

Endpoints principaux :

- `GET /health` : statut, labels d'observation et labels d'action.
- `POST /predict` : retourne l'action predite pour une observation LunarLander-v3.
- `GET /play` : joue un episode et retourne la trajectoire.

Exemple de prediction :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"observation":[0,0,0,0,0,0,0,0]}'
```

Lancer la GUI et le dashboard Streamlit :

```bash
uv run streamlit run src/app.py
```

Entrainer le DQN PyTorch manuel :

```bash
uv run python scripts/train.py --config configs/experiments/manual_dqn_default.toml
```

Evaluer le DQN PyTorch manuel et generer une video sample :

```bash
uv run python scripts/evaluate.py --config configs/experiments/manual_dqn_default.toml
```

Lancer TensorBoard :

```bash
uv run tensorboard --logdir runs/tensorboard
```

## Organisation Des Configs

Les experiences gardent l'identite du run, le type d'agent et les choix de configs a utiliser. Les parametres d'entrainement et d'algorithme restent separes dans `configs/model_params/`.

```text
configs/
|-- envs/
|   |-- default.toml
|   |-- wind_light.toml
|   |-- wind_medium.toml
|   `-- wind_hard.toml
|-- evaluation/
|   `-- standard.toml
|-- video/
|   `-- standard.toml
|-- paths/
|   `-- default.toml
|-- model_params/
|   |-- sb3_dqn/
|   |   `-- default.toml
|   `-- manual_dqn/
|       `-- default.toml
`-- experiments/
    |-- random_default.toml
    |-- sb3_dqn_default.toml
    `-- manual_dqn_default.toml
```

Exemple d'experience :

```toml
[run]
name = "sb3_dqn_default"
seed = 42

[agent]
type = "sb3_dqn"

[experiment]
env = "default"
evaluation = "standard"
video = "standard"
paths = "default"
model_params = "default"
```

Exemple de parametres modele :

```toml
[train]
total_timesteps = 200000

[algorithm]
learning_rate = 0.0001
buffer_size = 100000
learning_starts = 1000
batch_size = 64
gamma = 0.99
train_freq = 4
gradient_steps = 1
target_update_interval = 1000
exploration_fraction = 0.2
exploration_initial_eps = 1.0
exploration_final_eps = 0.05
net_arch = [256, 256]
verbose = 1
```

## Chemins Derives

Les artefacts sont derives automatiquement de `run.name`.

Pour `run.name = "sb3_dqn_default"` :

```text
models/sb3_dqn_default.zip
models/sb3_dqn_default.json
reports/evaluations/sb3_dqn_default.csv
reports/videos/sb3_dqn_default.mp4
runs/tensorboard/sb3_dqn_default_*/
runs/monitor/sb3_dqn_default_monitor.csv
```

Pour `run.name = "manual_dqn_default"` :

```text
models/manual_dqn_default.pt
models/manual_dqn_default.json
reports/training/manual_dqn_default.csv
reports/evaluations/manual_dqn_default.csv
reports/videos/manual_dqn_default.mp4
runs/tensorboard/manual_dqn_default/
```

## Comparaison

Workflow de comparaison simple :

```bash
uv run python scripts/evaluate.py --config configs/experiments/random_default.toml
uv run python scripts/train.py --config configs/experiments/sb3_dqn_default.toml
uv run python scripts/train.py --config configs/experiments/manual_dqn_default.toml
uv run python scripts/evaluate.py --config configs/experiments/sb3_dqn_default.toml
uv run python scripts/evaluate.py --config configs/experiments/manual_dqn_default.toml
```

Comparer ensuite :

- Reward moyen.
- Ecart-type.
- Reward min/max.
- Nombre moyen de steps.
- Courbes TensorBoard.
- Videos des episodes.

## Smoke Tests

Pour verifier le chargement de config et l'environnement sans long entrainement :

```bash
uv run python scripts/inspect.py --env configs/envs/default.toml
```
