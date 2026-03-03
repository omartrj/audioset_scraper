Bad lables:
- Narration, monologue
- Speech synthesizer
- Music

## STEP 1: 
Audio di persone che parlano senza sottofondo stradale
**IMPORTANTE**: Usa `balanced_train_segments.csv` perché sennò scarichi solo audio con label `Speech`, che è di gran lunga la più comune
TARGET_CATEGORIES = ["Human voice"]
AVOID_CATEGORIES = ["Narration, monologue", "Speech synthesizer", "Music", "Vehicle", "Siren"]

## STEP 2:
Audio di rumore ambientale, anche con voci umani o folle, preferibilmente in strada
Come prima però non voglio assolutamente sirene o veicoli d'emergenza
**IMPORTANTE**: Usa `unbalanced_train_segments.csv` perchè il rumore stradale è raro quindi meglio prendere tutto quello che c'è
TARGET_CATEGORIES = ["Traffic noise, roadway noise"]
AVOID_CATEGORIES = ["Narration, monologue", "Speech synthesizer", "Music", "Siren", "Emergency vehicle"]