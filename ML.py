import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path

nlp = spacy.load('en_core_web_sm')
TRAIN_DATA = [
    ("Kinder Capital has raised  £81 million in its second seed fund", {"entities": [(0, 14, "ORG")]}),
    ("Airmeet has raised $12 million in Series A funding led by Sequoia Capital", {"entities": [(0, 7, "ORG")]}),
    ("veem raised $31M from Truist Ventures", {"entities": [(0, 4, "ORG")]}),
    ("AppDirect raised $185 million in growth funding", {"entities": [(0, 9, "ORG")]}),
    ("ApplyBoard has raised C$70M (US$55M) in Series C extension funding", {"entities": [(0, 10, "ORG")]}),
    ("eFileCabinet closed an $11.5m Series C funding", {"entities": [(0, 12, "ORG")]}),
    ("AnalytIQ Sports Technologies has raised angel funding of an undisclosed amount", {"entities": [(0, 27, "ORGT")]}),
    ("WM Motor  raised 10 billion yuan ($1.47 billion)", {"entities": [(0, 9, "ORG")]}),
    ("KRE8.TV $1.1 million Seed round led by Mellanox co-founder and President Eyal Waldman",
     {"entities": [(0, 7, "ORG")]}),
    ("Phytolon secured $4.1m", {"entities": [(0, 8, "ORG")]}),
    ("Fourth Partner Energy raised $16 million in mezzanine funding from investment funds",
     {"entities": [(0, 21, "ORG")]}),
]
ner = nlp.get_pipe("ner")

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(30):

        # shuufling examples  before every iteration
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  # batch of texts
                annotations,  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                losses=losses,
            )
doc = nlp("daksh raised $1 from daksh in seed funding")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


# Save the  model to directory
from pathlib import Path


output_dir = Path('/Desktop/')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)


# Save the  model to directory
output_dir = Path('/content/')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

# Load the saved model and predict
print("Loading from", output_dir)
nlp_updated = spacy.load(output_dir)
doc = nlp_updated("Fridge can be ordered in FlipKart" )
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

