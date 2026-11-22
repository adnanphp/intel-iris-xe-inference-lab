"""Map Imagenette WordNet IDs to ImageNet-1k class indices."""

# Imagenette's 10 classes and their ImageNet-1k indices
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

def wordnet_to_imagenet_index(wordnet_id):
    return IMAGENETTE_TO_IMAGENET.get(wordnet_id)
