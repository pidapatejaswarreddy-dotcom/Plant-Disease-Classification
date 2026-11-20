"""Class-name parsing and cautious, educational disease descriptions.

Only classes listed in DISEASE_INFO get a description. Any other class name (for example
from a re-trained model on a different dataset) is shown without disease claims.
Nothing here is a treatment prescription.
"""
from __future__ import annotations

CATEGORY_ADVICE = {
    "fungal": [
        "Remove and dispose of heavily affected leaves or fallen debris (do not compost them).",
        "Improve air circulation and avoid wetting foliage late in the day.",
        "Rotate crops where practical and prefer resistant varieties.",
        "Ask your local agricultural extension service whether a treatment is appropriate and which products are permitted in your region.",
    ],
    "oomycete": [
        "Act quickly: this kind of disease can spread fast in cool, wet weather. Isolate or remove affected plants following local guidance.",
        "Avoid overhead irrigation and keep foliage dry.",
        "Do not save seed tubers or fruit from affected plants.",
        "Contact your local agricultural extension service for region-specific control advice.",
    ],
    "bacterial": [
        "Avoid handling plants when foliage is wet and clean tools between plants.",
        "Use disease-free seed or planting material and rotate crops.",
        "Remove badly affected plant parts and avoid overhead watering.",
        "Ask a local plant-health service about options; bacterial diseases are often hard to cure once established.",
    ],
    "viral": [
        "Viral diseases generally cannot be cured. Local guidance usually involves removing infected plants.",
        "Wash hands and disinfect tools after handling affected plants.",
        "Control insect vectors where relevant and use certified virus-free planting material.",
        "Confirm the diagnosis with a plant-health laboratory or extension service.",
    ],
    "pest": [
        "Look closely at the underside of leaves (a hand lens helps) to confirm the pest.",
        "Water-stressed, dusty plants are often more affected; keep plants evenly watered.",
        "Encourage natural predators and ask a local extension service about suitable control options.",
    ],
    "vector_bacterial": [
        "This disease is typically serious and is subject to official monitoring in many regions. Report suspected cases to your local agricultural authority.",
        "It is spread by insect vectors and infected planting material; do not move plant material from suspect trees.",
        "Rely on an expert/laboratory diagnosis before taking any action.",
    ],
}

HEALTHY_ADVICE = [
    "The model did not match this leaf to any of the diseases it knows for this plant.",
    "This is not a guarantee of plant health: early infections, nutrient problems and pests the model was not trained on can look healthy.",
    "Keep monitoring the plant and re-check if new spots, discolouration or wilting appear.",
]

# class name -> (category, description)
DISEASE_INFO = {
    "Apple___Apple_scab": ("fungal", "Fungal disease (Venturia inaequalis) causing olive-green to brown scabby spots on leaves and fruit."),
    "Apple___Black_rot": ("fungal", "Fungal disease (Botryosphaeria obtusa) causing 'frogeye' leaf spots, fruit rot and cankers on branches."),
    "Apple___Cedar_apple_rust": ("fungal", "Fungal disease (Gymnosporangium juniperi-virginianae) causing yellow-orange leaf spots; it needs both apple and juniper/cedar hosts."),
    "Cherry_(including_sour)___Powdery_mildew": ("fungal", "Fungal disease producing a white, powdery growth on leaf surfaces, which can curl or distort."),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("fungal", "Fungal disease causing rectangular tan-to-grey lesions between leaf veins."),
    "Corn_(maize)___Common_rust_": ("fungal", "Fungal disease (Puccinia sorghi) causing small reddish-brown pustules on both leaf surfaces."),
    "Corn_(maize)___Northern_Leaf_Blight": ("fungal", "Fungal disease (Exserohilum turcicum) causing long, cigar-shaped grey-green to tan lesions."),
    "Grape___Black_rot": ("fungal", "Fungal disease (Guignardia bidwellii) causing tan leaf spots with dark borders and shrivelled berries."),
    "Grape___Esca_(Black_Measles)": ("fungal", "A trunk disease complex caused by several fungi; leaves can show 'tiger-stripe' discolouration and berries may spot."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("fungal", "Fungal leaf spot disease causing dark brown, angular spots on grape leaves."),
    "Orange___Haunglongbing_(Citrus_greening)": ("vector_bacterial", "Bacterial disease spread by the Asian citrus psyllid; typical signs are blotchy yellow leaf mottling and small, misshapen fruit."),
    "Peach___Bacterial_spot": ("bacterial", "Bacterial disease (Xanthomonas) causing small, dark, angular leaf spots that may drop out, giving a 'shot-hole' look."),
    "Pepper,_bell___Bacterial_spot": ("bacterial", "Bacterial disease (Xanthomonas) causing water-soaked spots that turn brown on leaves and fruit."),
    "Potato___Early_blight": ("fungal", "Fungal disease (Alternaria solani) causing brown spots with concentric rings, usually on older leaves first."),
    "Potato___Late_blight": ("oomycete", "Water-mould disease (Phytophthora infestans) causing dark, water-soaked lesions that can spread rapidly in cool, wet weather."),
    "Squash___Powdery_mildew": ("fungal", "Fungal disease producing white, powdery patches on leaves and stems."),
    "Strawberry___Leaf_scorch": ("fungal", "Fungal disease (Diplocarpon earlianum) causing small purple spots that merge and scorch the leaf."),
    "Tomato___Bacterial_spot": ("bacterial", "Bacterial disease (Xanthomonas species) causing small, dark, water-soaked leaf and fruit spots."),
    "Tomato___Early_blight": ("fungal", "Fungal disease (Alternaria solani) causing brown 'target-like' spots with concentric rings."),
    "Tomato___Late_blight": ("oomycete", "Water-mould disease (Phytophthora infestans) causing large, greasy-looking dark lesions, often fast-spreading in humid conditions."),
    "Tomato___Leaf_Mold": ("fungal", "Fungal disease (Passalora fulva) causing yellow patches on upper leaf surfaces and olive-coloured mould underneath, common in humid conditions."),
    "Tomato___Septoria_leaf_spot": ("fungal", "Fungal disease (Septoria lycopersici) causing many small circular spots with dark borders and grey centres."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("pest", "Damage by two-spotted spider mites (Tetranychus urticae): fine yellow stippling, bronzing and sometimes fine webbing."),
    "Tomato___Target_Spot": ("fungal", "Fungal disease (Corynespora cassiicola) causing brown spots with concentric rings on leaves and fruit."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("viral", "Virus spread by whiteflies; typical signs are upward-curled, yellowed leaves and stunted growth."),
    "Tomato___Tomato_mosaic_virus": ("viral", "Virus causing mottled light/dark green leaves and distortion; it spreads easily by contact and contaminated tools."),
}


def parse_class_name(class_name: str) -> dict:
    """'Tomato___Early_blight' -> plant 'Tomato', condition 'Early blight', status."""
    plant_raw, _, cond_raw = class_name.partition("___")
    plant = plant_raw.replace("_", " ").strip()
    condition = cond_raw.replace("_", " ").strip() or "Unknown"
    is_healthy = condition.lower() == "healthy"
    return {
        "class_name": class_name,
        "plant": plant,
        "condition": condition,
        "status": "healthy" if is_healthy else ("diseased" if cond_raw else "unknown"),
        "is_healthy": is_healthy,
    }


def get_class_details(class_name: str) -> dict:
    d = parse_class_name(class_name)
    if d["is_healthy"]:
        d.update(known=True, category="healthy",
                 description=f"No sign of the {d['plant']} diseases this model knows was detected in the image.",
                 recommendations=list(HEALTHY_ADVICE))
    elif class_name in DISEASE_INFO:
        category, description = DISEASE_INFO[class_name]
        d.update(known=True, category=category, description=description,
                 recommendations=list(CATEGORY_ADVICE[category]))
    else:
        d.update(known=False, category="unknown",
                 description="No curated description is available for this class.",
                 recommendations=["Consult a local agricultural extension service or plant pathologist."])
    return d
