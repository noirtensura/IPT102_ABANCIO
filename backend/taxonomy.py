"""Unified dermatological taxonomy mapping Dataset A (19 fine classes) and Dataset B (6 broad classes)
into a consistent 2-level hierarchy (7 broad categories -> 20 specific sub-conditions).
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- Level 1: 7 Broad Clinical Categories
BROAD_CLASSES: tuple[str, ...] = (
    "Infectious Skin Conditions",
    "Eczema and Dermatitis",
    "Acne and Rosacea",
    "Pigmentation Disorders",
    "Benign Skin Lesions",
    "Malignant and Pre-Malignant Lesions",
    "Inflammatory and Systemic Conditions",
)

# --------------------------------------------------------------------------- Level 2: 20 Specific Sub-Conditions
FINE_TO_BROAD: dict[str, str] = {
    # Infectious Skin Conditions
    "Tinea, Ringworm and Fungal Infections": "Infectious Skin Conditions",
    "Warts, Molluscum and Viral Growths": "Infectious Skin Conditions",
    "Herpes and Viral Skin Infections": "Infectious Skin Conditions",
    "Bacterial Skin Infections (Cellulitis and Impetigo)": "Infectious Skin Conditions",
    # Eczema and Dermatitis
    "Eczema": "Eczema and Dermatitis",
    "Atopic Dermatitis": "Eczema and Dermatitis",
    "Contact Dermatitis": "Eczema and Dermatitis",
    # Acne and Rosacea
    "Acne and Rosacea": "Acne and Rosacea",
    # Pigmentation Disorders
    "Vitiligo and Depigmentation": "Pigmentation Disorders",
    "Hyperpigmentation, Melasma and Light Disorders": "Pigmentation Disorders",
    # Benign Skin Lesions
    "Seborrheic Keratoses and Benign Growths": "Benign Skin Lesions",
    "Vascular Skin Lesions": "Benign Skin Lesions",
    "Benign Moles and Nevi": "Benign Skin Lesions",
    # Malignant and Pre-Malignant Lesions
    "Actinic Keratosis, Basal and Squamous Cell Carcinoma": "Malignant and Pre-Malignant Lesions",
    "Melanoma and Suspicious Pigmented Lesions": "Malignant and Pre-Malignant Lesions",
    # Inflammatory and Systemic Conditions
    "Psoriasis, Lichen Planus and Related Conditions": "Inflammatory and Systemic Conditions",
    "Lupus and Connective Tissue Conditions": "Inflammatory and Systemic Conditions",
    "Exanthems and Drug Eruptions": "Inflammatory and Systemic Conditions",
    "Urticaria (Hives)": "Inflammatory and Systemic Conditions",
    "Vasculitis": "Inflammatory and Systemic Conditions",
    "Systemic Skin Manifestations": "Inflammatory and Systemic Conditions",
}

FINE_CLASSES: tuple[str, ...] = tuple(FINE_TO_BROAD.keys())

BROAD_TO_FINE: dict[str, tuple[str, ...]] = {
    broad: tuple(fine for fine, parent in FINE_TO_BROAD.items() if parent == broad)
    for broad in BROAD_CLASSES
}

# --------------------------------------------------------------------------- Raw Folder Mappings
DATASET_A_FOLDER_TO_FINE: dict[str, str] = {
    "Acne and Rosacea Photos": "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": (
        "Actinic Keratosis, Basal and Squamous Cell Carcinoma"
    ),
    "Atopic Dermatitis Photos": "Atopic Dermatitis",
    "Cellulitis Impetigo and other Bacterial Infections": "Bacterial Skin Infections (Cellulitis and Impetigo)",
    "Eczema Photos": "Eczema",
    "Exanthems and Drug Eruptions": "Exanthems and Drug Eruptions",
    "Herpes HPV and other STDs Photos": "Herpes and Viral Skin Infections",
    "Light Diseases and Disorders of Pigmentation": "Hyperpigmentation, Melasma and Light Disorders",
    "Lupus and other Connective Tissue diseases": "Lupus and Connective Tissue Conditions",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma and Suspicious Pigmented Lesions",
    "Poison Ivy Photos and other Contact Dermatitis": "Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis, Lichen Planus and Related Conditions",
    "Seborrheic Keratoses and other Benign Tumors": "Seborrheic Keratoses and Benign Growths",
    "Systemic Disease": "Systemic Skin Manifestations",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Tinea, Ringworm and Fungal Infections",
    "Urticaria Hives": "Urticaria (Hives)",
    "Vascular Tumors": "Vascular Skin Lesions",
    "Vasculitis Photos": "Vasculitis",
    "Warts Molluscum and other Viral Infections": "Warts, Molluscum and Viral Growths",
}

DATASET_B_FOLDER_TO_BROAD: dict[str, str] = {
    "1. Enfeksiyonel": "Infectious Skin Conditions",
    "1. Infectious": "Infectious Skin Conditions",
    "2. Ekzama": "Eczema and Dermatitis",
    "2. Eczema": "Eczema and Dermatitis",
    "3. Akne": "Acne and Rosacea",
    "3. Acne": "Acne and Rosacea",
    "4. Pigment": "Pigmentation Disorders",
    "4. Pigmentation": "Pigmentation Disorders",
    "5. Benign": "Benign Skin Lesions",
    "6. Malign": "Malignant and Pre-Malignant Lesions",
    "6. Malignant": "Malignant and Pre-Malignant Lesions",
}

FOLDER_ALIASES: tuple[tuple[str, str], ...] = (
    ("/1. Enfeksiyonel/", "/1. Infectious/"),
    ("/2. Ekzama/", "/2. Eczema/"),
    ("/3. Akne/", "/3. Acne/"),
    ("/4. Pigment/", "/4. Pigmentation/"),
    ("/6. Malign/", "/6. Malignant/"),
)

# High-precision filename rules to recover specific sub-conditions from Dataset B clinical images.
_SLUG_RULES: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "Infectious Skin Conditions": (
        (
            re.compile(r"(tinea|candida|candidiasis|ringworm|onychomycosis|pityriasis-versicolor|fungal)"),
            "Tinea, Ringworm and Fungal Infections",
        ),
        (
            re.compile(r"(wart|molluscum|condyloma|verruca)"),
            "Warts, Molluscum and Viral Growths",
        ),
        (
            re.compile(r"(herpes|zoster|shingles|varicella|syphilis|chancroid)"),
            "Herpes and Viral Skin Infections",
        ),
        (
            re.compile(
                r"(impetigo|cellulitis|folliculitis|erysipelas|furuncle|carbuncle|abscess|ecthyma|paronychia|mycobacterium|actinomycosis|staph|erythrasma|pitted-keratolysis)"
            ),
            "Bacterial Skin Infections (Cellulitis and Impetigo)",
        ),
    ),
    "Eczema and Dermatitis": (
        (
            re.compile(r"(atopic|05atopic)"),
            "Atopic Dermatitis",
        ),
        (
            re.compile(r"(contact-?derm|poison-?ivy|allergic-contact)"),
            "Contact Dermatitis",
        ),
        (
            re.compile(r"(eczema|stasis-dermatitis|dyshidros|nummular|lichen-simplex|neurotic-excor|hand-dermatitis|chapped-fissured|03eczema|03chronic)"),
            "Eczema",
        ),
    ),
    "Acne and Rosacea": (
        (re.compile(r".+"), "Acne and Rosacea"),
    ),
    "Pigmentation Disorders": (
        (
            re.compile(r"(vitiligo|albinism|hypomelan|depigment|nevus-anemicus|pityriasis-alba)"),
            "Vitiligo and Depigmentation",
        ),
        (
            re.compile(r"(melasma|lentigo|sun-damaged|phototoxic|porphyr|polymorphous|poikiloderma|hyperpigment|freckle|erythema-ab-igne|actinic-comedones|milia)"),
            "Hyperpigmentation, Melasma and Light Disorders",
        ),
    ),
    "Benign Skin Lesions": (
        (
            re.compile(r"^isic_"),
            "Benign Moles and Nevi",
        ),
        (
            re.compile(r"(hemangioma|angioma|pyogenic-granuloma|vascular|venous-lake|cherry-angioma|telangiect)"),
            "Vascular Skin Lesions",
        ),
        (
            re.compile(
                r"(seborrheic|dermatofibroma|keratoacanthoma|cyst|skin-tag|sebaceous-hyperplasia|keloid|chondrodermatitis|lipoma|cylindroma|collagenoma|syringoma|milium|milia|scar)"
            ),
            "Seborrheic Keratoses and Benign Growths",
        ),
        (
            re.compile(r"(nevus|nevi|mole|lentigo)"),
            "Benign Moles and Nevi",
        ),
    ),
    "Malignant and Pre-Malignant Lesions": (
        (
            re.compile(r"(basal-cell|squamous-cell|actinic-keratosis|actinic-cheilitis|bowen|keratoacanthoma|ctcl|carcinoma|metastasis)"),
            "Actinic Keratosis, Basal and Squamous Cell Carcinoma",
        ),
        (
            re.compile(r"(^isic_|melanoma|atypical-nevi|dysplastic|lentigo-maligna)"),
            "Melanoma and Suspicious Pigmented Lesions",
        ),
    ),
}


def infer_fine_label_from_filename(broad_label: str, filename: str) -> str | None:
    """Infer a high-confidence specific sub-condition label from a Dataset B filename."""
    low = filename.lower()
    for pattern, fine_label in _SLUG_RULES.get(broad_label, ()):
        if pattern.search(low):
            return fine_label
    return None


def normalize_stem_group(filename: str) -> str:
    """Normalize thumbnail/copy filename variants so originals and copies stay in the same CV/split group."""
    stem = filename.rsplit(".", 1)[0].lower().strip()
    # Strip DermNet thumbnail prefix ("t-03..." / "t-tinea-foot-1")
    if stem.startswith("t-"):
        stem = stem[2:]
    # Strip copy suffixes ("- copy", "(1)", Turkish "kopyası")
    stem = re.sub(r"(\s*-\s*copy(\s*\(\d+\))?|\s*\(\d+\)|\s+kopyas[ıi])$", "", stem).strip()
    return stem


# --------------------------------------------------------------------------- Condition Guides (Broad + Specific + Legacy)
CONDITION_GUIDES: dict[str, tuple[str, str, str]] = {
    "Uncertain": (
        "Uncertain",
        "The model could not reliably distinguish a single condition from this photo. Lighting, focus, framing, or overlapping visual features across skin conditions can cause low certainty.",
        "https://dermnetnz.org/topics",
    ),
    # ---- 7 Broad Clinical Categories ----
    "Infectious Skin Conditions": (
        "Infectious Skin Conditions",
        "Skin infections caused by fungi, viruses, or bacteria can present as ring-shaped scaly patches, firm warts or dome-shaped bumps, grouped blisters, or warm red crusted sores.",
        "https://dermnetnz.org/topics/skin-infections",
    ),
    "Eczema and Dermatitis": (
        "Eczema and Dermatitis",
        "Eczema and dermatitis commonly cause itchy, inflamed, dry, cracked, or scaly patches that may weep, crust, or thicken over time.",
        "https://www.aad.org/public/diseases/eczema/types/atopic-dermatitis",
    ),
    "Acne and Rosacea": (
        "Acne and Rosacea",
        "Acne and rosacea commonly cause facial redness, whiteheads, blackheads, tender pimples, cysts, or visible small blood vessels.",
        "https://dermnetnz.org/topics/acne-vulgaris",
    ),
    "Pigmentation Disorders": (
        "Pigmentation Disorders",
        "Pigmentation and light-related skin disorders produce areas of lighter or darker skin tone, patchy discoloration, or sun-triggered rashes.",
        "https://dermnetnz.org/topics/pigmentation-disorders",
    ),
    "Benign Skin Lesions": (
        "Benign Skin Lesions",
        "Benign growths include common moles, stuck-on waxy seborrheic keratoses, firm dermatofibromas, cysts, and harmless vascular spots.",
        "https://dermnetnz.org/topics/benign-skin-lesions",
    ),
    "Malignant and Pre-Malignant Lesions": (
        "Malignant and Pre-Malignant Lesions",
        "Sun-damaged pre-cancerous or malignant skin lesions can appear as rough scaly patches, pearly or non-healing sores, or pigmented spots with asymmetry, irregular borders, or multiple colors. Prompt dermatologic evaluation is important.",
        "https://www.aad.org/public/diseases/skin-cancer/find/at-risk/abcdes",
    ),
    "Inflammatory and Systemic Conditions": (
        "Inflammatory and Systemic Conditions",
        "Non-infectious inflammatory and systemic conditions include well-defined scaly plaques (such as psoriasis), hives, sun-sensitive facial rashes, drug eruptions, and purpuric spots.",
        "https://dermnetnz.org/topics",
    ),
    # ---- 20 Specific Sub-Conditions ----
    "Tinea, Ringworm and Fungal Infections": (
        "Tinea, Ringworm and Fungal Infections",
        "Superficial fungal infections frequently form ring-shaped scaly red patches with a clearer center, itchy foot/groin rashes, or discolored scaly patches.",
        "https://dermnetnz.org/topics/tinea-corporis",
    ),
    "Warts, Molluscum and Viral Growths": (
        "Warts, Molluscum and Viral Growths",
        "Common viral skin growths include rough, firm warts with tiny pinpoint dots and smooth, dome-shaped molluscum papules with a central dimple.",
        "https://dermnetnz.org/topics/viral-wart",
    ),
    "Herpes and Viral Skin Infections": (
        "Herpes and Viral Skin Infections",
        "Cutaneous viral infections such as herpes simplex or shingles typically present as grouped fluid-filled blisters on a red base that crust over.",
        "https://dermnetnz.org/topics/herpes-simplex",
    ),
    "Bacterial Skin Infections (Cellulitis and Impetigo)": (
        "Bacterial Skin Infections (Cellulitis and Impetigo)",
        "Bacterial skin infections can cause expanding red, warm, tender skin or honey-colored crusted sores and pus-filled bumps.",
        "https://dermnetnz.org/topics/impetigo",
    ),
    "Eczema": (
        "Eczema",
        "Eczema can cause itchy, inflamed, dry, scaly, or coin-shaped skin patches that may ooze, crust, or thicken from scratching.",
        "https://www.aad.org/public/diseases/eczema/types/atopic-dermatitis",
    ),
    "Atopic Dermatitis": (
        "Atopic Dermatitis",
        "Atopic dermatitis often presents as chronic dry, intensely itchy, red, or thickened patches in skin folds, hands, or face.",
        "https://dermnetnz.org/topics/atopic-dermatitis",
    ),
    "Contact Dermatitis": (
        "Contact Dermatitis",
        "Allergic or irritant contact dermatitis causes itchy red patches, linear streaks, bumps, or blisters where skin touched a trigger.",
        "https://dermnetnz.org/topics/contact-dermatitis",
    ),
    "Vitiligo and Depigmentation": (
        "Vitiligo and Depigmentation",
        "Vitiligo causes well-defined milky-white or lighter patches of skin due to loss of pigment cells, often symmetrically on the hands, face, elbows, or knees.",
        "https://dermnetnz.org/topics/vitiligo",
    ),
    "Hyperpigmentation, Melasma and Light Disorders": (
        "Hyperpigmentation, Melasma and Light Disorders",
        "Pigmentation and photosensitive conditions cause darker patches (such as melasma or solar lentigines), uneven sun-damaged skin tone, or sun-provoked rashes.",
        "https://dermnetnz.org/topics/melasma",
    ),
    "Seborrheic Keratoses and Benign Growths": (
        "Seborrheic Keratoses and Benign Growths",
        "Benign skin growths such as seborrheic keratoses, cysts, and dermatofibromas often appear as waxy, stuck-on tan/brown plaques or firm skin-colored bumps.",
        "https://dermnetnz.org/topics/seborrhoeic-keratosis",
    ),
    "Vascular Skin Lesions": (
        "Vascular Skin Lesions",
        "Vascular lesions such as cherry angiomas or hemangiomas typically appear as bright red, purple, or bluish spots or soft nodules.",
        "https://dermnetnz.org/topics/benign-skin-lesions",
    ),
    "Benign Moles and Nevi": (
        "Benign Moles and Nevi",
        "Common melanocytic nevi (moles) are typically round or oval with smooth borders and fairly uniform tan, brown, or skin-toned color.",
        "https://dermnetnz.org/topics/melanocytic-naevus",
    ),
    "Actinic Keratosis, Basal and Squamous Cell Carcinoma": (
        "Actinic Keratosis, Basal and Squamous Cell Carcinoma",
        "Sun-damaged or non-melanoma skin cancers may look like persistent rough scaly patches, pearly raised bumps with rolled edges, or sores that bleed and do not heal.",
        "https://www.aad.org/public/diseases/skin-cancer/basal-cell-carcinoma",
    ),
    "Melanoma and Suspicious Pigmented Lesions": (
        "Melanoma and Suspicious Pigmented Lesions",
        "Suspicious pigmented lesions can show asymmetry, irregular or notched borders, multiple colors, evolving size/shape, or stand out from surrounding moles.",
        "https://www.aad.org/public/diseases/skin-cancer/abcdes-melanoma",
    ),
    "Psoriasis, Lichen Planus and Related Conditions": (
        "Psoriasis, Lichen Planus and Related Conditions",
        "Psoriasis typically forms well-defined red plaques topped with thick silvery-white scales (often on elbows, knees, or scalp), while lichen planus causes flat-topped purplish itchy bumps.",
        "https://dermnetnz.org/topics/psoriasis",
    ),
    "Lupus and Connective Tissue Conditions": (
        "Lupus and Connective Tissue Conditions",
        "Connective tissue skin findings can include sun-sensitive red plaques, scaly discoid lesions, or butterfly-shaped facial redness.",
        "https://dermnetnz.org/topics/cutaneous-lupus-erythematosus",
    ),
    "Exanthems and Drug Eruptions": (
        "Exanthems and Drug Eruptions",
        "Viral exanthems and medication reactions typically appear as widespread, symmetrical red spots or papules across the trunk and limbs.",
        "https://dermnetnz.org/topics/drug-eruptions",
    ),
    "Urticaria (Hives)": (
        "Urticaria (Hives)",
        "Hives appear as raised, itchy, red or skin-colored welts (wheals) that shift shape and location over hours.",
        "https://dermnetnz.org/topics/acute-urticaria",
    ),
    "Vasculitis": (
        "Vasculitis",
        "Cutaneous vasculitis commonly presents as non-blanching red or purple spots (palpable purpura), most often on the lower legs.",
        "https://dermnetnz.org/topics/cutaneous-vasculitis",
    ),
    "Systemic Skin Manifestations": (
        "Systemic Skin Manifestations",
        "Internal or systemic medical conditions can produce skin signs such as localized plaques, firm nodules, discoloration, or rashes.",
        "https://dermnetnz.org/topics",
    ),
    # ---- Legacy HAM10000 7-class keys (backward compatibility) ----
    "akiec": (
        "Actinic keratosis or Bowen disease",
        "Actinic keratoses can appear as rough, scaly spots. Bowen disease can appear as a scaly patch.",
        "https://dermnetnz.org/cme/teledermatology-skin-cancer/clinical-features-of-skin-lesions",
    ),
    "bcc": (
        "Basal cell carcinoma",
        "Some basal cell carcinomas look like shiny, raised bumps. Others appear as flat, scaly areas.",
        "https://www.aad.org/public/diseases/skin-cancer/basal-cell-carcinoma",
    ),
    "bkl": (
        "Benign keratosis-like lesion",
        "This group includes several types of growth. One example is a waxy or warty patch that appears to sit on top of the skin.",
        "https://dermnetnz.org/topics/seborrhoeic-keratosis",
    ),
    "df": (
        "Dermatofibroma",
        "Dermatofibromas often look like small, firm bumps. Their color can vary.",
        "https://dermnetnz.org/topics/dermatofibroma",
    ),
    "mel": (
        "Melanoma",
        "Some melanomas have an uneven shape, irregular edges, or more than one color.",
        "https://www.aad.org/public/diseases/skin-cancer/abcdes-melanoma",
    ),
    "nv": (
        "Mole (melanocytic nevus)",
        "Common moles are often round or oval, with a fairly uniform color. They can be flat or raised.",
        "https://dermnetnz.org/topics/melanocytic-naevus",
    ),
    "vasc": (
        "Vascular skin lesion",
        "Some vascular lesions look red, blue, or purple. Appearance varies with the type of lesion.",
        "https://dermnetnz.org/topics/benign-skin-lesions",
    ),
}

# Also register legacy Dataset A raw folder names pointing to their clean guides.
for _raw_folder, _clean_fine in DATASET_A_FOLDER_TO_FINE.items():
    if _raw_folder not in CONDITION_GUIDES and _clean_fine in CONDITION_GUIDES:
        _name, _desc, _ref = CONDITION_GUIDES[_clean_fine]
        CONDITION_GUIDES[_raw_folder] = (_raw_folder, _desc, _ref)
