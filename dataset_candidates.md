# ISL Dataset Candidates (unverified — links only, not downloaded)

Found via web search on 2026-08-21. **Nothing here has been downloaded or content-inspected** — these are dataset pages/repos I could confirm exist via search/fetch, with metadata as described on their own pages. Verify before committing to one (check actual file structure, class list, and license terms firsthand).

---

## Pre-extracted landmarks (matches our pipeline exactly — highest synergy)

| Dataset | Scope | Access | Notes |
|---|---|---|---|
| [Indian Sign Language Hand Landmarks Dataset](https://www.kaggle.com/datasets/eraakash/indian-sign-language-hand-landmarks-dataset) (Kaggle) | A-Z alphabet, ~2k records/class, ~50k total | Free, Kaggle account | Landmarks already extracted (not raw video/images) — could plug directly into our feature format with far less preprocessing. Verify exact landmark schema matches or is convertible to MediaPipe's 21-point hand format. |
| [iSign / CISLR](https://huggingface.co/datasets/IIT-K/CISLR) (IIT-K, via Exploration-Lab) | ~4,700-word isolated recognition task, includes pose data alongside video | Free, Hugging Face | Part of the larger iSign benchmark (below). Pose data extraction already done for some portion — worth checking overlap with our target vocab. |

## Isolated-sign video (directly usable — one label per clip)

| Dataset | Scope | Access | Notes |
|---|---|---|---|
| [INCLUDE](https://huggingface.co/datasets/ai4bharat/INCLUDE) (AI4Bharat/IIT Madras) | 263 classes, 15 categories, 5,250 videos total (3,820 train / 425 val / 1,010 test) | **Free, no registration**, hosted on Zenodo, CC-BY-4.0 | Best-verified option — confirmed free, confirmed license, confirmed direct download. Also see [GitHub repo](https://github.com/AI4Bharat/INCLUDE) (pretrained models + code) and [project page](https://sign-language.ai4bharat.org/#/INCLUDE). |
| INCLUDE-50 (subset of INCLUDE) | 50 words, ~25 videos/word w/ augmentation | Same as above | Smaller, easier starting subset — good fit for our alphabet+20-30-word MVP scope specifically. |
| FDMSE-ISL (from [HWGAT paper](https://arxiv.org/abs/2407.14224)) | 2,002 words, 40,033 videos, 20 signers (gender-balanced), multi-view | **⚠ No public download link found** — paper describes the dataset but doesn't link a release. Would need to email the authors. | Largest/most rigorous dataset found, but access unconfirmed — don't rely on this without contacting authors first. |

## Continuous / sentence-level video (needs our own segmentation work to use)

| Dataset | Scope | Access | Notes |
|---|---|---|---|
| [ISL-CSLTR](https://data.mendeley.com/datasets/kcmpdxky7p/1) (SASTRA University + Navajeevan School for the Deaf) | 700 annotated videos, 100 sentences, 7 signers, plus 1,036 word-level images | Free, Mendeley Data. Also mirrored on [Kaggle](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset) | Includes word-level *images* too (see below) — mixed-format dataset. |
| [iSign](https://exploration-lab.github.io/iSign/) (main benchmark) | 10,000+ video-sentence/phrase pairs, continuous translation focus | Free, [Hugging Face](https://huggingface.co/datasets/Exploration-Lab/iSign) + OneDrive for some task files | Broader benchmark that CISLR (isolated words, above) is part of. |
| ISLTranslate | 118K+ video-sentence pairs | Referenced in search results, no link independently verified — check the [ResearchGate paper](https://www.researchgate.net/publication/372918922_ISLTranslate_Dataset_for_Translating_Indian_Sign_Language) for the actual release link | Largest sentence-level ISL dataset found, but I could not confirm a direct download URL — verify before relying on it. |

## Static images (alphabet / fingerspelling only)

| Dataset | Scope | Access | Notes |
|---|---|---|---|
| [ISL Alphabet Dataset](https://github.com/ayeshatasnim-h/Indian-Sign-Language-dataset) (GitHub) | 12,700 images, 26 letters A-Z | Free, GitHub | Directly usable for our alphabet-only MVP static poses. |
| [Indian Sign Language Alphabet Dataset](https://www.kaggle.com/datasets/rushilverma07/indian-sign-language-alphabet-dataset) (Kaggle) | Alphabet-focused | Free, Kaggle account | |
| [Indian Sign Language (ISL) - character level](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) (Kaggle) | Character-level | Free, Kaggle account | |
| [Indian Sign Language](https://www.kaggle.com/datasets/sujoy2000/indian-sign-language) (Kaggle) | Letter-based, ~38 classes (per search snippet, includes digits) | Free, Kaggle account | |
| [Indian Sign Language Dataset](https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset) (Kaggle) | Not independently verified beyond title | Free, Kaggle account | Lower confidence — check contents before use. |
| [Indian Sign Language Dataset](https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset) (Kaggle) | Not independently verified beyond title | Free, Kaggle account | Lower confidence — check contents before use. |
| Mendeley "Indian Sign Language_Dataset" | Not independently verified beyond title | Free, [Mendeley Data](https://data.mendeley.com/datasets/yx7kdssfjp/1) | Lower confidence — separate from ISL-CSLTR above, check contents before use. |

## Commercial / paid (flagging friction, probably skip for a hackathon)

| Dataset | Access | Notes |
|---|---|---|
| [GTS.ai ISL Dataset](https://gts.ai/dataset-download/indian-sign-language-isl-dataset-ai-data-collection/) | Requires submitting contact details via a lead-gen form; pricing unclear, "Get a Quote" for commercial use | Vendor/commercial data broker, not a research release — lowest priority given free alternatives above. |

## Reference / not directly datasets

- [AI4Bharat indicnlp_catalog issue #127](https://github.com/AI4Bharat/indicnlp_catalog/issues/127) — curated list this research partly drew from; also links several finger-spelling recognition *code* repos (not datasets): [Karthikeyu](https://github.com/Karthikeyu/Indian-sign-language-recognition), [imRishabhGupta](https://github.com/imRishabhGupta/Indian-Sign-Language-Recognition), [yatharth77](https://github.com/yatharth77/Indian-Sign-Language-Gesture-Recognition), [abhishekdudhal](https://github.com/abhishekdudhal/Indian-Sign-Language-Recognition-System) — worth a skim for preprocessing/training-code ideas, not for data itself.
- [IEEE DataPort — "Indian Sign Language" keyword search](https://ieee-dataport.org/keywords/indian-sign-language) — an index of multiple listings, some likely requiring IEEE membership; browse directly rather than treating as one dataset.

---

## My read, given our MVP scope (alphabet + ~20-30 common words)

**Most promising for immediate use:**
1. **INCLUDE-50** — smallest, cleanest, confirmed-free, confirmed-license isolated-word dataset; good chance several of its 50 words overlap with a sensible starter vocab.
2. **ISL alphabet image dataset(s)** (GitHub + Kaggle) — covers the alphabet half of the MVP scope directly, static images so no segmentation needed.
3. **eraakash's Hand Landmarks dataset (Kaggle)** — worth checking first specifically because it's *already* landmark-extracted, matching our pipeline's native format; if the schema is compatible (or convertible), this could skip our own extraction step entirely for the alphabet portion.
