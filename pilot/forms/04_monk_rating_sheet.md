# Monk Skin Tone rating sheet (one or two raters)

The analysis supports **one rater** (`raters.n: 1`) or **two independent raters** (`raters.n: 2`).
With one rater the pilot reports the ratings but cannot report agreement; with two it reports
quadratic-weighted kappa, ICC(2,1) and the share of exact and within-one-shade matches, and uses the
mean of the two (or rater 1 only, `raters.consensus: rater1`).

## Procedure (the same for every participant)
1. Use the official Monk Skin Tone scale (https://skintone.google), the **same printed copy or the
   same screen** for every participant, under **light 1** (desk LED).
2. Rate the **cheek** (not the forehead or nose), from about 50 cm, without make-up correction.
3. Choose the single closest shade, 1 (lightest) to 10 (darkest). If between two shades, choose the
   darker one and write "between" in the notes.
4. With two raters: rate **independently**, without discussing and without seeing the other's
   sheet, and **before** any camera skin-tone estimate is shown.
5. Do not change a rating afterwards.

## Sheet

| Participant | Rater id | Monk (1–10) | Light | Notes |
|---|---|---|---|---|
| P01 | R1 | | light1_500lux | |
| P01 | R2 (optional) | | light1_500lux | |
| P02 | R1 | | light1_500lux | |
| P02 | R2 (optional) | | light1_500lux | |

Enter the rows in `pilot_data/monk_ratings.csv` exactly as above (columns `participant_id,
rater_id, monk, light, notes`). The participant's own rating goes in `participants.csv`
(`monk_self`), not here.

**Why Monk and not only Fitzpatrick:** the Fitzpatrick scale classifies sun reaction, not colour, and
annotators apply it inconsistently to darker skin. The Monk scale has 10 colour shades designed to
cover the full range. Neither replaces the colorimeter ITA° of the main study; the pilot reports how
the camera ITA° estimate agrees with the Monk ratings.
