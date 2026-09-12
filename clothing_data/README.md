# Trueform Clothing: retail strategy case study assets

This folder supports the fifth portfolio project, a case study built on a real
MSc coursework submission: an operational business proposal for Trueform
Clothing, an adaptive and inclusive clothing retailer in Norwich. The full
write-up is `trueform-case-study.html` at the root of the repository.

Unlike the other portfolio projects, this one is not built on synthetic or
reconstructed data. `Trueform_Clothing_Retail_Strategy_Report.pdf` is the
actual report submitted for the Retail Marketing and Management module
(University of East Anglia, MSc Business Management), which received a
distinction grade.

## Why there is a CSV here at all

The original report is a qualitative strategy document: it argues its
competitive position in prose rather than in a spreadsheet. Section 4,
Competitive Environment and Positioning, compares Trueform against four
alternatives a Norwich shopper actually has today, on two dimensions: price
and how adaptive the product range is.

`competitor_positioning.csv` turns that prose comparison into a small,
structured dataset, so it can be charted rather than just described:

| Column | Meaning |
|---|---|
| `retailer` | The retailer being compared |
| `price_tier` | 1 (budget) to 5 (premium), scored from the report's own price comparisons |
| `adaptive_specialisation` | 1 (none) to 5 (full specialist), scored from the report's own product comparisons |
| `note` | The specific claim and reference the score is drawn from |

Every score in the `note` column traces back to a sentence and a citation in
the original report. Nothing here is invented data standing in for real
figures, it is the report's own qualitative argument, made explicit enough to
plot.

## The charts

`generate_charts.py` reads `competitor_positioning.csv` and renders two
charts with Matplotlib, saved to `images/`. Run it yourself with:

```bash
python3 clothing_data/generate_charts.py
```

| File | Chart |
|---|---|
| `01_positioning_map.png` | Price positioning against adaptive specialisation, all five retailers |
| `02_pricing_ladder.png` | The same price tier, read as a single ranked ladder |

## The report images

`images/` also holds the concept renders and diagrams from the original
report, cropped from the submitted PDF for use on the case study page:

| File | From the report |
|---|---|
| `storefront.png` | Figure 1, storefront concept |
| `store_interior.png` | Figure 2, store interior concept |
| `trial_room.png` | Figure 3, accessible trial room layout |
| `product_designs.png` | Appendix A, adaptive product designs |
| `loyalty_card.png` | Appendix C, loyalty card concept |
| `five_s_original.png` | Appendix B, the Five S's fishbone diagram as submitted (the case study page recreates this framework as a clean web component; this file is kept for reference) |
