# LaTeX Document Setup - Missing Files

This document requires the following image files to compile successfully:

## Required Figures (place in `figs/` directory):

1. **class_distribution_splits.png**
   - Class distribution across training, validation, and test splits
   - Expected: Bar chart showing box/bag/barcode instances in each split

2. **training_analysis.png**
   - 4-panel figure showing:
     - Instances per class (bar chart)
     - Images per class (bar chart)
     - Annotations per image (histogram)
     - Object size distribution (histogram)

3. **training_loss_curve.png**
   - Training and validation loss curves over 50 epochs
   - Should show convergence around epoch 8

4. **dataset_examples.png**
   - Real examples from your datasets
   - Left side: barcode examples (blue labels)
   - Right side: warehouse box examples (red boxes)

5. **qualitative_results.png**
   - Model prediction results
   - Preferably showing input, ground truth, and prediction side-by-side
   - Multiple rows for different object classes

## If figures are not ready:

You can temporarily disable figures for testing by commenting them out:
- Find lines with `\includegraphics`
- Add `%` before the line to comment it out
- Or comment the entire figure environment

## Bibliography

`references.bib` has been created with all required citations.

## How to run LaTeX:

```bash
cd ~/...AMLCV_Project_2025/REPORT/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The first pdflatex generates references, bibtex processes the bibliography, 
and the final two pdflatex runs resolve all references and citations.
